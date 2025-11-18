import asyncio
import os
import json
import time # Needed for buffer timing logic
import io
import base64 # Needed for TTS encoding
import numpy as np
import torch
import requests
from dotenv import load_dotenv
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import edge_tts # For TTS generation

# Check for torch device compatibility (for older torch versions)
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Model Setup ---
WHISPER_EN_MODEL = "Veronica1NW/en_whisper_nonstandard_small"
WHISPER_SW_MODEL = "Veronica1NW/sw_whisper_nonstandard_small"

TTS_VOICES = {
    'en': {'male': "en-KE-ChilembaNeural", 'female': "en-KE-AsiliaNeural"},
    'sw': {'male': "sw-KE-RafikiNeural", 'female': "sw-KE-RehemaNeural"}
}

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 5 # Transcription trigger time
MIN_PROCESS_SEC = 0.6
OVERLAP_SEC = 0.1

# --- Flask & SocketIO ---
app = Flask(__name__)
# NOTE: SECRET_KEY should be set in .env file for production
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key_if_not_set')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

client_sessions = {}

# --- Device ---
# Assuming these variables are initialized outside the function as per the original code structure
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load Processor and Models ---
    processor_en = WhisperProcessor.from_pretrained(WHISPER_EN_MODEL)
    processor_sw = WhisperProcessor.from_pretrained(WHISPER_SW_MODEL)
    model_en = WhisperForConditionalGeneration.from_pretrained(WHISPER_EN_MODEL).to(device)
    model_sw = WhisperForConditionalGeneration.from_pretrained(WHISPER_SW_MODEL).to(device)
    app.logger.info("Whisper models and processors loaded.")
except Exception as e:
    app.logger.error(f"Failed to load models: {e}")
    # Define placeholder functions/variables if models fail to load, for debugging
    device = torch.device("cpu")
    processor_en = processor_sw = None
    model_en = model_sw = None

def get_processor_and_model(lang):
    if lang.lower() == 'en' and processor_en:
        return processor_en, model_en
    elif lang.lower() == 'sw' and processor_sw:
        return processor_sw, model_sw
    else:
        # Fallback if models failed to load
        app.logger.error(f"Cannot get model for {lang}. Check model loading step.")
        raise ValueError(f"Unsupported or unloaded language: {lang}")

# --- TTS Helper ---
async def generate_and_stream_tts(text, language, gender, sid):
    """
    Generate TTS with edge_tts and stream base64-encoded audio chunks to client.
    """
    voice = TTS_VOICES.get(language, {}).get(gender, TTS_VOICES['en']['male'])
    app.logger.info(f"TTS start for {sid}: voice={voice}, text[:40]={text[:40]!r}")

    try:
        communicate = edge_tts.Communicate(text, voice)
        # iterate async generator returned by edge_tts
        async for chunk in communicate.stream():
            try:
                if chunk.get("type") == "audio" and chunk.get("data"):
                    # chunk["data"] is bytes
                    base64_audio = base64.b64encode(chunk["data"]).decode("utf-8")
                    # emit without unsupported kwargs (no 'binary' param)
                    socketio.emit("tts_audio_chunk", {"data": base64_audio}, room=sid)
                    app.logger.debug(f"Sent audio chunk to {sid}, size={len(chunk['data'])}")
            except Exception as inner:
                app.logger.exception(f"Error while sending tts chunk for {sid}: {inner}")
        # end of stream
        socketio.emit("tts_audio_end", room=sid)
        app.logger.info(f"TTS finished for {sid}")
    except Exception as e:
        app.logger.exception(f"TTS streaming error for session {sid}: {e}")
        socketio.emit("error", {"message": f"TTS error: {e}"}, room=sid)


# --- Audio decoding ---
def decode_audio_bytes(raw_bytes):
    try:
        # Attempt 1: 16-bit PCM (standard client format: signed 16-bit integers)
        # We normalize by 32768.0 to get floats between -1.0 and 1.0
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Fallback check (less likely to be hit now)
        if audio.size == 0 or np.abs(audio).max() < 1e-5:
            # Attempt 2: Float32 (if the client somehow sent raw floats)
            audio = np.frombuffer(raw_bytes, dtype=np.float32)
            
        return audio
    except Exception as e:
        app.logger.error(f"Audio decoding failed: {e}")
        return np.zeros(0, dtype=np.float32)

# --- Transcription Worker ---
def process_audio_buffer(sid):
    session = client_sessions.get(sid)
    if not session:
        return

    # Pop accumulated audio
    audio_buffer = session.pop('buffer', [])
    if not audio_buffer:
        return

    lang = session.get('lang', 'en')
    output_audio = session.get('output_audio', False)
    gender = session.get('gender', 'male')

    # Combine audio bytes
    try:
        combined_bytes = b"".join(audio_buffer)
    except Exception as e:
        app.logger.error(f"Failed to join audio buffer for {sid}: {e}")
        return

    # Decode to float32 numpy array
    audio_np = decode_audio_bytes(combined_bytes)
    
    # --- ENHANCED LOGGING AND RELAXED SILENCE CHECK ---
    
    buffer_size_bytes = len(combined_bytes)
    energy = np.abs(audio_np).mean()
    duration_sec = audio_np.shape[0] / SAMPLE_RATE
    
    app.logger.info(f"Audio Buffer for {sid}: Size={buffer_size_bytes} bytes, Duration={duration_sec:.2f}s, Energy={energy:.6f}")
    
    # RELAXED THRESHOLD: Check if energy is too low
    if energy < 0.00005: 
        app.logger.info(f"Silence or near-silence detected for {sid}, skipping transcription.")
        # Re-add buffer to avoid losing speech that starts after a pause
        session.setdefault('buffer', []).extend(audio_buffer)
        return

    if audio_np.size == 0:
        app.logger.warning(f"Decoded audio is empty for {sid}, skipping.")
        return

    if duration_sec < MIN_PROCESS_SEC:
        app.logger.warning(f"Audio too short ({duration_sec:.2f}s) for {sid}, skipping.")
        session.setdefault('buffer', []).extend(audio_buffer) # Keep in buffer for next chunk
        return

    try:
        # --- Load processor & model ---
        processor, model = get_processor_and_model(lang)

        # --- Convert audio to model input features (no padding, no attention mask) ---
        input_features = processor(audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)

        # --- Generate transcription ---
        pred_ids = model.generate(input_features)
        transcribed_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        app.logger.info(f"[{lang.upper()}] Transcription for {sid}: {transcribed_text}")
        socketio.emit('transcription_text', {'text': transcribed_text}, room=sid)

        session['buffer'] = [] # Clear buffer only on successful transcription

        app.logger.info(f"TTS trigger for {sid}: {output_audio}, text={transcribed_text[:30]}")
        # --- Optionally generate TTS ---
        if output_audio and transcribed_text.strip() and any(c.isalnum() for c in transcribed_text):
            socketio.start_background_task(
                lambda: asyncio.run(generate_and_stream_tts(transcribed_text, lang, gender, sid))
            )
        else:
            app.logger.info(f"Skipped TTS for {sid}: no meaningful text ('{transcribed_text}')")

    except Exception as e:
        import traceback
        app.logger.error(f"Transcription error for {sid}: {repr(e)}")
        app.logger.error(traceback.format_exc())
        socketio.emit('error', {'message': f"Transcription failed: {e}"}, room=sid)
        session.setdefault('buffer', []).extend(audio_buffer) # Keep buffer on error
        return

    # --- Keep overlap tail for smooth continuity ---
    try:
        tail_samples = int(OVERLAP_SEC * SAMPLE_RATE)
        # We need to account for 2 bytes per sample (int16)
        if audio_np.size * 2 > tail_samples * 2: 
            tail_bytes = combined_bytes[-(tail_samples * 2):] 
            session['buffer'] = [tail_bytes]
            session['buffer_start'] = time.time()
        else:
            session['buffer_start'] = None
    except Exception:
        session['buffer_start'] = None

# --- Flask Routes ---
@app.route('/')
def index():
    # In a real Flask app, you would have a templates/index.html file.
    # Since we are using an inline frontend, we return it here.
    return render_template('index.html')

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    client_sessions[sid] = {
        'buffer': [],
        'buffer_start': None,
        'lang': 'en',
        'output_audio': False,
        'gender': 'male',
        'last_chunk_time': time.time()
    }
    app.logger.info(f"Client connected: {sid}")
    emit('config_ack', {'sid': sid, 'message': 'Connection established. Ready for options.'}, room=sid)

@socketio.on('set_options')
def handle_set_options(data):
    sid = request.sid
    session = client_sessions.get(sid)
    if not session:
        return
    session['lang'] = data.get('lang', session['lang']).lower()
    session['output_audio'] = data.get('output', 'text_only').lower() == 'audio'
    session['gender'] = data.get('gender', session['gender']).lower()
    app.logger.info(f"Session {sid} options updated: {session['lang']}, Audio: {session['output_audio']}, Gender: {session['gender']}")
    emit('options_updated', {'lang': session['lang'], 'output_audio': session['output_audio']}, room=sid)

@socketio.on('audio_chunk')
def handle_audio_chunk(*args):
    data = args[0] if len(args) > 0 else None
    sid = request.sid
    if sid not in client_sessions:
        client_sessions[sid] = {'buffer': [], 'buffer_start': None, 'lang':'en', 'output_audio':False,'gender':'male'}

    session = client_sessions[sid]

    # Convert incoming data (which might be memoryview or ArrayBuffer) to bytes
    if isinstance(data, (memoryview, bytearray)):
        chunk_bytes = bytes(data)
    elif isinstance(data, bytes):
        chunk_bytes = data
    else:
        # Attempt to convert other types if necessary
        try:
            chunk_bytes = bytes(data)
        except Exception:
            app.logger.error(f"Unsupported audio chunk type from {sid}: {type(data)}")
            return

    session.setdefault('buffer', []).append(chunk_bytes)
    session['last_chunk_time'] = time.time()

    buffer_start = session.get('buffer_start')
    if buffer_start is None:
        buffer_start = session['last_chunk_time']
        session['buffer_start'] = buffer_start
    
    # Check if we have enough data to trigger transcription (5 seconds)
    if time.time() - buffer_start >= CHUNK_DURATION_SEC:
        socketio.start_background_task(process_audio_buffer, sid)
        # Reset buffer start time after triggering processing
        session['buffer_start'] = None 

@socketio.on('stop_recording')
def handle_stop_recording():
    sid = request.sid
    # Force process any remaining buffer when recording stops
    socketio.start_background_task(process_audio_buffer, sid)
    app.logger.info(f"Forcing final transcription process for {sid}.")


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    client_sessions.pop(sid, None)
    app.logger.info(f"Client disconnected: {sid}")

if __name__ == '__main__':
    if not app.config['SECRET_KEY']:
        app.logger.warning("SECRET_KEY not set. Using default.")
    print("Starting Flask-SocketIO server...")
    # Using eventlet or gevent is required for async_mode
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=app.config['DEBUG'])
    except Exception as e:
        print(f"Server startup failed: {e}")