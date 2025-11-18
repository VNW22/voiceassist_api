import os
import json
import numpy as np
import torch
import requests
import pydub
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# --- CONFIG ---
TARGET_SAMPLE_RATE = 16000
# IMPORTANT: Ensure these environment variables are set in your execution environment!
REQUIRED_TOKEN = os.environ.get("STREAMING_AUTH_TOKEN")
# ----------------------

# --- GLOBAL AUDIO HANDLERS ---

vad = webrtcvad.Vad()
vad.set_mode(2) # 0-3, 3 is most aggressive

def rms(audio_segment: pydub.AudioSegment):
    """Calculates the Root Mean Square (loudness) of an audio segment."""
    samples = np.array(audio_segment.get_array_of_samples())
    return np.sqrt(np.mean(samples.astype(np.float32) ** 2))

async def handle_audio_chunk(transcriber, chunk: bytes, buffer: pydub.AudioSegment = None,
                             rms_threshold=500, min_speech_ms=300, max_speech_ms=3000):
    """
    Processes an incoming 16kHz PCM audio chunk for VAD, buffering, and transcription.
    """
    if buffer is None:
        buffer = pydub.AudioSegment.empty()

    if len(chunk) % 2 != 0:
        chunk += b"\0"

    # Convert raw 16kHz PCM bytes to pydub segment
    segment = pydub.AudioSegment(
        data=chunk,
        sample_width=2,
        channels=1,
        frame_rate=TARGET_SAMPLE_RATE
    )

    if rms(segment) < rms_threshold:
        return buffer, None

    buffer += segment
    
    if len(buffer) < min_speech_ms:
        return buffer, None

    # Run VAD
    samples = np.array(buffer.get_array_of_samples())
    samples16 = (samples.astype(np.int16)).tobytes()
    frame_duration = 30
    frame_size = int(TARGET_SAMPLE_RATE * frame_duration / 1000 * 2)
    speech_detected = False

    for i in range(0, len(samples16), frame_size):
        frame = samples16[i:i+frame_size]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate=TARGET_SAMPLE_RATE):
            speech_detected = True
            break

    if not speech_detected:
        overlap_ms = 300
        buffer = buffer[-overlap_ms:] if len(buffer) > overlap_ms else pydub.AudioSegment.empty()
        return buffer, None

    # Transcribe if speech is confirmed and buffer exceeds max time
    if len(buffer) > max_speech_ms:
        audio_bytes = buffer.raw_data
        try:
            transcription = transcriber.transcribe(audio_bytes)
        except Exception as e:
            print(f"Transcription error: {e}")
            transcription = None

        overlap_ms = 300
        buffer = buffer[-overlap_ms:] if len(buffer) > overlap_ms else pydub.AudioSegment.empty()

        return buffer, transcription.strip() if transcription else None

    return buffer, None


# --- ASR/TTS CLASS (Standard Python Class) ---

class WhisperASR:
    """Handles model loading, transcription, language setting, and TTS calls."""
    def __init__(self):
        # NOTE: Model paths assume standard HuggingFace caching or pre-installed models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.current_lang = "en"
        
        # Load English model (using model ID directly for portability)
        en_model_id = "Veronica1NW/en_whisper_nonstandard_medium"
        self.processor_en = WhisperProcessor.from_pretrained(en_model_id)
        self.model_en = WhisperForConditionalGeneration.from_pretrained(en_model_id).to(device)
        print("✅ English model ready.")

        # Load Swahili model
        sw_model_id = "Veronica1NW/sw_whisper_nonstandard_medium"
        self.processor_sw = WhisperProcessor.from_pretrained(sw_model_id)
        self.model_sw = WhisperForConditionalGeneration.from_pretrained(sw_model_id).to(device)
        print("✅ Swahili model ready.")

    def eleven_tts(self, text: str, voice_id: str = "SAz9YHcvj6GT2YYXdXww") -> bytes:
        """Calls the Eleven Labs API for TTS."""
        api_key = os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVEN_API_KEY is not set.")
            
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/wav",
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "text": text,
            "voice_settings": {"stability": 0.75, "similarity_boost": 0.75},
            "model_id": "eleven_multilingual_v2"
        })
        
        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 200:
            return response.content
        else:
            raise RuntimeError(f"TTS request failed: {response.status_code}, {response.text}")

    def set_language(self, lang: str):
        self.current_lang = lang.lower()

    def transcribe(self, audio_bytes):
        processor, model = (self.processor_sw, self.model_sw) if self.current_lang == "sw" else (self.processor_en, self.model_en)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data /= np.max(np.abs(audio_data))

        inputs = processor(audio_data, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_features.to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=128,      
                num_beams=5,             
                repetition_penalty=2.0,  
                no_repeat_ngram_size=3   
            )
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return transcription

# --- FASTAPI APP INITIALIZATION ---

app = FastAPI(title="Streaming ASR API")
transcriber = WhisperASR() # Instantiate the class once

# Global CORS settings for maximal compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def status():
    """Health check endpoint."""
    return Response(status_code=200)

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    
    # --- SECURITY: TOKEN VALIDATION ---
    message = await ws.receive()
    if "text" in message and message["text"]:
        try:
            init_data = json.loads(message["text"])
            client_token = init_data.get("auth_token")
            
            if not client_token or client_token != REQUIRED_TOKEN:
                print(f"🚨 Authentication failed. Client token: {client_token}")
                await ws.close(code=1008, reason="Unauthorized: Invalid Token")
                return 
            
            # Auth successful, set initial state
            lang = init_data.get("lang", "en").lower()
            mode = init_data.get("mode", "text") 
            transcriber.set_language(lang)
            print(f"✅ Auth Success. Language: {lang}, Mode: {mode}")

        except json.JSONDecodeError:
            await ws.close(code=1008, reason="Bad Request")
            return
    else:
        await ws.close(code=1008, reason="Missing Initial Config")
        return

    buffer = pydub.AudioSegment.empty()
    print("🔗 WebSocket connection secured.")

    try:
        while True:
            message = await ws.receive()

            if "bytes" in message:
                chunk = message["bytes"]
                buffer, text = await handle_audio_chunk(transcriber, chunk, buffer)
                
                if text:
                    await ws.send_text(text)
                    
                    if mode == "audio+text":
                        tts_audio_bytes = None
                        try:
                            tts_audio_bytes = transcriber.eleven_tts(text)
                            if tts_audio_bytes:
                                await ws.send_bytes(tts_audio_bytes)
                        except Exception as e:
                            print(f"⚠️ TTS error: {e}")
            
            elif "text" in message:
                # Allows mid-stream config updates
                try:
                    init_data = json.loads(message["text"])
                    lang = init_data.get("lang", lang).lower()
                    mode = init_data.get("mode", mode) 
                    transcriber.set_language(lang)
                    print(f"🌐 Updated config: {lang}, {mode}")
                except:
                    pass

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected.")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        # Note: Added print to ensure error visibility
        import traceback
        traceback.print_exc()
        try:
            await ws.close(code=1011, reason=str(e))
        except Exception:
            pass

# --- STANDARD ENTRY POINT ---
if __name__ == "__main__":
    # Ensure this only runs if the environment variables are set
    if not REQUIRED_TOKEN:
        print("FATAL: STREAMING_AUTH_TOKEN environment variable is not set. Cannot run.")
    elif not os.environ.get("ELEVEN_API_KEY"):
         print("WARNING: ELEVEN_API_KEY is not set. TTS calls will fail.")
         uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run Uvicorn server (typical for local development)
        uvicorn.run(app, host="0.0.0.0", port=8000)