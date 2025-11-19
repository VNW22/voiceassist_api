import asyncio
import os
import json
from pathlib import Path
import pydub
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import numpy as np
import torch
import webrtcvad
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ---------------------- CONFIG ----------------------
# Set these in your environment or .env file
# os.environ["AZURE_SPEECH_KEY"] = "..."
# os.environ["STREAMING_AUTH_TOKEN"] = "..."

TARGET_SAMPLE_RATE = 16000
MODEL_DIR = Path("./models")  # Local directory for models
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------- AUDIO HANDLER UTILS ----------------------
def rms(audio_segment: pydub.AudioSegment):
    samples = np.array(audio_segment.get_array_of_samples())
    return np.sqrt(np.mean(samples.astype(np.float32) ** 2))

vad = webrtcvad.Vad()
vad.set_mode(2)

async def handle_audio_chunk(service, chunk: bytes, buffer: pydub.AudioSegment = None,
                             current_lang: str = "en",
                             rms_threshold=500, min_speech_ms=300, max_speech_ms=3000):
    if buffer is None:
        buffer = pydub.AudioSegment.empty()
        
    if chunk is None or len(chunk) == 0:
        return buffer, None

    if len(chunk) % 2 != 0:
        chunk += b"\0"

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

    if len(buffer) > max_speech_ms:
        audio_bytes = buffer.raw_data
        try:
            # Pass the specific language to the global service
            transcription = service.transcribe(audio_bytes, current_lang)
        except Exception as e:
            print("Transcription error:", e)
            transcription = None

        overlap_ms = 300
        buffer = buffer[-overlap_ms:] if len(buffer) > overlap_ms else pydub.AudioSegment.empty()

        return buffer, transcription.strip() if transcription else None

    return buffer, None


# ---------------------- GLOBAL MODEL SERVICE ----------------------
class WhisperService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading models on {self.device}...")
        self.load_models()

    def maybe_download_whisper_model(self, model_storage_dir: Path, model_id: str):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        model_dir = model_storage_dir / model_id.replace("/", "_")
        if not model_dir.exists():
            print(f"Downloading Whisper model {model_id}...")
            processor = WhisperProcessor.from_pretrained(model_id)
            model = WhisperForConditionalGeneration.from_pretrained(model_id)
            processor.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            print("✅ Model downloaded.")
        else:
            print(f"Model '{model_id}' already exists locally.")
        return model_dir

    def load_models(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        # Load English
        en_path = self.maybe_download_whisper_model(MODEL_DIR, "Veronica1NW/en_whisper_nonstandard_medium")
        self.processor_en = WhisperProcessor.from_pretrained(en_path)
        self.model_en = WhisperForConditionalGeneration.from_pretrained(en_path).to(self.device)
        print("✅ English model ready.")

        # Load Swahili
        sw_path = self.maybe_download_whisper_model(MODEL_DIR, "Veronica1NW/sw_whisper_nonstandard_medium")
        self.processor_sw = WhisperProcessor.from_pretrained(sw_path)
        self.model_sw = WhisperForConditionalGeneration.from_pretrained(sw_path).to(self.device)
        print("✅ Swahili model ready.")

    def transcribe(self, audio_bytes, lang: str):
        processor, model = (self.processor_sw, self.model_sw) if lang == "sw" else (self.processor_en, self.model_en)
        
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

    def azure_tts(self, text: str, lang: str, gender: str) -> bytes:
        import azure.cognitiveservices.speech as speechsdk

        TTS_VOICES = {
            "en": {"male": "en-KE-ChilembaNeural", "female": "en-KE-AsiliaNeural"},
            "sw": {"male": "sw-KE-RafikiNeural", "female": "sw-KE-ZuriNeural"},
        }

        speech_key = os.environ.get("AZURE_SPEECH_KEY")
        service_region = "eastus"
        
        if not speech_key:
            print("⚠️ AZURE_SPEECH_KEY not found.")
            return None

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

        lang_code = lang if lang in TTS_VOICES else "en"
        gender_key = gender if gender in ["male", "female"] else "female"
        
        voice_name = TTS_VOICES[lang_code][gender_key] 
        speech_config.speech_synthesis_voice_name = voice_name

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        print(f"🗣️ Azure TTS ({voice_name}): '{text[:30]}...'")
        
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            print(f"⚠️ TTS Failed: {result.reason}")
            return None

# ---------------------- FASTAPI APP ----------------------

# Global variable to hold the service
whisper_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global whisper_service
    whisper_service = WhisperService()
    yield
    # Clean up on shutdown (if needed)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def status():
    return Response(status_code=200)

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    REQUIRED_TOKEN = os.environ.get("STREAMING_AUTH_TOKEN")
    
    # Connection State
    current_lang = "en"
    current_mode = "text"
    current_gender = "female"

    try:
        message = await ws.receive()
    except Exception:
        await ws.close()
        return

    if "text" in message and message["text"]:
        try:
            init_data = json.loads(message["text"])
            
            client_token = init_data.get("auth_token")
            if not client_token or client_token != REQUIRED_TOKEN:
                await ws.close(code=1008, reason="Unauthorized")
                return 
            
            current_lang = init_data.get("lang", "en").lower()
            current_mode = init_data.get("mode", "text") 
            current_gender = init_data.get("gender", "female").lower()

            print(f"✅ Auth Success. Lang: {current_lang}, Mode: {current_mode}, Gender: {current_gender}")

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

            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                if len(chunk) == 0: continue
                    
                # Pass current_lang to the handler
                buffer, text = await handle_audio_chunk(
                    whisper_service, 
                    chunk, 
                    buffer, 
                    current_lang=current_lang
                )
                
                if text:
                    print("📝 Transcribed:", text)
                    await ws.send_text(text)
                    
                    if current_mode == "audio+text":
                        try:
                            # Use global service for TTS
                            tts_audio_bytes = whisper_service.azure_tts(text, current_lang, current_gender)
                            if tts_audio_bytes:
                                await ws.send_bytes(tts_audio_bytes)
                        except Exception as e:
                            print(f"⚠️ TTS error: {e}")
            
            elif "text" in message and message["text"] is not None:
                try:
                    update_data = json.loads(message["text"])
                    
                    if "lang" in update_data:
                        current_lang = update_data["lang"].lower()
                    
                    if "mode" in update_data:
                        current_mode = update_data["mode"]
                    
                    if "gender" in update_data:
                        current_gender = update_data["gender"].lower()

                    print(f"🌐 Config Updated: {current_lang}, {current_mode}, {current_gender}")
                    
                    await ws.send_text(json.dumps({
                        "type": "config_updated",
                        "lang": current_lang,
                        "mode": current_mode,
                        "gender": current_gender
                    }))
                    
                except Exception as e:
                    print(f"⚠️ Config update ignored: {e}")

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected.")
    except Exception as e:
        try: await ws.close(code=1011, reason=str(e)[:100])
        except: pass

if __name__ == "__main__":
    import uvicorn
    # Run on all interfaces (0.0.0.0) at port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)