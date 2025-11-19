import pydub
import numpy as np
import webrtcvad

vad = webrtcvad.Vad()
vad.set_mode(2)
TARGET_SAMPLE_RATE = 16000

def rms(audio_segment: pydub.AudioSegment):
    samples = np.array(audio_segment.get_array_of_samples())
    return np.sqrt(np.mean(samples.astype(np.float32) ** 2))

async def handle_audio_chunk(transcriber, chunk: bytes, buffer: pydub.AudioSegment = None,
                             rms_threshold=100, min_speech_ms=2000, max_speech_ms=5000):
    if buffer is None:
        buffer = pydub.AudioSegment.empty()
    
    if len(chunk) % 2 != 0:
        chunk += b"\0"
    segment = pydub.AudioSegment(data=chunk, sample_width=2, channels=1, frame_rate=TARGET_SAMPLE_RATE)

    if rms(segment) < rms_threshold:
        return buffer, None

    buffer += segment

    samples = np.array(buffer.get_array_of_samples())
    samples16 = (samples.astype(np.int16)).tobytes()
    frame_size = int(TARGET_SAMPLE_RATE * 30 / 1000 * 2)
    speech_detected = any(vad.is_speech(samples16[i:i+frame_size], sample_rate=TARGET_SAMPLE_RATE)
                          for i in range(0, len(samples16), frame_size) if len(samples16[i:i+frame_size]) == frame_size)
    
    if not speech_detected:
        buffer = buffer[-1500:] if len(buffer) > 1500 else pydub.AudioSegment.empty()
        return buffer, None

    if len(buffer) >= min_speech_ms or len(buffer) >= max_speech_ms:
        audio_bytes = buffer.raw_data
        try:
            transcription = transcriber.transcribe(audio_bytes)
        except Exception as e:
            print("Transcription error:", e)
            transcription = None
        buffer = pydub.AudioSegment.empty()
        return buffer, transcription.strip() if transcription else None

    return buffer, None
