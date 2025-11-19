from pathlib import Path
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

TARGET_SAMPLE_RATE = 16000

def maybe_download_whisper_model(model_storage_dir: Path, model_id: str):
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

def transcribe(audio_bytes, processor, model, device="cpu"):
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    if np.max(np.abs(audio_data)) > 0:
        audio_data /= np.max(np.abs(audio_data))
    inputs = processor(audio_data, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True).input_features.to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=128, num_beams=5, repetition_penalty=2.0, no_repeat_ngram_size=3)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]
