
---

# Voicessist API

![Modal Hosted](https://img.shields.io/badge/Hosted%20on-Modal-blueviolet)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ASR](https://img.shields.io/badge/Speech--to--Text-Whisper%20Finetuned-success)
![Languages](https://img.shields.io/badge/Languages-Kenyan%20English%20%26%20Swahili-yellow)
![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-orange)
![WebSocket](https://img.shields.io/badge/API-WebSocket%20Streaming-informational)
![Purpose](https://img.shields.io/badge/Purpose-Speech%20Impaired%20Accessibility-%23FF69B4)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/github/license/VNW22/voicessist-api)
![Stars](https://img.shields.io/github/stars/VNW22/voicessist-api?style=social)

---

Voicessist is a real-time Speech-to-Text (STT) and Text-to-Speech (TTS) API designed for **speech-impaired users**.
It supports **Kenyan English** and **Swahili**, and is optimized for **non-standard speech patterns**.
The service is fully hosted on **Modal**, providing low-latency, scalable inference.

---

                

## 🚀 Features

* Real-time **STT** (speech → text)
* Real-time **TTS** (text → speech)
* Supports **Kenyan English + Swahili**
* Optimized for **non-standard and impaired speech**
* WebSocket-based streaming
* Hosted on **Modal** for reliable cloud inference

---

## 🧠 Models Used

* **English Whisper (Finetuned):** `Veronica1NW/en_whisper_nonstandard_medium`
* **Swahili Whisper (Finetuned):** `Veronica1NW/sw_whisper_nonstandard_medium`

Models are automatically downloaded from Hugging Face at runtime.

---

## 🌐 WebSocket Endpoint (Modal Hosted)

Use this WebSocket URL to connect to the API:

```javascript
const WS_URL = "wss://veronicahwags-cdli--streaming-whisper-whisperasr-web.modal.run/ws";
```

This endpoint handles streaming audio input and returns:

* live STT transcriptions
* real-time TTS audio (base64)

---

## 🎧 Sending Audio Chunks

To send streaming audio:

* **Encoding:** Base64
* **Format:** raw audio or PCM WAV
* **Recommended rate:** 16 kHz
* **Chunk size:** ~1–3 seconds

### Audio Chunk Payload

```json
{
  "type": "audio_chunk",
  "audio": "<base64-encoded-audio>"
}
```

### Stream Control Messages

```json
{"type": "start_stream"}
{"type": "end_stream"}
```

---

## 📥 Receiving Responses

### Transcription Response

```json
{
  "type": "transcription",
  "text": "Hello, how are you?"
}
```

### TTS Audio Response

```json
{
  "type": "tts",
  "audio": "<base64-encoded-audio>"
}
```

---

## 📂 Frontend / Backend Compatibility

All detailed instructions for:

* formatting audio for the backend
* converting microphone audio to **16 kHz**
* streaming PCM data
* implementing STT + TTS with **FastAPI** and **Uvicorn**

are located in the **`main/` folder** of this repository.

> The `main/` folder provides the full working example of how the STT + TTS pipeline is implemented.

---

## 🔧 Local Installation

```bash
git clone https://github.com/VNW22/voicessist-api.git
cd voicessist-api
pip install -r requirements.txt
```

---

## 🐍 Python Example Client

```python
import asyncio
import websockets
import base64

async def stream():
    uri = "wss://veronicahwags-cdli--streaming-whisper-whisperasr-web.modal.run/ws"
    async with websockets.connect(uri) as ws:

        await ws.send('{"type": "start_stream"}')

        with open("chunk.wav", "rb") as f:
            chunk = f.read()

        b64 = base64.b64encode(chunk).decode()
        await ws.send(f'{{"type":"audio_chunk","audio":"{b64}"}}')

        print(await ws.recv())

        await ws.send('{"type":"end_stream"}')

asyncio.run(stream())
```

---


error: Receives operational errors.