# Voicessist API

Voicessist is a real-time speech-to-text (STT) and text-to-speech (TTS) API built to help speech-impaired users communicate clearly. It supports Kenyan English and Swahili and is optimized for non-standard speech patterns. The API is hosted on Modal for fast, scalable cloud access.

## Features

- Real-time STT (speech → text)
- Real-time TTS (text → speech)
- Supports Kenyan English + Swahili
- Optimized for speech-impaired users
- Hosted on Modal for reliable cloud inference

## Models

This API uses finetuned Hugging Face Whisper models:

- English: `Veronica1NW/en_whisper_nonstandard_medium`
- Swahili: `Veronica1NW/sw_whisper_nonstandard_medium`

> Models are downloaded automatically at runtime via Hugging Face Hub.

## WebSocket Endpoint

```
wss://<your-modal-endpoint>
```

The API communicates over WebSocket for real-time streaming of audio chunks.

## Sending Audio Chunks

To get accurate real-time transcription, send audio in small chunks:

- **Format:** 16-bit PCM WAV or raw audio
- **Sample rate:** 16 kHz recommended
- **Chunk length:** ~1–3 seconds (2500–5000 ms)
- **Encoding:** Base64

**Example payload:**

```json
{
  "type": "audio_chunk",
  "audio": "<base64-encoded-audio>"
}
```

**Optional control messages:**

```json
{"type": "start_stream"}  // Begin streaming
{"type": "end_stream"}    // End streaming and force final transcription/TTS
```

## Receiving Responses

The API sends JSON responses for each chunk.

**Transcription:**

```json
{
  "type": "transcription",
  "text": "Hello, how are you?"
}
```

**TTS audio:**

```json
{
  "type": "tts",
  "audio": "<base64-encoded-audio>"
}
```

You can play the TTS audio immediately on the client.

## Setup / Installation

Even though the API is hosted on Modal, you can run a local test:

```bash
git clone https://github.com/YOUR_USERNAME/voicessist-api.git
cd voicessist-api
pip install -r requirements.txt
```

## Python Usage Example

```python
import asyncio
import websockets
import base64

async def send_audio():
    uri = "wss://<your-modal-endpoint>"
    async with websockets.connect(uri) as ws:
        # Start streaming
        await ws.send('{"type":"start_stream"}')

        # Send audio chunk
        with open("chunk.wav", "rb") as f:
            chunk_bytes = f.read()
        chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")
        await ws.send(f'{{"type":"audio_chunk","audio":"{chunk_b64}"}}')

        # Receive transcription
        response = await ws.recv()
        print(response)

        # End streaming
        await ws.send('{"type":"end_stream"}')

asyncio.run(send_audio())
```

## Notes for Users

- Always send audio in small, consistent chunks.
- Keep sample rate at 16 kHz for best performance.
- Base64-encode audio before sending over WebSocket.
- The API automatically detects and handles English and Swahili speech.



Clone this repo:https://github.com/VNW22/voicessist-api.git
cd voicessist-api
