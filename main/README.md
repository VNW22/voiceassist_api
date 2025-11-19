# 🎙️ Streaming Voice Agent (FastAPI + Whisper + Azure)

A real-time Speech-to-Text and Text-to-Speech agent supporting **Kenyan English** and **Swahili** (Male/Female voices).

## ⚡ Features
- **STT:** Fine-tuned Whisper Medium (running locally on GPU/CPU).
- **TTS:** Azure Cognitive Services (Neural Voices).
- **Streaming:** WebSocket-based real-time communication.
- **VAD:** Voice Activity Detection to ignore silence.

## 🛠️ Setup

### 1. Install System Dependencies
You need `ffmpeg` installed on your machine.
- **Ubuntu:** `sudo apt install ffmpeg libasound2-dev`
- **Mac:** `brew install ffmpeg`

### 2. Install Python Requirements
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


API Endpoints
WebSocket: ws://localhost:8000/ws

Status: GET http://localhost:8000/status