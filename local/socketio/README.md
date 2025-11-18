Real-Time Whisper ASR and Edge-TTS Server

This project implements a real-time Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) server using Whisper (via Hugging Face Transformers) for transcription and Edge-TTS for audio response. It utilizes Flask-SocketIO for low-latency, bidirectional audio streaming.

🚀 Getting Started

Follow these steps to set up and run the server on your local machine.

Prerequisites

Python 3.8+

Git

GPU (Optional but Highly Recommended): The Whisper models are computationally intensive. For real-time performance, a CUDA-enabled GPU is strongly recommended.

1. Repository Setup

Clone the repository and navigate into the project directory:

git clone <your-repository-url>
cd <your-repository-name>


2. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

# Create a virtual environment
python3 -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
.\venv\Scripts\activate


3. Install Dependencies

Install all required Python packages using the requirements.txt file:

pip install -r requirements.txt


4. Configure Environment Variables

This project requires a .env file for configuration, specifically for the Flask secret key and potentially for Hugging Face authentication if your models are private.

Create a file named .env in the root directory of the project and add the following lines:

# .env file

# Required for Flask-SocketIO security
SECRET_KEY="your_strong_random_secret_key_here"

# Optional: Set to False for production
FLASK_DEBUG=True

# OPTIONAL: If using private models on Hugging Face, include your token
# HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"


5. Running the Server

Start the Flask-SocketIO server. The application will automatically attempt to load the fine-tuned Whisper models from the specified Hugging Face repository IDs:

WHISPER_EN_MODEL: Veronica1NW/en_whisper_nonstandard_small

WHISPER_SW_MODEL: Veronica1NW/sw_whisper_nonstandard_small

<!-- end list -->

python your_main_script_name.py


The server will typically start on http://0.0.0.0:5000. You can access the application via your browser using http://localhost:5000/.

6. Client Connection

This server acts as a backend API and does not include a frontend. To use the application, you need a separate frontend client (e.g., an HTML page with JavaScript, or a mobile app) that connects to the server via SocketIO:

Server Address: The preferred address for client applications (especially those running in Chrome/Google-based browsers which require secure contexts for microphone access) is ws://localhost:5000. You can also use ws://127.0.0.1:5000 or ws://your-ip-address:5000.

Note on Security: Modern browsers like Chrome often enforce strict security policies, requiring audio/microphone access to be initiated from a secure origin (https:). When testing locally, http://localhost is usually the only non-secure HTTP address that is treated as a secure context, allowing microphone access without https.

Key Events:

Outgoing (Client to Server):

connect: Initiates connection.

set_options: Sends configuration (lang, output, gender).

audio_chunk: Streams raw audio bytes (16-bit PCM, 16000 Hz sample rate).

stop_recording: Triggers final processing of the audio buffer.

Incoming (Server to Client):

transcription_text: Receives the transcribed text.

tts_audio_chunk: Receives a base64-encoded chunk of synthesized audio.

tts_audio_end: Signals the end of the TTS stream.

error: Receives operational errors.