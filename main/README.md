# FRONTEND INSTRUCTIONS

This document explains the critical components of the client-side JavaScript for handling audio streaming, mode switching, and secure authentication to ensure high-accuracy, reliable communication with the backend service.

## 1. Audio Conversion for Backend Acceptance

The backend, which is typically configured to accept raw audio data for highly accurate transcription, expects a very specific format. The frontend is responsible for capturing raw microphone data and converting it to the required specification: 16-bit Pulse Code Modulation (PCM) at 16kHz.

| Aspect | Frontend Implementation | Importance to Backend |
| :--- | :--- | :--- |
| **Target Sample Rate** | `const SAMPLE_RATE = 16000;` | Backend must accept audio at **16 kHz**, which is standard for accurate ASR models. |
| **Conversion Process** | `startRecording` uses `processor.onaudioprocess` to capture mic audio. | Ensures raw audio is captured in real time for transformation. |
| **Downsampling** | `const downsampled = downsampleBuffer(inputData, nativeSampleRate, SAMPLE_RATE);` | Converts audio from 44.1kHz/48kHz → **16kHz**. Backend expects exactly 16kHz input. |
| **Format Encoding** | `const pcm16 = floatTo16BitPCM(downsampled);` | Converts 32-bit float → **16-bit PCM**, the raw format your backend consumes. |
| **Transmission** | `ws.send(pcm16); // Send raw 16kHz binary data` | Sends continuous **16-bit PCM binary stream** via WebSocket. Backend reads it as a stream of 16-bit integers. |


## 2. Handling Text Only vs. Text + Audio Modes

The frontend communicates the desired response type to the backend via the initial configuration message, allowing the backend to know whether to perform Speech-to-Text (STT) only or STT followed by Text-to-Speech (TTS).
```javascript
Mode Element: The <select id="modeSelect"> element controls the behavior.
```
Configuration Message (Crucial!): The sendConfig() function transmits the selected mode (and language) immediately upon connection and whenever the selection changes:
```javascript
const config = {
    lang: languageSelect.value,
    mode: modeSelect.value, // This is the key value for the backend ('text' or 'audio+text')
    auth_token: AUTH_TOKEN 
};
ws.send(JSON.stringify(config));
```

Data Reception: The ws.onmessage handler uses the data type to distinguish between messages:

String Data: Handled as the text transcription and displayed in the text box. (Used in both modes).

Binary Data (ArrayBuffer): Handled as audio chunks and pushed to the audioQueue. (Used only in audio+text mode).

## 3. Text Output Box

The text area provides the primary output for the user.

HTML Element: <textarea id="transcriptionBox" readonly placeholder="Transcription will appear here..."></textarea>

Update Logic: Any string message received from the server (which is the transcription) is immediately written to this box, providing a real-time feedback mechanism.

## 4. 🛑 Critical Note: TTS Playback Button

To ensure the microphone only captures the user's speech and prevents acoustic echo cancellation (AEC) issues—where the output audio is transcribed back into the input—playback is deliberately made manual.

Manual Trigger: The TTS audio received from the backend is only queued (audioQueue.push(event.data);) and is NOT played immediately.

Clickable Button: The playback is initiated only when the user clicks the <button id="playTTSButton">.

Recording Interlock: The play button is disabled during the entire recording phase (startRecording), ensuring no sound is emitted while the microphone is active. This is vital for maintaining the accuracy of the Speech-to-Text (STT) process.

## 5. Security Instructions (Secure Authentication)

The frontend must provide a shared secret to the WebSocket service for basic authentication. This is passed alongside the configuration settings.

Auth Token Variable: Locate and REPLACE the placeholder value for the AUTH_TOKEN in the JavaScript:

// IMPORTANT: REPLACE THIS PLACEHOLDER WITH YOUR ACTUAL TOKEN 
```javascript
const STREAMING_AUTH_TOKEN = "YOUR_SECURE_TOKEN_HERE"; 
```

Token Transmission: The token is securely attached to the configuration object sent on ws.onopen via the sendConfig() function:

// CRITICAL: Attach the authentication token here
```javascript
auth_token: AUTH_TOKEN 
```

Security Best Practice: The value of this token MUST match the secret token expected by your backend service (e.g., the STREAMING_AUTH_TOKEN environment variable in your service deployment). For production use, it is highly recommended to fetch this token securely from a short-lived, authenticated API endpoint rather than hardcoding it into the client-side source code.