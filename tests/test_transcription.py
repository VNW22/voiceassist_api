# tests/test_transcription.py
import base64
import asyncio
import websockets

async def test_transcription():
    uri = "wss://<your-modal-endpoint>"
    async with websockets.connect(uri) as ws:
        await ws.send('{"type":"start_stream"}')

        # Send a tiny test audio file
        with open("tests/test_audio.wav", "rb") as f:
            chunk_bytes = f.read()
        chunk_b64 = base64.b64encode(chunk_bytes).decode("utf-8")
        await ws.send(f'{{"type":"audio_chunk","audio":"{chunk_b64}"}}')

        # Receive transcription
        response = await ws.recv()
        assert "text" in response  # check the API returned text

        await ws.send('{"type":"end_stream"}')

if __name__ == "__main__":
    asyncio.run(test_transcription())
