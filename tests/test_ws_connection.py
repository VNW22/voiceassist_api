import asyncio
import websockets

async def test_ws_connect():
    uri = "wss://<your-modal-endpoint>"
    async with websockets.connect(uri) as ws:
        await ws.send('{"type":"start_stream"}')
        response = await ws.recv()
        assert response is not None  # basic check that we got something
        await ws.send('{"type":"end_stream"}')

if __name__ == "__main__":
    asyncio.run(test_ws_connect())
