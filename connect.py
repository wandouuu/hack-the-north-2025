import asyncio
import websockets

# Replace with your Electron app's WebSocket URL
ELECTRON_WS_URL = "ws://localhost:3000"

async def connect_to_electron():
    async with websockets.connect(ELECTRON_WS_URL) as websocket:
        print("Connected to Electron app")
        # Example: send a message
        await websocket.send("Hello from Python!")
        # Example: receive a message
        response = await websocket.recv()
        print(f"Received: {response}")

if __name__ == "__main__":
    asyncio.run(connect_to_electron())