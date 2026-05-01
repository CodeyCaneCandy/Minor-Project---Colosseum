from fastapi import WebSocket
from typing import List

class ConnectionManager:
    def __init__(self):
        # Keeps track of all active browser connections
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Sends the JSON payload to the front end."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # If a user closed their browser tab mid-stream, just ignore it
                pass

# We create a single instance here so we can import it across different routers
manager = ConnectionManager()