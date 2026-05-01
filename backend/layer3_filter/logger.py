from datetime import datetime
from typing import Optional, Callable, Awaitable

class FilterStreamLogger:
    def __init__(self):
        # This is a hook. Later, FastAPI will attach its WebSocket broadcast function here.
        self.broadcast_callback: Optional[Callable[[dict], Awaitable[None]]] = None

    def _format_payload(self, event_type: str, action: str, target: str, reason: str) -> dict:
        """Structures the data perfectly for the UI's 'Filter reasoning logger' component."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_type,  # e.g., "rule_triggered"
            "action": action,     # "exclude", "warn", "restrict"
            "target": target,     # e.g., "SVM", "KNN"
            "reason": reason
        }

    async def log_event(self, action: str, target: str, reason: str):
        """
        1. Prints to the Uvicorn terminal for your backend debugging.
        2. Streams the JSON payload to the front end if the WebSocket is connected.
        """
        payload = self._format_payload("rule_triggered", action, target, reason)
        
        # Backend terminal view (makes debugging easy)
        print(f"\033[96m[LAYER 3 - {action.upper()}]\033[0m {target}: {reason}")
        
        # Front-end live stream view
        if self.broadcast_callback:
            # We await the broadcast so the UI animates exactly as the engine runs
            await self.broadcast_callback(payload)