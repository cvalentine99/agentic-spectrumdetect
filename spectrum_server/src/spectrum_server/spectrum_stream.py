from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from fastapi import WebSocket
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpectrumFrame(BaseModel):
    """Represents a single spectrum slice ready for broadcasting to the UI."""

    center_freq_hz: int = Field(..., description="Tuned center frequency.")
    sample_rate_hz: int = Field(..., description="Sample rate used for this capture.")
    fft_bin_hz: float = Field(..., gt=0, description="FFT bin resolution.")
    magnitudes_db: list[float] = Field(
        ..., description="Magnitude values (dB) ordered from low to high frequency bins."
    )
    timestamp_ns: int = Field(
        default_factory=lambda: int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        description="Capture time in nanoseconds since epoch.",
    )


@dataclass
class _PendingMessage:
    """Internal helper to keep queued messages lightweight."""

    payload: str


class SpectrumStreamManager:
    """Tracks active WebSocket clients and fan-outs spectrum frames to them."""

    def __init__(self, queue_size: int = 256) -> None:
        self._connections: set[WebSocket] = set()
        self._queue: asyncio.Queue[_PendingMessage] = asyncio.Queue(maxsize=queue_size)
        self._broadcast_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the background broadcaster if it is not already running."""
        async with self._lock:
            if self._broadcast_task and not self._broadcast_task.done():
                return
            self._loop = asyncio.get_running_loop()
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
            logger.info("Spectrum stream broadcaster started.")

    async def stop(self) -> None:
        """Stop broadcasting and clean up WebSocket connections."""
        async with self._lock:
            if self._broadcast_task:
                self._broadcast_task.cancel()
                try:
                    await self._broadcast_task
                except asyncio.CancelledError:
                    pass
                self._broadcast_task = None
            await asyncio.gather(*(self._safe_close(ws) for ws in list(self._connections)), return_exceptions=True)
            self._connections.clear()
            logger.info("Spectrum stream broadcaster stopped.")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket client."""
        await websocket.accept()
        await self.start()
        self._connections.add(websocket)
        logger.debug("WebSocket client connected. Total clients: %s", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket client."""
        self._connections.discard(websocket)
        await self._safe_close(websocket)
        logger.debug("WebSocket client disconnected. Total clients: %s", len(self._connections))

    async def publish_frame(self, frame: SpectrumFrame) -> None:
        """Enqueue a SpectrumFrame for broadcasting."""
        await self._publish(self._encode(frame))

    async def publish_payload(self, payload: dict[str, Any]) -> None:
        """Enqueue a pre-built payload for broadcasting."""
        await self._publish(self._encode(payload))

    def publish_threadsafe(self, payload: SpectrumFrame | dict[str, Any]) -> None:
        """Thread-safe publish helper for producers outside the event loop."""
        if not self._loop:
            raise RuntimeError("SpectrumStreamManager has not been started.")
        message = self._encode(payload)
        try:
            self._loop.call_soon_threadsafe(asyncio.create_task, self._publish(message))
        except RuntimeError:
            logger.exception("Failed to publish spectrum payload from thread.")

    async def _publish(self, message: _PendingMessage) -> None:
        """Insert payload into the queue, dropping the oldest on overflow."""
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
            await self._queue.put(message)

    async def _broadcast_loop(self) -> None:
        """Background task to fan-out queued messages to every subscriber."""
        while True:
            message = await self._queue.get()
            await self._fan_out(message.payload)
            self._queue.task_done()

    async def _fan_out(self, payload: str) -> None:
        stale: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        if stale:
            await asyncio.gather(*(self.disconnect(ws) for ws in stale), return_exceptions=True)

    def _encode(self, payload: SpectrumFrame | dict[str, Any]) -> _PendingMessage:
        """Convert models/dicts into a JSON string for transport."""
        if isinstance(payload, SpectrumFrame):
            json_payload = payload.model_dump_json()
        else:
            json_payload = json.dumps(payload, default=self._json_default)
        return _PendingMessage(payload=json_payload)

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    async def _safe_close(self, websocket: WebSocket) -> None:
        try:
            await websocket.close()
        except Exception:
            logger.debug("Ignoring WebSocket close error", exc_info=True)


# Single shared manager instance used by the FastAPI app.
spectrum_stream = SpectrumStreamManager()
