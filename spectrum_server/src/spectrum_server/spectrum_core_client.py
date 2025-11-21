import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CoreSpectrumFrame:
    center_freq_hz: int
    sample_rate_hz: int
    fft_size: int
    averaging: int
    magnitudes_dbm: list[float]
    timestamp_ns: int
    detections: list[dict]
    schema_version: int | None = None

    @property
    def magnitudes_db(self) -> list[float]:
        """Alias for code paths that still expect magnitudes_db."""
        return self.magnitudes_dbm


class SpectrumCoreClient:
    """
    Async ZeroMQ client for the C++ spectrum_core service.

    Protocol (REQ/REP, multipart):
    Frame 0: JSON header
      {
        "op": "measure",
        "center_freq_hz": 2450000000,
        "sample_rate_hz": 50000000,
        "fft_size": 2048,
        "averaging": 10
      }
    Frame 1: raw float32 power array (length fft_size), optional

    The server is expected to emit a JSON reply with the same fields plus
    `timestamp_ns`; the binary frame carries float32 log-power values.
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        self._endpoint = endpoint or os.getenv("SPECTRUM_CORE_ENDPOINT")
        self._timeout_ms = int(os.getenv("SPECTRUM_CORE_RESPONSE_TIMEOUT_MS", "200"))
        self._lock = asyncio.Lock()
        self._ctx = None
        self._sock = None

    @property
    def enabled(self) -> bool:
        return bool(self._endpoint)

    def _reset(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
        self._sock = None

    def _ensure_socket(self):
        if self._ctx is None:
            try:
                import zmq
            except ImportError as exc:
                raise RuntimeError("pyzmq is required for SpectrumCoreClient") from exc
            self._ctx = zmq.Context.instance()

        if self._sock is None:
            import zmq

            self._sock = self._ctx.socket(zmq.REQ)
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.connect(self._endpoint)

    async def measure_spectrum(
        self,
        *,
        center_freq_hz: int,
        sample_rate_hz: int,
        fft_size: int,
        averaging: int,
    ) -> Optional[CoreSpectrumFrame]:
        """
        Request a GPU-side FFT/averaging pass. Returns None when the service is
        not configured or times out so callers can fall back to simulated data.
        """
        if not self.enabled:
            return None

        try:
            self._ensure_socket()
        except Exception as exc:
            logger.warning("SpectrumCoreClient disabled (pyzmq missing?): %s", exc)
            return None

        import zmq

        payload = {
            "op": "measure",
            "schema_version": 1,
            "center_freq_hz": center_freq_hz,
            "sample_rate_hz": sample_rate_hz,
            "fft_size": fft_size,
            "averaging": averaging,
        }

        async with self._lock:
            try:
                self._sock.send_multipart([json.dumps(payload).encode("utf-8")])
                poller = zmq.Poller()
                poller.register(self._sock, zmq.POLLIN)
                socks = dict(await asyncio.get_event_loop().run_in_executor(
                    None, poller.poll, self._timeout_ms
                ))
                if self._sock not in socks:
                    logger.warning("SpectrumCoreClient timeout after %d ms", self._timeout_ms)
                    self._reset()
                    return None

                msg = self._sock.recv_multipart(flags=zmq.NOBLOCK)
            except Exception as exc:
                logger.warning("SpectrumCoreClient request failed: %s", exc)
                self._reset()
                return None

        if not msg:
            return None
        return self._decode_reply(
            msg,
            default_center=center_freq_hz,
            default_rate=sample_rate_hz,
            fft_size=fft_size,
            averaging=averaging,
        )

    def _decode_reply(
        self,
        msg,
        *,
        default_center: int,
        default_rate: int,
        fft_size: int,
        averaging: int,
    ) -> Optional[CoreSpectrumFrame]:
        """Parse a spectrum_core multipart reply into a CoreSpectrumFrame."""
        try:
            header = json.loads(msg[0].decode("utf-8"))
            power = None
            if len(msg) > 1 and msg[1]:
                power = np.frombuffer(msg[1], dtype=np.float32)
                power = power.tolist()

            fft_size_reply = int(header.get("fft_size", fft_size))
            if power is not None and len(power) != fft_size_reply:
                logger.warning(
                    "SpectrumCoreClient received power length %d but fft_size %d",
                    len(power),
                    fft_size_reply,
                )
                power = power[:fft_size_reply]

            return CoreSpectrumFrame(
                center_freq_hz=int(header.get("center_freq_hz", default_center)),
                sample_rate_hz=int(header.get("sample_rate_hz", default_rate)),
                fft_size=fft_size_reply,
                averaging=int(header.get("averaging", averaging)),
                magnitudes_dbm=power if power is not None else [],
                timestamp_ns=int(header.get("timestamp_ns", 0)),
                detections=header.get("detections", []),
                schema_version=int(header["schema_version"]) if "schema_version" in header else None,
            )
        except Exception as exc:
            logger.warning("SpectrumCoreClient parse error: %s", exc)
            return None


spectrum_core_client = SpectrumCoreClient()
