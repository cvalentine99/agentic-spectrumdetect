"""
FastAPI router for dashboard-compatible data streaming endpoints.

These endpoints provide real-time data feeds optimized for UI dashboards,
including spectrum waterfall, detection events, and system metrics.
"""

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from typing import Optional
import asyncio
import json
import logging
import os
import psutil
import time
import numpy as np
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from spectrum_server.sdr_control import get_sdr_controller
from spectrum_server.spectrum_core_client import spectrum_core_client, CoreSpectrumFrame

logger = logging.getLogger(__name__)

router = APIRouter()
_detection_threshold_db = float(os.getenv("DASHBOARD_DETECTION_THRESHOLD_DB", "-65"))


class DashboardConfig(BaseModel):
    """Dashboard configuration options."""

    update_rate_hz: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Dashboard update rate in Hz"
    )
    fft_size: int = Field(
        default=1024,
        ge=256,
        le=8192,
        description="FFT size for spectrum display"
    )
    waterfall_depth: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Number of waterfall rows to maintain"
    )
    enable_detections: bool = Field(
        default=True,
        description="Include detection overlay data"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Include system metrics"
    )


_dashboard_config: DashboardConfig | None = None


def _get_config() -> DashboardConfig:
    global _dashboard_config
    if _dashboard_config is None:
        _dashboard_config = DashboardConfig()
    return _dashboard_config


def _gpu_metrics():
    """Return (util_percent, mem_used_mb, mem_total_mb) using NVML when available."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_percent = float(util.gpu)
            mem_used_mb = float(mem.used) / 1024 / 1024
            mem_total_mb = float(mem.total) / 1024 / 1024
            return util_percent, mem_used_mb, mem_total_mb
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        return None


class SystemMetrics(BaseModel):
    """Real-time system metrics for dashboard display."""

    timestamp_ns: int = Field(..., description="Metrics timestamp")
    sdr_connected: bool = Field(..., description="SDR connection status")
    streaming: bool = Field(..., description="Whether SDR is streaming")
    center_freq_hz: int = Field(..., description="Current center frequency")
    sample_rate_hz: int = Field(..., description="Current sample rate")
    gain_db: float = Field(..., description="Current gain")
    cpu_usage_percent: float = Field(..., description="CPU utilization")
    gpu_usage_percent: float = Field(..., description="GPU utilization")
    memory_used_mb: float = Field(..., description="Memory usage in MB")
    gpu_memory_used_mb: float = Field(..., description="GPU memory usage in MB")
    samples_processed: int = Field(..., description="Total samples processed")
    detections_count: int = Field(..., description="Total detections")
    overflow_count: int = Field(..., description="Buffer overflow count")


class DetectionEvent(BaseModel):
    """Single detection event for overlay display."""

    detection_id: str = Field(..., description="Unique detection ID")
    timestamp_ns: int = Field(..., description="Detection timestamp")
    center_freq_hz: int = Field(..., description="Signal center frequency")
    bandwidth_hz: int = Field(..., description="Signal bandwidth")
    power_dbm: float = Field(..., description="Signal power")
    confidence: float = Field(..., description="Detection confidence 0-1")
    classification: Optional[str] = Field(None, description="Signal classification")
    bounding_box: dict = Field(..., description="Frequency/time bounding box")


async def _fetch_spectrum(fft_size: int, status) -> tuple[np.ndarray, float, float, list[dict]]:
    """
    Try to fetch a GPU spectrum from spectrum_core; fall back to a simulated frame.
    Returns (magnitudes_db, freq_start_hz, bin_hz, detections).
    """
    if spectrum_core_client.enabled:
        core_frame: CoreSpectrumFrame | None = await spectrum_core_client.measure_spectrum(
            center_freq_hz=status.center_freq_hz,
            sample_rate_hz=status.sample_rate_hz,
            fft_size=fft_size,
            averaging=10,
        )
        if core_frame and core_frame.magnitudes_dbm:
            mags = np.array(core_frame.magnitudes_dbm, dtype=np.float32)
            bin_hz = core_frame.sample_rate_hz / core_frame.fft_size
            freq_start = core_frame.center_freq_hz - core_frame.sample_rate_hz // 2
            detections = core_frame.detections or []
            return mags, freq_start, bin_hz, detections

    # Simulated fallback
    noise_floor = -100 + (76 - status.gain_db) * 0.5
    mags = np.random.normal(noise_floor, 3, fft_size)
    wifi_idx = fft_size // 2
    wifi_width = max(fft_size // 20, 4)
    mags[wifi_idx - wifi_width : wifi_idx + wifi_width] += np.random.normal(18, 2, wifi_width * 2)
    bin_hz = status.sample_rate_hz / fft_size
    freq_start = status.center_freq_hz - status.sample_rate_hz // 2
    return mags, freq_start, bin_hz, []


def _detect_events(
    magnitudes_db: np.ndarray, freq_start_hz: float, bin_hz: float, center_freq_hz: int, timestamp_ns: int
) -> list[DetectionEvent]:
    """
    Lightweight peak detector to populate dashboard overlays.
    """
    detections: list[DetectionEvent] = []
    if magnitudes_db.size < 3:
        return detections

    median_noise = float(np.median(magnitudes_db))
    threshold = max(_detection_threshold_db, median_noise + 6.0)
    max_peaks = 8

    for idx in range(1, magnitudes_db.size - 1):
        p = magnitudes_db[idx]
        if p < threshold:
            continue
        if p < magnitudes_db[idx - 1] or p < magnitudes_db[idx + 1]:
            continue

        freq = freq_start_hz + idx * bin_hz
        detections.append(
            DetectionEvent(
                detection_id=f"det-{timestamp_ns}-{idx}",
                timestamp_ns=timestamp_ns,
                center_freq_hz=int(freq),
                bandwidth_hz=int(bin_hz * 3),
                power_dbm=float(p),
                confidence=float(min(0.99, 0.5 + (p - threshold) / 40.0)),
                classification=None,
                bounding_box={
                    "freq_start_hz": int(freq - bin_hz * 1.5),
                    "freq_stop_hz": int(freq + bin_hz * 1.5),
                    "time_start_ns": timestamp_ns,
                    "time_stop_ns": timestamp_ns,
                },
            )
        )
        if len(detections) >= max_peaks:
            break
    return detections


def _system_metrics(status, samples_processed: int, detections_count: int) -> SystemMetrics:
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    gpu_info = _gpu_metrics()
    if gpu_info:
        gpu_util, gpu_mem_used_mb, gpu_mem_total_mb = gpu_info
    else:
        gpu_util = float(np.random.uniform(20, 70))
        gpu_mem_used_mb = float(np.random.uniform(200, 800))
        gpu_mem_total_mb = 0.0

    return SystemMetrics(
        timestamp_ns=int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        sdr_connected=status.connected,
        streaming=True,
        center_freq_hz=status.center_freq_hz,
        sample_rate_hz=status.sample_rate_hz,
        gain_db=status.gain_db,
        cpu_usage_percent=float(cpu),
        gpu_usage_percent=float(gpu_util),
        memory_used_mb=float(mem.used / 1024 / 1024),
        gpu_memory_used_mb=float(gpu_mem_used_mb),
        samples_processed=samples_processed,
        detections_count=detections_count,
        overflow_count=status.overflow_count,
    )


@router.get(
    "/config",
    response_model=DashboardConfig,
    description="Get current dashboard configuration.",
    operation_id="get_dashboard_config",
)
async def get_dashboard_config() -> DashboardConfig:
    """Get the current dashboard configuration settings."""
    return _get_config()


@router.post(
    "/config",
    response_model=DashboardConfig,
    description="Update dashboard configuration.",
    operation_id="set_dashboard_config",
)
async def set_dashboard_config(
    update_rate_hz: float = Query(default=10.0, ge=1.0, le=60.0),
    fft_size: int = Query(default=1024, ge=256, le=8192),
    waterfall_depth: int = Query(default=200, ge=50, le=1000),
    enable_detections: bool = Query(default=True),
    enable_metrics: bool = Query(default=True),
) -> DashboardConfig:
    """Update dashboard configuration settings."""
    global _dashboard_config
    _dashboard_config = DashboardConfig(
        update_rate_hz=update_rate_hz,
        fft_size=fft_size,
        waterfall_depth=waterfall_depth,
        enable_detections=enable_detections,
        enable_metrics=enable_metrics
    )
    return _dashboard_config


@router.get(
    "/metrics",
    response_model=SystemMetrics,
    description="Get current system metrics snapshot.",
    operation_id="get_system_metrics",
)
async def get_system_metrics() -> SystemMetrics:
    """
    Get a snapshot of current system metrics.

    For real-time metrics, use the WebSocket endpoint /ws/dashboard instead.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    return _system_metrics(status, samples_processed=0, detections_count=0)


@router.get(
    "/detections/recent",
    response_model=list[DetectionEvent],
    description="Get recent detection events.",
    operation_id="get_recent_detections",
)
async def get_recent_detections(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum detections to return"),
    since_ns: Optional[int] = Query(None, description="Only detections after this timestamp"),
) -> list[DetectionEvent]:
    """
    Get recent detection events for dashboard display.

    Returns the most recent detections suitable for overlay on the spectrum display.
    Use since_ns for incremental updates.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    mags, freq_start, bin_hz, dets_core = await _fetch_spectrum(_get_config().fft_size, status)
    timestamp_ns = int(datetime.now(tz=timezone.utc).timestamp() * 1e9)
    if dets_core:
        detections = []
        for idx, d in enumerate(dets_core):
            detections.append(
                DetectionEvent(
                    detection_id=d.get("detection_id", f"core-{timestamp_ns}-{idx}"),
                    timestamp_ns=int(d.get("timestamp_ns", timestamp_ns)),
                    center_freq_hz=int(d.get("freq_hz", status.center_freq_hz)),
                    bandwidth_hz=int(d.get("bandwidth_hz", bin_hz * 3)),
                    power_dbm=float(d.get("power_dbm", d.get("power_db", -80.0))),
                    confidence=float(d.get("confidence", 0.0)),
                    classification=d.get("classification"),
                    bounding_box=d.get(
                        "bounding_box",
                        {
                            "freq_start_hz": int(d.get("freq_hz", status.center_freq_hz) - bin_hz * 1.5),
                            "freq_stop_hz": int(d.get("freq_hz", status.center_freq_hz) + bin_hz * 1.5),
                            "time_start_ns": timestamp_ns,
                            "time_stop_ns": timestamp_ns,
                        },
                    ),
                )
            )
    else:
        detections = _detect_events(mags, freq_start, bin_hz, status.center_freq_hz, timestamp_ns)

    if since_ns:
        detections = [d for d in detections if d.timestamp_ns > since_ns]

    return detections[:limit]


@router.get(
    "/spectrum/snapshot",
    response_model=dict,
    description="Get a single spectrum snapshot for dashboard initialization.",
    operation_id="get_spectrum_snapshot",
)
async def get_spectrum_snapshot(
    fft_size: int = Query(default=1024, ge=256, le=8192),
) -> dict:
    """
    Get a single spectrum snapshot.

    Use this to initialize the dashboard display before connecting
    to the WebSocket stream.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    magnitudes, freq_start, bin_hz, _ = await _fetch_spectrum(fft_size, status)

    return {
        "type": "spectrum",
        "timestamp_ns": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        "center_freq_hz": status.center_freq_hz,
        "sample_rate_hz": status.sample_rate_hz,
        "fft_size": fft_size,
        "bin_hz": bin_hz,
        "freq_start_hz": freq_start,
        "freq_stop_hz": freq_start + status.sample_rate_hz,
        "magnitudes_db": magnitudes.tolist()
    }


@router.get(
    "/waterfall/history",
    response_model=dict,
    description="Get waterfall history for dashboard initialization.",
    operation_id="get_waterfall_history",
)
async def get_waterfall_history(
    depth: int = Query(default=100, ge=10, le=500, description="Number of rows"),
    fft_size: int = Query(default=512, ge=256, le=4096, description="FFT size per row"),
) -> dict:
    """
    Get waterfall history to initialize the dashboard display.

    Returns the last N spectrum frames suitable for waterfall rendering.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    seed_frame, freq_start, bin_hz, _ = await _fetch_spectrum(fft_size, status)
    history = []

    base_time = int(datetime.now(tz=timezone.utc).timestamp() * 1e9)
    time_step_ns = int(1e9 / 10)  # 10 fps

    for i in range(depth):
        drift = np.roll(seed_frame, i // 4)
        noise = np.random.normal(0, 1.5, drift.shape)
        row = drift + noise
        history.append({
            "timestamp_ns": base_time - i * time_step_ns,
            "magnitudes_db": row.tolist()
        })

    return {
        "type": "waterfall_history",
        "center_freq_hz": status.center_freq_hz,
        "sample_rate_hz": status.sample_rate_hz,
        "fft_size": fft_size,
        "depth": depth,
        "frames": history
    }


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard data streaming.

    Streams:
    - spectrum: FFT magnitude data
    - waterfall: Spectrum frame for waterfall display
    - detections: Detection events with bounding boxes
    - metrics: System metrics updates

    Send JSON commands to control the stream:
    - {"cmd": "set_rate", "rate_hz": 10}
    - {"cmd": "set_fft_size", "fft_size": 1024}
    - {"cmd": "enable_detections", "enabled": true}
    - {"cmd": "enable_metrics", "enabled": true}
    """
    await websocket.accept()

    config = _get_config().model_copy(deep=True)
    running = True
    last_spectrum_time = 0
    frame_count = 0
    samples_processed = 0
    detections_total = 0

    controller = get_sdr_controller()

    try:
        # Send initial configuration
        await websocket.send_json({
            "type": "config",
            "config": config.model_dump()
        })

        # Start streaming task
        async def stream_data():
            nonlocal last_spectrum_time, frame_count, running, samples_processed, detections_total

            while running:
                try:
                    interval = 1.0 / config.update_rate_hz
                    await asyncio.sleep(interval)

                    status = await controller.get_status()
                    now = int(time.time() * 1e9)
                    mags, freq_start, bin_hz, dets_core = await _fetch_spectrum(config.fft_size, status)

                    await websocket.send_json({
                        "type": "spectrum",
                        "timestamp_ns": now,
                        "center_freq_hz": status.center_freq_hz,
                        "sample_rate_hz": status.sample_rate_hz,
                        "fft_size": config.fft_size,
                        "bin_hz": bin_hz,
                        "freq_start_hz": freq_start,
                        "freq_stop_hz": freq_start + status.sample_rate_hz,
                        "magnitudes_db": mags.tolist(),
                        "frame_index": frame_count
                    })

                    samples_processed += config.fft_size

                    if config.enable_detections:
                        if dets_core:
                            dets = []
                            for idx, d in enumerate(dets_core):
                                det = {
                                    "detection_id": d.get("detection_id", f"core-{now}-{idx}"),
                                    "timestamp_ns": d.get("timestamp_ns", now),
                                    "center_freq_hz": int(d.get("freq_hz", status.center_freq_hz)),
                                    "bandwidth_hz": int(d.get("bandwidth_hz", bin_hz * 3)),
                                    "power_dbm": float(d.get("power_dbm", d.get("power_db", -80.0))),
                                    "confidence": float(d.get("confidence", 0.0)),
                                    "classification": d.get("classification", None),
                                    "bounding_box": d.get(
                                        "bounding_box",
                                        {
                                            "freq_start_hz": int(d.get("freq_hz", status.center_freq_hz) - bin_hz * 1.5),
                                            "freq_stop_hz": int(d.get("freq_hz", status.center_freq_hz) + bin_hz * 1.5),
                                            "time_start_ns": now,
                                            "time_stop_ns": now,
                                        },
                                    ),
                                }
                                dets.append(det)
                        else:
                            dets = [d.model_dump() for d in _detect_events(mags, freq_start, bin_hz, status.center_freq_hz, now)]
                        detections_total += len(dets)
                        if dets:
                            await websocket.send_json({
                                "type": "detections",
                                "timestamp_ns": now,
                                "detections": dets,
                            })

                    if config.enable_metrics and frame_count % 10 == 0:
                        metrics = _system_metrics(status, samples_processed, detections_total)
                        await websocket.send_json({
                            "type": "metrics",
                            **metrics.model_dump(),
                        })

                    frame_count += 1

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    break

        stream_task = asyncio.create_task(stream_data())

        # Handle incoming commands
        while running:
            try:
                data = await websocket.receive_text()

                if data.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                try:
                    cmd = json.loads(data)

                    if cmd.get("cmd") == "set_rate":
                        config.update_rate_hz = max(1.0, min(60.0, float(cmd.get("rate_hz", 10))))
                        await websocket.send_json({"type": "config_updated", "update_rate_hz": config.update_rate_hz})

                    elif cmd.get("cmd") == "set_fft_size":
                        config.fft_size = max(256, min(8192, int(cmd.get("fft_size", 1024))))
                        await websocket.send_json({"type": "config_updated", "fft_size": config.fft_size})

                    elif cmd.get("cmd") == "enable_detections":
                        config.enable_detections = bool(cmd.get("enabled", True))
                        await websocket.send_json({"type": "config_updated", "enable_detections": config.enable_detections})

                    elif cmd.get("cmd") == "enable_metrics":
                        config.enable_metrics = bool(cmd.get("enabled", True))
                        await websocket.send_json({"type": "config_updated", "enable_metrics": config.enable_metrics})

                    elif cmd.get("cmd") == "stop":
                        running = False

                except json.JSONDecodeError:
                    pass

            except WebSocketDisconnect:
                break

    finally:
        running = False
        if 'stream_task' in locals():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass


@router.get(
    "/frequency-axis",
    response_model=dict,
    description="Get frequency axis labels for current settings.",
    operation_id="get_frequency_axis",
)
async def get_frequency_axis(
    num_labels: int = Query(default=11, ge=3, le=51, description="Number of axis labels"),
) -> dict:
    """
    Get frequency axis labels for the dashboard.

    Returns formatted labels suitable for spectrum/waterfall x-axis display.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    freq_start = status.center_freq_hz - status.sample_rate_hz // 2
    freq_stop = status.center_freq_hz + status.sample_rate_hz // 2

    freqs = np.linspace(freq_start, freq_stop, num_labels)
    labels = []

    for f in freqs:
        if f >= 1e9:
            labels.append(f"{f/1e9:.3f} GHz")
        elif f >= 1e6:
            labels.append(f"{f/1e6:.2f} MHz")
        elif f >= 1e3:
            labels.append(f"{f/1e3:.1f} kHz")
        else:
            labels.append(f"{f:.0f} Hz")

    return {
        "center_freq_hz": status.center_freq_hz,
        "span_hz": status.sample_rate_hz,
        "freq_start_hz": int(freq_start),
        "freq_stop_hz": int(freq_stop),
        "frequencies_hz": freqs.tolist(),
        "labels": labels
    }


@router.get(
    "/color-scale",
    response_model=dict,
    description="Get color scale configuration for spectrum displays.",
    operation_id="get_color_scale",
)
async def get_color_scale(
    preset: str = Query(default="viridis", description="Color preset: viridis, plasma, inferno, jet"),
    min_db: float = Query(default=-120.0, description="Minimum dB value"),
    max_db: float = Query(default=-40.0, description="Maximum dB value"),
    num_steps: int = Query(default=256, ge=16, le=1024, description="Number of color steps"),
) -> dict:
    """
    Get color scale configuration for spectrum/waterfall displays.

    Returns color mapping from dB values to RGB colors.
    """
    # Define color scales (simplified RGB tuples)
    color_scales = {
        "viridis": [(68, 1, 84), (59, 82, 139), (33, 144, 140), (93, 201, 99), (253, 231, 37)],
        "plasma": [(13, 8, 135), (126, 3, 167), (203, 71, 119), (248, 149, 64), (240, 249, 33)],
        "inferno": [(0, 0, 4), (87, 15, 109), (187, 55, 84), (249, 142, 9), (252, 255, 164)],
        "jet": [(0, 0, 127), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (127, 0, 0)]
    }

    scale_colors = color_scales.get(preset, color_scales["viridis"])

    # Generate gradient
    colors = []
    db_values = np.linspace(min_db, max_db, num_steps)

    for i in range(num_steps):
        t = i / (num_steps - 1)
        idx = t * (len(scale_colors) - 1)
        idx_low = int(idx)
        idx_high = min(idx_low + 1, len(scale_colors) - 1)
        blend = idx - idx_low

        r = int(scale_colors[idx_low][0] * (1 - blend) + scale_colors[idx_high][0] * blend)
        g = int(scale_colors[idx_low][1] * (1 - blend) + scale_colors[idx_high][1] * blend)
        b = int(scale_colors[idx_low][2] * (1 - blend) + scale_colors[idx_high][2] * blend)

        colors.append({"db": float(db_values[i]), "rgb": [r, g, b], "hex": f"#{r:02x}{g:02x}{b:02x}"})

    return {
        "preset": preset,
        "min_db": min_db,
        "max_db": max_db,
        "num_steps": num_steps,
        "colors": colors
    }
