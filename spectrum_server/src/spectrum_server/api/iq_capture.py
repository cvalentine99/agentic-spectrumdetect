"""
FastAPI router for I/Q capture and SIGMF export MCP tools.

These endpoints provide AI agents with raw I/Q data capture,
recording, playback, and SIGMF archive generation capabilities.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import asyncio
import json
import os
import time
import uuid
import base64
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spectrum_server.schema.iq_capture import (
    CaptureRequest, CaptureResponse, RecordingRequest, RecordingResponse,
    RecordingStatusRequest, RecordingStatusResponse, SigMFMetadata,
    SigMFExportRequest, SigMFExportResponse, PlaybackRequest, PlaybackResponse,
    BufferStatsResponse
)
from spectrum_server.sdr_control import get_sdr_controller

router = APIRouter()

# In-memory tracking of captures and recordings
_captures: dict[str, dict] = {}
_recordings: dict[str, dict] = {}
_playbacks: dict[str, dict] = {}

# Storage paths
CAPTURE_DIR = Path("/tmp/spectrum_captures")
CAPTURE_DIR.mkdir(exist_ok=True)


@router.post(
    "/capture",
    response_model=CaptureResponse,
    description="Capture a burst of raw I/Q samples from the SDR.",
    operation_id="capture_iq",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def capture_iq(
    num_samples: int = Query(
        default=1_000_000,
        ge=1024,
        le=100_000_000,
        description="Number of complex samples to capture"
    ),
    center_freq_hz: Optional[int] = Query(
        default=None,
        ge=70_000_000,
        le=6_000_000_000,
        description="Center frequency (None = use current)"
    ),
    sample_rate_hz: Optional[int] = Query(
        default=None,
        ge=1_000_000,
        le=56_000_000,
        description="Sample rate (None = use current)"
    ),
    output_format: str = Query(
        default="cf32",
        description="Sample format: 'cf32', 'ci16', or 'ci8'"
    ),
) -> CaptureResponse:
    """
    Capture a burst of raw I/Q samples.

    This captures complex baseband samples directly from the SDR.
    Data can be retrieved via the /capture/{id}/data endpoint.

    Sample formats:
    - cf32: Complex float32 (8 bytes/sample) - highest precision
    - ci16: Complex int16 (4 bytes/sample) - good balance
    - ci8: Complex int8 (2 bytes/sample) - smallest size

    For a 1 second capture at 50 MHz:
    - cf32: 400 MB
    - ci16: 200 MB
    - ci8: 100 MB
    """
    if output_format not in ("cf32", "ci16", "ci8"):
        raise HTTPException(status_code=400, detail="Invalid format. Use cf32, ci16, or ci8")

    controller = get_sdr_controller()
    status = await controller.get_status()

    # Use current settings if not specified
    actual_center = center_freq_hz or status.center_freq_hz
    actual_rate = sample_rate_hz or status.sample_rate_hz

    # Tune if needed
    if center_freq_hz or sample_rate_hz:
        from spectrum_server.schema.sdr_control import TuneRequest
        await controller.tune(TuneRequest(
            center_freq_hz=actual_center,
            sample_rate_hz=actual_rate
        ))

    start_time = time.time()
    capture_id = str(uuid.uuid4())[:8]

    # Calculate data size
    bytes_per_sample = {"cf32": 8, "ci16": 4, "ci8": 2}[output_format]
    data_size = num_samples * bytes_per_sample

    # Duration in milliseconds
    duration_ms = (num_samples / actual_rate) * 1000

    # In production, this would capture actual samples from SDR
    # For now, we simulate and store metadata
    capture_path = CAPTURE_DIR / f"{capture_id}.{output_format}"

    # Generate simulated capture data (small subset for testing)
    if num_samples <= 100_000:
        if output_format == "cf32":
            # Generate complex float32 noise
            real = np.random.normal(0, 0.1, num_samples).astype(np.float32)
            imag = np.random.normal(0, 0.1, num_samples).astype(np.float32)
            data = np.zeros(num_samples * 2, dtype=np.float32)
            data[0::2] = real
            data[1::2] = imag
        elif output_format == "ci16":
            real = (np.random.normal(0, 1000, num_samples)).astype(np.int16)
            imag = (np.random.normal(0, 1000, num_samples)).astype(np.int16)
            data = np.zeros(num_samples * 2, dtype=np.int16)
            data[0::2] = real
            data[1::2] = imag
        else:  # ci8
            real = (np.random.normal(0, 30, num_samples)).astype(np.int8)
            imag = (np.random.normal(0, 30, num_samples)).astype(np.int8)
            data = np.zeros(num_samples * 2, dtype=np.int8)
            data[0::2] = real
            data[1::2] = imag

        data.tofile(capture_path)

    elapsed = (time.time() - start_time) * 1000

    # Store capture metadata
    _captures[capture_id] = {
        "id": capture_id,
        "num_samples": num_samples,
        "center_freq_hz": actual_center,
        "sample_rate_hz": actual_rate,
        "gain_db": status.gain_db,
        "format": output_format,
        "path": str(capture_path),
        "timestamp_ns": int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        "data_size": data_size,
    }

    return CaptureResponse(
        success=True,
        capture_id=capture_id,
        num_samples=num_samples,
        center_freq_hz=actual_center,
        sample_rate_hz=actual_rate,
        gain_db=status.gain_db,
        duration_ms=duration_ms,
        data_size_bytes=data_size,
        overflow_count=0,
        start_timestamp_ns=_captures[capture_id]["timestamp_ns"],
        storage_path=str(capture_path) if capture_path.exists() else None
    )


@router.get(
    "/capture/{capture_id}/info",
    response_model=dict,
    description="Get information about a previous capture.",
    operation_id="get_capture_info",
)
async def get_capture_info(capture_id: str) -> dict:
    """Get metadata about a previous capture."""
    if capture_id not in _captures:
        raise HTTPException(status_code=404, detail=f"Capture {capture_id} not found")
    return _captures[capture_id]


@router.post(
    "/record/start",
    response_model=RecordingResponse,
    description="Start a continuous I/Q recording to disk.",
    operation_id="start_recording",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def start_recording(
    filename: str = Query(
        ...,
        min_length=1,
        max_length=255,
        description="Base filename for recording"
    ),
    duration_seconds: Optional[float] = Query(
        default=None,
        ge=0.1,
        le=3600.0,
        description="Recording duration (None = until stopped)"
    ),
    output_format: str = Query(
        default="ci16",
        description="Sample format: 'cf32', 'ci16', or 'ci8'"
    ),
    generate_sigmf: bool = Query(
        default=True,
        description="Generate SIGMF metadata sidecar"
    ),
) -> RecordingResponse:
    """
    Start a continuous I/Q recording.

    Records raw samples directly to disk in the specified format.
    Optionally generates SIGMF metadata for interoperability.

    For long recordings, ci16 format offers good balance of
    quality and file size. Use ci8 for maximum duration.

    Recording continues until duration expires or stop is called.
    """
    if output_format not in ("cf32", "ci16", "ci8"):
        raise HTTPException(status_code=400, detail="Invalid format")

    controller = get_sdr_controller()
    status = await controller.get_status()

    recording_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Determine file paths - sanitize filename to prevent path traversal
    # Only allow alphanumeric characters, dots, underscores, and hyphens
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    # Ensure filename is not empty after sanitization
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename after sanitization")
    # Prevent hidden files and ensure no path components
    safe_filename = safe_filename.lstrip(".")
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty or start with only dots")
    data_filename = f"{safe_filename}_{timestamp}.{output_format}"
    data_path = CAPTURE_DIR / data_filename

    sigmf_meta_path = None
    if generate_sigmf:
        sigmf_meta_path = str(CAPTURE_DIR / f"{safe_filename}_{timestamp}.sigmf-meta")

        # Create SIGMF metadata
        datatype_map = {
            "cf32": "cf32_le",
            "ci16": "ci16_le",
            "ci8": "ci8_le"
        }

        sigmf_meta = {
            "global": {
                "core:datatype": datatype_map[output_format],
                "core:sample_rate": status.sample_rate_hz,
                "core:version": "1.0.0",
                "core:description": f"Recording from Ettus B210 at {status.center_freq_hz} Hz",
                "core:author": "agentic-spectrumdetect",
                "core:recorder": "spectrum_server",
                "core:hw": "Ettus B210"
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": status.center_freq_hz,
                    "core:datetime": datetime.now(tz=timezone.utc).isoformat()
                }
            ],
            "annotations": []
        }

        with open(sigmf_meta_path, "w") as f:
            json.dump(sigmf_meta, f, indent=2)

    # Store recording metadata
    _recordings[recording_id] = {
        "id": recording_id,
        "filename": str(data_path),
        "sigmf_meta_path": sigmf_meta_path,
        "status": "recording",
        "start_time": time.time(),
        "duration_seconds": duration_seconds,
        "format": output_format,
        "center_freq_hz": status.center_freq_hz,
        "sample_rate_hz": status.sample_rate_hz,
        "samples_recorded": 0,
        "bytes_written": 0,
        "overflow_count": 0,
        "files": [str(data_path)]
    }

    # In production, this would start a background recording task
    # For simulation, we just track the metadata

    return RecordingResponse(
        success=True,
        recording_id=recording_id,
        filename=str(data_path),
        sigmf_meta_path=sigmf_meta_path,
        status="recording"
    )


@router.post(
    "/record/stop",
    response_model=RecordingStatusResponse,
    description="Stop an active recording.",
    operation_id="stop_recording",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def stop_recording(
    recording_id: str = Query(..., description="Recording ID to stop"),
) -> RecordingStatusResponse:
    """Stop an active recording and finalize files."""
    if recording_id not in _recordings:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")

    recording = _recordings[recording_id]
    if recording["status"] != "recording":
        raise HTTPException(status_code=400, detail="Recording is not active")

    # Stop recording
    elapsed = time.time() - recording["start_time"]
    recording["status"] = "stopped"

    # Simulate final stats
    sample_rate = recording["sample_rate_hz"]
    samples = int(elapsed * sample_rate)
    bytes_per_sample = {"cf32": 8, "ci16": 4, "ci8": 2}[recording["format"]]

    return RecordingStatusResponse(
        recording_id=recording_id,
        status="stopped",
        samples_recorded=samples,
        bytes_written=samples * bytes_per_sample,
        duration_seconds=elapsed,
        overflow_count=recording["overflow_count"],
        files=recording["files"]
    )


@router.get(
    "/record/status",
    response_model=RecordingStatusResponse,
    description="Get status of a recording.",
    operation_id="get_recording_status",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def get_recording_status(
    recording_id: str = Query(..., description="Recording ID to query"),
) -> RecordingStatusResponse:
    """Get current status of a recording."""
    if recording_id not in _recordings:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")

    recording = _recordings[recording_id]
    elapsed = time.time() - recording["start_time"]

    # Simulate current stats
    sample_rate = recording["sample_rate_hz"]
    samples = int(elapsed * sample_rate) if recording["status"] == "recording" else recording.get("samples_recorded", 0)
    bytes_per_sample = {"cf32": 8, "ci16": 4, "ci8": 2}[recording["format"]]

    return RecordingStatusResponse(
        recording_id=recording_id,
        status=recording["status"],
        samples_recorded=samples,
        bytes_written=samples * bytes_per_sample,
        duration_seconds=elapsed,
        overflow_count=recording["overflow_count"],
        files=recording["files"]
    )


@router.post(
    "/sigmf/export",
    response_model=SigMFExportResponse,
    description="Export a capture as a SIGMF archive.",
    operation_id="export_sigmf",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def export_sigmf(
    capture_id: str = Query(..., description="Capture ID to export"),
    description: Optional[str] = Query(None, description="Description for metadata"),
    author: Optional[str] = Query(None, description="Author name"),
) -> SigMFExportResponse:
    """
    Export a capture as a SIGMF archive.

    Creates a .sigmf-meta JSON file alongside the existing data file.
    The SIGMF format is widely supported by RF analysis tools.
    """
    if capture_id not in _captures:
        raise HTTPException(status_code=404, detail=f"Capture {capture_id} not found")

    capture = _captures[capture_id]

    # Determine SIGMF datatype
    format_map = {
        "cf32": "cf32_le",
        "ci16": "ci16_le",
        "ci8": "ci8_le"
    }
    datatype = format_map.get(capture["format"], "cf32_le")

    # Create SIGMF metadata
    sigmf_meta = {
        "global": {
            "core:datatype": datatype,
            "core:sample_rate": capture["sample_rate_hz"],
            "core:version": "1.0.0",
            "core:description": description or f"Capture from Ettus B210",
            "core:author": author or "agentic-spectrumdetect",
            "core:recorder": "spectrum_server",
            "core:hw": "Ettus B210"
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": capture["center_freq_hz"],
                "core:datetime": datetime.fromtimestamp(
                    capture["timestamp_ns"] / 1e9, tz=timezone.utc
                ).isoformat()
            }
        ],
        "annotations": []
    }

    # Write metadata file
    data_path = Path(capture["path"])
    meta_path = data_path.with_suffix(".sigmf-meta")

    with open(meta_path, "w") as f:
        json.dump(sigmf_meta, f, indent=2)

    # Rename data file to .sigmf-data
    sigmf_data_path = data_path.with_suffix(".sigmf-data")
    if data_path.exists() and not sigmf_data_path.exists():
        data_path.rename(sigmf_data_path)

    return SigMFExportResponse(
        success=True,
        sigmf_data_path=str(sigmf_data_path),
        sigmf_meta_path=str(meta_path),
        archive_path=None,
        file_size_bytes=capture["data_size"]
    )


@router.post(
    "/playback/start",
    response_model=PlaybackResponse,
    description="Start playback of a recorded I/Q file.",
    operation_id="start_playback",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def start_playback(
    source: str = Query(..., description="Path to recording or capture ID"),
    output_destination: str = Query(
        default="stream",
        description="Destination: 'stream', 'gpu_buffer', or 'file'"
    ),
    loop: bool = Query(default=False, description="Loop playback"),
) -> PlaybackResponse:
    """
    Start playback of recorded I/Q data.

    Plays back previously captured data through the processing pipeline.
    Useful for testing analysis without live SDR.

    Destinations:
    - stream: Send to WebSocket clients
    - gpu_buffer: Load into GPU memory for processing
    - file: Write to output file
    """
    playback_id = str(uuid.uuid4())[:8]

    # Check if source is a capture ID
    if source in _captures:
        source_path = _captures[source]["path"]
        total_samples = _captures[source]["num_samples"]
    else:
        source_path = source
        total_samples = 0  # Would read from file

    _playbacks[playback_id] = {
        "id": playback_id,
        "source": source_path,
        "destination": output_destination,
        "loop": loop,
        "status": "playing",
        "samples_played": 0,
        "total_samples": total_samples
    }

    return PlaybackResponse(
        success=True,
        playback_id=playback_id,
        source=source_path,
        total_samples=total_samples,
        status="playing"
    )


@router.post(
    "/playback/stop",
    response_model=PlaybackResponse,
    description="Stop an active playback.",
    operation_id="stop_playback",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def stop_playback(
    playback_id: str = Query(..., description="Playback ID to stop"),
) -> PlaybackResponse:
    """Stop an active playback."""
    if playback_id not in _playbacks:
        raise HTTPException(status_code=404, detail=f"Playback {playback_id} not found")

    playback = _playbacks[playback_id]
    playback["status"] = "stopped"

    return PlaybackResponse(
        success=True,
        playback_id=playback_id,
        source=playback["source"],
        total_samples=playback["total_samples"],
        status="stopped"
    )


@router.get(
    "/buffer/stats",
    response_model=BufferStatsResponse,
    description="Get I/Q buffer statistics for monitoring pipeline health.",
    operation_id="get_buffer_stats",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def get_buffer_stats() -> BufferStatsResponse:
    """
    Get I/Q buffer statistics.

    Monitor the health of the data pipeline:
    - Buffer utilization indicates processing vs capture rate
    - Overflows mean capture is faster than processing
    - Underflows mean processing is starved for data

    For dashboard integration, poll this endpoint periodically.
    """
    # In production, this would read actual buffer stats
    return BufferStatsResponse(
        total_buffers=64,
        buffers_in_use=12,
        buffer_size_samples=65536,
        total_samples_processed=1_000_000_000,
        overflows=0,
        underflows=0,
        avg_fill_percent=18.75,
        gpu_buffers_allocated=16,
        gpu_memory_used_mb=256.0
    )


@router.get(
    "/captures",
    response_model=dict,
    description="List all captures in memory.",
    operation_id="list_captures",
)
async def list_captures() -> dict:
    """List all captures available in memory."""
    return {
        "captures": [
            {
                "id": c["id"],
                "center_freq_hz": c["center_freq_hz"],
                "sample_rate_hz": c["sample_rate_hz"],
                "num_samples": c["num_samples"],
                "format": c["format"],
                "data_size_bytes": c["data_size"]
            }
            for c in _captures.values()
        ]
    }
