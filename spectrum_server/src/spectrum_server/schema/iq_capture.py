"""
Pydantic schemas for I/Q capture and SIGMF export MCP tools.

These schemas define the API contracts for raw I/Q data capture,
recording, playback, and SIGMF metadata generation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class CaptureRequest(BaseModel):
    """Request to capture raw I/Q samples."""

    num_samples: int = Field(
        ...,
        ge=1024,
        le=100_000_000,
        description="Number of complex samples to capture"
    )
    center_freq_hz: Optional[int] = Field(
        None,
        ge=70_000_000,
        le=6_000_000_000,
        description="Center frequency (None = use current)"
    )
    sample_rate_hz: Optional[int] = Field(
        None,
        ge=1_000_000,
        le=56_000_000,
        description="Sample rate (None = use current)"
    )
    gain_db: Optional[float] = Field(
        None,
        ge=0.0,
        le=76.0,
        description="RF gain (None = use current)"
    )
    output_format: Literal["cf32", "ci16", "ci8"] = Field(
        default="cf32",
        description="Sample format: cf32 (complex float32), ci16 (complex int16), ci8 (complex int8)"
    )


class CaptureResponse(BaseModel):
    """Response from capture operation."""

    success: bool = Field(..., description="Whether capture succeeded")
    capture_id: str = Field(..., description="Unique identifier for this capture")
    num_samples: int = Field(..., description="Actual number of samples captured")
    center_freq_hz: int = Field(..., description="Center frequency used")
    sample_rate_hz: int = Field(..., description="Sample rate used")
    gain_db: float = Field(..., description="Gain used")
    duration_ms: float = Field(..., description="Capture duration in milliseconds")
    data_size_bytes: int = Field(..., description="Size of captured data in bytes")
    overflow_count: int = Field(0, description="Number of overflows during capture")
    start_timestamp_ns: int = Field(..., description="Capture start timestamp")
    storage_path: Optional[str] = Field(None, description="Path to stored data file")


class RecordingRequest(BaseModel):
    """Request to start a continuous I/Q recording."""

    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Base filename for recording (without extension)"
    )
    duration_seconds: Optional[float] = Field(
        None,
        ge=0.1,
        le=3600.0,
        description="Recording duration (None = record until stopped)"
    )
    max_file_size_mb: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum file size before rotation in MB"
    )
    center_freq_hz: Optional[int] = Field(
        None,
        description="Center frequency for recording"
    )
    sample_rate_hz: Optional[int] = Field(
        None,
        description="Sample rate for recording"
    )
    output_format: Literal["cf32", "ci16", "ci8"] = Field(
        default="ci16",
        description="Sample format for recording"
    )
    generate_sigmf: bool = Field(
        default=True,
        description="Generate SIGMF metadata sidecar"
    )


class RecordingResponse(BaseModel):
    """Response from recording operation."""

    success: bool = Field(..., description="Whether recording started")
    recording_id: str = Field(..., description="Unique identifier for this recording")
    filename: str = Field(..., description="Full path to recording file")
    sigmf_meta_path: Optional[str] = Field(None, description="Path to SIGMF metadata file")
    status: Literal["recording", "stopped", "error"] = Field(..., description="Recording status")


class RecordingStatusRequest(BaseModel):
    """Request to check recording status."""

    recording_id: str = Field(..., description="Recording ID to query")


class RecordingStatusResponse(BaseModel):
    """Recording status response."""

    recording_id: str = Field(..., description="Recording identifier")
    status: Literal["recording", "stopped", "error"] = Field(..., description="Current status")
    samples_recorded: int = Field(..., description="Total samples recorded so far")
    bytes_written: int = Field(..., description="Total bytes written")
    duration_seconds: float = Field(..., description="Recording duration so far")
    overflow_count: int = Field(..., description="Number of overflows")
    files: list[str] = Field(..., description="List of files created")


class SigMFMetadata(BaseModel):
    """SIGMF metadata structure (simplified)."""

    # Global metadata
    datatype: str = Field(..., description="Sample format (e.g., 'cf32_le', 'ci16_le')")
    sample_rate: float = Field(..., description="Sample rate in Hz")
    version: str = Field(default="1.0.0", description="SIGMF version")
    description: Optional[str] = Field(None, description="Recording description")
    author: Optional[str] = Field(None, description="Recording author")
    hw: Optional[str] = Field(None, description="Hardware description")
    recorder: str = Field(default="agentic-spectrumdetect", description="Recording software")

    # Capture metadata
    center_freq_hz: float = Field(..., description="Center frequency at start")
    datetime_start: str = Field(..., description="ISO 8601 start timestamp")
    global_index: int = Field(default=0, description="Starting sample index")

    # Annotations (optional)
    annotations: list[dict] = Field(default_factory=list, description="Signal annotations")


class SigMFExportRequest(BaseModel):
    """Request to export capture as SIGMF archive."""

    capture_id: str = Field(..., description="Capture ID to export")
    description: Optional[str] = Field(None, description="Description for metadata")
    author: Optional[str] = Field(None, description="Author name")
    annotations: list[dict] = Field(
        default_factory=list,
        description="Signal annotations to include"
    )
    output_path: Optional[str] = Field(
        None,
        description="Output path (None = auto-generate)"
    )
    archive_format: Literal["sigmf", "tar", "zip"] = Field(
        default="sigmf",
        description="Archive format"
    )


class SigMFExportResponse(BaseModel):
    """Response from SIGMF export."""

    success: bool = Field(..., description="Whether export succeeded")
    sigmf_data_path: str = Field(..., description="Path to .sigmf-data file")
    sigmf_meta_path: str = Field(..., description="Path to .sigmf-meta file")
    archive_path: Optional[str] = Field(None, description="Path to archive if created")
    file_size_bytes: int = Field(..., description="Total file size")


class PlaybackRequest(BaseModel):
    """Request to play back recorded I/Q data."""

    source: str = Field(..., description="Path to SIGMF recording or capture ID")
    output_destination: Literal["stream", "gpu_buffer", "file"] = Field(
        default="stream",
        description="Where to send playback data"
    )
    loop: bool = Field(
        default=False,
        description="Loop playback continuously"
    )
    start_sample: int = Field(
        default=0,
        ge=0,
        description="Sample index to start playback"
    )
    num_samples: Optional[int] = Field(
        None,
        description="Number of samples to play (None = all)"
    )
    playback_rate: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Playback speed multiplier"
    )


class PlaybackResponse(BaseModel):
    """Response from playback operation."""

    success: bool = Field(..., description="Whether playback started")
    playback_id: str = Field(..., description="Unique identifier for this playback")
    source: str = Field(..., description="Source being played")
    total_samples: int = Field(..., description="Total samples in source")
    status: Literal["playing", "paused", "stopped", "error"] = Field(..., description="Playback status")


class BufferStatsResponse(BaseModel):
    """I/Q buffer statistics for monitoring."""

    total_buffers: int = Field(..., description="Total buffer slots")
    buffers_in_use: int = Field(..., description="Currently used buffers")
    buffer_size_samples: int = Field(..., description="Samples per buffer")
    total_samples_processed: int = Field(..., description="Total samples processed")
    overflows: int = Field(..., description="Buffer overflow count")
    underflows: int = Field(..., description="Buffer underflow count")
    avg_fill_percent: float = Field(..., description="Average buffer utilization")
    gpu_buffers_allocated: int = Field(..., description="GPU-side buffers allocated")
    gpu_memory_used_mb: float = Field(..., description="GPU memory used for buffers")
