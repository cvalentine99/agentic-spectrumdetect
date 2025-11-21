"""
Pydantic schemas for spectrum analysis MCP tools.

These schemas define the API contracts for spectrum monitoring, sweeps,
power measurements, and occupancy analysis.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SpectrumMeasurement(BaseModel):
    """Single spectrum measurement result."""

    center_freq_hz: int = Field(..., description="Center frequency of measurement")
    span_hz: int = Field(..., description="Frequency span covered")
    fft_size: int = Field(..., description="Number of FFT bins")
    bin_hz: float = Field(..., description="Frequency resolution per bin (Hz)")
    magnitudes_dbm: list[float] = Field(..., description="Magnitude values in dBm per bin")
    freq_start_hz: int = Field(..., description="Start frequency of span")
    freq_stop_hz: int = Field(..., description="Stop frequency of span")
    timestamp_ns: int = Field(..., description="Measurement timestamp (ns since epoch)")
    integration_count: int = Field(1, description="Number of FFTs averaged")


class PowerMeasurementRequest(BaseModel):
    """Request for integrated power measurement in a frequency band."""

    start_freq_hz: int = Field(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Start frequency in Hz"
    )
    stop_freq_hz: int = Field(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Stop frequency in Hz"
    )
    integration_time_ms: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Integration time in milliseconds"
    )
    detector_type: str = Field(
        default="rms",
        description="Detector type: 'peak', 'rms', 'average', 'sample'"
    )


class PowerMeasurementResponse(BaseModel):
    """Response from power measurement."""

    success: bool = Field(..., description="Whether measurement succeeded")
    total_power_dbm: float = Field(..., description="Total integrated power in dBm")
    peak_power_dbm: float = Field(..., description="Peak power detected in dBm")
    peak_freq_hz: int = Field(..., description="Frequency of peak power")
    average_power_dbm: float = Field(..., description="Average power across band")
    bandwidth_hz: int = Field(..., description="Measurement bandwidth")
    noise_floor_dbm: float = Field(..., description="Estimated noise floor")
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio")
    timestamp_ns: int = Field(..., description="Measurement timestamp")


class SpectrumSweepRequest(BaseModel):
    """Request for a spectrum sweep across multiple frequencies."""

    start_freq_hz: int = Field(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Sweep start frequency"
    )
    stop_freq_hz: int = Field(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Sweep stop frequency"
    )
    step_hz: Optional[int] = Field(
        None,
        ge=1_000,
        description="Step size between measurements (None = auto based on bandwidth)"
    )
    rbw_hz: int = Field(
        default=10_000,
        ge=100,
        le=1_000_000,
        description="Resolution bandwidth in Hz"
    )
    averaging: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of sweeps to average"
    )


class SpectrumSweepResponse(BaseModel):
    """Response from spectrum sweep."""

    success: bool = Field(..., description="Whether sweep completed")
    start_freq_hz: int = Field(..., description="Actual sweep start frequency")
    stop_freq_hz: int = Field(..., description="Actual sweep stop frequency")
    num_points: int = Field(..., description="Number of frequency points")
    frequencies_hz: list[int] = Field(..., description="Frequency points (Hz)")
    magnitudes_dbm: list[float] = Field(..., description="Power at each frequency (dBm)")
    sweep_time_ms: float = Field(..., description="Total sweep time")
    rbw_hz: int = Field(..., description="Actual resolution bandwidth used")


class OccupancyRequest(BaseModel):
    """Request for channel occupancy measurement."""

    center_freq_hz: int = Field(..., description="Center frequency of channel")
    bandwidth_hz: int = Field(..., description="Channel bandwidth in Hz")
    threshold_dbm: float = Field(
        default=-80.0,
        description="Power threshold for 'occupied' detection"
    )
    measurement_time_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Total measurement period in milliseconds"
    )
    sample_interval_ms: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Interval between samples"
    )


class OccupancyResponse(BaseModel):
    """Response from occupancy measurement."""

    success: bool = Field(..., description="Whether measurement completed")
    occupancy_percent: float = Field(..., description="Percentage of time signal detected")
    duty_cycle: float = Field(..., description="Duty cycle of detected signal")
    num_transmissions: int = Field(..., description="Number of distinct transmissions detected")
    avg_tx_duration_ms: float = Field(..., description="Average transmission duration")
    avg_power_when_active_dbm: float = Field(..., description="Average power during active periods")
    max_power_dbm: float = Field(..., description="Maximum power observed")
    min_power_dbm: float = Field(..., description="Minimum power observed")
    measurement_time_ms: int = Field(..., description="Actual measurement duration")


class SignalDetection(BaseModel):
    """Description of a detected signal."""

    freq_start_hz: int = Field(..., description="Signal start frequency")
    freq_stop_hz: int = Field(..., description="Signal stop frequency")
    center_freq_hz: int = Field(..., description="Signal center frequency")
    bandwidth_hz: int = Field(..., description="Signal bandwidth")
    peak_power_dbm: float = Field(..., description="Peak signal power")
    avg_power_dbm: float = Field(..., description="Average signal power")
    snr_db: float = Field(..., description="Signal-to-noise ratio")
    classification: Optional[str] = Field(None, description="Signal classification if available")
    confidence: Optional[float] = Field(None, description="Classification confidence 0-1")
    modulation: Optional[str] = Field(None, description="Detected modulation type")
    timestamp_ns: int = Field(..., description="Detection timestamp")


class SignalSearchRequest(BaseModel):
    """Request to search for signals in a frequency range."""

    start_freq_hz: int = Field(..., description="Search start frequency")
    stop_freq_hz: int = Field(..., description="Search stop frequency")
    min_snr_db: float = Field(
        default=6.0,
        ge=0.0,
        description="Minimum SNR threshold for detection"
    )
    min_bandwidth_hz: int = Field(
        default=1000,
        ge=100,
        description="Minimum signal bandwidth to detect"
    )
    classify: bool = Field(
        default=False,
        description="Run classification on detected signals"
    )


class SignalSearchResponse(BaseModel):
    """Response from signal search."""

    success: bool = Field(..., description="Whether search completed")
    signals: list[SignalDetection] = Field(..., description="List of detected signals")
    search_time_ms: float = Field(..., description="Time taken for search")
    noise_floor_dbm: float = Field(..., description="Estimated noise floor")
    span_hz: int = Field(..., description="Total frequency span searched")


class WaterfallFrame(BaseModel):
    """Single frame of waterfall data for streaming to dashboard."""

    center_freq_hz: int = Field(..., description="Center frequency")
    span_hz: int = Field(..., description="Frequency span")
    timestamp_ns: int = Field(..., description="Frame timestamp")
    magnitudes_db: list[float] = Field(..., description="Magnitude data (low to high freq)")
    frame_index: int = Field(..., description="Sequential frame index")


class SpectrogramData(BaseModel):
    """Spectrogram data structure for bulk transfer or storage."""

    center_freq_hz: int = Field(..., description="Center frequency")
    sample_rate_hz: int = Field(..., description="Sample rate")
    fft_size: int = Field(..., description="FFT size")
    hop_size: int = Field(..., description="Hop size between FFTs")
    num_frames: int = Field(..., description="Number of time frames")
    start_time_ns: int = Field(..., description="Start timestamp")
    end_time_ns: int = Field(..., description="End timestamp")
    data_format: str = Field(default="float32", description="Data format (float32, int16)")
    data_base64: Optional[str] = Field(None, description="Base64-encoded magnitude data")
    data_url: Optional[str] = Field(None, description="URL to fetch data if not inline")
