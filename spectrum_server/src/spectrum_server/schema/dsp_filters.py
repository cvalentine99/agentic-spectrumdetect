"""
Pydantic schemas for GPU-accelerated DSP filter MCP tools.

These schemas define the API contracts for real-time signal processing
operations including filtering, decimation, channelization, and demodulation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class FilterType(str, Enum):
    """Available filter types."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"
    FIR = "fir"
    IIR = "iir"


class WindowType(str, Enum):
    """FFT window types."""
    RECTANGULAR = "rectangular"
    HANNING = "hanning"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    BLACKMAN_HARRIS = "blackman_harris"
    KAISER = "kaiser"
    FLAT_TOP = "flat_top"


class FilterDesignRequest(BaseModel):
    """Request to design a digital filter."""

    filter_type: FilterType = Field(..., description="Type of filter to design")
    cutoff_freq_hz: float = Field(..., gt=0, description="Cutoff frequency in Hz")
    cutoff_freq_high_hz: Optional[float] = Field(
        None, gt=0,
        description="Upper cutoff for bandpass/bandstop filters"
    )
    sample_rate_hz: int = Field(..., gt=0, description="Sample rate in Hz")
    num_taps: int = Field(
        default=101,
        ge=3,
        le=4097,
        description="Number of filter taps (FIR) or order (IIR)"
    )
    window: WindowType = Field(
        default=WindowType.HAMMING,
        description="Window function for FIR design"
    )
    passband_ripple_db: float = Field(
        default=0.1,
        ge=0.01,
        le=3.0,
        description="Passband ripple in dB (IIR only)"
    )
    stopband_atten_db: float = Field(
        default=60.0,
        ge=20.0,
        le=120.0,
        description="Stopband attenuation in dB"
    )


class FilterDesignResponse(BaseModel):
    """Response from filter design."""

    success: bool = Field(..., description="Whether design succeeded")
    filter_id: str = Field(..., description="Unique identifier for this filter")
    filter_type: FilterType = Field(..., description="Type of filter designed")
    num_taps: int = Field(..., description="Actual number of taps")
    cutoff_freq_hz: float = Field(..., description="Actual cutoff frequency")
    passband_ripple_db: float = Field(..., description="Achieved passband ripple")
    stopband_atten_db: float = Field(..., description="Achieved stopband attenuation")
    group_delay_samples: float = Field(..., description="Group delay in samples")
    coefficients_b: list[float] = Field(..., description="Numerator coefficients")
    coefficients_a: Optional[list[float]] = Field(
        None, description="Denominator coefficients (IIR only)"
    )


class FilterApplyRequest(BaseModel):
    """Request to apply a filter to data."""

    filter_id: str = Field(..., description="Filter ID to apply")
    source: str = Field(..., description="Data source: capture ID or 'live'")
    output_destination: Literal["stream", "capture", "gpu_buffer"] = Field(
        default="stream",
        description="Where to send filtered output"
    )
    gpu_accelerated: bool = Field(
        default=True,
        description="Use GPU acceleration if available"
    )


class FilterApplyResponse(BaseModel):
    """Response from filter application."""

    success: bool = Field(..., description="Whether filter application succeeded")
    processing_rate_msps: float = Field(..., description="Processing rate in MSPS")
    latency_us: float = Field(..., description="Filter latency in microseconds")
    gpu_used: bool = Field(..., description="Whether GPU was used")
    output_capture_id: Optional[str] = Field(
        None, description="Capture ID if output was saved"
    )


class DecimationRequest(BaseModel):
    """Request to decimate (downsample) data."""

    decimation_factor: int = Field(
        ...,
        ge=2,
        le=1024,
        description="Decimation factor (output_rate = input_rate / factor)"
    )
    source: str = Field(..., description="Data source: capture ID or 'live'")
    filter_type: Literal["cic", "fir", "halfband"] = Field(
        default="fir",
        description="Anti-aliasing filter type"
    )
    num_stages: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of decimation stages"
    )
    gpu_accelerated: bool = Field(
        default=True,
        description="Use GPU acceleration"
    )


class DecimationResponse(BaseModel):
    """Response from decimation."""

    success: bool = Field(..., description="Whether decimation succeeded")
    input_rate_hz: int = Field(..., description="Input sample rate")
    output_rate_hz: int = Field(..., description="Output sample rate")
    decimation_factor: int = Field(..., description="Actual decimation factor")
    processing_rate_msps: float = Field(..., description="Processing throughput")
    latency_us: float = Field(..., description="Decimation latency")


class ChannelizerRequest(BaseModel):
    """Request to channelize a wideband signal."""

    num_channels: int = Field(
        ...,
        ge=2,
        le=4096,
        description="Number of output channels"
    )
    channel_bandwidth_hz: int = Field(
        ...,
        gt=0,
        description="Bandwidth per channel in Hz"
    )
    source: str = Field(..., description="Data source")
    overlap_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Channel overlap percentage"
    )
    window: WindowType = Field(
        default=WindowType.HAMMING,
        description="Window function for polyphase filterbank"
    )
    output_channels: Optional[list[int]] = Field(
        None,
        description="Specific channels to output (None = all)"
    )


class ChannelizerResponse(BaseModel):
    """Response from channelization."""

    success: bool = Field(..., description="Whether channelization succeeded")
    num_channels: int = Field(..., description="Number of channels created")
    channel_bandwidth_hz: int = Field(..., description="Bandwidth per channel")
    channel_spacing_hz: int = Field(..., description="Spacing between channels")
    channel_freqs_hz: list[int] = Field(..., description="Center frequency of each channel")
    processing_rate_msps: float = Field(..., description="Processing throughput")


class SpectrogramRequest(BaseModel):
    """Request to compute a spectrogram."""

    fft_size: int = Field(
        default=2048,
        ge=64,
        le=65536,
        description="FFT size"
    )
    hop_size: int = Field(
        default=512,
        ge=1,
        description="Hop size between FFTs"
    )
    window: WindowType = Field(
        default=WindowType.BLACKMAN_HARRIS,
        description="Window function"
    )
    source: str = Field(..., description="Data source")
    num_frames: Optional[int] = Field(
        None,
        ge=1,
        description="Number of frames (None = all available)"
    )
    output_format: Literal["magnitude", "power", "log_power"] = Field(
        default="log_power",
        description="Output format for spectrogram values"
    )
    gpu_accelerated: bool = Field(
        default=True,
        description="Use GPU for FFT computation"
    )


class SpectrogramResponse(BaseModel):
    """Response from spectrogram computation."""

    success: bool = Field(..., description="Whether computation succeeded")
    fft_size: int = Field(..., description="FFT size used")
    hop_size: int = Field(..., description="Hop size used")
    num_frames: int = Field(..., description="Number of time frames")
    freq_resolution_hz: float = Field(..., description="Frequency resolution")
    time_resolution_ms: float = Field(..., description="Time resolution")
    processing_time_ms: float = Field(..., description="Computation time")
    gpu_used: bool = Field(..., description="Whether GPU was used")
    data_url: Optional[str] = Field(None, description="URL to fetch spectrogram data")


class DemodulationRequest(BaseModel):
    """Request to demodulate a signal."""

    modulation_type: Literal["am", "fm", "pm", "ask", "fsk", "psk", "qam"] = Field(
        ...,
        description="Modulation type to demodulate"
    )
    source: str = Field(..., description="Data source")
    center_freq_offset_hz: float = Field(
        default=0.0,
        description="Frequency offset from center for signal"
    )
    symbol_rate_hz: Optional[float] = Field(
        None, gt=0,
        description="Symbol rate for digital modulations"
    )
    fm_deviation_hz: Optional[float] = Field(
        None, gt=0,
        description="FM deviation for FM demodulation"
    )
    output_audio: bool = Field(
        default=False,
        description="Output demodulated audio (AM/FM only)"
    )


class DemodulationResponse(BaseModel):
    """Response from demodulation."""

    success: bool = Field(..., description="Whether demodulation succeeded")
    modulation_type: str = Field(..., description="Modulation type demodulated")
    estimated_snr_db: float = Field(..., description="Estimated SNR")
    symbol_rate_hz: Optional[float] = Field(None, description="Detected symbol rate")
    bits_demodulated: Optional[int] = Field(None, description="Number of bits recovered")
    audio_sample_rate_hz: Optional[int] = Field(None, description="Audio output sample rate")
    output_capture_id: Optional[str] = Field(None, description="ID of output data")


class RSSIMeasurement(BaseModel):
    """RSSI (Received Signal Strength Indicator) measurement."""

    rssi_dbm: float = Field(..., description="RSSI value in dBm")
    rssi_linear: float = Field(..., description="Linear power value")
    peak_dbm: float = Field(..., description="Peak power in measurement window")
    min_dbm: float = Field(..., description="Minimum power in measurement window")
    timestamp_ns: int = Field(..., description="Measurement timestamp")


class RSSIRequest(BaseModel):
    """Request for RSSI measurement."""

    integration_samples: int = Field(
        default=65536,
        ge=1024,
        le=10_000_000,
        description="Number of samples to integrate"
    )
    center_freq_hz: Optional[int] = Field(
        None,
        description="Specific frequency to measure (None = current)"
    )
    bandwidth_hz: Optional[int] = Field(
        None,
        description="Bandwidth to measure (None = full)"
    )


class PipelineConfig(BaseModel):
    """Configuration for a real-time processing pipeline."""

    name: str = Field(..., description="Pipeline name")
    stages: list[dict] = Field(
        ...,
        description="List of processing stages with their configurations"
    )
    input_sample_rate_hz: int = Field(..., description="Input sample rate")
    output_sample_rate_hz: int = Field(..., description="Output sample rate")
    buffer_size_samples: int = Field(
        default=65536,
        description="Buffer size between stages"
    )
    gpu_device_id: int = Field(
        default=0,
        description="GPU device to use for processing"
    )


class PipelineStatus(BaseModel):
    """Status of a processing pipeline."""

    pipeline_id: str = Field(..., description="Pipeline identifier")
    name: str = Field(..., description="Pipeline name")
    running: bool = Field(..., description="Whether pipeline is running")
    stages: list[str] = Field(..., description="List of stage names")
    samples_processed: int = Field(..., description="Total samples processed")
    processing_rate_msps: float = Field(..., description="Current processing rate")
    latency_us: float = Field(..., description="Total pipeline latency")
    gpu_utilization_percent: float = Field(..., description="GPU utilization")
    buffer_utilization_percent: float = Field(..., description="Buffer utilization")
