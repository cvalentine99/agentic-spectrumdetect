"""
Pydantic schemas for SDR control MCP tools.

These schemas define the API contracts for Ettus B210 SDR control,
including tuning, gain management, antenna selection, and configuration.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from enum import Enum


class AntennaPort(str, Enum):
    """B210 antenna port options."""
    TX_RX = "TX/RX"
    RX2 = "RX2"
    AUTO = "AUTO"


class GainMode(str, Enum):
    """Gain control mode."""
    MANUAL = "manual"
    AUTO = "auto"


class SDRStatus(BaseModel):
    """Current SDR hardware status."""

    connected: bool = Field(..., description="Whether SDR is connected and responding")
    device_name: str = Field(..., description="Device identifier (e.g., 'Ettus B210')")
    serial: Optional[str] = Field(None, description="Device serial number")
    center_freq_hz: int = Field(..., description="Current center frequency in Hz")
    sample_rate_hz: int = Field(..., description="Current sample rate in Hz")
    bandwidth_hz: int = Field(..., description="Current analog bandwidth in Hz")
    gain_db: float = Field(..., description="Current RF gain in dB")
    gain_mode: GainMode = Field(..., description="Gain control mode (manual/auto)")
    antenna: AntennaPort = Field(..., description="Selected antenna port")
    lo_locked: bool = Field(..., description="Local oscillator lock status")
    temperature_c: Optional[float] = Field(None, description="Device temperature in Celsius")
    overflow_count: int = Field(0, description="Number of overflow events detected")


class TuneRequest(BaseModel):
    """Request to tune SDR to specific frequency and bandwidth."""

    center_freq_hz: int = Field(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Center frequency in Hz (70 MHz - 6 GHz for B210)"
    )
    sample_rate_hz: int = Field(
        default=50_000_000,
        ge=1_000_000,
        le=56_000_000,
        description="Sample rate in Hz (1-56 MHz for B210)"
    )
    bandwidth_hz: Optional[int] = Field(
        None,
        ge=200_000,
        le=56_000_000,
        description="Analog filter bandwidth in Hz (None = auto-select based on sample rate)"
    )

    @field_validator('bandwidth_hz')
    @classmethod
    def validate_bandwidth(cls, v, info):
        """Ensure bandwidth doesn't exceed sample rate."""
        if v is not None and 'sample_rate_hz' in info.data:
            sr = info.data['sample_rate_hz']
            if v > sr * 1.5:
                raise ValueError(f"Bandwidth {v} Hz exceeds 1.5x sample rate {sr} Hz")
        return v


class TuneResponse(BaseModel):
    """Response from tune operation."""

    success: bool = Field(..., description="Whether tune operation succeeded")
    actual_center_freq_hz: int = Field(..., description="Actual tuned frequency (may differ slightly)")
    actual_sample_rate_hz: int = Field(..., description="Actual sample rate achieved")
    actual_bandwidth_hz: int = Field(..., description="Actual analog bandwidth set")
    lo_offset_hz: Optional[int] = Field(None, description="LO offset applied to avoid DC spike")
    tune_time_ms: float = Field(..., description="Time taken to complete tune in milliseconds")


class GainRequest(BaseModel):
    """Request to set RF gain."""

    gain_db: float = Field(
        ...,
        ge=0.0,
        le=76.0,
        description="RF gain in dB (0-76 dB for B210)"
    )
    gain_mode: GainMode = Field(
        default=GainMode.MANUAL,
        description="Gain control mode"
    )
    apply_to_channel: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Which RX channel to configure (0 or 1)"
    )


class GainResponse(BaseModel):
    """Response from gain operation."""

    success: bool = Field(..., description="Whether gain set succeeded")
    actual_gain_db: float = Field(..., description="Actual gain value set")
    gain_range_db: tuple[float, float] = Field(..., description="Valid gain range (min, max)")


class AntennaRequest(BaseModel):
    """Request to select antenna port."""

    antenna: AntennaPort = Field(..., description="Antenna port to select")
    channel: int = Field(
        default=0,
        ge=0,
        le=1,
        description="RX channel (0 or 1)"
    )


class AntennaResponse(BaseModel):
    """Response from antenna selection."""

    success: bool = Field(..., description="Whether antenna selection succeeded")
    antenna: AntennaPort = Field(..., description="Selected antenna port")
    available_antennas: list[str] = Field(..., description="List of available antenna ports")


class CalibrationRequest(BaseModel):
    """Request to run SDR calibration."""

    calibrate_dc_offset: bool = Field(True, description="Calibrate DC offset")
    calibrate_iq_balance: bool = Field(True, description="Calibrate IQ balance")
    center_freq_hz: Optional[int] = Field(None, description="Frequency for calibration (None = current)")


class CalibrationResponse(BaseModel):
    """Response from calibration."""

    success: bool = Field(..., description="Whether calibration succeeded")
    dc_offset_i: Optional[float] = Field(None, description="DC offset correction for I channel")
    dc_offset_q: Optional[float] = Field(None, description="DC offset correction for Q channel")
    iq_balance: Optional[float] = Field(None, description="IQ balance correction value")
    calibration_time_ms: float = Field(..., description="Time taken for calibration")


class StreamControlRequest(BaseModel):
    """Request to start/stop I/Q data streaming."""

    action: Literal["start", "stop"] = Field(..., description="Stream control action")
    num_samples: Optional[int] = Field(
        None,
        ge=1024,
        description="Number of samples to capture (None = continuous)"
    )
    channel: int = Field(
        default=0,
        ge=0,
        le=1,
        description="RX channel to stream from"
    )


class StreamControlResponse(BaseModel):
    """Response from stream control operation."""

    success: bool = Field(..., description="Whether stream control succeeded")
    streaming: bool = Field(..., description="Current streaming state")
    samples_received: int = Field(0, description="Total samples received since stream start")
    overflow_count: int = Field(0, description="Number of overflows detected")
    data_rate_mbps: Optional[float] = Field(None, description="Current data rate in Mbps")
