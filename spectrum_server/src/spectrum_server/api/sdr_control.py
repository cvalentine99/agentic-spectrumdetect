"""
FastAPI router for SDR control MCP tools.

These endpoints are automatically exposed via FastMCP to provide
AI agents with direct Ettus B210 SDR control capabilities.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from spectrum_server.sdr_control import get_sdr_controller, SDRConfig, SDRBackend
from spectrum_server.schema.sdr_control import (
    SDRStatus, TuneRequest, TuneResponse, GainRequest, GainResponse,
    AntennaRequest, AntennaResponse, CalibrationRequest, CalibrationResponse,
    StreamControlRequest, StreamControlResponse, AntennaPort, GainMode
)

router = APIRouter()


@router.get(
    "/status",
    response_model=SDRStatus,
    description="Get current SDR hardware status including frequency, gain, antenna, and connection state.",
    operation_id="get_sdr_status",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def get_sdr_status() -> SDRStatus:
    """
    Get comprehensive SDR status.

    Returns current configuration and state of the Ettus B210 SDR including:
    - Connection status and device info
    - Current frequency, sample rate, and bandwidth
    - Gain settings and mode
    - Antenna selection
    - LO lock status and overflow counts
    """
    controller = get_sdr_controller()
    return await controller.get_status()


@router.post(
    "/tune",
    response_model=TuneResponse,
    description="Tune SDR to specified center frequency and sample rate. For Ettus B210: freq range 70 MHz - 6 GHz, sample rate up to 56 MHz.",
    operation_id="tune_sdr",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def tune_sdr(
    center_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Center frequency in Hz (70 MHz - 6 GHz)"
    ),
    sample_rate_hz: int = Query(
        default=50_000_000,
        ge=1_000_000,
        le=56_000_000,
        description="Sample rate in Hz (1-56 MHz)"
    ),
    bandwidth_hz: Optional[int] = Query(
        default=None,
        ge=200_000,
        le=56_000_000,
        description="Analog filter bandwidth in Hz (None = auto)"
    ),
) -> TuneResponse:
    """
    Tune the SDR to a new center frequency.

    This operation reconfigures the RF frontend:
    - Sets the local oscillator to the specified center frequency
    - Configures the ADC sample rate
    - Optionally sets the analog filter bandwidth

    Common frequency bands:
    - WiFi 2.4 GHz: 2400-2500 MHz (center: 2450 MHz)
    - WiFi 5 GHz: 5150-5850 MHz
    - ISM 900 MHz: 902-928 MHz (center: 915 MHz)
    - GPS L1: 1575.42 MHz

    Note: Tuning causes a brief interruption in data streaming.
    """
    controller = get_sdr_controller()
    request = TuneRequest(
        center_freq_hz=center_freq_hz,
        sample_rate_hz=sample_rate_hz,
        bandwidth_hz=bandwidth_hz
    )
    return await controller.tune(request)


@router.post(
    "/gain",
    response_model=GainResponse,
    description="Set RF gain for the SDR. For Ettus B210: 0-76 dB range.",
    operation_id="set_sdr_gain",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def set_sdr_gain(
    gain_db: float = Query(
        ...,
        ge=0.0,
        le=76.0,
        description="RF gain in dB (0-76 dB for B210)"
    ),
    gain_mode: str = Query(
        default="manual",
        description="Gain mode: 'manual' or 'auto'"
    ),
    channel: int = Query(
        default=0,
        ge=0,
        le=1,
        description="RX channel (0 or 1)"
    ),
) -> GainResponse:
    """
    Set RF gain for signal reception.

    Gain settings affect signal amplitude and noise floor:
    - Higher gain (60-76 dB): Better sensitivity for weak signals
    - Medium gain (30-60 dB): Balanced for moderate signals
    - Lower gain (0-30 dB): Prevents saturation for strong signals

    Auto mode lets the hardware optimize gain automatically.
    Manual mode gives precise control for consistent measurements.
    """
    controller = get_sdr_controller()
    mode = GainMode.AUTO if gain_mode.lower() == "auto" else GainMode.MANUAL
    request = GainRequest(gain_db=gain_db, gain_mode=mode, apply_to_channel=channel)
    return await controller.set_gain(request)


@router.post(
    "/antenna",
    response_model=AntennaResponse,
    description="Select antenna port for the SDR. B210 supports TX/RX and RX2 ports.",
    operation_id="set_sdr_antenna",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def set_sdr_antenna(
    antenna: str = Query(
        ...,
        description="Antenna port: 'TX/RX', 'RX2', or 'AUTO'"
    ),
    channel: int = Query(
        default=0,
        ge=0,
        le=1,
        description="RX channel (0 or 1)"
    ),
) -> AntennaResponse:
    """
    Select which antenna port to use for reception.

    Ettus B210 antenna ports:
    - TX/RX: Combined transmit/receive port, full frequency range
    - RX2: Receive-only port, may have different characteristics

    Select based on your antenna connection and measurement needs.
    """
    controller = get_sdr_controller()
    try:
        port = AntennaPort(antenna)
    except ValueError:
        port = AntennaPort.TX_RX
    request = AntennaRequest(antenna=port, channel=channel)
    return await controller.set_antenna(request)


@router.post(
    "/calibrate",
    response_model=CalibrationResponse,
    description="Run DC offset and IQ balance calibration on the SDR.",
    operation_id="calibrate_sdr",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def calibrate_sdr(
    calibrate_dc_offset: bool = Query(
        default=True,
        description="Enable DC offset calibration"
    ),
    calibrate_iq_balance: bool = Query(
        default=True,
        description="Enable IQ balance calibration"
    ),
    center_freq_hz: Optional[int] = Query(
        default=None,
        description="Frequency for calibration (None = current frequency)"
    ),
) -> CalibrationResponse:
    """
    Run calibration routines to improve signal quality.

    DC Offset Calibration:
    - Removes DC bias from I and Q channels
    - Reduces center-frequency spike in spectrum

    IQ Balance Calibration:
    - Corrects amplitude and phase imbalance between I and Q
    - Reduces image frequency artifacts

    Best practice: Run calibration after changing frequency significantly.
    """
    controller = get_sdr_controller()
    request = CalibrationRequest(
        calibrate_dc_offset=calibrate_dc_offset,
        calibrate_iq_balance=calibrate_iq_balance,
        center_freq_hz=center_freq_hz
    )
    return await controller.calibrate(request)


@router.post(
    "/stream",
    response_model=StreamControlResponse,
    description="Start or stop I/Q data streaming from the SDR.",
    operation_id="control_sdr_stream",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def control_sdr_stream(
    action: str = Query(
        ...,
        description="Action: 'start' or 'stop'"
    ),
    num_samples: Optional[int] = Query(
        default=None,
        ge=1024,
        description="Number of samples to capture (None = continuous)"
    ),
    channel: int = Query(
        default=0,
        ge=0,
        le=1,
        description="RX channel to stream from"
    ),
) -> StreamControlResponse:
    """
    Control I/Q data streaming from the SDR.

    Start: Begin streaming raw I/Q samples from the SDR.
    - Data flows to the real-time processing pipeline
    - Samples are available via WebSocket at /ws/spectrum

    Stop: Halt streaming and free resources.

    For continuous monitoring, start streaming without num_samples.
    For burst capture, specify the exact sample count needed.
    """
    controller = get_sdr_controller()
    if action.lower() not in ("start", "stop"):
        raise HTTPException(status_code=400, detail="Action must be 'start' or 'stop'")

    request = StreamControlRequest(
        action=action.lower(),  # type: ignore
        num_samples=num_samples,
        channel=channel
    )
    return await controller.stream_control(request)


@router.get(
    "/frequency-bands",
    response_model=dict,
    description="Get predefined frequency band configurations for common use cases.",
    operation_id="get_frequency_bands",
)
async def get_frequency_bands() -> dict:
    """
    Get predefined frequency band configurations.

    Returns common frequency bands with recommended settings
    for the Ettus B210 SDR. Use these as starting points
    for spectrum monitoring tasks.
    """
    return {
        "bands": [
            {
                "name": "WiFi 2.4 GHz",
                "center_freq_hz": 2_450_000_000,
                "sample_rate_hz": 50_000_000,
                "bandwidth_hz": 50_000_000,
                "description": "IEEE 802.11 b/g/n/ax 2.4 GHz band"
            },
            {
                "name": "WiFi 5 GHz Lower",
                "center_freq_hz": 5_250_000_000,
                "sample_rate_hz": 56_000_000,
                "bandwidth_hz": 56_000_000,
                "description": "IEEE 802.11 a/n/ac/ax 5 GHz UNII-1/UNII-2"
            },
            {
                "name": "WiFi 5 GHz Upper",
                "center_freq_hz": 5_700_000_000,
                "sample_rate_hz": 56_000_000,
                "bandwidth_hz": 56_000_000,
                "description": "IEEE 802.11 5 GHz UNII-3/ISM"
            },
            {
                "name": "ISM 900 MHz",
                "center_freq_hz": 915_000_000,
                "sample_rate_hz": 26_000_000,
                "bandwidth_hz": 26_000_000,
                "description": "902-928 MHz ISM band (LoRa, Zigbee, etc.)"
            },
            {
                "name": "GPS L1",
                "center_freq_hz": 1_575_420_000,
                "sample_rate_hz": 10_000_000,
                "bandwidth_hz": 10_000_000,
                "description": "GPS L1 C/A signal at 1575.42 MHz"
            },
            {
                "name": "LTE Band 7 (2600 MHz)",
                "center_freq_hz": 2_655_000_000,
                "sample_rate_hz": 50_000_000,
                "bandwidth_hz": 50_000_000,
                "description": "LTE FDD Band 7 downlink"
            },
            {
                "name": "FM Broadcast",
                "center_freq_hz": 98_000_000,
                "sample_rate_hz": 25_000_000,
                "bandwidth_hz": 25_000_000,
                "description": "FM radio broadcast 88-108 MHz"
            },
            {
                "name": "ADS-B (1090 MHz)",
                "center_freq_hz": 1_090_000_000,
                "sample_rate_hz": 10_000_000,
                "bandwidth_hz": 10_000_000,
                "description": "Aircraft transponder signals"
            },
            {
                "name": "Bluetooth/ZigBee",
                "center_freq_hz": 2_440_000_000,
                "sample_rate_hz": 25_000_000,
                "bandwidth_hz": 25_000_000,
                "description": "2.4 GHz ISM for BLE and ZigBee"
            },
            {
                "name": "DECT (1880-1900 MHz)",
                "center_freq_hz": 1_890_000_000,
                "sample_rate_hz": 25_000_000,
                "bandwidth_hz": 25_000_000,
                "description": "Digital Enhanced Cordless Telecommunications"
            }
        ],
        "device_limits": {
            "min_freq_hz": 70_000_000,
            "max_freq_hz": 6_000_000_000,
            "max_sample_rate_hz": 56_000_000,
            "max_bandwidth_hz": 56_000_000,
            "gain_range_db": [0.0, 76.0]
        }
    }


@router.post(
    "/quick-tune",
    response_model=TuneResponse,
    description="Quickly tune to a predefined frequency band by name.",
    operation_id="quick_tune_sdr",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def quick_tune_sdr(
    band_name: str = Query(
        ...,
        description="Band name (e.g., 'WiFi 2.4 GHz', 'ISM 900 MHz', 'GPS L1')"
    ),
) -> TuneResponse:
    """
    Quickly tune to a predefined frequency band.

    Use get_frequency_bands to see available options.
    This is a convenience endpoint that applies optimal
    settings for common monitoring scenarios.
    """
    bands = (await get_frequency_bands())["bands"]
    band = next((b for b in bands if b["name"].lower() == band_name.lower()), None)

    if not band:
        raise HTTPException(
            status_code=404,
            detail=f"Band '{band_name}' not found. Use /sdr/frequency-bands to see available bands."
        )

    controller = get_sdr_controller()
    request = TuneRequest(
        center_freq_hz=band["center_freq_hz"],
        sample_rate_hz=band["sample_rate_hz"],
        bandwidth_hz=band["bandwidth_hz"]
    )
    return await controller.tune(request)
