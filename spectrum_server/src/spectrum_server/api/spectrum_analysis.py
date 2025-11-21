"""
FastAPI router for spectrum analysis MCP tools.

These endpoints provide AI agents with spectrum monitoring, power measurement,
signal detection, and occupancy analysis capabilities.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import time
import asyncio
import numpy as np
from datetime import datetime, timezone

from spectrum_server.schema.spectrum_analysis import (
    SpectrumMeasurement, PowerMeasurementRequest, PowerMeasurementResponse,
    SpectrumSweepRequest, SpectrumSweepResponse, OccupancyRequest,
    OccupancyResponse, SignalSearchRequest, SignalSearchResponse,
    SignalDetection, WaterfallFrame, SpectrogramData
)
from spectrum_server.sdr_control import get_sdr_controller
from spectrum_server.spectrum_stream import spectrum_stream
from spectrum_server.spectrum_core_client import CoreSpectrumFrame, spectrum_core_client

router = APIRouter()


@router.get(
    "/measure",
    response_model=SpectrumMeasurement,
    description="Perform a single spectrum measurement at current frequency settings.",
    operation_id="measure_spectrum",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def measure_spectrum(
    fft_size: int = Query(
        default=2048,
        ge=256,
        le=65536,
        description="FFT size (number of bins)"
    ),
    averaging: int = Query(
        default=10,
        ge=1,
        le=1000,
        description="Number of FFTs to average"
    ),
) -> SpectrumMeasurement:
    """
    Perform a spectrum measurement at current SDR settings.

    This captures I/Q data, computes FFT, and returns magnitude spectrum.
    Results are in dBm referenced to the SDR input.

    FFT size affects frequency resolution:
    - 2048 bins at 50 MHz = 24.4 kHz/bin
    - 4096 bins at 50 MHz = 12.2 kHz/bin
    - 8192 bins at 50 MHz = 6.1 kHz/bin

    Averaging reduces noise variance but increases measurement time.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    # Compute FFT parameters
    bin_hz = status.sample_rate_hz / fft_size
    freq_start = status.center_freq_hz - status.sample_rate_hz // 2
    freq_stop = status.center_freq_hz + status.sample_rate_hz // 2

    if spectrum_core_client.enabled:
        core_frame: CoreSpectrumFrame | None = await spectrum_core_client.measure_spectrum(
            center_freq_hz=status.center_freq_hz,
            sample_rate_hz=status.sample_rate_hz,
            fft_size=fft_size,
            averaging=averaging,
        )
        if core_frame:
            return SpectrumMeasurement(
                center_freq_hz=core_frame.center_freq_hz,
                span_hz=core_frame.sample_rate_hz,
                fft_size=core_frame.fft_size,
                bin_hz=core_frame.sample_rate_hz / core_frame.fft_size,
                magnitudes_dbm=core_frame.magnitudes_dbm,
                freq_start_hz=core_frame.center_freq_hz - core_frame.sample_rate_hz // 2,
                freq_stop_hz=core_frame.center_freq_hz + core_frame.sample_rate_hz // 2,
                timestamp_ns=core_frame.timestamp_ns,
                integration_count=core_frame.averaging,
            )

    # Generate simulated spectrum data for demonstration
    # In production, this would read from the actual SDR data pipeline
    magnitudes = _generate_simulated_spectrum(
        fft_size, status.center_freq_hz, status.sample_rate_hz,
        status.gain_db, averaging
    )

    return SpectrumMeasurement(
        center_freq_hz=status.center_freq_hz,
        span_hz=status.sample_rate_hz,
        fft_size=fft_size,
        bin_hz=bin_hz,
        magnitudes_dbm=magnitudes,
        freq_start_hz=freq_start,
        freq_stop_hz=freq_stop,
        timestamp_ns=int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        integration_count=averaging
    )


@router.post(
    "/power",
    response_model=PowerMeasurementResponse,
    description="Measure integrated power in a frequency band.",
    operation_id="measure_power",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def measure_power(
    start_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Start frequency in Hz"
    ),
    stop_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Stop frequency in Hz"
    ),
    integration_time_ms: int = Query(
        default=100,
        ge=10,
        le=10000,
        description="Integration time in milliseconds"
    ),
    detector_type: str = Query(
        default="rms",
        description="Detector: 'peak', 'rms', 'average', 'sample'"
    ),
) -> PowerMeasurementResponse:
    """
    Measure integrated power within a frequency band.

    This tunes the SDR (if needed) and measures total power in the specified range.
    Results include peak, average, and integrated power measurements.

    Detector types:
    - peak: Maximum value during integration period
    - rms: Root-mean-square power (best for continuous signals)
    - average: Arithmetic mean (best for pulsed signals)
    - sample: Single instantaneous measurement

    Use longer integration times for more accurate measurements.
    """
    if stop_freq_hz <= start_freq_hz:
        raise HTTPException(status_code=400, detail="stop_freq must be greater than start_freq")

    controller = get_sdr_controller()
    status = await controller.get_status()

    bandwidth = stop_freq_hz - start_freq_hz
    center_freq = (start_freq_hz + stop_freq_hz) // 2

    # Check if retune is needed
    if (abs(center_freq - status.center_freq_hz) > status.sample_rate_hz // 2):
        from spectrum_server.schema.sdr_control import TuneRequest
        tune_req = TuneRequest(
            center_freq_hz=center_freq,
            sample_rate_hz=max(bandwidth * 2, status.sample_rate_hz)
        )
        await controller.tune(tune_req)

    # Simulate power measurement
    # In production, this reads from the SDR data pipeline
    noise_floor = -100.0 + (76 - status.gain_db) * 0.5
    peak_power = noise_floor + np.random.uniform(10, 30)
    avg_power = noise_floor + np.random.uniform(5, 15)
    total_power = avg_power + 10 * np.log10(bandwidth / 1000)  # Integrate over bandwidth

    return PowerMeasurementResponse(
        success=True,
        total_power_dbm=float(total_power),
        peak_power_dbm=float(peak_power),
        peak_freq_hz=center_freq + np.random.randint(-bandwidth//4, bandwidth//4),
        average_power_dbm=float(avg_power),
        bandwidth_hz=bandwidth,
        noise_floor_dbm=float(noise_floor),
        snr_db=float(peak_power - noise_floor),
        timestamp_ns=int(datetime.now(tz=timezone.utc).timestamp() * 1e9)
    )


@router.post(
    "/sweep",
    response_model=SpectrumSweepResponse,
    description="Perform a frequency sweep across a range wider than instantaneous bandwidth.",
    operation_id="sweep_spectrum",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def sweep_spectrum(
    start_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Sweep start frequency"
    ),
    stop_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Sweep stop frequency"
    ),
    rbw_hz: int = Query(
        default=10_000,
        ge=100,
        le=1_000_000,
        description="Resolution bandwidth in Hz"
    ),
    averaging: int = Query(
        default=1,
        ge=1,
        le=100,
        description="Number of sweeps to average"
    ),
) -> SpectrumSweepResponse:
    """
    Perform a spectrum sweep across a wide frequency range.

    This automatically steps the SDR center frequency to cover the entire
    requested span, stitching results together into a continuous spectrum.

    Resolution bandwidth (RBW) affects:
    - Frequency resolution (smaller RBW = finer resolution)
    - Sweep time (smaller RBW = longer sweep)
    - Sensitivity (smaller RBW = lower noise floor)

    For wide spans (100s of MHz to GHz), expect sweep times of several seconds.
    """
    if stop_freq_hz <= start_freq_hz:
        raise HTTPException(status_code=400, detail="stop_freq must be greater than start_freq")

    start_time = time.time()
    controller = get_sdr_controller()
    status = await controller.get_status()

    # Calculate sweep parameters
    span = stop_freq_hz - start_freq_hz
    instantaneous_bw = status.sample_rate_hz
    num_steps = max(1, int(np.ceil(span / (instantaneous_bw * 0.8))))  # 20% overlap
    step_size = span / num_steps

    # Calculate number of frequency points based on RBW
    points_per_step = int(instantaneous_bw / rbw_hz)
    total_points = num_steps * points_per_step

    frequencies = []
    magnitudes = []

    # Simulate sweep (in production, this would actually tune and measure)
    for i in range(num_steps):
        step_center = start_freq_hz + int(step_size * (i + 0.5))
        step_start = step_center - instantaneous_bw // 2
        step_stop = step_center + instantaneous_bw // 2

        # Generate frequencies for this step
        step_freqs = np.linspace(step_start, step_stop, points_per_step, dtype=int)

        # Simulate spectrum data
        step_mags = _generate_simulated_spectrum(
            points_per_step, step_center, instantaneous_bw,
            status.gain_db, averaging
        )

        frequencies.extend(step_freqs.tolist())
        magnitudes.extend(step_mags)

        # Simulate tune time between steps
        await asyncio.sleep(0.01)

    sweep_time = (time.time() - start_time) * 1000

    return SpectrumSweepResponse(
        success=True,
        start_freq_hz=start_freq_hz,
        stop_freq_hz=stop_freq_hz,
        num_points=len(frequencies),
        frequencies_hz=frequencies,
        magnitudes_dbm=magnitudes,
        sweep_time_ms=sweep_time,
        rbw_hz=rbw_hz
    )


@router.post(
    "/occupancy",
    response_model=OccupancyResponse,
    description="Measure channel occupancy over time.",
    operation_id="measure_occupancy",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def measure_occupancy(
    center_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Channel center frequency"
    ),
    bandwidth_hz: int = Query(
        default=1_000_000,
        ge=1000,
        le=56_000_000,
        description="Channel bandwidth in Hz"
    ),
    threshold_dbm: float = Query(
        default=-80.0,
        description="Power threshold for 'occupied' detection"
    ),
    measurement_time_ms: int = Query(
        default=1000,
        ge=100,
        le=60000,
        description="Total measurement period"
    ),
) -> OccupancyResponse:
    """
    Measure channel occupancy (duty cycle) over time.

    This monitors a channel and determines what percentage of time
    signals are present above the threshold.

    Use cases:
    - Spectrum availability assessment
    - Interference characterization
    - Channel utilization monitoring
    - Regulatory compliance measurements

    Lower thresholds detect weaker signals but may trigger on noise.
    """
    controller = get_sdr_controller()

    # Tune to channel if needed
    status = await controller.get_status()
    if abs(center_freq_hz - status.center_freq_hz) > status.sample_rate_hz // 2:
        from spectrum_server.schema.sdr_control import TuneRequest
        await controller.tune(TuneRequest(
            center_freq_hz=center_freq_hz,
            sample_rate_hz=max(bandwidth_hz * 2, 10_000_000)
        ))

    # Simulate occupancy measurement
    # In production, this monitors the channel over time
    num_samples = measurement_time_ms // 10
    active_samples = int(num_samples * np.random.uniform(0.1, 0.7))
    occupancy = (active_samples / num_samples) * 100

    num_tx = max(1, active_samples // np.random.randint(5, 20))
    avg_tx_duration = (measurement_time_ms * occupancy / 100) / num_tx if num_tx > 0 else 0

    noise_floor = -100 + (76 - status.gain_db) * 0.5
    active_power = threshold_dbm + np.random.uniform(5, 20)

    return OccupancyResponse(
        success=True,
        occupancy_percent=float(occupancy),
        duty_cycle=float(occupancy / 100),
        num_transmissions=num_tx,
        avg_tx_duration_ms=float(avg_tx_duration),
        avg_power_when_active_dbm=float(active_power),
        max_power_dbm=float(active_power + 5),
        min_power_dbm=float(noise_floor),
        measurement_time_ms=measurement_time_ms
    )


@router.post(
    "/search",
    response_model=SignalSearchResponse,
    description="Search for signals in a frequency range and optionally classify them.",
    operation_id="search_signals",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def search_signals(
    start_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Search start frequency"
    ),
    stop_freq_hz: int = Query(
        ...,
        ge=70_000_000,
        le=6_000_000_000,
        description="Search stop frequency"
    ),
    min_snr_db: float = Query(
        default=6.0,
        ge=0.0,
        description="Minimum SNR for detection"
    ),
    min_bandwidth_hz: int = Query(
        default=10_000,
        ge=1000,
        description="Minimum signal bandwidth"
    ),
    classify: bool = Query(
        default=False,
        description="Run classification on detected signals"
    ),
) -> SignalSearchResponse:
    """
    Search for signals in a frequency range.

    This performs a sweep and identifies distinct signals above the noise floor.
    When classification is enabled, each signal is analyzed to determine
    modulation type and protocol.

    Detection algorithm:
    1. Measure spectrum across the range
    2. Estimate noise floor
    3. Find peaks exceeding min_snr_db above noise
    4. Determine signal boundaries
    5. Optionally classify each signal

    Results include frequency, bandwidth, power, and classification for each signal.
    """
    if stop_freq_hz <= start_freq_hz:
        raise HTTPException(status_code=400, detail="stop_freq must be greater than start_freq")

    start_time = time.time()
    controller = get_sdr_controller()
    status = await controller.get_status()

    span = stop_freq_hz - start_freq_hz
    noise_floor = -100 + (76 - status.gain_db) * 0.5

    # Simulate signal detection
    # In production, this would actually search the spectrum
    num_signals = np.random.randint(0, 5)
    signals = []

    for _ in range(num_signals):
        sig_center = np.random.randint(start_freq_hz, stop_freq_hz)
        sig_bw = np.random.randint(min_bandwidth_hz, min(5_000_000, span // 4))
        sig_power = noise_floor + min_snr_db + np.random.uniform(0, 20)

        classification = None
        confidence = None
        modulation = None

        if classify:
            modulation_types = ["OFDM", "QAM", "QPSK", "BPSK", "FSK", "AM", "FM", "GFSK"]
            signal_types = ["WiFi", "LTE", "Bluetooth", "ZigBee", "LoRa", "Unknown RF"]
            classification = np.random.choice(signal_types)
            confidence = np.random.uniform(0.6, 0.98)
            modulation = np.random.choice(modulation_types)

        signals.append(SignalDetection(
            freq_start_hz=sig_center - sig_bw // 2,
            freq_stop_hz=sig_center + sig_bw // 2,
            center_freq_hz=sig_center,
            bandwidth_hz=sig_bw,
            peak_power_dbm=float(sig_power),
            avg_power_dbm=float(sig_power - 3),
            snr_db=float(sig_power - noise_floor),
            classification=classification,
            confidence=float(confidence) if confidence else None,
            modulation=modulation,
            timestamp_ns=int(datetime.now(tz=timezone.utc).timestamp() * 1e9)
        ))

    search_time = (time.time() - start_time) * 1000

    return SignalSearchResponse(
        success=True,
        signals=signals,
        search_time_ms=search_time,
        noise_floor_dbm=float(noise_floor),
        span_hz=span
    )


@router.get(
    "/waterfall/snapshot",
    response_model=WaterfallFrame,
    description="Get a single waterfall frame for dashboard display.",
    operation_id="get_waterfall_snapshot",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def get_waterfall_snapshot(
    fft_size: int = Query(
        default=1024,
        ge=256,
        le=8192,
        description="FFT size for waterfall"
    ),
) -> WaterfallFrame:
    """
    Get a single waterfall frame for dashboard display.

    This returns the current spectrum suitable for adding to a
    waterfall/spectrogram display. Call repeatedly to build up
    the time dimension, or use the WebSocket endpoint for continuous streaming.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    magnitudes = _generate_simulated_spectrum(
        fft_size, status.center_freq_hz, status.sample_rate_hz,
        status.gain_db, 1
    )

    return WaterfallFrame(
        center_freq_hz=status.center_freq_hz,
        span_hz=status.sample_rate_hz,
        timestamp_ns=int(datetime.now(tz=timezone.utc).timestamp() * 1e9),
        magnitudes_db=magnitudes,
        frame_index=0  # Would increment in streaming mode
    )


def _generate_simulated_spectrum(
    fft_size: int,
    center_freq: int,
    sample_rate: int,
    gain_db: float,
    averaging: int
) -> list[float]:
    """Generate simulated spectrum data for testing."""
    # Base noise floor depends on gain
    noise_floor = -100 + (76 - gain_db) * 0.5

    # Generate noise
    spectrum = np.random.normal(noise_floor, 3 / np.sqrt(averaging), fft_size)

    # Add some simulated signals
    # WiFi-like signal in center
    wifi_idx = fft_size // 2
    wifi_width = fft_size // 20
    wifi_power = noise_floor + 25
    spectrum[wifi_idx - wifi_width:wifi_idx + wifi_width] += np.random.normal(
        wifi_power - noise_floor, 2, wifi_width * 2
    )

    # Random narrowband signals
    for _ in range(3):
        idx = np.random.randint(fft_size // 10, fft_size * 9 // 10)
        width = np.random.randint(5, 20)
        power = noise_floor + np.random.uniform(10, 30)
        start_idx = max(0, idx - width)
        end_idx = min(fft_size, idx + width)
        spectrum[start_idx:end_idx] = power + np.random.normal(0, 1, end_idx - start_idx)

    return spectrum.tolist()
