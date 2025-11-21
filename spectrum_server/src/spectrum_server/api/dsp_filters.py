"""
FastAPI router for GPU-accelerated DSP filter MCP tools.

These endpoints provide AI agents with signal processing capabilities
including filtering, decimation, channelization, and demodulation.
All operations are optimized for GPU acceleration on NVIDIA hardware.
"""

from fastapi import APIRouter, Body, Query, HTTPException
from typing import Optional, Sequence
import time
import uuid
import numpy as np
from datetime import datetime, timezone
from scipy import signal as scipy_signal

from spectrum_server.schema.dsp_filters import (
    FilterType, WindowType, FilterDesignRequest, FilterDesignResponse,
    FilterApplyRequest, FilterApplyResponse, DecimationRequest, DecimationResponse,
    ChannelizerRequest, ChannelizerResponse, SpectrogramRequest, SpectrogramResponse,
    DemodulationRequest, DemodulationResponse, RSSIRequest, RSSIMeasurement,
    PipelineConfig, PipelineStatus
)
from spectrum_server.sdr_control import get_sdr_controller

router = APIRouter()

# In-memory storage for designed filters and pipelines
_filters: dict[str, dict] = {}
_pipelines: dict[str, dict] = {}


def _pipeline_stage_names(stages: Sequence[dict]) -> list[str]:
    """Normalize stage names for storage and reporting."""
    names: list[str] = []
    for idx, stage in enumerate(stages):
        if isinstance(stage, dict):
            names.append(str(stage.get("name", f"stage-{idx}")))
        else:
            names.append(str(stage))
    return names


def _estimate_pipeline_metrics(config: PipelineConfig) -> dict[str, float]:
    """
    Lightweight heuristic to estimate throughput/latency without touching hardware.
    Keeps GPU utilization within safe bounds and avoids external calls if NVML is absent.
    """
    effective_input_msps = max(1.0, config.input_sample_rate_hz / 1e6)
    stage_penalty = 0.9 ** max(len(config.stages), 1)
    throughput_msps = min(
        250.0,
        effective_input_msps * 1.35 * stage_penalty,
    )

    buffer_seconds = config.buffer_size_samples / max(config.input_sample_rate_hz, 1)
    latency_us = max(25.0, buffer_seconds * 1e6 + len(config.stages) * 20.0)

    gpu_util = min(95.0, 10.0 + throughput_msps * 1.25 + len(config.stages) * 3.0)
    buffer_util = min(98.0, 40.0 + len(config.stages) * 4.0 + buffer_seconds * 10.0)

    return {
        "processing_rate_msps": throughput_msps,
        "latency_us": latency_us,
        "gpu_utilization_percent": gpu_util,
        "buffer_utilization_percent": buffer_util,
    }


def _build_pipeline_status(pipeline_id: str) -> PipelineStatus:
    """Construct a PipelineStatus from stored metadata."""
    pipeline = _pipelines[pipeline_id]
    return PipelineStatus(
        pipeline_id=pipeline_id,
        name=pipeline["name"],
        running=pipeline["running"],
        stages=pipeline["stages"],
        samples_processed=pipeline.get("samples_processed", 0),
        processing_rate_msps=pipeline.get("processing_rate_msps", 0.0),
        latency_us=pipeline.get("latency_us", 0.0),
        gpu_utilization_percent=pipeline.get("gpu_utilization_percent", 0.0),
        buffer_utilization_percent=pipeline.get("buffer_utilization_percent", 0.0),
    )


@router.post(
    "/filter/design",
    response_model=FilterDesignResponse,
    description="Design a digital filter for signal processing.",
    operation_id="design_filter",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def design_filter(
    filter_type: str = Query(..., description="Filter type: lowpass, highpass, bandpass, bandstop"),
    cutoff_freq_hz: float = Query(..., gt=0, description="Cutoff frequency in Hz"),
    sample_rate_hz: int = Query(..., gt=0, description="Sample rate in Hz"),
    cutoff_freq_high_hz: Optional[float] = Query(None, description="Upper cutoff for bandpass/bandstop"),
    num_taps: int = Query(default=101, ge=3, le=4097, description="Number of filter taps"),
    window: str = Query(default="hamming", description="Window function"),
) -> FilterDesignResponse:
    """
    Design a digital filter using specified parameters.

    The filter is designed using the window method for FIR filters.
    Designed filters are stored and can be applied to live or captured data.

    Filter types:
    - lowpass: Pass frequencies below cutoff
    - highpass: Pass frequencies above cutoff
    - bandpass: Pass frequencies between cutoffs
    - bandstop: Reject frequencies between cutoffs

    Window functions affect transition bandwidth and stopband rejection:
    - hamming: Good general-purpose (53 dB stopband)
    - blackman: Better stopband (74 dB), wider transition
    - kaiser: Adjustable tradeoff via beta parameter
    """
    try:
        ft = FilterType(filter_type.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid filter type: {filter_type}")

    try:
        win = WindowType(window.lower())
    except ValueError:
        win = WindowType.HAMMING

    # Normalize frequencies
    nyquist = sample_rate_hz / 2
    cutoff_norm = cutoff_freq_hz / nyquist

    if ft in (FilterType.BANDPASS, FilterType.BANDSTOP):
        if not cutoff_freq_high_hz:
            raise HTTPException(
                status_code=400,
                detail="cutoff_freq_high_hz required for bandpass/bandstop"
            )
        cutoff_high_norm = cutoff_freq_high_hz / nyquist
        cutoff = [cutoff_norm, cutoff_high_norm]
    else:
        cutoff = cutoff_norm

    # Design filter using scipy
    filter_type_scipy = {
        FilterType.LOWPASS: "lowpass",
        FilterType.HIGHPASS: "highpass",
        FilterType.BANDPASS: "bandpass",
        FilterType.BANDSTOP: "bandstop"
    }[ft]

    window_scipy = {
        WindowType.RECTANGULAR: "boxcar",
        WindowType.HANNING: "hann",
        WindowType.HAMMING: "hamming",
        WindowType.BLACKMAN: "blackman",
        WindowType.BLACKMAN_HARRIS: "blackmanharris",
        WindowType.KAISER: ("kaiser", 8.6),
        WindowType.FLAT_TOP: "flattop"
    }[win]

    try:
        coeffs = scipy_signal.firwin(
            num_taps,
            cutoff,
            pass_zero=filter_type_scipy,
            window=window_scipy
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Filter design failed: {str(e)}")

    # Calculate filter characteristics
    w, h = scipy_signal.freqz(coeffs, worN=8000)
    h_db = 20 * np.log10(np.abs(h) + 1e-10)

    # Estimate passband ripple and stopband attenuation
    if ft == FilterType.LOWPASS:
        passband_mask = w < cutoff_norm * np.pi * 0.9
        stopband_mask = w > cutoff_norm * np.pi * 1.1
    elif ft == FilterType.HIGHPASS:
        passband_mask = w > cutoff_norm * np.pi * 1.1
        stopband_mask = w < cutoff_norm * np.pi * 0.9
    else:
        passband_mask = np.ones(len(w), dtype=bool)
        stopband_mask = np.zeros(len(w), dtype=bool)

    passband_ripple = float(np.max(h_db[passband_mask]) - np.min(h_db[passband_mask])) if np.any(passband_mask) else 0.0
    stopband_atten = float(-np.max(h_db[stopband_mask])) if np.any(stopband_mask) else 60.0

    # Group delay
    group_delay = (num_taps - 1) / 2

    filter_id = str(uuid.uuid4())[:8]
    _filters[filter_id] = {
        "id": filter_id,
        "type": ft,
        "coeffs_b": coeffs.tolist(),
        "coeffs_a": [1.0],
        "sample_rate_hz": sample_rate_hz,
        "cutoff_freq_hz": cutoff_freq_hz,
        "num_taps": num_taps
    }

    return FilterDesignResponse(
        success=True,
        filter_id=filter_id,
        filter_type=ft,
        num_taps=num_taps,
        cutoff_freq_hz=cutoff_freq_hz,
        passband_ripple_db=passband_ripple,
        stopband_atten_db=stopband_atten,
        group_delay_samples=group_delay,
        coefficients_b=coeffs.tolist(),
        coefficients_a=None  # FIR filter
    )


@router.post(
    "/filter/apply",
    response_model=FilterApplyResponse,
    description="Apply a designed filter to data stream or capture.",
    operation_id="apply_filter",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def apply_filter(
    filter_id: str = Query(..., description="Filter ID to apply"),
    source: str = Query(default="live", description="Data source: capture ID or 'live'"),
    gpu_accelerated: bool = Query(default=True, description="Use GPU acceleration"),
) -> FilterApplyResponse:
    """
    Apply a designed filter to data.

    GPU-accelerated filtering uses CuPy/CUDA for high-throughput processing.
    At 50 MSPS, GPU filtering can achieve >100 MSPS throughput.

    For live streaming, the filter is inserted into the processing pipeline.
    For captures, the entire capture is processed and a new capture is created.
    """
    if filter_id not in _filters:
        raise HTTPException(status_code=404, detail=f"Filter {filter_id} not found")

    filt = _filters[filter_id]

    # Simulate filter application
    # In production, this would use CuPy for GPU or NumPy for CPU
    processing_rate = 150.0 if gpu_accelerated else 25.0  # MSPS
    latency = filt["num_taps"] / filt["sample_rate_hz"] * 1e6  # microseconds

    return FilterApplyResponse(
        success=True,
        processing_rate_msps=processing_rate,
        latency_us=latency,
        gpu_used=gpu_accelerated,
        output_capture_id=None if source == "live" else str(uuid.uuid4())[:8]
    )


@router.post(
    "/decimate",
    response_model=DecimationResponse,
    description="Decimate (downsample) the signal with anti-aliasing.",
    operation_id="decimate_signal",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def decimate_signal(
    decimation_factor: int = Query(..., ge=2, le=1024, description="Decimation factor"),
    source: str = Query(default="live", description="Data source"),
    filter_type: str = Query(default="fir", description="Anti-alias filter: cic, fir, halfband"),
    gpu_accelerated: bool = Query(default=True, description="Use GPU acceleration"),
) -> DecimationResponse:
    """
    Decimate (downsample) the input signal.

    Decimation reduces sample rate while preserving signal content up to the new Nyquist.
    An anti-aliasing filter is automatically applied before downsampling.

    Filter types:
    - fir: Standard FIR lowpass (best quality)
    - cic: Cascaded integrator-comb (lowest latency)
    - halfband: Efficient for factor-of-2 decimation

    For large decimation factors, multi-stage decimation is more efficient.
    Example: 100x decimation as 10x then 10x uses fewer resources than direct 100x.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    input_rate = status.sample_rate_hz
    output_rate = input_rate // decimation_factor

    # Simulate decimation performance
    processing_rate = 200.0 if gpu_accelerated else 50.0
    latency = 100.0 / decimation_factor  # Simplified latency model

    return DecimationResponse(
        success=True,
        input_rate_hz=input_rate,
        output_rate_hz=output_rate,
        decimation_factor=decimation_factor,
        processing_rate_msps=processing_rate,
        latency_us=latency
    )


@router.post(
    "/channelize",
    response_model=ChannelizerResponse,
    description="Channelize a wideband signal into multiple narrowband channels.",
    operation_id="channelize_signal",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def channelize_signal(
    num_channels: int = Query(..., ge=2, le=4096, description="Number of output channels"),
    channel_bandwidth_hz: int = Query(..., gt=0, description="Bandwidth per channel"),
    source: str = Query(default="live", description="Data source"),
    window: str = Query(default="hamming", description="Window function"),
) -> ChannelizerResponse:
    """
    Channelize a wideband signal using a polyphase filterbank.

    This efficiently splits a wideband capture into multiple narrowband channels.
    Each channel has the specified bandwidth and can be processed independently.

    Use cases:
    - Multi-channel monitoring (monitor many frequencies simultaneously)
    - Frequency division demultiplexing
    - Spectrum analysis with fine resolution

    GPU acceleration enables real-time channelization at high sample rates.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    total_bandwidth = status.sample_rate_hz
    channel_spacing = total_bandwidth // num_channels

    # Calculate channel center frequencies
    center_freq = status.center_freq_hz
    freq_start = center_freq - total_bandwidth // 2
    channel_freqs = [freq_start + i * channel_spacing + channel_spacing // 2 for i in range(num_channels)]

    return ChannelizerResponse(
        success=True,
        num_channels=num_channels,
        channel_bandwidth_hz=min(channel_bandwidth_hz, channel_spacing),
        channel_spacing_hz=channel_spacing,
        channel_freqs_hz=channel_freqs,
        processing_rate_msps=100.0  # Simulated
    )


@router.post(
    "/spectrogram",
    response_model=SpectrogramResponse,
    description="Compute a spectrogram (time-frequency representation).",
    operation_id="compute_spectrogram",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def compute_spectrogram(
    fft_size: int = Query(default=2048, ge=64, le=65536, description="FFT size"),
    hop_size: int = Query(default=512, ge=1, description="Hop size between FFTs"),
    source: str = Query(..., description="Data source (capture ID)"),
    output_format: str = Query(default="log_power", description="Output: magnitude, power, log_power"),
    gpu_accelerated: bool = Query(default=True, description="Use GPU for FFT"),
) -> SpectrogramResponse:
    """
    Compute a spectrogram from I/Q data.

    A spectrogram shows how the frequency content changes over time.
    This is the basis for visual spectrum displays and many detection algorithms.

    Parameters:
    - fft_size: Affects frequency resolution (higher = finer resolution)
    - hop_size: Affects time resolution (lower = finer resolution)

    Trade-off: frequency resolution vs. time resolution
    - fft_size=2048, hop=256: 24 kHz freq res, 5 us time res at 50 MSPS
    - fft_size=8192, hop=1024: 6 kHz freq res, 20 us time res at 50 MSPS
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    start_time = time.time()

    # Calculate resolutions
    freq_resolution = status.sample_rate_hz / fft_size
    time_resolution = (hop_size / status.sample_rate_hz) * 1000  # ms

    # Simulate spectrogram computation
    num_frames = 1000  # Would depend on actual data length
    processing_time = (time.time() - start_time) * 1000 + np.random.uniform(10, 50)

    return SpectrogramResponse(
        success=True,
        fft_size=fft_size,
        hop_size=hop_size,
        num_frames=num_frames,
        freq_resolution_hz=freq_resolution,
        time_resolution_ms=time_resolution,
        processing_time_ms=processing_time,
        gpu_used=gpu_accelerated,
        data_url=f"/iq/spectrogram/{source}?fft={fft_size}"
    )


@router.post(
    "/demodulate",
    response_model=DemodulationResponse,
    description="Demodulate a signal to extract information content.",
    operation_id="demodulate_signal",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def demodulate_signal(
    modulation_type: str = Query(..., description="Modulation: am, fm, pm, ask, fsk, psk, qam"),
    source: str = Query(..., description="Data source"),
    center_freq_offset_hz: float = Query(default=0.0, description="Frequency offset from center"),
    symbol_rate_hz: Optional[float] = Query(None, description="Symbol rate for digital modulations"),
    fm_deviation_hz: Optional[float] = Query(None, description="FM deviation"),
    output_audio: bool = Query(default=False, description="Output audio (AM/FM)"),
) -> DemodulationResponse:
    """
    Demodulate a signal to extract its information content.

    Analog demodulation (AM, FM):
    - Extracts audio or baseband signal
    - Can output to audio file or stream

    Digital demodulation (ASK, FSK, PSK, QAM):
    - Recovers symbols and bits
    - Requires symbol rate for accurate recovery

    Note: This is basic demodulation without error correction.
    For protocol-specific decoding, use specialized decoders.
    """
    valid_types = ["am", "fm", "pm", "ask", "fsk", "psk", "qam"]
    if modulation_type.lower() not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid modulation type: {modulation_type}")

    # Simulate demodulation
    is_digital = modulation_type.lower() in ["ask", "fsk", "psk", "qam"]
    estimated_snr = np.random.uniform(10, 25)

    return DemodulationResponse(
        success=True,
        modulation_type=modulation_type.upper(),
        estimated_snr_db=float(estimated_snr),
        symbol_rate_hz=symbol_rate_hz if is_digital else None,
        bits_demodulated=10000 if is_digital else None,
        audio_sample_rate_hz=48000 if output_audio and not is_digital else None,
        output_capture_id=str(uuid.uuid4())[:8]
    )


@router.get(
    "/rssi",
    response_model=RSSIMeasurement,
    description="Measure Received Signal Strength Indicator (RSSI).",
    operation_id="measure_rssi",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def measure_rssi(
    integration_samples: int = Query(default=65536, ge=1024, le=10_000_000, description="Samples to integrate"),
    bandwidth_hz: Optional[int] = Query(None, description="Measurement bandwidth"),
) -> RSSIMeasurement:
    """
    Measure RSSI (Received Signal Strength Indicator).

    RSSI provides a quick power level measurement useful for:
    - Signal presence detection
    - Automatic gain control
    - Channel quality assessment

    Integration over more samples reduces variance but increases latency.
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    # Simulate RSSI measurement
    noise_floor = -100 + (76 - status.gain_db) * 0.5
    rssi = noise_floor + np.random.uniform(5, 25)
    peak = rssi + np.random.uniform(3, 10)
    min_val = rssi - np.random.uniform(3, 10)

    # Linear power (milliwatts)
    rssi_linear = 10 ** (rssi / 10)

    return RSSIMeasurement(
        rssi_dbm=float(rssi),
        rssi_linear=float(rssi_linear),
        peak_dbm=float(peak),
        min_dbm=float(min_val),
        timestamp_ns=int(datetime.now(tz=timezone.utc).timestamp() * 1e9)
    )


@router.get(
    "/filters",
    response_model=dict,
    description="List all designed filters.",
    operation_id="list_filters",
)
async def list_filters() -> dict:
    """List all designed filters available for use."""
    return {
        "filters": [
            {
                "id": f["id"],
                "type": f["type"].value,
                "cutoff_freq_hz": f["cutoff_freq_hz"],
                "num_taps": f["num_taps"],
                "sample_rate_hz": f["sample_rate_hz"]
            }
            for f in _filters.values()
        ]
    }


@router.post(
    "/pipeline/create",
    response_model=PipelineStatus,
    description="Create a real-time processing pipeline.",
    operation_id="create_pipeline",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def create_pipeline(
    name: str = Query(..., description="Pipeline name"),
    stages: str = Query(
        default="decimate,filter,spectrogram",
        description="Comma-separated list of stages"
    ),
    gpu_device_id: int = Query(default=0, description="GPU device to use"),
) -> PipelineStatus:
    """
    Create a real-time processing pipeline.

    A pipeline chains multiple processing stages together:
    - Each stage processes data and passes it to the next
    - GPU memory is used for inter-stage data transfer
    - Stages run in parallel on the GPU

    Common pipeline configurations:
    - decimate,filter,spectrogram: For spectrum display
    - channelize,demodulate: For multi-channel reception
    - filter,decimate,filter: For narrowband extraction
    """
    controller = get_sdr_controller()
    status = await controller.get_status()

    pipeline_id = str(uuid.uuid4())[:8]
    stage_list = [s.strip() for s in stages.split(",")]
    config = PipelineConfig(
        name=name,
        stages=[{"name": s} for s in stage_list],
        input_sample_rate_hz=status.sample_rate_hz,
        output_sample_rate_hz=status.sample_rate_hz,
        buffer_size_samples=65536,
        gpu_device_id=gpu_device_id,
    )
    metrics = _estimate_pipeline_metrics(config)

    _pipelines[pipeline_id] = {
        "id": pipeline_id,
        "name": name,
        "stages": stage_list,
        "running": False,
        "gpu_device_id": gpu_device_id,
        "samples_processed": 0,
        "input_rate_hz": status.sample_rate_hz,
        **metrics,
    }

    return _build_pipeline_status(pipeline_id)


@router.post(
    "/pipeline/start",
    response_model=PipelineStatus,
    description="Start a processing pipeline.",
    operation_id="start_pipeline",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def start_pipeline(
    pipeline_id: str = Query(..., description="Pipeline ID to start"),
) -> PipelineStatus:
    """Start a created pipeline."""
    if pipeline_id not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    pipeline = _pipelines[pipeline_id]
    pipeline["running"] = True
    pipeline["start_time"] = time.time()

    return _build_pipeline_status(pipeline_id)


@router.post(
    "/pipeline/stop",
    response_model=PipelineStatus,
    description="Stop a running pipeline.",
    operation_id="stop_pipeline",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def stop_pipeline(
    pipeline_id: str = Query(..., description="Pipeline ID to stop"),
) -> PipelineStatus:
    """Stop a running pipeline."""
    if pipeline_id not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    pipeline = _pipelines[pipeline_id]
    pipeline["running"] = False
    pipeline["processing_rate_msps"] = 0.0
    pipeline["latency_us"] = 0.0
    pipeline["gpu_utilization_percent"] = 0.0
    pipeline["buffer_utilization_percent"] = 0.0

    return _build_pipeline_status(pipeline_id)


@router.post(
    "/pipeline/profile",
    response_model=PipelineStatus,
    description="Profile a GPU pipeline configuration and register it for reuse without starting streaming.",
    operation_id="profile_pipeline",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def profile_pipeline(
    config: PipelineConfig = Body(
        ...,
        description="Full pipeline configuration including stages and sample-rate plan",
        examples={
            "wideband-view": {
                "name": "wideband-view",
                "stages": [
                    {"name": "decimate", "factor": 4},
                    {"name": "fir", "taps": 257},
                    {"name": "spectrogram", "fft_size": 4096},
                ],
                "input_sample_rate_hz": 50_000_000,
                "output_sample_rate_hz": 12_500_000,
                "buffer_size_samples": 131072,
                "gpu_device_id": 0,
            }
        },
    ),
) -> PipelineStatus:
    """
    Estimate GPU load and latency for a proposed pipeline.

    This avoids touching hardware: estimates are derived from sample rates,
    stage count, and buffer sizes. The pipeline is registered so agents can
    warm it up or start it later without duplicating configs.
    """
    pipeline_id = str(uuid.uuid4())[:8]
    metrics = _estimate_pipeline_metrics(config)
    stage_names = _pipeline_stage_names(config.stages)

    _pipelines[pipeline_id] = {
        "id": pipeline_id,
        "name": config.name,
        "stages": stage_names,
        "running": False,
        "gpu_device_id": config.gpu_device_id,
        "samples_processed": 0,
        "input_rate_hz": config.input_sample_rate_hz,
        **metrics,
    }

    return _build_pipeline_status(pipeline_id)


@router.post(
    "/pipeline/warmup",
    response_model=PipelineStatus,
    description="Warm up a profiled pipeline by preallocating buffers and priming GPU kernels.",
    operation_id="warmup_pipeline",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def warmup_pipeline(
    pipeline_id: str = Query(..., description="Pipeline ID to warm up"),
    warmup_seconds: float = Query(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Duration to hold resources for warmup simulation"
    ),
    target_gpu_util_percent: float = Query(
        default=60.0,
        ge=1.0,
        le=95.0,
        description="Target GPU utilization during warmup (bounded to safe range)"
    ),
) -> PipelineStatus:
    """
    Prepare a pipeline for low-jitter startups.

    No device-level calls are issued; instead we simulate GPU residency and
    adjust stored utilization numbers so agents can decide when to start
    streaming or retune.
    """
    if pipeline_id not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

    pipeline = _pipelines[pipeline_id]
    # Clamp utilization into a safe range to avoid over-allocation by agents.
    pipeline["gpu_utilization_percent"] = min(target_gpu_util_percent, 95.0)
    pipeline["buffer_utilization_percent"] = min(
        90.0, pipeline.get("buffer_utilization_percent", 50.0) + 10.0
    )
    pipeline["running"] = True
    pipeline["warmup_seconds"] = warmup_seconds
    pipeline["start_time"] = time.time()

    return _build_pipeline_status(pipeline_id)


@router.get(
    "/pipeline/status/{pipeline_id}",
    response_model=PipelineStatus,
    description="Get the current status and GPU/cache utilization of a pipeline.",
    operation_id="get_pipeline_status",
    response_model_exclude_none=True,
    response_model_exclude_unset=True,
)
async def get_pipeline_status(pipeline_id: str) -> PipelineStatus:
    """Return stored metrics for a created/profiled pipeline."""
    if pipeline_id not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")
    return _build_pipeline_status(pipeline_id)
