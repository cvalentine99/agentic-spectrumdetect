# Spectrum Server MCP

**Spectrum Server** is a FastAPI application exposing Model Context Protocol (MCP) tools for AI-driven spectrum analysis. It provides comprehensive Ettus B210 SDR control, GPU-accelerated signal processing, and real-time dashboard streaming.

## Features

### SDR Control (`/sdr`)
- **Hardware Status**: Query SDR connection state, frequency, gain, antenna
- **Frequency Tuning**: Tune to any frequency from 70 MHz - 6 GHz
- **Gain Control**: Manual or automatic gain (0-76 dB)
- **Antenna Selection**: TX/RX or RX2 port selection
- **Calibration**: DC offset and IQ balance calibration
- **Quick Tune**: Preset frequency bands (WiFi, LTE, GPS, ISM, etc.)

### Spectrum Analysis (`/spectrum`)
- **Spectrum Measurement**: Single FFT measurement with configurable averaging
- **Power Measurement**: Integrated power in frequency bands
- **Frequency Sweep**: Wide-band spectrum sweep across multiple tune steps
- **Occupancy Analysis**: Channel duty cycle and transmission statistics
- **Signal Search**: Find and optionally classify signals in a range
- **Waterfall Snapshot**: Single frame for spectrogram display

### I/Q Capture (`/iq`)
- **Burst Capture**: Capture configurable number of samples
- **Continuous Recording**: Stream to disk with optional duration limit
- **SIGMF Export**: Generate SIGMF-compliant metadata files
- **Playback**: Replay captured data through processing pipeline
- **Buffer Statistics**: Monitor I/Q buffer health and GPU memory

### DSP Filters (`/dsp`)
- **Filter Design**: Design FIR lowpass/highpass/bandpass/bandstop filters
- **Filter Application**: GPU-accelerated filter execution
- **Decimation**: Downsample with anti-aliasing (CIC, FIR, halfband)
- **Channelization**: Polyphase filterbank for multi-channel extraction
- **Spectrogram**: GPU-accelerated STFT computation
- **Demodulation**: AM/FM/PSK/QAM demodulation
- **RSSI**: Real-time signal strength measurement
- **Processing Pipelines**: Chain multiple DSP stages

### Dashboard (`/dashboard`)
- **Real-time WebSocket**: Stream spectrum, detections, and metrics
- **Waterfall History**: Initialize displays with historical data
- **Detection Events**: Signal detection overlay data
- **System Metrics**: CPU, GPU, memory, throughput statistics
- **Color Scales**: Configurable visualization presets

## API Endpoints

| Prefix | Description | Tags |
|--------|-------------|------|
| `/sdr` | SDR hardware control | sdr-control |
| `/spectrum` | Spectrum measurements | spectrum-analysis |
| `/iq` | I/Q capture and SIGMF | iq-capture |
| `/dsp` | DSP filter operations | dsp-filters |
| `/dashboard` | Dashboard data streaming | dashboard |
| `/radio` | Legacy UDP protocol | radio |
| `/llm/mcp` | MCP tool endpoint | - |

## MCP Integration

All API endpoints (except `/health`) are exposed via FastMCP at `/llm/mcp/`. AI agents can discover and invoke tools using the Model Context Protocol.

Example MCP tool invocation:
```json
{
  "method": "tools/call",
  "params": {
    "name": "tune_sdr",
    "arguments": {
      "center_freq_hz": 2450000000,
      "sample_rate_hz": 50000000
    }
  }
}
```

## WebSocket Endpoints

### Spectrum Stream
`ws://<host>:8000/ws/spectrum`

Receives raw spectrum frames from the processing pipeline.

### Dashboard Stream
`ws://<host>:8000/dashboard/ws/dashboard`

Full-featured dashboard data including spectrum, detections, and metrics.

Commands:
```json
{"cmd": "set_rate", "rate_hz": 10}
{"cmd": "set_fft_size", "fft_size": 2048}
{"cmd": "enable_detections", "enabled": true}
{"cmd": "enable_metrics", "enabled": true}
```

## Prerequisites

* Python 3.12+
* MongoDB instance (default port 27018)
* Network access to SDR/radio on UDP ports
* NVIDIA GPU with CUDA 12+ (for GPU acceleration)

### Optional Dependencies

```bash
# GPU acceleration
pip install cupy-cuda12x

# UHD for Ettus devices (system package)
sudo apt install python3-uhd

# SoapySDR abstraction (system package)
sudo apt install python3-soapysdr
```

### C++ spectrum_core bridge

Set `SPECTRUM_CORE_ENDPOINT` to point at the C++ GPU pipeline service (ZeroMQ/gRPC). When set, `/spectrum/measure` will request GPU FFT frames and fall back to the built-in simulator if the service does not respond within `SPECTRUM_CORE_RESPONSE_TIMEOUT_MS` (default 200 ms).
`DASHBOARD_DETECTION_THRESHOLD_DB` (default -65 dB) tunes the lightweight peak picker used for dashboard overlays.
`SPECTRUM_CORE_BIND` controls where the C++ `spectrum_core_server` binds (default `tcp://*:5555`) if you run it directly.
`SPECTRUM_CORE_IQ_ENDPOINT` (PULL cf32) and `SPECTRUM_CORE_DET_ENDPOINT` (PULL detections JSON) can feed live IQ/detections into the C++ server instead of synthetic sources.

#### ZeroMQ protocol (current)

The Python client speaks REQ/REP to `SPECTRUM_CORE_ENDPOINT`. Messages are multipart:

* Request frame 0 (JSON):
  ```json
  {"op": "measure", "center_freq_hz": 2450000000, "sample_rate_hz": 50000000, "fft_size": 2048, "averaging": 10}
  ```
* Response frame 0 (JSON): same fields plus `timestamp_ns`
* Response frame 1 (binary): float32 log-power array of length `fft_size`

If the reply is missing or times out, the server falls back to simulated spectra so agent flows stay responsive.

GPU metrics: the dashboard will use NVML if `pynvml`/`nvidia-ml-py` is available; otherwise it will emit placeholder GPU utilization.

## Quick Start

```bash
# Install dependencies
cd spectrum_server
pip install -e .

# Run server
uvicorn spectrum_server.server:app --host 0.0.0.0 --port 8000

# Or via Docker
docker compose up spectrum_server
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage

### Tune to WiFi 2.4 GHz
```bash
curl -X POST "http://localhost:8000/sdr/quick-tune?band_name=WiFi%202.4%20GHz"
```

### Measure Spectrum
```bash
curl "http://localhost:8000/spectrum/measure?fft_size=2048&averaging=10"
```

### Capture I/Q Data
```bash
curl -X POST "http://localhost:8000/iq/capture?num_samples=1000000&output_format=cf32"
```

### Design Lowpass Filter
```bash
curl -X POST "http://localhost:8000/dsp/filter/design?filter_type=lowpass&cutoff_freq_hz=5000000&sample_rate_hz=50000000"
```

### Search for Signals
```bash
curl -X POST "http://localhost:8000/spectrum/search?start_freq_hz=2400000000&stop_freq_hz=2500000000&classify=true"
```

## Architecture

```
spectrum_server/
├── api/
│   ├── sdr_control.py      # SDR hardware control endpoints
│   ├── spectrum_analysis.py # Spectrum measurement endpoints
│   ├── iq_capture.py       # I/Q capture and SIGMF endpoints
│   ├── dsp_filters.py      # DSP filter endpoints
│   ├── dashboard.py        # Dashboard streaming endpoints
│   └── radios_work.py      # Legacy UDP protocol
├── schema/
│   ├── sdr_control.py      # Pydantic models for SDR
│   ├── spectrum_analysis.py # Pydantic models for spectrum
│   ├── iq_capture.py       # Pydantic models for capture
│   └── dsp_filters.py      # Pydantic models for DSP
├── sdr_control.py          # SDR controller (UHD/SoapySDR)
├── spectrum_stream.py      # WebSocket spectrum broadcaster
└── server.py               # FastAPI + FastMCP application
```

## License

MIT License - see LICENSE.md for details.
