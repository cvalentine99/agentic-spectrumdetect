# SDR Integration Guide

This document describes the Ettus B210 SDR integration with the Spectrum Server MCP platform.

## Overview

The spectrum_server provides comprehensive SDR control through a unified interface supporting:
- **UHD** (USRP Hardware Driver) for direct Ettus device access
- **SoapySDR** abstraction layer for broader hardware support
- **UDP retune protocol** for tye_sp compatibility
- **Simulated mode** for development without hardware

## Hardware Requirements

### Supported Devices
- Ettus B210 (primary target)
- Ettus B200/B200mini
- Any SoapySDR-compatible device

### System Requirements
- USB 3.0 connection (USB 2.0 limits sample rate to ~8 MSPS)
- UHD 4.x with Python bindings
- Docker with privileged mode for USB access

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SDR_BACKEND` | `uhd` | Backend driver: `uhd`, `soapy`, or `simulated` |
| `SDR_DEVICE_SERIAL` | `` | Device serial number (e.g., `194919`) |
| `SDR_CENTER_FREQ_HZ` | `2450000000` | Initial center frequency |
| `SDR_SAMPLE_RATE_HZ` | `50000000` | Initial sample rate |
| `SDR_BANDWIDTH_HZ` | `50000000` | Initial analog bandwidth |
| `SDR_GAIN_DB` | `38.0` | Initial gain setting |
| `SDR_ANTENNA` | `TX/RX` | Antenna port: `TX/RX`, `RX2`, or `AUTO` |

### Docker Compose Configuration

```yaml
spectrum_server:
  build:
    context: spectrum_server
    dockerfile: Dockerfile
  privileged: true
  network_mode: "host"
  environment:
    SDR_DEVICE_SERIAL: "194919"
    SDR_BACKEND: "uhd"
  devices:
    - /dev/bus/usb:/dev/bus/usb
  volumes:
    - /usr/share/uhd:/usr/share/uhd:ro
```

Key requirements:
- `privileged: true` - Required for USB device access
- `devices` - Passthrough USB bus for B210
- `volumes` - UHD firmware images

## API Reference

### SDR Control (`/sdr/*`)

#### GET /sdr/status
Returns current SDR hardware status.

```bash
curl http://localhost:8000/sdr/status
```

Response:
```json
{
  "connected": true,
  "device_name": "Ettus B210",
  "serial": "194919",
  "center_freq_hz": 2449999999,
  "sample_rate_hz": 50000000,
  "bandwidth_hz": 50000000,
  "gain_db": 38.0,
  "gain_mode": "manual",
  "antenna": "TX/RX",
  "lo_locked": false,
  "overflow_count": 0
}
```

#### POST /sdr/tune
Tune to a new center frequency.

```bash
curl -X POST "http://localhost:8000/sdr/tune?center_freq_hz=915000000&sample_rate_hz=20000000"
```

Parameters:
- `center_freq_hz` (required): 70 MHz - 6 GHz
- `sample_rate_hz` (optional): 1-56 MHz, default 50 MHz
- `bandwidth_hz` (optional): Analog filter bandwidth

Response:
```json
{
  "success": true,
  "actual_center_freq_hz": 914999999,
  "actual_sample_rate_hz": 20000000,
  "actual_bandwidth_hz": 20000000,
  "tune_time_ms": 1066.6
}
```

#### POST /sdr/gain
Set receiver gain.

```bash
curl -X POST "http://localhost:8000/sdr/gain?gain_db=50"
```

Response:
```json
{
  "success": true,
  "actual_gain_db": 50.0,
  "gain_range_db": [0.0, 76.0]
}
```

#### POST /sdr/quick-tune
Tune to a preset frequency band.

```bash
curl -X POST "http://localhost:8000/sdr/quick-tune?band_name=WiFi%202.4%20GHz"
```

Available bands:
- WiFi 2.4 GHz (2450 MHz, 50 MSPS)
- WiFi 5 GHz Lower (5250 MHz, 56 MSPS)
- WiFi 5 GHz Upper (5700 MHz, 56 MSPS)
- ISM 900 MHz (915 MHz, 26 MSPS)
- GPS L1 (1575.42 MHz, 10 MSPS)
- LTE Band 7 (2655 MHz, 50 MSPS)
- LTE Band 3 (1842.5 MHz, 50 MSPS)
- Bluetooth (2441 MHz, 50 MSPS)

#### GET /sdr/frequency-bands
List all available preset frequency bands.

```bash
curl http://localhost:8000/sdr/frequency-bands
```

### Spectrum Analysis (`/spectrum/*`)

#### GET /spectrum/measure
Perform FFT measurement.

```bash
curl "http://localhost:8000/spectrum/measure?fft_size=1024&averaging=4"
```

Parameters:
- `fft_size`: 256, 512, 1024, 2048, 4096, 8192
- `averaging`: Number of FFTs to average (1-100)

Response:
```json
{
  "center_freq_hz": 2449999999,
  "span_hz": 50000000,
  "fft_size": 1024,
  "bin_hz": 48828.125,
  "magnitudes_dbm": [-87.5, -86.5, ...]
}
```

#### POST /spectrum/power
Measure integrated power in a frequency band.

```bash
curl -X POST "http://localhost:8000/spectrum/power?start_freq_hz=2400000000&stop_freq_hz=2500000000"
```

Response:
```json
{
  "success": true,
  "total_power_dbm": -25.56,
  "peak_power_dbm": -76.89,
  "peak_freq_hz": 2445919316,
  "average_power_dbm": -75.55,
  "bandwidth_hz": 100000000,
  "noise_floor_dbm": -87.0,
  "snr_db": 10.1
}
```

### DSP Operations (`/dsp/*`)

#### GET /dsp/rssi
Real-time signal strength measurement.

```bash
curl http://localhost:8000/dsp/rssi
```

Response:
```json
{
  "rssi_dbm": -64.63,
  "rssi_linear": 3.44e-07,
  "peak_dbm": -59.04,
  "min_dbm": -72.31,
  "timestamp_ns": 1763749817525000960
}
```

#### POST /dsp/filter/design
Design FIR filter.

```bash
curl -X POST "http://localhost:8000/dsp/filter/design?filter_type=lowpass&cutoff_freq_hz=5000000&sample_rate_hz=50000000"
```

#### POST /dsp/spectrogram
Compute STFT spectrogram.

```bash
curl -X POST "http://localhost:8000/dsp/spectrogram" \
  -H "Content-Type: application/json" \
  -d '{"fft_size": 1024, "hop_size": 256, "window": "hann"}'
```

### I/Q Capture (`/iq/*`)

#### POST /iq/capture
Capture I/Q samples.

```bash
curl -X POST "http://localhost:8000/iq/capture?num_samples=1000000&output_format=cf32"
```

#### POST /iq/sigmf/export
Export capture with SIGMF metadata.

```bash
curl -X POST "http://localhost:8000/iq/sigmf/export?capture_id=abc123"
```

## MCP Integration

All API endpoints are exposed via FastMCP at `/llm/mcp/`. AI agents can discover and invoke tools using the Model Context Protocol.

### Example MCP Tool Call

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

### Available MCP Tools (52 total)

**SDR Control:**
- `get_sdr_status` - Query hardware state
- `tune_sdr` - Set frequency and sample rate
- `set_sdr_gain` - Adjust receiver gain
- `set_sdr_antenna` - Select antenna port
- `calibrate_sdr` - DC offset/IQ balance calibration
- `control_sdr_stream` - Start/stop streaming
- `get_frequency_bands` - List preset bands
- `quick_tune_sdr` - Tune to preset band

**Spectrum Analysis:**
- `measure_spectrum` - Single FFT measurement
- `measure_power` - Band power measurement
- `sweep_spectrum` - Wide-band sweep
- `analyze_occupancy` - Channel duty cycle
- `search_signals` - Find and classify signals
- `get_waterfall_snapshot` - Spectrogram frame

**DSP Filters:**
- `design_filter` - FIR filter design
- `apply_filter` - GPU-accelerated filtering
- `decimate` - Downsample with anti-aliasing
- `channelize` - Polyphase filterbank
- `compute_spectrogram` - STFT computation
- `demodulate` - AM/FM/PSK/QAM demod
- `get_rssi` - Signal strength measurement

**I/Q Capture:**
- `capture_iq` - Burst capture
- `start_recording` - Continuous recording
- `stop_recording` - Stop recording
- `export_sigmf` - SIGMF metadata export
- `get_buffer_stats` - Buffer health metrics

## Troubleshooting

### SDR Not Connecting

1. Check USB connection:
```bash
uhd_find_devices
```

2. Verify device serial:
```bash
uhd_usrp_probe
```

3. Check container has USB access:
```bash
docker exec spectrum_server lsusb | grep Ettus
```

### UHD Import Errors

Ensure PYTHONPATH includes system packages:
```bash
export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"
```

### Overflow Errors

Reduce sample rate or FFT size if seeing overflow_count > 0:
```bash
curl -X POST "http://localhost:8000/sdr/tune?sample_rate_hz=20000000"
```

### Permission Denied

Run container with privileged mode:
```yaml
privileged: true
devices:
  - /dev/bus/usb:/dev/bus/usb
```

## Architecture

```
spectrum_server/
├── src/spectrum_server/
│   ├── api/
│   │   ├── sdr_control.py      # SDR control endpoints
│   │   ├── spectrum_analysis.py # Spectrum measurement endpoints
│   │   ├── iq_capture.py       # I/Q capture endpoints
│   │   ├── dsp_filters.py      # DSP filter endpoints
│   │   └── dashboard.py        # Dashboard streaming
│   ├── schema/
│   │   ├── sdr_control.py      # Pydantic models for SDR
│   │   ├── spectrum_analysis.py # Pydantic models for spectrum
│   │   ├── iq_capture.py       # Pydantic models for capture
│   │   └── dsp_filters.py      # Pydantic models for DSP
│   ├── sdr_control.py          # SDR controller (UHD/SoapySDR)
│   ├── spectrum_stream.py      # WebSocket broadcaster
│   └── server.py               # FastAPI + FastMCP application
└── Dockerfile                  # Ubuntu 24.04 + UHD build
```

## Performance

Tested on NVIDIA DGX Spark with Ettus B210:

| Operation | Typical Time |
|-----------|--------------|
| Tune frequency | ~1000 ms |
| Set gain | ~10 ms |
| 1024-pt FFT | ~5 ms |
| RSSI measurement | ~2 ms |
| Quick-tune preset | ~950 ms |

Maximum sample rates:
- USB 3.0: 56 MSPS
- USB 2.0: ~8 MSPS
