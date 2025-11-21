# Beta Dashboard Integration Guide

This document captures the current state of the dashboard APIs, how they obtain real spectra, and what remains to reach a stable beta.

## Overview
- `/dashboard` REST + WebSocket endpoints now attempt to pull real spectra from the C++ GPU pipeline via `spectrum_core_client` (ZeroMQ REQ/REP).
- If the C++ service is unavailable or times out, endpoints fall back to simulated data to keep the UI responsive.
- Lightweight peak detections (for overlays) are computed from the returned spectrum frames; thresholds are tunable via an env var.
- Metrics use system telemetry from `psutil`; GPU metrics are placeholders until NVML/DCGM wiring is added.

## Data Flow
1) Client requests `/dashboard/config` (or opens `/dashboard/ws/dashboard`).
2) Server fetches SDR status (center freq, sample rate, gain).
3) Server calls `spectrum_core_client.measure_spectrum(...)` with fft_size, averaging=10.
   - Transport: ZeroMQ REQ/REP to `SPECTRUM_CORE_ENDPOINT`.
   - Request frame 0 (JSON):
     ```json
     {"op": "measure", "center_freq_hz": 2450000000, "sample_rate_hz": 50000000, "fft_size": 2048, "averaging": 10}
     ```
   - Response frame 0 (JSON): same fields plus `timestamp_ns`.
   - Response frame 1 (binary): float32 log-power array of length `fft_size`.
4) On success, the dashboard endpoints emit spectrum payloads with `fft_size`, `bin_hz`, `freq_start_hz`, `freq_stop_hz`, and `magnitudes_db`.
5) A lightweight peak picker generates detection overlays (`detections` message) from the same frame.

## WebSocket Messages (`/dashboard/ws/dashboard`)
- `config`: initial config snapshot.
- `spectrum`: `{timestamp_ns, center_freq_hz, sample_rate_hz, fft_size, bin_hz, freq_start_hz, freq_stop_hz, magnitudes_db[], frame_index}`.
- `detections`: `{timestamp_ns, detections: [{detection_id, center_freq_hz, bandwidth_hz, power_dbm, confidence, bounding_box}...]}`.
- `metrics`: `{timestamp_ns, cpu_usage_percent, gpu_usage_percent (placeholder), memory_used_mb, samples_processed, detections_count, ...}`.
- Commands: `{"cmd": "set_rate", "rate_hz": 5|...}`, `{"cmd": "set_fft_size", ...}`, `{"cmd": "enable_detections": true|false}`, `{"cmd": "enable_metrics": true|false}`, `{"cmd": "stop": true}`.

## Environment Variables
- `SPECTRUM_CORE_ENDPOINT`: ZeroMQ endpoint (e.g., `tcp://127.0.0.1:5555`). If unset, simulated data is used.
- `SPECTRUM_CORE_RESPONSE_TIMEOUT_MS` (default `200`): REQ/REP timeout before falling back.
- `DASHBOARD_DETECTION_THRESHOLD_DB` (default `-65`): Peak picker floor; median noise + 6 dB is also applied.
- `SPECTRUM_CORE_IQ_ENDPOINT`: Optional ZeroMQ PULL endpoint for raw IQ frames (cf32 interleaved). When set, the C++ server will ingest live IQ; otherwise it synthesizes a tone.
- `SPECTRUM_CORE_DET_ENDPOINT`: Optional ZeroMQ PULL endpoint for detection JSON arrays (fields: detection_id, timestamp_ns, freq_hz, bandwidth_hz, power_dbm, confidence, class_id). When set, detections are forwarded to the dashboard header.
- `SPECTRUM_CORE_BIND` (C++ server): bind address (default `tcp://*:5555`).

## Status: Ready for Beta
- Real spectra are consumed when available; deterministic fallbacks keep the UI alive.
- Overlay detections are at least consistent with the spectra shown.
- Config is persistent across REST calls; WebSocket sessions maintain their own runtime config derived from the shared defaults.

## Remaining Gaps
- GPU metrics: Wire NVML/DCGM to replace placeholders. (NVML now used when `pynvml` is available; DCGM still pending.)
- End-to-end detections: Replace lightweight peaks with engine-derived detections (YOLO/threshold outputs from the C++ pipeline). Core service can return detection stubs; dashboard consumes them when present.
- Backpressure & buffering: Integrate `spectrum_stream` (WebSocket broadcaster) to avoid duplicate streams and to fan out the same frames to all clients.
- Error reporting: Surface transport errors/timeouts to the UI more explicitly.
- Auth/rate limits: Not yet addressed.

## How to Test
1) Export env vars:
   ```bash
   export SPECTRUM_CORE_ENDPOINT=tcp://127.0.0.1:5555
   export DASHBOARD_DETECTION_THRESHOLD_DB=-65
   ```
2) Run `uvicorn spectrum_server.server:app --host 0.0.0.0 --port 8000`.
3) Open `/dashboard/ws/dashboard` and verify:
   - `spectrum` messages carry correct center_freq/sample_rate/bin_hz.
   - `detections` appear near visible peaks.
   - If the C++ service is offline, spectra still stream via fallback (log warning in server logs).

## Server-Side Expectations (C++ service)
- Respond to `op=measure` with:
  - JSON header: center_freq_hz, sample_rate_hz, fft_size, averaging, timestamp_ns.
  - Binary frame: float32 log-power spectrum (length fft_size).
- Optional: `detections` array in the header. The dashboard will surface detections if provided; otherwise it will run a lightweight peak picker.

## spectrum_core_server (ZeroMQ producer)
- New C++ app `spectrum_core_server` binds `tcp://*:5555` by default (override with `SPECTRUM_CORE_BIND`).
- Accepts the JSON request described above and responds with header + float32 power payload computed by `GpuSpectrumPipeline`.
- Detections are included in the header (peak-based) and converted to absolute frequency before sending.

## GPU Metrics
- Dashboard now attempts NVML via `pynvml` (`nvidia-ml-py` dependency). If NVML is unavailable, it falls back to synthetic GPU metrics; DCGM integration remains a TODO.
