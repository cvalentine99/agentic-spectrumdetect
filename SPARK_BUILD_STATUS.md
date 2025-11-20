# NVIDIA Spark + Ettus B210 Build Status

## Hardware Configuration
- **Platform**: NVIDIA Spark (Grace-Blackwell)
- **CPU**: ARM64 (aarch64)
- **GPU**: NVIDIA GB10 (Blackwell architecture)
- **SDR**: Ettus B210 (Serial: 194919, Name: MyB210)

## Software Stack Installed ✅

### Core Components
- [x] **CUDA**: 13.0.88
- [x] **TensorRT**: 10.14.1.48-1+cuda13.0
- [x] **OpenCV**: 4.6.0+dfsg-13.1ubuntu1
- [x] **UHD**: 4.6.0.0 (Ettus USRP Hardware Driver)
- [x] **Boost**: 1.83 (required by UHD)
- [ ] **MongoDB C++ Driver**: 4.1.0 (currently building...)

### Python Environment
- [x] Virtual environment created at: `/home/cvalentine/agentic-spectrumdetect/venv`
- [ ] `spectrum_server` dependencies (pending)
- [ ] `agent` dependencies (pending)

## Ettus B210 Status ✅
```
Device Address:
    serial: 194919
    name: MyB210
    product: B210
    type: b200
```
**Connection**: Working
**FPGA Images**: Available via `uhd_images_downloader`

## Build Configuration
- **Target**: TYE Stream Processor (`tye_sp`)
- **Radio**: Ettus B210 via UHD
- **CUDA Path**: `/usr/local/cuda-13.0`
- **Build Command**:
  ```bash
  ./do_run_cmake -t tye_sp -c /usr/local/cuda-13.0 -r ettus_b210
  ```

## Next Steps

### 1. Complete MongoDB C++ Driver Build
Currently building in background. When complete:
```bash
cd /tmp/mongo-cxx-driver-r4.1.0/build
sudo make install
sudo ldconfig
```

### 2. Build TensorRT YOLO Engine
```bash
cd ~/agentic-spectrumdetect/tensorrt_yolo_engine
./do_run_cmake -t tye_sp -c /usr/local/cuda-13.0 -r ettus_b210
cd build
make -j$(nproc)
```

### 3. Install Python Dependencies
```bash
cd ~/agentic-spectrumdetect
source venv/bin/activate

# Spectrum server
cd spectrum_server
pip install -e .

# Agent
cd ../agent
pip install -e .
```

### 4. Run the System
```bash
# Terminal 1: Start MongoDB
docker-compose up -d mongodb

# Terminal 2: Start TensorRT YOLO Engine
cd tensorrt_yolo_engine/bin
./tye_sp \
  --gpus 1 \
  --engine-path ../11s.int8.gpu1.engine \
  --engines-per-gpu 2 \
  --sample-rate-mhz 50 \
  --center-freq 2450000000 \
  --atten-db 10

# Terminal 3: Start Spectrum Server
cd spectrum_server
source ../venv/bin/activate
uvicorn src.spectrum_server.main:app --host 0.0.0.0 --port 8000
```

## B210 Configuration for Spark

### Frequency Ranges (70 MHz - 6 GHz)
| Band | Frequency | Use Case |
|------|-----------|----------|
| FM Radio | 88-108 MHz | Broadcast monitoring |
| ISM 900 | 902-928 MHz | LoRa, Zigbee, ISM devices |
| GPS L1 | 1575.42 MHz | GPS signals |
| WiFi 2.4 | 2400-2500 MHz | WiFi, Bluetooth |
| WiFi 5 | 5150-5850 MHz | 5GHz WiFi |

### Sample Rates
- **Maximum**: 56 MHz (61.44 MSPS)
- **Recommended**: 25-50 MHz for balance of coverage and CPU load
- **Bandwidth**: Matches sample rate

### Gain Settings
B210 uses **gain** (0-76 dB), converted from attenuation:
- `--atten-db 0` → 76 dB gain (max sensitivity)
- `--atten-db 10` → 50.67 dB gain (moderate)
- `--atten-db 30` → 0 dB gain (min sensitivity)

## Performance Optimization for Spark

### USB 3.0 Tuning
```bash
# Increase USB buffer (recommended for B210)
echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

# Disable USB autosuspend
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend
```

### GPU Performance
```bash
# Set GPU to maximum performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 500  # Adjust for GB10 power limit
```

### System Tuning
```bash
# Increase network buffers for UDP retune
sudo sysctl -w net.core.rmem_max=67108864
sudo sysctl -w net.core.wmem_max=67108864
```

## Troubleshooting

### Issue: UHD Overflow
**Symptom**: `WARNING: Overflow detected`
**Solution**:
1. Reduce sample rate to 25 MHz
2. Ensure USB 3.0 connection (blue port)
3. Apply USB tuning above
4. Check CPU usage

### Issue: GPU Out of Memory
**Symptom**: CUDA out of memory errors
**Solution**:
1. Reduce `--engines-per-gpu` to 1
2. Monitor with `nvidia-smi`
3. Check TensorRT engine size

### Issue: Build Errors
**Missing Headers**: Ensure all -dev packages are installed
**CUDA Not Found**: Verify `/usr/local/cuda-13.0` exists and nvcc is in PATH
**TensorRT Issues**: Check `libnvinfer-dev` version matches CUDA 13.0

## Current Build Status

**Last Updated**: 2025-01-19
**Build Stage**: Dependencies installation
**Status**: MongoDB C++ driver compiling...

---

For detailed Ettus B210 usage, see: `ETTUS_B210_SETUP.md`
