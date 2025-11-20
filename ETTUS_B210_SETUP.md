# Ettus B210 SDR Integration Guide

This guide describes how to build and use the Agentic Spectrum Analyzer with the Ettus B210 SDR on Nvidia Spark.

## Table of Contents
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Building the Project](#building-the-project)
- [Running with B210](#running-with-b210)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)

---

## Overview

The Agentic Spectrum Analyzer has been enhanced to support the **Ettus B210 SDR** via the USRP Hardware Driver (UHD) library. The B210 offers:

- **Frequency Range**: 70 MHz - 6 GHz
- **Max Sample Rate**: 56 MHz (61.44 MSPS)
- **Bandwidth**: 56 MHz
- **Channels**: 2 RX, 2 TX (dual-channel)
- **Gain Range**: 0-76 dB
- **Connection**: USB 3.0

### Key Differences from Signal Hound

| Feature | Signal Hound SM200B | Ettus B210 |
|---------|---------------------|------------|
| Frequency Range | 100 kHz - 20 GHz | 70 MHz - 6 GHz |
| Max Sample Rate | 50 MHz | 56 MHz |
| Data Type | 32-bit complex float | 16-bit complex int (auto-converted) |
| Gain Control | Attenuation (0-30 dB) | Gain (0-76 dB) |
| Antenna | Auto | TX/RX, RX2 (selectable) |
| Channels | 1 RX | 2 RX, 2 TX |

---

## Hardware Requirements

### Minimum System Requirements
- **CPU**: Intel Core i7 or AMD Ryzen 7 (8+ cores recommended)
- **RAM**: 16 GB minimum, 32 GB recommended
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada, or Hopper)
- **Storage**: 50 GB free space
- **USB**: USB 3.0 port (SuperSpeed, 5 Gbps)

### Ettus B210 Setup
1. Connect B210 to USB 3.0 port (blue port)
2. Verify connection:
   ```bash
   lsusb | grep Ettus
   # Should show: Bus XXX Device XXX: ID 2500:0020 Ettus Research LLC USRP B200
   ```
3. Test basic functionality:
   ```bash
   uhd_find_devices
   uhd_usrp_probe
   ```

---

## Software Dependencies

### Core Dependencies
- **CUDA**: 12.9.1 or compatible
- **cuDNN**: Compatible with CUDA version
- **TensorRT**: 8.x or newer
- **UHD**: 4.0.0+ (installed via apt or built from source)
- **Boost**: 1.70+ (required by UHD)
- **OpenCV**: 4.12.0 (CUDA-enabled)
- **MongoDB C++ Driver**: 4.1.0+

### Python Dependencies
All Python dependencies are managed in the virtual environment:
```bash
# Activate virtual environment
source venv/bin/activate

# Install spectrum_server dependencies
cd spectrum_server
uv pip install -e .

# Install agent dependencies
cd ../agent
uv pip install -e .
```

---

## Building the Project

### Step 1: Set Up Virtual Environment
```bash
cd /home/cvalentine/agentic-spectrumdetect
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Build with Docker (Recommended)

#### Build Docker Image
```bash
cd tensorrt_yolo_engine
docker build -t agentic-spectrumdetect:b210 .
```

The Dockerfile automatically:
- Installs UHD libraries
- Downloads B210 FPGA images
- Builds the project with Ettus B210 support

#### Run Docker Container
```bash
docker run --gpus all \
  --device=/dev/bus/usb \
  --privileged \
  -v $(pwd):/build/tensorrt_yolo_engine \
  -it agentic-spectrumdetect:b210 /bin/bash
```

### Step 3: Build Manually (Alternative)

If building outside Docker:

#### Install UHD
```bash
sudo apt update
sudo apt install -y libuhd-dev uhd-host libboost-all-dev
sudo uhd_images_downloader -t b2xx
```

#### Build TensorRT YOLO Engine
```bash
cd tensorrt_yolo_engine

# Build with Ettus B210 support (default)
./do_run_cmake -t tye_sp -c /usr/local/cuda-12.9

# Or explicitly specify radio type
./do_run_cmake -t tye_sp -c /usr/local/cuda-12.9 -r ettus_b210

# Compile
cd build
make -j$(nproc)
```

#### Build Options
- `-r ettus_b210` - Ettus B210 SDR only (default)
- `-r signalhound` - Signal Hound SM series only
- `-r both` - Support both radio types

---

## Running with B210

### Step 1: Start MongoDB
```bash
docker-compose up -d mongodb
```

### Step 2: Run TensorRT YOLO Engine
```bash
cd tensorrt_yolo_engine/bin

# Basic usage
./tye_sp \
  --gpus 1 \
  --engine-path ../11s.int8.gpu1.engine \
  --engines-per-gpu 2 \
  --sample-rate-mhz 50 \
  --center-freq 2450000000 \
  --atten-db 10

# Advanced usage with all options
./tye_sp \
  --gpus 1 \
  --engine-path ../11s.int8.gpu1.engine \
  --engines-per-gpu 2 \
  --sample-rate-mhz 50 \
  --center-freq 2450000000 \
  --atten-db 0 \
  --ref-level -20.0 \
  --retune-port 63333 \
  --display-plots
```

### Step 3: Start Spectrum Server
```bash
cd spectrum_server
source ../venv/bin/activate
uvicorn src.spectrum_server.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Start Agent (Optional)
```bash
cd agent
source ../venv/bin/activate
python -m agent
```

---

## Configuration Reference

### Command-Line Arguments

#### Required Arguments
- `--gpus N` - Number of GPUs to use
- `--engine-path PATH` - Path to TensorRT engine file
- `--engines-per-gpu N` - Number of inference engines per GPU
- `--sample-rate-mhz RATE` - Sample rate in MHz (1-56)
- `--center-freq HZ` - Center frequency in Hz (70 MHz - 6 GHz)

#### Optional Arguments
- `--atten-db DB` - Attenuation in dB (0-30, or -1 for auto)
  - **Note**: B210 uses gain internally. Formula: `gain_db = 76 - atten_db`
  - `--atten-db 0` → 76 dB gain (maximum)
  - `--atten-db 10` → 50.67 dB gain
  - `--atten-db 30` → 0 dB gain (minimum)
- `--ref-level LEVEL` - Reference level (default: -20.0)
- `--retune-port PORT` - UDP port for retune commands (default: 63333)
- `--display-plots` - Enable OpenGL visualization

### Gain/Attenuation Mapping

The B210 uses **gain** (higher is more sensitive), while Signal Hound uses **attenuation** (higher reduces sensitivity). The software automatically converts:

| Attenuation (dB) | B210 Gain (dB) | Use Case |
|------------------|----------------|----------|
| 0 | 76 | Maximum sensitivity (weak signals) |
| 10 | 50.67 | Moderate signals |
| 20 | 25.33 | Strong signals |
| 30 | 0 | Minimum sensitivity (very strong signals) |
| -1 (auto) | 38 | Auto gain (middle range) |

### Sample Rate Recommendations

The B210 supports flexible sample rates up to 56 MHz:

| Sample Rate | Bandwidth | Best For |
|-------------|-----------|----------|
| 56 MHz | 56 MHz | Maximum coverage |
| 50 MHz | 50 MHz | Compatible with Signal Hound |
| 25 MHz | 25 MHz | Moderate coverage, lower CPU |
| 10 MHz | 10 MHz | Narrow band, low CPU usage |
| 5 MHz | 5 MHz | Very narrow band |

### Frequency Range Constraints

- **Minimum**: 70 MHz (no HF/VHF below 70 MHz)
- **Maximum**: 6 GHz (no microwave above 6 GHz)
- **Common Bands**:
  - FM Broadcast: 88-108 MHz
  - ISM 900: 902-928 MHz
  - GPS L1: 1575.42 MHz
  - WiFi 2.4 GHz: 2400-2500 MHz
  - WiFi 5 GHz: 5150-5850 MHz

---

## Troubleshooting

### Issue: Device Not Found
```
ERROR: UHD Could not find a device
```

**Solution**:
1. Check USB connection:
   ```bash
   lsusb | grep Ettus
   ```
2. Verify USB 3.0 port (blue port)
3. Check device permissions:
   ```bash
   sudo uhd_find_devices
   ```
4. Add udev rules:
   ```bash
   sudo cp /usr/lib/uhd/utils/uhd-usrp.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

### Issue: Sample Rate Error
```
ERROR: Sample rate XXXXX Hz exceeds maximum 56000000 Hz
```

**Solution**: Reduce sample rate to ≤56 MHz
```bash
./tye_sp ... --sample-rate-mhz 50
```

### Issue: Frequency Out of Range
```
ERROR: Frequency XXXXX Hz out of range [70000000, 6000000000]
```

**Solution**: Use frequency between 70 MHz and 6 GHz
```bash
# Invalid: 50 MHz (too low)
# Valid: 100 MHz
./tye_sp ... --center-freq 100000000
```

### Issue: Overflow Warnings
```
WARNING: Overflow detected
```

**Solution**:
1. Reduce sample rate
2. Enable USB 3.0 SuperSpeed
3. Disable USB autosuspend:
   ```bash
   echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend
   ```
4. Set CPU governor to performance:
   ```bash
   sudo cpupower frequency-set -g performance
   ```

### Issue: Compilation Errors

**UHD Not Found**:
```bash
sudo apt install -y libuhd-dev uhd-host
```

**Boost Not Found**:
```bash
sudo apt install -y libboost-all-dev
```

**CUDA Version Mismatch**: Ensure CUDA 12.9 is installed and `nvcc` is in PATH
```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
```

---

## Performance Tuning

### USB Performance
```bash
# Increase USB buffer size
echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

# Disable USB autosuspend
echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend
```

### CPU Performance
```bash
# Set CPU governor to performance
sudo apt install -y cpufrequtils
sudo cpufreq-set -g performance
```

### GPU Performance
```bash
# Set GPU to maximum performance
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit (adjust for your GPU)
```

### System Tuning
```bash
# Increase UDP buffer sizes
sudo sysctl -w net.core.rmem_max=67108864
sudo sysctl -w net.core.wmem_max=67108864

# Disable CPU scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Example Workflows

### WiFi 2.4 GHz Monitoring
```bash
./tye_sp \
  --gpus 1 \
  --engine-path ../11s.int8.gpu1.engine \
  --engines-per-gpu 2 \
  --sample-rate-mhz 50 \
  --center-freq 2450000000 \
  --atten-db 10
```

### ISM 900 MHz Scanning
```bash
./tye_sp \
  --gpus 1 \
  --engine-path ../11s.int8.gpu1.engine \
  --engines-per-gpu 2 \
  --sample-rate-mhz 25 \
  --center-freq 915000000 \
  --atten-db 5
```

### Wide-Band Survey (56 MHz)
```bash
./tye_sp \
  --gpus 1 \
  --engine-path ../11s.int8.gpu1.engine \
  --engines-per-gpu 2 \
  --sample-rate-mhz 56 \
  --center-freq 3000000000 \
  --atten-db 15
```

---

## Additional Resources

### UHD Documentation
- [UHD Manual](https://files.ettus.com/manual/)
- [B200/B210 User Guide](https://www.ettus.com/all-products/usrp-b210/)

### Support
- **GitHub Issues**: https://github.com/anthropics/claude-code/issues
- **UHD Support**: https://files.ettus.com/manual/page_support.html

---

## Version History

- **v1.0.0** (2025-01-19): Initial Ettus B210 support
  - Added UHD B210 radio driver
  - Implemented gain/attenuation conversion
  - Docker integration with UHD
  - Build system enhancements

---

## License

This project inherits the license from the Agentic Spectrum Analyzer project.

---

**Last Updated**: 2025-01-19
**Maintainer**: Claude Code Integration
