#!/bin/bash
#
# Dependency Installation Script for NVIDIA Spark (ARM64 + GB10)
# For Agentic Spectrum Analyzer with Ettus B210 support
#

set -e  # Exit on error

echo "=================================="
echo "NVIDIA Spark Dependencies Installer"
echo "=================================="
echo ""

# Check if running on ARM64
if [ "$(uname -m)" != "aarch64" ]; then
    echo "ERROR: This script is for ARM64 architecture only!"
    exit 1
fi

echo "[1/7] Installing system dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    libssl-dev \
    ca-certificates \
    gnupg \
    lsb-release

echo "[2/7] Installing CUDA build tools (if needed)..."
sudo apt install -y \
    cuda-toolkit-13-0 \
    cuda-compiler-13-0 \
    cuda-libraries-dev-13-0 || echo "CUDA already installed"

echo "[3/7] Installing UHD development files..."
sudo apt install -y \
    libuhd-dev \
    uhd-host \
    libboost-all-dev

# Download B210 FPGA images
echo "Downloading B210 FPGA images..."
sudo uhd_images_downloader -t b2xx || echo "Images already downloaded"

echo "[4/7] Installing OpenCV dependencies..."
sudo apt install -y \
    libopencv-dev \
    libopencv-contrib-dev \
    python3-opencv || echo "Will build OpenCV from source if needed"

echo "[5/7] Installing TensorRT (if available)..."
sudo apt install -y \
    tensorrt \
    libnvinfer-dev \
    libnvinfer-plugin-dev || echo "TensorRT may need manual installation"

echo "[6/7] Installing MongoDB C++ driver dependencies..."
sudo apt install -y \
    libmongoc-1.0-0 \
    libmongoc-dev \
    libbson-1.0-0 \
    libbson-dev \
    libssl-dev

echo "[7/7] Installing graphics libraries..."
sudo apt install -y \
    libgl-dev \
    libglx-dev \
    libglfw3-dev \
    freeglut3-dev \
    rapidjson-dev

echo ""
echo "=================================="
echo "Checking MongoDB C++ driver..."
echo "=================================="

if [ ! -f "/usr/local/lib/libbsoncxx.so" ] && [ ! -f "/usr/lib/aarch64-linux-gnu/libbsoncxx.so" ]; then
    echo "MongoDB C++ driver not found. Building from source..."

    cd /tmp
    wget https://github.com/mongodb/mongo-cxx-driver/releases/download/r4.1.0/mongo-cxx-driver-r4.1.0.tar.gz
    tar -xzf mongo-cxx-driver-r4.1.0.tar.gz
    cd mongo-cxx-driver-r4.1.0/build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 ../
    make -j$(nproc)
    sudo make install
    sudo ldconfig

    echo "MongoDB C++ driver installed!"
else
    echo "MongoDB C++ driver already installed."
fi

echo ""
echo "=================================="
echo "Dependency installation complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Verify CUDA: nvcc --version"
echo "2. Verify B210: uhd_find_devices"
echo "3. Build project: cd tensorrt_yolo_engine && ./do_run_cmake -t tye_sp -c /usr/local/cuda-13.0 -r ettus_b210"
echo ""
