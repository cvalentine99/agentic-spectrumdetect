#!/bin/bash
#
# Build OpenCV 4.x (latest) with CUDA 13 support for NVIDIA Spark
#

set -e

echo "=================================="
echo "OpenCV CUDA Build for CUDA 13"
echo "=================================="
echo ""

# Clean previous attempts
rm -rf /tmp/opencv_build
mkdir -p /tmp/opencv_build
cd /tmp/opencv_build

# Download OpenCV 4.x branch (supports CUDA 13)
echo "[1/5] Downloading OpenCV 4.x (latest, CUDA 13 compatible)..."
git clone --depth 1 -b 4.x https://github.com/opencv/opencv.git
git clone --depth 1 -b 4.x https://github.com/opencv/opencv_contrib.git

# Configure build
echo "[2/5] Configuring CMake for CUDA 13..."
cd opencv
mkdir -p build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_TBB=ON \
      -D CUDA_FAST_MATH=1 \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D ENABLE_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D CUDA_ARCH_BIN="100" \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0 \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      ..

echo ""
echo "[3/5] Building OpenCV (this will take ~2 hours on ARM64)..."
make -j4

echo ""
echo "[4/5] Installing OpenCV..."
sudo make install
sudo ldconfig

echo ""
echo "[5/5] Cleaning up..."
cd /
rm -rf /tmp/opencv_build

echo ""
echo "=================================="
echo "✅ OpenCV with CUDA 13 support installed!"
echo "=================================="
echo ""
opencv_version --version 2>/dev/null || echo "OpenCV installed at /usr/local"
echo ""
echo "Next: Build TensorRT YOLO Engine"
echo "  cd ~/agentic-spectrumdetect/tensorrt_yolo_engine"
echo "  ./do_run_cmake -t tye_sp -c /usr/local/cuda-13.0 -r ettus_b210"
echo "  cd build && make -j4"
echo ""
