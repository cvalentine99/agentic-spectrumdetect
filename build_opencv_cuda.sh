#!/bin/bash
#
# Build OpenCV 4.12.0 with CUDA support for NVIDIA Spark (ARM64 + GB10)
#

set -e

echo "=================================="
echo "OpenCV CUDA Build for NVIDIA Spark"
echo "=================================="
echo ""
echo "This will take 2-3 hours on ARM64."
echo "Building OpenCV 4.12.0 with CUDA 13.0 support..."
echo ""

# Create build directory
BUILD_DIR=/tmp/opencv_build
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Download OpenCV and contrib
if [ ! -d "opencv" ]; then
    echo "[1/5] Downloading OpenCV 4.12.0..."
    git clone --depth 1 -b 4.12.0 https://github.com/opencv/opencv.git
fi

if [ ! -d "opencv_contrib" ]; then
    echo "[2/5] Downloading OpenCV contrib modules..."
    git clone --depth 1 -b 4.12.0 https://github.com/opencv/opencv_contrib.git
fi

# Configure build
echo "[3/5] Configuring CMake..."
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

# Build
echo "[4/5] Building OpenCV (this will take ~2 hours)..."
make -j4

# Install
echo "[5/5] Installing OpenCV..."
sudo make install
sudo ldconfig

echo ""
echo "=================================="
echo "OpenCV with CUDA installed successfully!"
echo "=================================="
echo ""
echo "Next step: Build TensorRT YOLO Engine"
echo "  cd ~/agentic-spectrumdetect/tensorrt_yolo_engine"
echo "  ./do_run_cmake -t tye_sp -c /usr/local/cuda-13.0 -r ettus_b210"
echo "  cd build && make -j4"
echo ""
