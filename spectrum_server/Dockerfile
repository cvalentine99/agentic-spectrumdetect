# Stage 1: Build the application
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder

# Set the working directory in the container
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy the current directory contents into the container at /app
COPY . /app

RUN uv build --wheel


# Stage 2: Create the final container with UHD support
FROM ubuntu:24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12, UHD, and required system packages
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    python3-uhd \
    libuhd4.6.0 \
    uhd-host \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set the working directory in the container
WORKDIR /app

# Copy the built wheel from the builder stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=builder /app/dist/*.whl .

# Remove externally-managed marker to allow pip installs
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Install the application (using system packages for UHD)
RUN uv pip install --system --no-cache-dir *.whl

# Make UHD Python bindings available to Python 3.12
ENV PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"

# UHD images path
ENV UHD_IMAGES_DIR="/usr/share/uhd/images"

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["python", "-m", "spectrum_server"]
