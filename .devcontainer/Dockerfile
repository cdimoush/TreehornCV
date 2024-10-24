# Use Ubuntu 22.04
FROM ubuntu:22.04

# Install dependencies for both Python and C++
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbbmalloc2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-dev \
    && rm -rf /var/lib/apt/lists/*

# Clean up APT cache and remove CMake script
RUN apt-get clean

# Set default command to run Python
CMD ["python"]
