FROM pointcloudlibrary/env:20.04

ARG PCL_VERSION=1.13.0

# workaround for missing boost::date_time
RUN apt-get update && apt-get install -y libboost-date-time-dev && rm -rf /var/lib/apt/lists/*

# Minimum build with required modules:
# - filters
# - io
# - registration
# - visualization
# - common
RUN wget -qO- https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-${PCL_VERSION}.tar.gz | tar xz \
    && cd pcl-pcl-${PCL_VERSION} \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DWITH_LIBUSB=OFF \
        -DWITH_PNG=OFF \
        -DWITH_QHULL=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_PCAP=OFF \
        -DBUILD_ml=OFF \
        -DBUILD_stereo=OFF \
        -DBUILD_tracking=OFF \
        -DBUILD_keypoints=OFF \
        -DBUILD_segmentation=OFF \
        -DBUILD_outofcore=OFF \
        -DBUILD_surface=OFF \
    && make -j2 \
    && make install \
    && cd ../.. \
    && rm -rf pcl-pcl-${PCL_VERSION}/
