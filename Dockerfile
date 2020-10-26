FROM ubuntu:20.04
MAINTAINER Yosshi999

RUN apt-get update && apt-get install -y \
	git build-essential wget libssl-dev 

WORKDIR /opt

RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz && tar xzf cmake-3.18.4.tar.gz
RUN cd cmake-3.18.4 && ./bootstrap && make -j$(nproc) && make install

RUN apt-get install -y libpcl-dev=1.10.0

WORKDIR /workspace

