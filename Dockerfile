FROM ubuntu:20.04
MAINTAINER Yosshi999

RUN apt-get update && apt-get install -y \
	git build-essential wget libssl-dev 

WORKDIR /opt

RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz && tar xzf cmake-3.18.4.tar.gz
RUN cd cmake-3.18.4 && ./bootstrap && make -j$(nproc) && make install

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libpcl-dev

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:joseluisblancoc/mrpt-stable && \
    apt-get update && \
    apt-get install -y libmrpt-dev mrpt-apps

ARG uid
ARG gid
RUN groupadd -g $gid app && useradd -u $uid -g $gid app
USER app

WORKDIR /workspace

