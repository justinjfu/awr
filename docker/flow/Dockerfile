# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM justinfu/awr:0.1

RUN apt-get update
RUN apt-get install -y cmake swig libgtest-dev python-pygame python-scipy
RUN apt-get install -y autoconf libtool pkg-config libgdal-dev libxerces-c-dev
RUN apt-get install -y libproj-dev libfox-1.6-dev libxml2-dev libxslt1-dev
RUN apt-get install -y build-essential curl unzip flex bison python python-dev
RUN apt-get install -y python3-dev
RUN pip3 install cmake cython

WORKDIR /root/sumo_binaries/bin
RUN wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/binaries-ubuntu1804.tar.xz
RUN tar -xf binaries-ubuntu1804.tar.xz
RUN rm binaries-ubuntu1804.tar.xz
RUN chmod +x *
ENV PATH /root/sumo_binaries/bin:${PATH}
ENV SUMO_HOME /root/sumo_binaries/bin

# RUN pip install git+git://github.com/aravindr93/mjrl@master#egg=mjrl
RUN pip install https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/sumotools-0.4.0-py3-none-any.whl
RUN pip install git+https://github.com/flow-project/flow@master#egg=flow
WORKDIR /root/code

