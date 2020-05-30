# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update -q
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.7-dev python3.7 python3-pip
RUN virtualenv --python=python3.7 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.7 /usr/bin/python
RUN ln -s /env/bin/pip3.7 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get install -y ffmpeg

# For some reason this works despite an error showing up...
RUN DEBIAN_FRONTEND=noninteractive apt-get -qy install nvidia-384; exit 0
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-384

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
RUN ln -s /root/.mujoco/mujoco200 /root/.mujoco/mujoco200_linux
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN mkdir /root/install
WORKDIR /root/install

# For atari-py
RUN apt-get install -y zlib1g-dev swig cmake

# Previous versions installed from a requirements.txt, but direct pip
# install seems cleaner
RUN pip install glfw>=1.4.0
RUN pip install numpy>=1.11
RUN pip install scipy>=0.18.0
RUN pip install Cython>=0.27.2
RUN pip install matplotlib>=3.2.0
RUN pip install imageio>=2.1.2
RUN pip install cffi>=1.10
RUN pip install imagehash>=3.4
RUN pip install sk-video==1.1.10
RUN pip install h5py==2.10.0
RUN pip install ipdb
RUN pip install Pillow>=4.0.0
RUN pip install pycparser>=2.17.0
RUN pip install pytest>=3.0.5
RUN pip install pytest-instafail==0.3.0
RUN pip install sphinx
RUN pip install sphinx_rtd_theme
RUN pip install numpydoc
RUN pip install cloudpickle==1.3.0
RUN pip install cached-property==1.3.1
RUN pip install gitpython==3.1.0
RUN pip install gtimer==1.0.0b5
RUN pip install awscli==1.18.20
RUN pip install boto3==1.12.20
RUN pip install path.py==10.3.1
RUN pip install joblib==0.9.4
RUN pip install opencv-python==4.2.0.32
RUN pip install ray==0.8.2
RUN pip install mujoco_py==2.0.2.9
RUN pip install gym==0.17.1
RUN pip install http://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-linux_x86_64.whl
RUN pip install torchvision==0.4.1
RUN pip install click
RUN pip install absl-py
RUN pip install termcolor

RUN pip install git+git://github.com/deepmind/dm_control@master#egg=dm_control
RUN pip install git+https://github.com/vitchyr/multiworld.git
RUN pip install git+git://github.com/aravindr93/mjrl@master#egg=mjrl
RUN pip install git+git://github.com/rlworkgroup/metaworld.git@master#egg=metaworld

WORKDIR /root/installs
RUN git clone https://github.com/google-research/relay-policy-learning
ENV PYTHONPATH /root/installs/relay-policy-learning/adept_envs:${PYTHONPATH}

RUN pip install tensorflow==1.14.0

WORKDIR /root/code
