FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo 
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y git wget vim curl \
    # && apt-get install -y python3-pip \
    && apt-get install -y python3.8 python3.8-distutils \
    && apt-get install -y libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && curl https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install opencv-python \
    && pip3 install matplotlib \
    && pip3 install gradio \
    && pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install openmim \
#    && pip3 install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html \
    && pip3 install --upgrade pip \
    && pip3 install openmim \
    && mim install mmcv-full==1.4.0 \
    && pip3 install mmdet==2.18.0 \
    && pip3 install mmpose==0.20.0     

CMD [ "/bin/bash" ]
