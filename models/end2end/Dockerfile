FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y libglib2.0-0\
    libsm6 \
    libxext6 \
    libxrender-dev \
    vim \
    git \
    bash \
    python3-dev \
    python3.5-venv \
    curl \
    libmagickwand-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pip3
WORKDIR /home/pip_install

RUN curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

WORKDIR /home
RUN rm -rf pip_install

RUN pip install gdown

# Create two virtualenvs for separate CPU and GPU installations
RUN python3 -m venv /opt/venv-gpu
RUN /opt/venv-gpu/bin/pip install --upgrade "pip < 21.0"
RUN /opt/venv-gpu/bin/pip install pip install Keras==2.0.5 \
    numpy==1.16.4 \
    opencv-python==4.1.0.25 \
    pydicom==1.2.2 \
    h5py==2.9.0 \
    Wand==0.5.4 \
    pandas==0.24.1 \
    tensorflow-gpu==1.4.0 \
    tqdm==4.53.0

RUN python3 -m venv /opt/venv-cpu
RUN /opt/venv-cpu/bin/pip install --upgrade "pip < 21.0"
RUN /opt/venv-cpu/bin/pip install pip install Keras==2.0.5 \
    numpy==1.16.4 \
    opencv-python==4.1.0.25 \
    pydicom==1.2.2 \
    h5py==2.9.0 \
    Wand==0.5.4 \
    pandas==0.24.1 \
    tensorflow==1.4.0 \
    tqdm==4.53.0

# Copy in username file
COPY users.txt /home/
COPY create_users.sh /home/
RUN ./create_users.sh \
    && rm users.txt create_users.sh

ARG GROUPNAME
WORKDIR /home/predictions
RUN chown :$GROUPNAME $(pwd) && chmod 771 .

WORKDIR /home/end2end
RUN chown :$GROUPNAME $(pwd)

WORKDIR /home/end2end/weights
RUN chown :$GROUPNAME $(pwd)

# Download weight files from Google Drive
RUN pip install --upgrade setuptools wheel
RUN for i in "0B1PVLadG_dCKSUJYdzNyZjVsZHc" "0B1PVLadG_dCKdVQzbDRLNTZ4TXM" "0B1PVLadG_dCKYnREWlJQZ2JaSDQ" "0B1PVLadG_dCKVk9RM1dMeTkwcTg" "0B1PVLadG_dCKUnQwYVhOd2NfQlk" "0B1PVLadG_dCKN0ZxNFdCRWxHRFU"; do gdown "https://drive.google.com/uc?id=$i" ; sleep 5;  done

USER end2end_user
WORKDIR /home/end2end

