FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN apt-get update && apt-get install -y libglib2.0-0\
    libsm6 \
    libxext6 \
    libxrender-dev \
    zlib1g-dev \
    libjpeg-dev \
    vim \
    git \
    bash \
    python3.6 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/gmic

RUN git clone https://github.com/nyukat/GMIC.git \
    && cd GMIC \
    && git reset --hard 3bf4ce81dfa40553f108c8bfaf03bf006e082761 \
    && cd ..

# Modify src/scripts/run_model.py line 148/149
RUN sed -i "149i \                if len(datum[view]) == 0: continue" /home/gmic/GMIC/src/scripts/run_model.py

# Install dependencies
RUN pip3 install --no-cache-dir --trusted-host pypi.python.org h5py==2.8.0 \
    imageio==2.4.1 \
    numpy==1.14.3 \
    opencv-python==3.4.2.17 \
    pandas==0.22.0 \
    scipy==1.0.0 \
    torch==1.1.0 \
    torchvision==0.2.2 \
    tqdm==4.19.8 \
    matplotlib==3.0.2

WORKDIR /home
# Copy in username file
COPY users.txt /home/
COPY create_users.sh /home/
RUN ./create_users.sh && \
    rm users.txt create_users.sh

ARG GROUPNAME
WORKDIR /home/predictions
RUN chown :$GROUPNAME $(pwd) && chmod 771 .

USER nyu_gmic_user
WORKDIR /home/gmic/GMIC
