FROM nvidia/cuda:10.1-base-ubuntu18.04

# GPG key error fix
# ref: https://github.com/nyukat/mammography_metarepository/issues/24
# ref: https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list || true
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Proceed with building
RUN apt-get update
RUN apt-get install -y libglib2.0-0\
    libsm6 \
    libxext6 \
    libxrender-dev \
    zlib1g-dev \
    libjpeg-dev \
    vim \
    git \
    wget \
    bash \
    python3.6 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/bcc

RUN git clone https://github.com/nyukat/breast_cancer_classifier.git \
    && cd breast_cancer_classifier \
    && git reset --hard de2b0855d02984df0f516008bb4513ff71460e21 \
    && cd ..

RUN pip3 install --no-cache-dir --trusted-host pypi.python.org h5py==2.8.0 \
    imageio==2.4.1 \
    numpy==1.14.3 \
    opencv-python==3.4.2.17 \
    pandas==0.22.0 \
    scipy==1.0.0 \
    tqdm==4.19.8

RUN wget https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl && pip3 install torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip3 install --no-cache-dir --trusted-host pypi.python.org torchvision==0.2.1

RUN sed -i "179i \ \ \ \ folders = heatmap_save_path_malignant.split('/')\n    dirs = folders[:-1]\n    dir_path = '/'.join(dirs)\n    os.makedirs(dir_path, exist_ok=True)\n" /home/bcc/breast_cancer_classifier/src/heatmaps/run_producer.py \
    && sed -i "190i \ \ \ \ folders = heatmap_save_path_benign.split('/')\n    dirs = folders[:-1]\n    dir_path = '/'.join(dirs)\n    os.makedirs(dir_path, exist_ok=True)\n" /home/bcc/breast_cancer_classifier/src/heatmaps/run_producer.py \
    && sed -i "157s/.*/ \ \ \ return predictions_dict/" /home/bcc/breast_cancer_classifier/src/modeling/run_model_single.py

WORKDIR /home
# Copy in username file
COPY users.txt /home/
COPY create_users.sh /home/
RUN ./create_users.sh \
    && rm users.txt create_users.sh

ARG GROUPNAME
WORKDIR /home/predictions
RUN chown :$GROUPNAME $(pwd) && chmod 771 .

WORKDIR /home/bcc/breast_cancer_classifier
RUN chown :$GROUPNAME $(pwd) && chmod 771 .

USER nyu_model_single_user
WORKDIR /home/bcc/breast_cancer_classifier
