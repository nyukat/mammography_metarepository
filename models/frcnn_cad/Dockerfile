FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# Install Caffe dependencies
RUN apt-get update && apt-get install -y libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libgflags-dev \
    liblmdb-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    protobuf-compiler 

RUN apt-get install -y --no-install-recommends libboost-all-dev

RUN apt-get install -y libatlas-base-dev

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y curl \ 

    cmake \
    gcc \
    g++ \
    vim \
    git \
    python3-dev \
    build-essential \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libatlas-base-dev 

RUN apt-get install -y --no-install-recommends libboost-all-dev

# Install pip3
WORKDIR /home/pip_install
RUN curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

WORKDIR /home
RUN rm -rf pip_install

# Install numpy
RUN pip install numpy==1.14.0 \
    Cython \
    scikit-image \
    protobuf \
    easydict \
    pandas \
    pyyaml \
    pydicom \
    tqdm

# Install OpenCV
WORKDIR /home/opencv_install

RUN git clone https://github.com/opencv/opencv.git \
    && cd opencv \
    && git reset --hard 46e275dfe4616b07cd2492c17698d1a0be0aa6df \
    && cd ..

WORKDIR /home/opencv_install/opencv
RUN git checkout 3.4.2 && mkdir build
WORKDIR /home/opencv_install/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local  -D PYTHON_DEFAULT_EXECUTABLE="/usr/bin/python3" .. \
    && make -j8 \
    && make install

# Configure OpenCV
RUN echo "/usr/local/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/opencv.conf \
    && ldconfig
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/x86_64-linux-gnu"

# Install Cython and other dependencies
RUN pip install Cython \
    scikit-image \
    protobuf \
    easydict \
    pandas \
    pyyaml \
    pydicom \
    tqdm

# Clone frcnn_cad
WORKDIR /home/frcnn
RUN git clone https://github.com/riblidezso/frcnn_cad.git \
    && cd frcnn_cad \
    && git reset --hard 6c2deb57e37832c281588da11aed8808ba6ee1e9 \
    && cd ..

# Clone py-faster-rcnn
WORKDIR /home/frcnn/frcnn_cad
RUN git clone --recursive https://github.com/bstadnick/py-faster-rcnn.git \
    && cd py-faster-rcnn \
    && git reset --hard 0041be2631734fde273e36b3e4aabddb6875821e \
    && cd ..

# Edit Makefile.config
WORKDIR /home/frcnn/frcnn_cad/py-faster-rcnn/caffe-fast-rcnn
COPY ./models/frcnn_cad/help/Makefile.config .

# Build py-faster-rcnn
WORKDIR /home/frcnn/frcnn_cad/py-faster-rcnn/lib

# Edit Makefile for python3 and setup.py for python3
RUN sed -i 's/python/python3/' Makefile \
    && sed -i 's/iteritems/items/' setup.py \
    && make


# Configure git username and email
RUN git config --global user.name " " \
    && git config --global user.email " "

# Merge with Caffe for cudnn6
WORKDIR /home/frcnn/frcnn_cad/py-faster-rcnn/caffe-fast-rcnn
# Add remote
RUN git remote add caffe https://github.com/BVLC/caffe.git \
    && git fetch caffe \
    && git merge -X theirs caffe/master -m "Caffe for cudnn 6"

# Remove line from python_layer.hpp
RUN sed -i '/self_.attr("phase") = static_cast<int>(this->phase_);/d' include/caffe/layers/python_layer.hpp

# Create symbolic links to find files
WORKDIR /usr/lib/x86_64-linux-gnu
RUN ln -s libhdf5_serial.so libhdf5.so \
    && ln -s libhdf5_serial_hl.so libhdf5_hl.so

# Build caffe-faster-rcnn
WORKDIR /home/frcnn/frcnn_cad/py-faster-rcnn/caffe-fast-rcnn
RUN make -j8 && make pycaffe

# Update PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/home/frcnn/frcnn_cad/py-faster-rcnn/caffe-fast-rcnn/python/:/home/frcnn/frcnn_cad/py-faster-rcnn/lib:/home/frcnn/frcnn_cad/py-faster-rcnn/lib/rpn:/usr/local/lib/python3.5/site-packages"

# Copy in username file
COPY users.txt /home/
COPY create_users.sh /home/
WORKDIR /home
RUN ./create_users.sh \
    && rm users.txt create_users.sh

ARG GROUPNAME
WORKDIR /home/predictions
RUN chown :$GROUPNAME $(pwd) && chmod 771 .

# Download model weights
WORKDIR /home/frcnn/frcnn_cad
RUN chown :$GROUPNAME $(pwd)

WORKDIR /home/frcnn/frcnn_cad/weights
RUN chown :$GROUPNAME $(pwd) \
    && curl http://dkrib.web.elte.hu/cad_faster_rcnn/vgg16_frcnn_cad.caffemodel > vgg16_frcnn_cad.caffemodel

USER frcnn_cad_user

# Set WORKDIR
WORKDIR /home/frcnn/frcnn_cad

