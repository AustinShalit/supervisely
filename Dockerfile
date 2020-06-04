FROM tensorflow/tensorflow:1.15.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgeos-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        unzip \
        wget \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN wget "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 > /dev/null && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models && \
  git -C /tensorflow/models checkout fe748d4a4a1576b57c279014ac0ceb47344399c4

RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

COPY . /workdir

WORKDIR /workdir/src
