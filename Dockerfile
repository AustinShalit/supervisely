FROM tensorflow/tensorflow:1.12.3-py3

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

RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6.5 \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN wget "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 > /dev/null && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models && \
  git -C /tensorflow/models checkout fe748d4a4a1576b57c279014ac0ceb47344399c4

RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim

ENV LANG C.UTF-8

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

COPY . /workdir

WORKDIR /workdir/src
