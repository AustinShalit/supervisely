FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN curl -s -OL "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip" > /dev/null && \
    unzip protoc-3.0.0-linux-x86_64.zip -d proto3 > /dev/null && \
    mv proto3/bin/* /usr/local/bin && \
    mv proto3/include/* /usr/local/include && \
    rm -rf proto3 protoc-3.0.0-linux-x86_64.zip

RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

RUN cd /tensorflow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

COPY requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

COPY /wpilib-supervisely-nn-tf-obj-det /tensorflow/models/research/wpilib-supervisely-nn-tf-obj-det/

ENV PYTHONPATH $PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim
WORKDIR /tensorflow/models/research
