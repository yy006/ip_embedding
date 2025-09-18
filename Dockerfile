FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
WORKDIR /workspace
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -U pip && pip install -r /tmp/requirements.txt