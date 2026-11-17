FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV HF_HOME=/work/.cache/huggingface
# torch en versión CPU (evita descargar la versión con GPU, mucho más pesada)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && git config --global --add safe.directory /work
WORKDIR /work
