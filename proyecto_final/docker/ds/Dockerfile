# Imagen de ciencia de datos: JupyterLab (notebooks) y trabajos de Kestra
# (simulación, monitoreo, reentrenamiento). Contexto: la carpeta proyecto_final/
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/work HF_HOME=/opt/hf EMB_CACHE=/tmp/emb_cache/embeddings.pkl PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/req/requirements.txt
COPY docker/ds/requirements.txt /tmp/req/docker/ds/requirements.txt
RUN pip install --no-cache-dir -r /tmp/req/docker/ds/requirements.txt \
    && git config --global --add safe.directory /work
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu')" \
    && chmod -R a+rwX /opt/hf
# Código del proyecto dentro de la imagen (Kestra la usa sin montar carpetas).
# En el servicio jupyter, la carpeta del proyecto se monta encima en /work.
WORKDIR /work
COPY src /work/src
COPY monitoreo /work/monitoreo
COPY api /work/api
COPY scripts /work/scripts
