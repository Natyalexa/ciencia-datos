# Imagen del servicio de predicción. Contexto de construcción: la carpeta proyecto_final/
#   docker build -f api/Dockerfile -t resenas-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app HF_HOME=/opt/hf EMB_CACHE="" \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODELO_URI=models:/resenas-bancos-ec-modelo@champion

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
# torch en versión CPU (la versión con GPU pesa varios GB y no se usa)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt
# El modelo de embeddings (pesos públicos, ~470 MB) queda dentro de la imagen: el servicio
# no depende de internet para arrancar. El modelo de clasificación NO va en la imagen:
# se descarga del Model Registry por alias al iniciar y cuando el alias cambia.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu')" \
    && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY src /app/src
COPY api /app/api

RUN useradd --create-home servicio && mkdir -p /tmp/registro && chown servicio /tmp/registro
USER servicio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD curl -fs http://localhost:8000/salud || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
