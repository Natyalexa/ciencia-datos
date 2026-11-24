FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "dbt-snowflake>=1.9,<2.0"
WORKDIR /usr/app
# El proyecto dbt va dentro de la imagen para que Kestra pueda ejecutarlo sin montar carpetas.
# Con docker compose run, la carpeta ./dbt se monta encima (cambios sin reconstruir).
COPY . /usr/app
ENV DBT_PROFILES_DIR=/usr/app
# Sin secretos: la llave privada se monta en /secrets o la escribe Kestra en SNOWFLAKE_KEY_PATH.
