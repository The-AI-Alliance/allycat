FROM python:3.11-slim

WORKDIR /allycat

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_RUN_ENV=local_ollama \
    VECTOR_DB_TYPE=local

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl git netcat-traditional \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .
RUN chmod +x ./docker-startup.sh
RUN rm -rf .env workspace/* __pycache__ *.pyc

EXPOSE 8080 11434

ENTRYPOINT ["./docker-startup.sh"]
CMD ["deploy"]
