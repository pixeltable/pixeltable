FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

RUN python -m spacy download en_core_web_sm

COPY config.py schema.py load_data.py initialize.py server.py .

RUN mkdir -p /app/doc_index

EXPOSE 8083

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8083"]
