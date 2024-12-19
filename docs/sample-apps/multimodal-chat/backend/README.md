# Run Multimodal API locally

```bash
uv venv .venv
source .venv/bin/activate
uv sync
uv run main.py
```

or using Docker

```bash
docker build -t multimodal-api .
docker run -p 8000:8000 multimodal-api
```
