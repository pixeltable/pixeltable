# Pixeltable Backend API

FastAPI-based backend service for Pixeltable providing multimodal processing capabilities.

## API Routes

### Core Endpoints
- `GET /` - Service info
- `GET /health` - Health check

### File Management
- `POST /api/upload` - Upload document files
- `POST /api/videos/upload` - Upload video files
- `POST /api/audio/upload` - Upload audio files
- `GET /api/files` - List all uploaded files

### Chat Interface
- `POST /api/chat` - Send message and get AI response

## Quick Start

### Environment Setup
1. Create a `.env` file in the `api` folder:
2. Add the following environment variables:
    - `OPENAI_API_KEY`: Your OpenAI API key

### Using Python
```bash
uv venv .venv # Or python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
fastapi dev api/main.py
```

### Using Docker
```bash
docker build -t multimodal-api .
docker run -p 8000:8000 multimodal-api
```

## API Documentation
Once running, visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

## Development
- Python 3.10+
- Dependencies managed with `uv`
- Configuration via environment variables (see `.env.example`)
