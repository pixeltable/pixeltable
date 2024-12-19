# Pixel Table Backend API

FastAPI-based backend service for Pixel Table providing multimodal processing capabilities.

## API Routes

### Core Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `GET /api/files` - List all uploaded files

### File Management
- `POST /api/upload` - Upload document/video files
- `POST /api/videos/upload` - Upload video files
- `POST /api/audio/upload` - Upload audio files
- `GET /api/videos` - List uploaded videos

### Chat Interface
- `POST /api/chat` - Send message and get AI response

## Quick Start

### Using Python
```bash
uv venv .venv
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
