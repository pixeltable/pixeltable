# Pixeltable Media Toolkit

**Technical demonstration of Pixeltable's multimodal data processing capabilities.** Showcases native video/image functions, CLIP semantic search, and query-based architecture.

## Architecture

- **Native Functions**: Video processing (`clip`, `extract_audio`, `segment_video`) and image operations (`crop`, `resize`, `rotate`) without external dependencies
- **Semantic Search**: CLIP embedding indices for similarity queries across images and video frames
- **Query-Based Processing**: On-demand execution with automatic optimization and caching
- **Universal Input**: Native handling of YouTube URLs, image URLs, and local files
- **Persistent Outputs**: All generated content saved to `outputs/` directory

## Setup

```bash
pip install -r requirements.txt
python init.py  # Initialize tables and CLIP indices
```

## Usage

```bash
# Add media
python cli.py add "https://youtube.com/watch?v=VIDEO_ID"
python cli.py add "https://example.com/image.jpg"

# Process with native functions
python cli.py extract-audio                    # → outputs/*.wav
python cli.py generate-short --start 10 --end 30  # → outputs/*.mp4
python cli.py extract-thumbnail --timestamp 5     # → outputs/*.jpg

# Semantic search
python cli.py search-frames "person walking" --save  # → outputs/*.json
python cli.py search-images "cat" --threshold 0.2

# View results
python cli.py list
ls outputs/
```

## Commands

| Command | Function | Output |
|---------|----------|--------|
| `add "source"` | Store media in table | Media indexed with metadata |
| `list` | Display stored media | Console output with duration/type |
| `extract-audio` | Extract audio tracks | WAV files in outputs/ |
| `extract-thumbnail --timestamp N` | Extract frame at timestamp | JPG files in outputs/ |
| `generate-short --start N --end M` | Create vertical video clips | MP4 files in outputs/ |
| `split-video --segment-length N` | Split into segments | Multiple MP4 files |
| `search-frames "query"` | CLIP similarity search | Console results |
| `search-images "query" --save` | Image similarity search | JSON results in outputs/ |
| `analyze` | Extract metadata | Analysis reports in outputs/ |

## Native Functions Demonstrated

| Category | Functions | Implementation |
|----------|-----------|----------------|
| **Video** | `clip()`, `extract_audio()`, `segment_video()`, `get_duration()`, `extract_frame()` | Direct Pixeltable operations |
| **Image** | `crop()`, `resize()`, `rotate()`, `convert()` | Native image processing |
| **Search** | `similarity()` with CLIP embeddings | Automatic index creation and querying |
| **Data** | Table operations, views, iterators | Structured multimodal data storage |

## Technical Implementation

**Query-Based Architecture:**
```python
# Example: Extract audio from all videos
results = table.select(
    table.title,
    audio_path=table.video.extract_audio(format='wav')
).collect()
```

**CLIP Semantic Search:**
```python
# Similarity search across video frames
results = frames_view.order_by(
    frames_view.frame.similarity(query, idx='frame_clip_idx'),
    asc=False
).limit(5).collect()
```

**Fast Initialization Pattern:**
- `init.py`: One-time table and index creation
- `cli.py`: Direct table access for operations
- Eliminates startup overhead for repeated operations

## API Documentation

- [Video Functions](https://pixeltable.github.io/pixeltable/pixeltable/functions/video/)
- [Image Functions](https://pixeltable.github.io/pixeltable/pixeltable/functions/image/) 
- [CLIP Integration](https://pixeltable.github.io/pixeltable/pixeltable/functions/huggingface/#pixeltable.functions.huggingface.clip)
- [Pixeltable Documentation](https://docs.pixeltable.com)