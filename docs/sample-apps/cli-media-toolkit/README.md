# Pixeltable Media Toolkit

**Technical demonstration of Pixeltable's multimodal data processing capabilities.** Showcases native video/image/audio functions, local Whisper transcription, CLIP semantic search, and query-based architecture.

## Architecture

- **Native Functions**: Video processing (`clip`, `extract_audio`, `segment_video`), image operations (`crop`, `resize`, `rotate`), and audio processing (`get_metadata`) without external dependencies
- **AI Audio Processing**: Local Whisper transcription with output files
- **Semantic Search**: CLIP embedding indices for similarity queries across images and video frames
- **Query-Based Processing**: On-demand execution with automatic optimization and caching
- **Universal Input**: Native handling of YouTube URLs, image URLs, audio files, and local files using standard Python tempfile patterns
- **Persistent Outputs**: All generated content saved to `outputs/` directory

## Setup

```bash
pip install -r requirements.txt    # Installs Pixeltable, Whisper, Click, PyTubeFix
python init.py                     # Initialize tables and CLIP indices
```

## Quick Demo

```bash
# Add a YouTube video and audio file
python cli.py add "https://youtube.com/watch?v=dQw4w9WgXcQ" 
python cli.py add "audio.flac"

# See what you added
python cli.py list
# Output: 2 items with durations: YouTube (213s), Audio (60s)

# Process the media
python cli.py transcribe --model tiny    # Transcribe → 349 words from YouTube + 120 from audio
python cli.py extract-audio              # Extract → 39MB WAV file
python cli.py search-frames "dancing"    # Search → finds dancing scenes at timestamps

# Check results
python cli.py status                     # Shows: 220 frames indexed, 13 output files
ls outputs/                              # See all generated files
```

## Usage

```bash
# Add different types of media
python cli.py add "https://youtube.com/watch?v=dQw4w9WgXcQ"  # YouTube videos
python cli.py add "image.jpg"                               # Local images
python cli.py add "audio.flac"                              # Audio files
python cli.py add "video.mp4"                               # Local videos

# AI-powered audio transcription (local Whisper)
python cli.py transcribe --model tiny           # → outputs/*_transcript.txt + *_full.json

# Native video/audio processing
python cli.py extract-audio                     # → outputs/*_audio.wav (39MB from YouTube)
python cli.py extract-thumbnail --timestamp 60  # → outputs/*_thumb_60s.jpg (13KB thumbnail)
python cli.py generate-short --start 1 --end 3  # → outputs/*_clip_1_3s.mp4 (video clips)
python cli.py split-video --segment-length 10   # → outputs/segment_1.mp4, segment_2.mp4...

# CLIP semantic search
python cli.py search-images "dog" --save        # → outputs/image_search_*.json
python cli.py search-frames "dancing" --save    # → outputs/frame_search_*.json

# Analysis and status
python cli.py analyze                           # → outputs/analysis_report_*.txt
python cli.py list                              # Shows: 4 items with durations
python cli.py status                            # Shows: 220 frames indexed, 13 output files
```

## Commands

| Command | Function | Output |
|---------|----------|--------|
| `add "source"` | Store media in table | Media indexed with metadata |
| `list` | Display stored media | Console output with duration/type |
| `transcribe --model <name>` | Transcribe audio/video | TXT and JSON files in outputs/ |
| `extract-audio` | Extract audio tracks | WAV files in outputs/ |
| `extract-thumbnail --timestamp N` | Extract frame at timestamp | JPG files in outputs/ |
| `generate-short --start N --end M` | Create video clips | MP4 files in outputs/ |
| `split-video --segment-length N` | Split into segments | Multiple MP4 files |
| `search-frames "query" --save` | CLIP similarity search on frames | Console results + JSON files |
| `search-images "query" --save` | CLIP similarity search on images | Console results + JSON files |
| `analyze` | Extract metadata | Analysis reports in outputs/ |

## Output Files Generated

After running the commands above, you'll find these types of files in `outputs/`:

| File Type | Example | Size | Description |
|-----------|---------|------|-------------|
| **Audio Extraction** | `Rick_Astley_audio.wav` | 39MB | High-quality WAV from YouTube video |
| **Transcriptions** | `*_transcript.txt` | 1-2KB | Clean text transcripts (120-349 words) |
| **Whisper Metadata** | `*_transcript_full.json` | 8-28KB | Complete Whisper output with timestamps |
| **Video Thumbnails** | `*_thumb_60s.jpg` | 2-13KB | Frame extracts at specific timestamps |
| **Video Clips** | `*_clip_1_3s.mp4` | 6KB | Short video segments |
| **Video Segments** | `segment_1.mp4` | 6-10KB | Auto-split video parts |
| **Search Results** | `frame_search_*.json` | 1KB | CLIP similarity results with scores |
| **Analysis Reports** | `analysis_report_*.txt` | 354B | Media metadata summary |

## Native Functions Demonstrated

| Category | Functions | Implementation |
|----------|-----------|----------------|
| **Video** | `clip()`, `extract_audio()`, `segment_video()`, `get_duration()`, `extract_frame()` | Direct Pixeltable operations |
| **Audio** | `get_metadata()`, `whisper.transcribe()` | Local Whisper processing |
| **Image** | `crop()`, `resize()`, `rotate()`, `convert()` | Native image processing |
| **Search** | `similarity()` with CLIP embeddings | CLIP indices for visual similarity |
| **Data** | Table operations, views, iterators | Structured multimodal data storage |

## Technical Implementation

**Query-Based Architecture:**
```python
# Example: Transcribe audio/video on-demand
results = table.select(
    table.title,
    transcription=whisper.transcribe(
        table.audio if table.audio else table.video.extract_audio(),
        model='base'
    )
).collect()
# Then save results to output files
```

**CLIP Semantic Search:**
```python
# Similarity search across video frames
results = frames_view.order_by(
    frames_view.frame.similarity(query, idx='frame_clip_idx'),
    asc=False
).limit(5).collect()
```

**File Output Pattern:**
```python
# Generate transcripts and save to files
for item in results:
    transcript_file = f"outputs/{safe_filename(item['title'])}_transcript.txt"
    with open(transcript_file, 'w') as f:
        f.write(item['transcription']['text'])
```

**Real-World Example Results:**
```bash
# Final status after testing
python cli.py status
Status:
  Total media items: 4
  - Videos: 2 (including YouTube)
  - Images: 1  
  - Audio files: 1
  Video frames indexed: 220     # Automatic CLIP indexing
  Output files: 13            # All processing results
```

**Fast Initialization Pattern:**
- `init.py`: One-time table and index creation
- `cli.py`: Direct table access for operations  
- `tempfile.mktemp()`: Standard Python temporary file handling for YouTube downloads
- Eliminates startup overhead for repeated operations

## API Documentation

- [Video Functions](https://pixeltable.github.io/pixeltable/pixeltable/functions/video/)
- [Image Functions](https://pixeltable.github.io/pixeltable/pixeltable/functions/image/) 
- [CLIP Integration](https://pixeltable.github.io/pixeltable/pixeltable/functions/huggingface/#pixeltable.functions.huggingface.clip)
- [Pixeltable Documentation](https://docs.pixeltable.com)