# 📋 Pixeltable Comprehensive Cheat Sheet

> **Pixeltable** is the open-source Python library providing declarative data infrastructure for multimodal AI applications—enabling incremental storage, transformation, indexing, retrieval, and orchestration of data.

**Version:** 0.5.x  
**Documentation:** [docs.pixeltable.com](https://docs.pixeltable.com)  
**GitHub:** [github.com/pixeltable/pixeltable](https://github.com/pixeltable/pixeltable)  
**API Reference:** [SDK Documentation](https://docs.pixeltable.com/sdk/latest)

---

## Table of Contents

1. [Installation](#installation)
2. [Core Operations](#core-operations)
3. [Type System](#type-system)
4. [Queries & Data Operations](#queries--data-operations)
5. [Computed Columns](#computed-columns)
6. [Custom Functions (UDFs)](#custom-functions-udfs)
7. [Embedding Indexes & Vector Search](#embedding-indexes--vector-search)
8. [Iterators](#iterators)
9. [Views & Snapshots](#views--snapshots)
10. [Version Control](#version-control)
11. [Data Sharing](#data-sharing)
12. [AI Integrations](#ai-integrations)
13. [Built-in Functions](#built-in-functions)
14. [Data Import/Export](#data-importexport)
15. [Configuration](#configuration)
16. [Quick Reference](#quick-reference)

---

## Installation

```bash
# Basic installation
pip install pixeltable

# With specific integrations
pip install pixeltable[openai]
pip install pixeltable[anthropic]
pip install pixeltable[huggingface]

# All integrations
pip install pixeltable[all]
```

```python
import pixeltable as pxt
```

📚 [Installation Guide](https://docs.pixeltable.com/overview/quick-start)

---

## Core Operations

### Directories

[📖 API Reference](https://docs.pixeltable.com/sdk/latest/pixeltable)

```python
# Create
pxt.create_dir('project')
pxt.create_dir('project.subdir')

# List
pxt.ls()                           # DataFrame of contents
pxt.list_dirs()                    # List of directory names
pxt.list_tables()                  # List of table names
contents = pxt.get_dir_contents('project')

# Move/rename
pxt.move('old.path', 'new.path')

# Delete
pxt.drop_dir('project', force=True)
```

### Tables

[📖 Tables Guide](https://docs.pixeltable.com/tutorials/tables-and-data-operations) | [📖 API Reference](https://docs.pixeltable.com/sdk/latest/table)

```python
# Create
t = pxt.create_table('project.users', {
    'id': pxt.Required[pxt.String],
    'name': pxt.String,
    'age': pxt.Int,
    'created_at': pxt.Timestamp,
}, primary_key='id')

# Get
t = pxt.get_table('project.users')

# Operations
t.describe()                       # Show schema
t.count()                          # Row count
t.columns()                        # Column names
metadata = t.get_metadata()        # Table metadata
t.list_views()                     # Views based on this table

# Modify schema
t.add_column(email=pxt.String)
t.rename_column('old', 'new')
t.drop_column('column')

# Delete
pxt.drop_table('project.users')
```

---

## Type System

[📖 Type System Guide](https://docs.pixeltable.com/platform/type-system)

### Core Types

| Python | Pixeltable | Usage |
|--------|------------|-------|
| `str` | `pxt.String` | Text |
| `int` | `pxt.Int` | 64-bit integer |
| `float` | `pxt.Float` | 64-bit float |
| `bool` | `pxt.Bool` | Boolean |
| `datetime` | `pxt.Timestamp` | Date/time with timezone |
| `date` | `pxt.Date` | Date only |
| `dict` | `pxt.Json` | JSON data |
| `bytes` | `pxt.Binary` | Binary data |
| `UUID` | `pxt.UUID` | UUID |

### Media & Arrays

```python
pxt.Image                          # Images
pxt.Video                          # Videos
pxt.Audio                          # Audio
pxt.Document                       # PDFs, Word docs
pxt.Array[(768,), pxt.Float]       # Fixed-dim vector
pxt.Array[(None,), pxt.Float]      # Variable-dim vector
pxt.Required[pxt.String]           # Non-nullable
```

---

## Queries & Data Operations

[📖 Queries Guide](https://docs.pixeltable.com/tutorials/queries-and-expressions) | [📖 API Reference](https://docs.pixeltable.com/sdk/latest/query)

### Insert, Update, Delete

```python
# Insert
t.insert([{'name': 'Alice', 'age': 30}])
t.insert([User(name='Alice', age=30)])  # Pydantic

# Update
t.where(t.name == 'Alice').update({'age': 31})
t.batch_update([{'_rowid': id1, 'score': 100}])

# Delete
t.where(t.age < 18).delete()

# Recompute
t.recompute_columns(['embedding'])
```

### Queries

```python
# Basic
t.collect()                        # All rows
t.head(10)                         # First 10
t.select(t.name, t.age).collect()  # Specific columns

# Filter
t.where(t.age > 25).collect()
t.where((t.age > 25) & (t.active == True)).collect()

# Order & limit
t.order_by(t.created_at, asc=False).limit(100).collect()

# Aggregate
t.group_by(t.category).select(
    t.category,
    total=count(t.id),
    avg=avg(t.score)
).collect()

# Join
orders.join(users, on=(orders.user_id == users.id)).collect()

# Sample
t.sample(fraction=0.1).collect()
t.sample(n=100, seed=42).collect()
```

📚 [Join Tables Cookbook](https://docs.pixeltable.com/howto/cookbooks/core/query-join-tables)

---

## Computed Columns

[📖 Computed Columns Guide](https://docs.pixeltable.com/tutorials/computed-columns)

```python
# Basic
t.add_computed_column(total=t.price * t.quantity)

# With LLM
from pixeltable.functions.openai import chat_completions

t.add_computed_column(
    summary=chat_completions(
        messages=[{'role': 'user', 'content': t.text}],
        model='gpt-4o-mini'
    ).choices[0].message.content
)

# With UDF
@pxt.udf
def word_count(text: str) -> int:
    return len(text.split())

t.add_computed_column(words=word_count(t.content))

# Options
t.add_computed_column(col=expr, stored=False)  # Compute on demand
t.add_computed_column(col=expr, destination='s3://bucket/path/')  # S3 storage
```

📚 [Iterative Workflow](https://docs.pixeltable.com/howto/cookbooks/core/dev-iterative-workflow)

---

## Custom Functions (UDFs)

[📖 UDFs Guide](https://docs.pixeltable.com/platform/udfs-in-pixeltable)

### Basic UDF

```python
@pxt.udf
def multiply(a: int, b: int) -> int:
    return a * b

t.select(result=multiply(t.x, t.y)).collect()
```

### Batched UDF

```python
from pixeltable.func import Batch

@pxt.udf(batch_size=32)
def embed(texts: Batch[str]) -> Batch[list[float]]:
    return model.encode(texts).tolist()
```

### Query Functions (@pxt.query)

[📚 RAG Pipeline](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline)

```python
@pxt.query
def top_k_search(query: str, k: int = 5):
    """Reusable query function."""
    sim = docs.content.similarity(string=query)
    return docs.order_by(sim, asc=False).select(docs.content, score=sim).limit(k)

# Use in computed column
t.add_computed_column(context=top_k_search(t.question))
```

### Retrieval UDF

```python
# Convert table to callable function
lookup = pxt.retrieval_udf(
    kb_table,
    name='search_kb',
    description='Search knowledge base',
    parameters=['topic'],
    limit=5
)

results = lookup(topic='python')
tools = pxt.tools(lookup)  # Use as LLM tool
```

### User-Defined Aggregates (UDA)

[📚 Custom Aggregates Cookbook](https://docs.pixeltable.com/howto/cookbooks/core/custom-aggregates-uda)

```python
@pxt.uda(value_type=pxt.Float, update_type=pxt.Json)
class RunningAvg:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
    def update(self, val: float):
        self.sum += val
        self.count += 1
    def value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
```

### MCP UDFs

```python
# Connect to MCP server
udfs = pxt.mcp_udfs('http://localhost:8000/mcp')
t.add_computed_column(result=udfs[0](t.data))
```

---

## Embedding Indexes & Vector Search

[📖 Embedding Indexes Guide](https://docs.pixeltable.com/platform/embedding-indexes) | [📖 Vector Database](https://docs.pixeltable.com/platform/vector-database)

```python
from pixeltable.functions.openai import embeddings

# Create index
t.add_embedding_index(
    'idx',
    column=t.content,
    embedding=embeddings(input=t.content, model='text-embedding-3-small'),
    metric='cosine'  # 'cosine', 'ip', 'l2'
)

# Search with .similarity()
sim = t.content.similarity(string='query')
results = t.order_by(sim, asc=False).select(t.content, score=sim).limit(10)

# Search with named index
results = t.idx.similarity_search('query', k=10)

# Get raw embeddings
t.select(t.content, emb=t.content.embedding()).collect()

# Drop
t.drop_embedding_index('idx')
```

📚 [Semantic Search Cookbook](https://docs.pixeltable.com/howto/cookbooks/search/search-semantic-text) | [Image Search](https://docs.pixeltable.com/howto/cookbooks/search/search-similar-images) | [Text Embeddings](https://docs.pixeltable.com/howto/cookbooks/search/embed-text-openai)

---

## Iterators

[📖 Iterators Guide](https://docs.pixeltable.com/platform/iterators) | [📚 Split Rows Cookbook](https://docs.pixeltable.com/howto/cookbooks/core/data-split-rows)

```python
from pixeltable.iterators import (
    DocumentSplitter,
    FrameIterator,
    VideoSplitter,
    AudioSplitter,
    StringSplitter,
    TileIterator
)

# Extract video frames
frames = pxt.create_view(
    'project.frames',
    videos,
    iterator=FrameIterator.create(video=videos.video, fps=1)
)

# Chunk documents for RAG
chunks = pxt.create_view(
    'project.chunks',
    docs,
    iterator=DocumentSplitter.create(
        document=docs.document,
        separators='paragraph',
        limit=500,
        overlap=50
    )
)
```

📚 [Document Chunking](https://docs.pixeltable.com/howto/cookbooks/text/doc-chunk-for-rag) | [Video Frames](https://docs.pixeltable.com/howto/cookbooks/video/video-extract-frames)

---

## Views & Snapshots

[📖 Views Guide](https://docs.pixeltable.com/platform/views)

```python
# Filtered view
active = pxt.create_view(
    'project.active_users',
    users,
    filter=users.status == 'active'
)

# View with iterator
frames = pxt.create_view(
    'project.frames',
    videos,
    iterator=FrameIterator.create(video=videos.video, fps=2)
)

# Snapshot (read-only)
snapshot = pxt.create_snapshot('project.backup', users)
```

---

## Version Control

[📖 Version Control Guide](https://docs.pixeltable.com/platform/version-control) | [📚 Cookbook](https://docs.pixeltable.com/howto/cookbooks/core/version-control-history)

```python
t.history()                        # Version history
t.history(n=10)                    # Last 10 versions
versions = t.get_versions()
t.revert()                         # Undo last change (cannot be undone!)
```

---

## Data Sharing

[📖 Data Sharing Guide](https://docs.pixeltable.com/platform/data-sharing)

```python
# Publish to cloud
pxt.publish(
    source='my.table',
    destination_uri='pxt://username/dataset',
    access='public'  # or 'private' (default)
)
t.push()  # Sync local → cloud

# Replicate (clone)
local = pxt.replicate(
    remote_uri='pxt://user/dataset',
    local_path='my-copy'
)
local.pull()  # Sync cloud → local
```

---

## AI Integrations

### OpenAI

[📖 Provider Guide](https://docs.pixeltable.com/howto/providers/working-with-openai) | [📖 API Reference](https://docs.pixeltable.com/sdk/latest/openai)

```python
from pixeltable.functions.openai import (
    chat_completions,
    embeddings,
    image_generations,
    speech,
    transcriptions,
    translations,
    moderations,
    vision,
    invoke_tools
)

# Chat
t.add_computed_column(
    response=chat_completions(
        messages=[{'role': 'user', 'content': t.prompt}],
        model='gpt-4o'
    ).choices[0].message.content
)

# Embeddings
t.add_computed_column(
    emb=embeddings(input=t.text, model='text-embedding-3-small').data[0].embedding
)

# Image generation
t.add_computed_column(
    img=image_generations(prompt=t.prompt, model='dall-e-3').data[0].image
)

# Text-to-speech
t.add_computed_column(
    audio=speech(input=t.text, model='tts-1', voice='alloy')
)

# Transcription
t.add_computed_column(
    text=transcriptions(audio=t.audio, model='whisper-1').text
)

# Translation
t.add_computed_column(
    english=translations(audio=t.audio, model='whisper-1').text
)

# Moderation
t.add_computed_column(
    flags=moderations(input=t.content, model='omni-moderation-latest')
)

# Vision
t.add_computed_column(
    desc=vision(
        prompt='Describe',
        image=t.image,
        model='gpt-4o'
    ).choices[0].message.content
)

# Tool calling
t.add_computed_column(
    tool_results=invoke_tools(tools, t.llm_response)
)
```

### Anthropic

[📖 Provider Guide](https://docs.pixeltable.com/howto/providers/working-with-anthropic) | [📖 API Reference](https://docs.pixeltable.com/sdk/latest/anthropic)

```python
from pixeltable.functions.anthropic import messages, invoke_tools

t.add_computed_column(
    response=messages(
        messages=[{'role': 'user', 'content': t.prompt}],
        model='claude-sonnet-4-20250514'
    ).content[0].text
)
```

### Google Gemini

[📖 Provider Guide](https://docs.pixeltable.com/howto/providers/working-with-gemini) | [📖 API Reference](https://docs.pixeltable.com/sdk/latest/gemini)

```python
from pixeltable.functions.gemini import (
    generate_content,
    generate_images,
    generate_videos,
    invoke_tools
)

# Text generation
t.add_computed_column(
    text=generate_content(contents=t.prompt, model='gemini-2.0-flash').text
)

# Image generation
t.add_computed_column(
    img=generate_images(prompt=t.prompt, model='imagen-3.0-generate-001')
)

# Video generation
t.add_computed_column(
    video=generate_videos(prompt=t.prompt, model='veo-001')
)
```

### AWS Bedrock

[📖 Provider Guide](https://docs.pixeltable.com/howto/providers/working-with-bedrock) | [📖 API Reference](https://docs.pixeltable.com/sdk/latest/bedrock)

```python
from pixeltable.functions.bedrock import converse, invoke_tools

t.add_computed_column(
    response=converse(
        messages=[{'role': 'user', 'content': [{'text': t.prompt}]}],
        model_id='anthropic.claude-3-5-sonnet-20241022-v2:0'
    ).output.message.content[0].text
)
```

### Other LLM Providers

| Provider | Module | Key Functions | Guide |
|----------|--------|---------------|-------|
| Together AI | `together` | `chat_completions`, `completions`, `embeddings`, `image_generations` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-together) |
| Groq | `groq` | `chat_completions`, `invoke_tools` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-groq) |
| Fireworks | `fireworks` | `chat_completions` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-fireworks) |
| Mistral AI | `mistralai` | `chat_completions`, `fim_completions`, `embeddings` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-mistralai) |
| DeepSeek | `deepseek` | `chat_completions` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-deepseek) |
| OpenRouter | `openrouter` | `chat_completions` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-openrouter) |
| Ollama | `ollama` | `chat`, `generate`, `embed` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-ollama) |
| Llama.cpp | `llama_cpp` | `create_chat_completion` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-llama-cpp) |
| Replicate | `replicate` | `run` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-replicate) |
| fal | `fal` | `run` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-fal) |
| Voyage AI | `voyageai` | `embeddings`, `rerank`, `multimodal_embed` | [Guide](https://docs.pixeltable.com/howto/providers/working-with-voyageai) |

### Hugging Face

[📖 Provider Guide](https://docs.pixeltable.com/howto/providers/working-with-hugging-face) | [📖 API Reference](https://docs.pixeltable.com/sdk/latest/huggingface)

```python
from pixeltable.functions.huggingface import (
    # Embeddings
    sentence_transformer,  # Text embeddings
    clip,                  # Image/text embeddings
    cross_encoder,         # Similarity scoring
    
    # Vision
    detr_for_object_detection,
    vit_for_image_classification,
    image_captioning,
    
    # Generation
    text_to_image,
    image_to_image,
    image_to_video,
    text_to_speech,
    text_generation,
    
    # NLP
    summarization,
    translation,
    text_classification,
    token_classification,  # NER
    question_answering,
    
    # Speech
    automatic_speech_recognition,
    speech2text_for_conditional_generation,
)

# Examples
t.add_computed_column(emb=sentence_transformer(t.text, model_id='all-MiniLM-L6-v2'))
t.add_computed_column(obj=detr_for_object_detection(t.img, model_id='facebook/detr-resnet-50'))
t.add_computed_column(img=text_to_image(t.prompt, model_id='stable-diffusion-v1-5/stable-diffusion-v1-5'))
```

📚 [Image-to-Image](https://docs.pixeltable.com/howto/cookbooks/images/img-image-to-image)

### Whisper & Speech

[📖 Whisper Guide](https://docs.pixeltable.com/sdk/latest/whisper) | [📖 WhisperX Guide](https://docs.pixeltable.com/sdk/latest/whisperx)

```python
from pixeltable.functions.whisper import transcribe
from pixeltable.functions.whisperx import transcribe_audio

t.add_computed_column(transcript=transcribe(t.audio, model='base'))
```

📚 [Audio Transcription](https://docs.pixeltable.com/howto/cookbooks/audio/audio-transcribe) | [TTS](https://docs.pixeltable.com/howto/cookbooks/audio/audio-text-to-speech)

### Vision Models

[📖 Vision API](https://docs.pixeltable.com/sdk/latest/vision) | [📖 YOLOX API](https://docs.pixeltable.com/sdk/latest/yolox)

```python
from pixeltable.functions.yolox import yolox, yolo_to_coco
from pixeltable.functions.vision import draw_bounding_boxes, eval_detections

t.add_computed_column(boxes=yolox(t.image, model_id='yolox_m'))
t.add_computed_column(coco=yolo_to_coco(t.image, t.boxes))
t.add_computed_column(annotated=draw_bounding_boxes(t.image, t.boxes['boxes'], t.boxes['labels']))
```

📚 [Object Detection](https://docs.pixeltable.com/howto/cookbooks/images/img-detect-objects) | [Visualize Detections](https://docs.pixeltable.com/howto/cookbooks/images/img-visualize-detections)

### Reve (Audio/Video Editing)

[📖 API Reference](https://docs.pixeltable.com/sdk/latest/reve)

```python
from pixeltable.functions.reve import create, edit, remix

t.add_computed_column(audio=create(script=t.script))
t.add_computed_column(edited=edit(audio=t.audio, edits=t.edits))
```

### TwelveLabs (Video Understanding)

[📖 API Reference](https://docs.pixeltable.com/sdk/latest/twelvelabs)

```python
from pixeltable.functions.twelvelabs import embed

t.add_computed_column(video_emb=embed(video=t.video))
```

---

## Built-in Functions

### String

[📖 API Reference](https://docs.pixeltable.com/sdk/latest/string)

```python
# Use as methods (no import needed)
t.name.lower()
t.name.upper()
t.name.strip()
t.name.replace('old', 'new')
t.name.split(' ')
t.name.contains('text')
t.name.startswith('A')
t.name.endswith('z')
t.name.len()

# Full list: capitalize, casefold, center, contains_re, count, fill, find, findall, 
# format, fullmatch, index, is*, join, ljust, lstrip, match, normalize, pad, 
# partition, removeprefix, removesuffix, repeat, replace_re, reverse, rfind, 
# rindex, rjust, rpartition, rstrip, slice, slice_replace, swapcase, title, 
# translate, upper, wrap, zfill
```

### Image

[📖 API Reference](https://docs.pixeltable.com/sdk/latest/image)

```python
# Properties
t.image.width
t.image.height
t.image.mode

# Transformations
t.image.resize((256, 256))
t.image.rotate(90)
t.image.crop((x1, y1, x2, y2))
t.image.convert('L')
t.image.transpose(method)

# Operations
t.image.blend(other, alpha=0.5)
t.image.composite(overlay, mask)

# Analysis
t.image.histogram()
t.image.entropy()
t.image.get_metadata()

# Full list: alpha_composite, b64_encode, getbands, getbbox, getcolors, 
# getchannel, getextrema, getpalette, getpixel, getprojection, point, 
# quantize, reduce, thumbnail, effect_spread
```

📚 [Apply Filters](https://docs.pixeltable.com/howto/cookbooks/images/img-apply-filters) | [PIL Transforms](https://docs.pixeltable.com/howto/cookbooks/images/img-pil-transforms) | [Brightness/Contrast](https://docs.pixeltable.com/howto/cookbooks/images/img-brightness-contrast)

### Video

[📖 API Reference](https://docs.pixeltable.com/sdk/latest/video)

```python
# Metadata
t.video.get_metadata()
t.video.get_duration()

# Extract
t.video.extract_frame(timestamp=5.0)
t.video.extract_audio()

# Edit
t.video.clip(start=10.0, end=30.0)
t.video.concat_videos([t.vid1, t.vid2])
t.video.segment_video(timestamps=[10, 20, 30])
t.video.overlay_text(text='Title', position='center')
t.video.with_audio(audio=t.audio_file)

# Scene detection
t.video.scene_detect_content()
t.video.scene_detect_adaptive()
t.video.scene_detect_threshold(threshold=30.0)
t.video.scene_detect_histogram()
t.video.scene_detect_hash()
```

📚 [Extract Frames](https://docs.pixeltable.com/howto/cookbooks/video/video-extract-frames) | [Scene Detection](https://docs.pixeltable.com/howto/cookbooks/video/video-scene-detection) | [Video Overlay](https://docs.pixeltable.com/howto/cookbooks/video/video-add-text-overlay) | [Generate Video AI](https://docs.pixeltable.com/howto/cookbooks/video/video-generate-ai)

### Audio

[📖 API Reference](https://docs.pixeltable.com/sdk/latest/audio)

```python
from pixeltable.functions.audio import get_metadata, encode_audio

t.audio.get_metadata()
t.add_computed_column(aac=encode_audio(t.audio, codec='aac'))
```

📚 [Extract Audio](https://docs.pixeltable.com/howto/cookbooks/audio/audio-extract-from-video) | [Summarize Podcast](https://docs.pixeltable.com/howto/cookbooks/audio/audio-summarize-podcast)

### Document

```python
# Document chunking via iterator (see Iterators section)
from pixeltable.iterators import DocumentSplitter
```

📚 [Extract Text](https://docs.pixeltable.com/howto/cookbooks/text/doc-extract-text-from-office-files)

### Timestamp & Date

[📖 Timestamp API](https://docs.pixeltable.com/sdk/latest/timestamp) | [📖 Date API](https://docs.pixeltable.com/sdk/latest/date)

```python
from pixeltable.functions import timestamp, date

# Timestamp properties
t.created_at.year
t.created_at.month
t.created_at.day
t.created_at.hour
t.created_at.minute
t.created_at.second

# Methods
t.created_at.weekday()
t.created_at.strftime('%Y-%m-%d')
t.created_at.astimezone('US/Pacific')

# Date operations
date.make_date(year=2024, month=1, day=1)
date.add_days(t.date_col, days=7)
```

📚 [Time Zones](https://docs.pixeltable.com/howto/cookbooks/core/time-zones)

### Math & JSON

[📖 Math API](https://docs.pixeltable.com/sdk/latest/math) | [📖 JSON API](https://docs.pixeltable.com/sdk/latest/json)

```python
from pixeltable.functions import math, json

# Math
math.abs(t.val)
math.ceil(t.val)
math.floor(t.val)
math.round(t.val, 2)
math.sqrt(t.val)
math.pow(t.base, t.exp)

# JSON
t.metadata['key']
t.metadata['nested']['field']
json.make_list([t.a, t.b, t.c])
```

### UUID & Net

[📖 UUID API](https://docs.pixeltable.com/sdk/latest/pixeltable#pixeltablefunctionsuuiduuid4)

```python
from pixeltable.functions.uuid import uuid4
from pixeltable.functions.net import presigned_url

t.add_computed_column(id=uuid4())
t.add_computed_column(url=presigned_url(t.s3_path, expiration=3600))
```

📚 [UUID Identity](https://docs.pixeltable.com/howto/cookbooks/core/workflow-uuid-identity)

---

## Data Import/Export

[📖 I/O API Reference](https://docs.pixeltable.com/sdk/latest/io)

### Import

```python
from pixeltable import io

# CSV, JSON, Parquet, Excel
t.insert(io.import_csv('data.csv'))
t.insert(io.import_json('data.json'))
t.insert(io.import_parquet('data.parquet'))
t.insert(io.import_excel('data.xlsx', sheet_name='Sheet1'))

# Pandas
import pandas as pd
t.insert(io.import_pandas(pd.read_csv('data.csv')))

# Hugging Face datasets
from datasets import load_dataset
ds = load_dataset('squad', split='train[:100]')
t.insert(io.import_huggingface_dataset(ds))
```

📚 [Import CSV](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) | [JSON](https://docs.pixeltable.com/howto/cookbooks/data/data-import-json) | [Parquet](https://docs.pixeltable.com/howto/cookbooks/data/data-import-parquet) | [Excel](https://docs.pixeltable.com/howto/cookbooks/data/data-import-excel) | [S3](https://docs.pixeltable.com/howto/cookbooks/data/data-import-s3) | [HuggingFace](https://docs.pixeltable.com/howto/cookbooks/data/data-import-huggingface)

### Export

```python
# Pandas
df = t.collect().to_pandas()

# Parquet
io.export_parquet(t, 'output.parquet')

# PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(t.to_pytorch_dataset(), batch_size=32)

# COCO format
coco_path = t.to_coco_dataset()

# LanceDB
io.export_lancedb(t, 'lancedb_uri', 'table_name')

# Label Studio
io.create_label_studio_project(t, media_column=t.image)

# FiftyOne
io.export_images_as_fo_dataset(t, img_column=t.image)
```

📚 [Export PyTorch](https://docs.pixeltable.com/howto/cookbooks/data/data-export-pytorch) | [Sampling](https://docs.pixeltable.com/howto/cookbooks/data/data-sampling)

### External Storage

[📖 Cloud Storage Guide](https://docs.pixeltable.com/integrations/cloud-storage)

```python
# Configure in config.toml or env vars
# PIXELTABLE_OUTPUT_MEDIA_DEST="s3://bucket/path/"

# Per-column destination
t.add_computed_column(
    thumbnail=t.image.resize((128, 128)),
    destination='s3://bucket/thumbnails/'
)
```

---

## Configuration

[📖 Configuration Guide](https://docs.pixeltable.com/platform/configuration)

```python
# API Keys
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'

# Config file: ~/.pixeltable/config.toml
[openai]
api_key = "sk-..."

# Logging
pxt.configure_logging(level='INFO')

# Custom database
pxt.init({'home': '/path/to/data'})
```

📚 [API Keys Workflow](https://docs.pixeltable.com/howto/cookbooks/core/workflow-api-keys)

---

## Quick Reference

### Essential Commands

| Task | Command | Docs |
|------|---------|------|
| Create table | `pxt.create_table('dir.table', schema)` | [Guide](https://docs.pixeltable.com/tutorials/tables-and-data-operations) |
| Get table | `pxt.get_table('dir.table')` | [API](https://docs.pixeltable.com/sdk/latest/pixeltable#pixeltableget_table) |
| Query | `t.where(condition).collect()` | [Guide](https://docs.pixeltable.com/tutorials/queries-and-expressions) |
| Add computed | `t.add_computed_column(name=expr)` | [Guide](https://docs.pixeltable.com/tutorials/computed-columns) |
| Embed index | `t.add_embedding_index('idx', column, embedding)` | [Guide](https://docs.pixeltable.com/platform/embedding-indexes) |
| Create view | `pxt.create_view('name', base, iterator=...)` | [Guide](https://docs.pixeltable.com/platform/views) |

### Decorators

| Decorator | Purpose | Docs |
|-----------|---------|------|
| `@pxt.udf` | Basic UDF | [Guide](https://docs.pixeltable.com/platform/udfs-in-pixeltable) |
| `@pxt.expr_udf` | SQL-optimized | [Guide](https://docs.pixeltable.com/platform/udfs-in-pixeltable) |
| `@pxt.uda` | Custom aggregate | [Cookbook](https://docs.pixeltable.com/howto/cookbooks/core/custom-aggregates-uda) |
| `@pxt.query` | Reusable query | [Cookbook](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) |

### Tool Calling

[📚 Tool Calling Cookbook](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling)

```python
# Create tools
tools = pxt.tools(my_udf1, my_udf2)

# Use with LLM
response = chat_completions(
    messages=[{'role': 'user', 'content': query}],
    model='gpt-4o',
    tools=tools,
    tool_choice=tools.choice(required=True)
)

# Execute tools
results = invoke_tools(tools, response)
```

**Providers with invoke_tools:**
OpenAI | Anthropic | Gemini | Bedrock | Groq

### AI Provider Summary

[📖 All Integrations](https://docs.pixeltable.com/integrations/models)

| Provider | Type | Module |
|----------|------|--------|
| OpenAI | Cloud | `openai` |
| Anthropic | Cloud | `anthropic` |
| Google Gemini | Cloud | `gemini` |
| AWS Bedrock | Cloud | `bedrock` |
| Together AI | Cloud | `together` |
| Groq | Cloud | `groq` |
| Fireworks | Cloud | `fireworks` |
| Mistral AI | Cloud | `mistralai` |
| DeepSeek | Cloud | `deepseek` |
| OpenRouter | Cloud | `openrouter` |
| Replicate | Cloud | `replicate` |
| fal | Cloud | `fal` |
| Voyage AI | Cloud | `voyageai` |
| Ollama | Local | `ollama` |
| Llama.cpp | Local | `llama_cpp` |
| Whisper | Local | `whisper` |
| WhisperX | Local | `whisperx` |
| Hugging Face | Local/Cloud | `huggingface` |
| YOLOX | Local | `yolox` |
| Reve | Cloud | `reve` |
| TwelveLabs | Cloud | `twelvelabs` |

---

## Common Patterns & Examples

### RAG Pipeline

[📚 RAG Pipeline](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) | [📚 RAG Operations](https://docs.pixeltable.com/howto/use-cases/rag-operations) | [📚 RAG Demo](https://docs.pixeltable.com/howto/use-cases/rag-demo)

```python
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.openai import embeddings, chat_completions

# 1. Chunk documents
chunks = pxt.create_view(
    'rag.chunks',
    docs,
    iterator=DocumentSplitter.create(document=docs.doc, separators='paragraph', limit=500)
)

# 2. Index
chunks.add_embedding_index(
    'idx',
    column=chunks.text,
    embedding=embeddings(input=chunks.text, model='text-embedding-3-small')
)

# 3. Query function
@pxt.query
def retrieve(query: str, k: int = 5):
    sim = chunks.text.similarity(string=query)
    return chunks.order_by(sim, asc=False).select(chunks.text, score=sim).limit(k)

# 4. Generate answers
qa = pxt.create_table('rag.qa', {'question': pxt.String})
qa.add_computed_column(context=retrieve(qa.question))
qa.add_computed_column(
    answer=chat_completions(
        messages=[
            {'role': 'system', 'content': 'Answer using context: ' + qa.context},
            {'role': 'user', 'content': qa.question}
        ],
        model='gpt-4o-mini'
    ).choices[0].message.content
)
```

### Agent with Memory

[📚 Agent Memory](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-agent-memory)

```python
# Memory store
memories = pxt.create_table('agent.memories', {'content': pxt.String})
memories.add_embedding_index('idx', column=memories.content, embedding=...)

# Retrieval
@pxt.query
def recall(context: str, k: int = 3):
    sim = memories.content.similarity(string=context)
    return memories.order_by(sim, asc=False).limit(k).select(memories.content)

# Use in conversation
conversations.add_computed_column(recalled=recall(conversations.message))
```

### Video Object Detection

[📚 Object Detection in Videos](https://docs.pixeltable.com/howto/use-cases/object-detection-in-videos)

```python
from pixeltable.iterators import FrameIterator
from pixeltable.functions.yolox import yolox

frames = pxt.create_view(
    'project.frames',
    videos,
    iterator=FrameIterator.create(video=videos.video, fps=1)
)
frames.add_computed_column(detections=yolox(frames.frame))
```

### Image Transformations

[📚 Watermarks](https://docs.pixeltable.com/howto/cookbooks/images/img-add-watermarks) | [📚 Opacity](https://docs.pixeltable.com/howto/cookbooks/images/img-adjust-opacity) | [📚 RGB to Grayscale](https://docs.pixeltable.com/howto/cookbooks/images/img-rgb-to-grayscale)

```python
t.add_computed_column(gray=t.image.convert('L'))
t.add_computed_column(small=t.image.resize((256, 256)))
```

### Text Processing

[📚 Summarization](https://docs.pixeltable.com/howto/cookbooks/text/text-summarize) | [📚 Translation](https://docs.pixeltable.com/howto/cookbooks/text/text-translate) | [📚 Entity Extraction](https://docs.pixeltable.com/howto/cookbooks/text/text-extract-entities)

### Multimodal Analysis

[📚 Vision Batch Analysis](https://docs.pixeltable.com/howto/cookbooks/images/vision-batch-analysis) | [📚 Vision Structured Output](https://docs.pixeltable.com/howto/cookbooks/images/vision-structured-output) | [📚 Generate Captions](https://docs.pixeltable.com/howto/cookbooks/images/img-generate-captions)

---

## Additional Resources

### Integration Guides

- [Label Studio](https://docs.pixeltable.com/howto/using-label-studio-with-pixeltable)
- [FiftyOne](https://docs.pixeltable.com/howto/working-with-fiftyone)
- [Pydantic](https://docs.pixeltable.com/howto/providers/working-with-pydantic)
- [Tigris](https://docs.pixeltable.com/howto/providers/working-with-tigris)

### Advanced Topics

- [JSON Extraction Workflow](https://docs.pixeltable.com/howto/cookbooks/core/workflow-json-extraction)
- [Data Lookup Patterns](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-data-lookup)
- [Table as UDF](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-table-as-udf)
- [Deployment Infrastructure](https://docs.pixeltable.com/howto/deployment/infrastructure)
- [Operations & Monitoring](https://docs.pixeltable.com/howto/deployment/operations)

### Community

- **Discord**: [discord.com/invite/QPyqFYx2UN](https://discord.com/invite/QPyqFYx2UN)
- **GitHub**: [github.com/pixeltable/pixeltable](https://github.com/pixeltable/pixeltable)
- **Changelog**: [docs.pixeltable.com/changelog/changelog](https://docs.pixeltable.com/changelog/changelog)

---

*For complete API documentation, see [docs.pixeltable.com/sdk/latest](https://docs.pixeltable.com/sdk/latest)*

*Last updated: December 2025 | Pixeltable v0.5.x*
