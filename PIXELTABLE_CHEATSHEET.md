# 📋 Pixeltable Comprehensive Cheat Sheet

> **Pixeltable** is the open-source Python library providing declarative data infrastructure for multimodal AI applications—enabling incremental storage, transformation, indexing, retrieval, and orchestration of data.

**Version:** 0.5.x  
**Documentation:** [docs.pixeltable.com](https://docs.pixeltable.com)  
**GitHub:** [github.com/pixeltable/pixeltable](https://github.com/pixeltable/pixeltable)

---

## Table of Contents

1. [Installation](#installation)
2. [Directories](#directories)
3. [Tables](#tables)
4. [Type System](#type-system)
5. [Data Operations](#data-operations)
6. [Queries](#queries)
7. [Computed Columns](#computed-columns)
8. [User-Defined Functions (UDFs)](#user-defined-functions-udfs)
9. [Embedding Indexes (Vector Search)](#embedding-indexes-vector-search)
10. [Iterators](#iterators)
11. [Views](#views)
12. [Version Control](#version-control)
13. [AI Integrations](#ai-integrations)
14. [Built-in Functions](#built-in-functions)
15. [Data Import/Export](#data-importexport)
16. [Configuration](#configuration)
17. [Common Patterns](#common-patterns)

---

## Installation

```bash
# Basic installation
pip install pixeltable

# With specific integrations
pip install pixeltable[openai]
pip install pixeltable[anthropic]
pip install pixeltable[huggingface]
pip install pixeltable[together]

# All integrations
pip install pixeltable[all]
```

```python
import pixeltable as pxt
```

---

## Directories

Pixeltable organizes tables into directories (like folders).

```python
# Create directories
pxt.create_dir('my_project')
pxt.create_dir('my_project.subdir')
pxt.create_dir('my_project.subdir.nested')

# List contents
pxt.ls()                          # Root level - returns DataFrame
pxt.ls('my_project')              # Specific directory

# List directories only
pxt.list_dirs()                   # All directories
pxt.list_dirs('my_project')       # Within specific path
pxt.list_dirs(recursive=True)     # Including nested

# List tables only
pxt.list_tables()
pxt.list_tables('my_project', recursive=True)

# Get detailed contents
contents = pxt.get_dir_contents('my_project')
contents['tables']    # List of table names
contents['dirs']      # List of directory names

# Move/rename
pxt.move('my_project.old_name', 'my_project.new_name')

# Delete
pxt.drop_dir('my_project', force=True)  # force=True for non-empty
pxt.drop_dir('my_project', if_not_exists='ignore')
```

---

## Tables

### Creating Tables

```python
# Basic table with schema
t = pxt.create_table('project.users', {
    'name': pxt.String,
    'age': pxt.Int,
    'score': pxt.Float,
    'active': pxt.Bool,
    'created_at': pxt.Timestamp,
})

# With required (non-nullable) columns
t = pxt.create_table('project.items', {
    'id': pxt.Required[pxt.String],
    'name': pxt.Required[pxt.String],
    'description': pxt.String,  # Nullable
})

# With media columns
t = pxt.create_table('project.media', {
    'image': pxt.Image,
    'video': pxt.Video,
    'audio': pxt.Audio,
    'document': pxt.Document,
})

# With array columns
t = pxt.create_table('project.vectors', {
    'embedding': pxt.Array[(768,), pxt.Float],  # Fixed dimension
    'features': pxt.Array[(None,), pxt.Float],  # Variable dimension
})

# With JSON columns
t = pxt.create_table('project.data', {
    'metadata': pxt.Json,
    'config': pxt.Json,
})

# With primary key
t = pxt.create_table('project.records', {
    'id': pxt.Required[pxt.String],
    'data': pxt.String,
}, primary_key='id')

# With comment
t = pxt.create_table('project.docs', {
    'content': pxt.String,
}, comment='Documentation table')

# If exists handling
t = pxt.create_table('project.t', schema, if_exists='ignore')  # Don't error if exists
t = pxt.create_table('project.t', schema, if_exists='replace')  # Drop and recreate
```

### Table Operations

```python
# Get existing table
t = pxt.get_table('project.users')
t = pxt.get_table('project.users', if_not_exists='ignore')  # Returns None if not found

# Describe schema
t.describe()

# Get metadata
metadata = t.get_metadata()
metadata['name']
metadata['version']
metadata['comment']

# Count rows
t.count()

# List columns
t.columns()  # Returns list of column names

# Add columns
t.add_column(email=pxt.String)
t.add_column(phone=pxt.String, if_exists='ignore')
t.add_columns({'col1': pxt.String, 'col2': pxt.Int})

# Rename column
t.rename_column('old_name', 'new_name')

# Drop column
t.drop_column('column_name')
t.drop_column('column_name', if_not_exists='ignore')

# Drop table
pxt.drop_table('project.users')
pxt.drop_table('project.users', force=True)  # Even with dependents
pxt.drop_table('project.users', if_not_exists='ignore')
```

---

## Type System

### Core Types

| Python Type | Pixeltable Type | Description |
|-------------|-----------------|-------------|
| `str` | `pxt.String` | Text strings |
| `int` | `pxt.Int` | 64-bit integers |
| `float` | `pxt.Float` | 64-bit floats |
| `bool` | `pxt.Bool` | Boolean values |
| `datetime.datetime` | `pxt.Timestamp` | Date and time with timezone |
| `datetime.date` | `pxt.Date` | Date only |
| `dict` | `pxt.Json` | JSON-serializable data |
| `bytes` | `pxt.Binary` | Binary data |
| `uuid.UUID` | `pxt.UUID` | UUID values |

### Media Types

| Type | Pixeltable Type | Description |
|------|-----------------|-------------|
| Image | `pxt.Image` | PIL.Image.Image |
| Video | `pxt.Video` | Video files |
| Audio | `pxt.Audio` | Audio files |
| Document | `pxt.Document` | PDFs, Word docs, etc. |

### Array Types

```python
# Fixed dimension
pxt.Array[(768,), pxt.Float]      # 768-dim float vector
pxt.Array[(224, 224, 3), pxt.Int] # 224x224 RGB image array

# Variable dimension
pxt.Array[(None,), pxt.Float]     # Variable-length float vector
```

### Required (Non-nullable)

```python
pxt.Required[pxt.String]  # Non-nullable string
pxt.Required[pxt.Int]     # Non-nullable integer
```

---

## Data Operations

### Insert

```python
# Single row
t.insert({'name': 'Alice', 'age': 30})

# Multiple rows
t.insert([
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
])

# From Pydantic models
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

t.insert([User(name='Alice', age=30), User(name='Bob', age=25)])

# Media files (URLs or local paths)
t.insert({'image': 'https://example.com/photo.jpg'})
t.insert({'image': '/path/to/local/image.png'})
t.insert({'video': 's3://bucket/video.mp4'})

# With primary key (upsert behavior)
t.insert({'id': 'user-1', 'name': 'Alice'}, on_conflict='update')
t.insert({'id': 'user-1', 'name': 'Alice'}, on_conflict='ignore')
```

### Update

```python
# Update specific rows
t.where(t.name == 'Alice').update({'age': 31})

# Update with expression
t.where(t.active == True).update({'score': t.score + 10})

# Update with dict
t.where(t.id == 'user-1').update({'name': 'Alice Smith', 'age': 31})

# Batch update (for multiple rows with different values)
t.batch_update([
    {'_rowid': row_id_1, 'score': 100},
    {'_rowid': row_id_2, 'score': 200},
])
```

### Delete

```python
# Delete specific rows
t.where(t.age < 18).delete()

# Delete with complex condition
t.where((t.active == False) & (t.last_login < cutoff_date)).delete()

# Delete all rows (keeps schema)
t.delete()
```

### Recompute

```python
# Recompute specific computed columns
t.recompute_columns(['embedding', 'summary'])

# Recompute with new parameters
t.recompute_columns(['summary'], cascade=True)
```

---

## Queries

### Basic Queries

```python
# Select all columns
t.collect()                      # Returns ResultSet
t.head(10)                       # First 10 rows
t.tail(10)                       # Last 10 rows
t.show(20)                       # Show 20 rows

# Select specific columns
t.select(t.name, t.age).collect()

# Rename in output
t.select(username=t.name, user_age=t.age).collect()

# Select with expressions
t.select(
    t.name,
    full_name=t.first_name + ' ' + t.last_name,
    age_group=t.age // 10 * 10
).collect()
```

### Filtering with where()

```python
# Single condition
t.where(t.age > 25).collect()

# Multiple conditions (AND)
t.where((t.age > 25) & (t.active == True)).collect()

# OR conditions
t.where((t.age > 65) | (t.is_vip == True)).collect()

# NOT condition
t.where(~(t.category == 'spam')).collect()

# Null checks
t.where(t.email == None).collect()
t.where(t.email != None).collect()

# String operations
t.where(t.name.contains('alice')).collect()
t.where(t.email.endswith('@gmail.com')).collect()

# Chaining
t.where(t.age > 25).where(t.active == True).collect()
```

### Comparison Operators

```python
t.age == 25      # Equal
t.age != 25      # Not equal
t.age > 25       # Greater than
t.age >= 25      # Greater than or equal
t.age < 25       # Less than
t.age <= 25      # Less than or equal
```

### Ordering

```python
# Ascending (default)
t.order_by(t.created_at).collect()

# Descending
t.order_by(t.created_at, asc=False).collect()

# Multiple columns
t.order_by(t.category, t.name).collect()
```

### Limiting & Sampling

```python
# Limit results
t.limit(100).collect()

# Random sample
t.sample(fraction=0.1).collect()   # 10% of rows
t.sample(n=100).collect()          # 100 random rows
t.sample(n=100, seed=42).collect() # Reproducible sample
```

### Grouping & Aggregation

```python
from pixeltable.functions import count, sum, avg, min, max

# Group by with aggregation
t.group_by(t.category).select(
    t.category,
    total=count(t.id),
    avg_price=avg(t.price),
    max_price=max(t.price)
).collect()

# Multiple grouping columns
t.group_by(t.year, t.month).select(
    t.year, t.month,
    revenue=sum(t.amount)
).collect()
```

### Distinct

```python
# Unique values
t.select(t.category).distinct().collect()

# Unique combinations
t.select(t.category, t.status).distinct().collect()
```

### Joins

```python
orders = pxt.get_table('project.orders')
users = pxt.get_table('project.users')

# Inner join
orders.join(users, on=(orders.user_id == users.id)).select(
    orders.order_id,
    users.name,
    orders.amount
).collect()

# With alias to avoid column name conflicts
orders.join(users, on=(orders.user_id == users.id)).select(
    order_id=orders.order_id,
    customer_name=users.name,
    order_amount=orders.amount
).collect()
```

### ResultSet Operations

```python
result = t.collect()

# Access data
len(result)                    # Number of rows
result[0]                      # First row as dict
result[0]['name']              # Specific field

# Convert to other formats
df = result.to_pandas()        # pandas DataFrame
for row in result:             # Iterate as dicts
    print(row['name'])

# Convert to Pydantic models
for user in result.to_pydantic(UserModel):
    print(user.name)
```

---

## Computed Columns

Computed columns automatically calculate values when data is inserted or updated.

### Basic Computed Columns

```python
# String concatenation
t.add_computed_column(
    full_name=t.first_name + ' ' + t.last_name
)

# Arithmetic
t.add_computed_column(
    total=t.price * t.quantity
)

# Conditional
t.add_computed_column(
    status='active' if t.is_enabled else 'inactive'
)

# Using apply() for custom logic
t.add_computed_column(
    word_count=t.content.apply(lambda x: len(x.split()), col_type=pxt.Int)
)
```

### With External Functions

```python
from pixeltable.functions.openai import chat_completions

# LLM-powered computed column
t.add_computed_column(
    summary=chat_completions(
        messages=[{'role': 'user', 'content': 'Summarize: ' + t.text}],
        model='gpt-4o-mini'
    ).choices[0].message.content
)
```

### With UDFs

```python
@pxt.udf
def extract_domain(email: str) -> str:
    return email.split('@')[1] if '@' in email else ''

t.add_computed_column(domain=extract_domain(t.email))
```

### Options

```python
# If column exists
t.add_computed_column(col=expr, if_exists='ignore')
t.add_computed_column(col=expr, if_exists='replace')

# Store vs compute on demand
t.add_computed_column(col=expr, stored=True)   # Default: stored
t.add_computed_column(col=expr, stored=False)  # Compute on query
```

---

## User-Defined Functions (UDFs)

### Basic UDF

```python
@pxt.udf
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

# Use in queries
t.select(result=multiply(t.x, t.y)).collect()

# Use in computed columns
t.add_computed_column(product=multiply(t.x, t.y))
```

### Batched UDF (for efficiency)

```python
from pixeltable.func import Batch

@pxt.udf(batch_size=32)
def embed_batch(texts: Batch[str]) -> Batch[list[float]]:
    """Embed multiple texts at once for efficiency."""
    return model.encode(texts).tolist()
```

### UDF with Media Types

```python
import PIL.Image

@pxt.udf
def get_dimensions(img: PIL.Image.Image) -> dict:
    """Extract image dimensions."""
    return {'width': img.width, 'height': img.height}

@pxt.udf
def resize_image(img: PIL.Image.Image, size: int) -> PIL.Image.Image:
    """Resize image to square."""
    return img.resize((size, size))
```

### Expression UDF (SQL-optimized)

```python
@pxt.expr_udf
def discount_price(price: float, discount: float) -> float:
    """Calculate discounted price."""
    return price * (1 - discount)
```

### UDF with Optional Parameters

```python
@pxt.udf
def format_name(
    first: str,
    last: str,
    middle: str | None = None,
    suffix: str = ''
) -> str:
    """Format a full name."""
    parts = [first]
    if middle:
        parts.append(middle)
    parts.append(last)
    if suffix:
        parts.append(suffix)
    return ' '.join(parts)
```

### User-Defined Aggregates (UDA)

```python
@pxt.uda(
    value_type=pxt.Float,
    update_type=pxt.Json,
    name='running_avg'
)
class RunningAverage:
    """Compute running average."""
    def __init__(self):
        self.sum = 0.0
        self.count = 0
    
    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1
    
    def value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

# Use in group_by
t.group_by(t.category).select(
    t.category,
    avg_score=running_avg(t.score)
).collect()
```

---

## Embedding Indexes (Vector Search)

### Create Embedding Index

```python
from pixeltable.functions.openai import embeddings

# Text embedding index
t.add_embedding_index(
    'content_idx',
    column=t.content,
    embedding=embeddings(input=t.content, model='text-embedding-3-small')
)

# With HuggingFace
from pixeltable.functions.huggingface import sentence_transformer

t.add_embedding_index(
    'text_idx',
    column=t.text,
    embedding=sentence_transformer(t.text, model_id='all-MiniLM-L6-v2')
)

# Image embedding with CLIP
from pixeltable.functions.huggingface import clip

t.add_embedding_index(
    'image_idx',
    column=t.image,
    embedding=clip(t.image, model_id='openai/clip-vit-base-patch32')
)

# Custom metric
t.add_embedding_index(
    'idx',
    column=t.content,
    embedding=embed_fn(t.content),
    metric='cosine'  # 'cosine', 'ip' (inner product), 'l2'
)
```

### Similarity Search

```python
# Basic search
results = t.content_idx.similarity_search('What is machine learning?', k=10)
results.select(t.content, results.similarity).collect()

# With filters
results = t.content_idx.similarity_search(
    'machine learning',
    k=10
)
results.where(t.category == 'tech').select(t.content, results.similarity).collect()

# Get similarity score
results.select(
    t.title,
    t.content,
    score=results.similarity
).collect()
```

### Manage Indexes

```python
# List indexes
t.describe()  # Shows indexes

# Drop index
t.drop_embedding_index('content_idx')
```

---

## Iterators

Iterators extract rows from media (video frames, document chunks, etc.).

### FrameIterator (Video Frames)

```python
from pixeltable.iterators import FrameIterator

videos = pxt.get_table('project.videos')

# Create view with extracted frames
frames = pxt.create_view(
    'project.frames',
    videos,
    iterator=FrameIterator.create(
        video=videos.video,
        fps=1  # 1 frame per second
    )
)

# Access frame data
frames.select(frames.frame, frames.pos).head(10)
```

### DocumentSplitter (Text Chunks)

```python
from pixeltable.iterators import DocumentSplitter

docs = pxt.get_table('project.documents')

# Split into chunks
chunks = pxt.create_view(
    'project.chunks',
    docs,
    iterator=DocumentSplitter.create(
        document=docs.document,
        separators='paragraph',  # 'sentence', 'token_limit', 'char_limit'
        limit=500,  # Max chars per chunk
        overlap=50  # Overlap between chunks
    )
)
```

### StringSplitter

```python
from pixeltable.iterators import StringSplitter

# Split text by delimiter
lines = pxt.create_view(
    'project.lines',
    t,
    iterator=StringSplitter.create(
        text=t.content,
        separators='\n'
    )
)
```

### AudioSplitter

```python
from pixeltable.iterators import AudioSplitter

# Split audio into segments
segments = pxt.create_view(
    'project.audio_segments',
    recordings,
    iterator=AudioSplitter.create(
        audio=recordings.audio,
        segment_duration=30.0  # 30 second segments
    )
)
```

### VideoSplitter

```python
from pixeltable.iterators import VideoSplitter

# Split video into clips
clips = pxt.create_view(
    'project.clips',
    videos,
    iterator=VideoSplitter.create(
        video=videos.video,
        segment_duration=60.0  # 60 second clips
    )
)
```

### TileIterator (Image Tiles)

```python
from pixeltable.iterators import TileIterator

# Split images into tiles
tiles = pxt.create_view(
    'project.tiles',
    images,
    iterator=TileIterator.create(
        image=images.image,
        tile_size=(256, 256),
        overlap=32
    )
)
```

---

## Views

Views are virtual tables based on queries or iterators.

### Filtered View

```python
# Create view from filter
active_users = pxt.create_view(
    'project.active_users',
    users,
    filter=users.status == 'active'
)

# View inherits columns and computed columns
active_users.select(active_users.name, active_users.email).collect()
```

### View with Iterator

```python
# Extract video frames as view
frames = pxt.create_view(
    'project.video_frames',
    videos,
    iterator=FrameIterator.create(video=videos.video, fps=2)
)

# Add computed columns to view
frames.add_computed_column(
    objects=detect_objects(frames.frame)
)
```

### Snapshot (Point-in-time)

```python
# Create snapshot of current state
snapshot = pxt.create_snapshot('project.users_backup', users)

# Snapshot is read-only
```

---

## Version Control

### View History

```python
# Get version history
t.history()  # Returns DataFrame with version info
t.history(n=10)  # Last 10 versions

# Get version metadata
versions = t.get_versions()
for v in versions:
    print(f"Version {v['version']}: {v['change_type']} at {v['created_at']}")
```

### Revert Changes

```python
# Revert to previous version
t.revert()

# WARNING: Revert cannot be undone!
```

---

## AI Integrations

### OpenAI

```python
from pixeltable.functions.openai import chat_completions, embeddings, image, audio

# Chat completions
t.add_computed_column(
    response=chat_completions(
        messages=[{'role': 'user', 'content': t.prompt}],
        model='gpt-4o'
    ).choices[0].message.content
)

# With system prompt
t.add_computed_column(
    response=chat_completions(
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': t.question}
        ],
        model='gpt-4o-mini',
        max_tokens=500,
        temperature=0.7
    ).choices[0].message.content
)

# Embeddings
t.add_computed_column(
    embedding=embeddings(
        input=t.text,
        model='text-embedding-3-small'
    ).data[0].embedding
)

# Image generation (DALL-E)
t.add_computed_column(
    generated_image=image.generate(
        prompt=t.prompt,
        model='dall-e-3',
        size='1024x1024'
    ).data[0].image
)

# Audio transcription (Whisper)
t.add_computed_column(
    transcript=audio.transcriptions(
        audio=t.audio_file,
        model='whisper-1'
    ).text
)

# Vision (GPT-4 Vision)
t.add_computed_column(
    description=chat_completions(
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe this image'},
                {'type': 'image_url', 'image_url': {'url': t.image}}
            ]
        }],
        model='gpt-4o'
    ).choices[0].message.content
)
```

### Anthropic

```python
from pixeltable.functions.anthropic import messages

t.add_computed_column(
    response=messages(
        messages=[{'role': 'user', 'content': t.prompt}],
        model='claude-sonnet-4-20250514',
        max_tokens=1000
    ).content[0].text
)
```

### Google Gemini

```python
from pixeltable.functions.gemini import generate_content

t.add_computed_column(
    response=generate_content(
        contents=t.prompt,
        model='gemini-2.0-flash'
    ).text
)
```

### Hugging Face

```python
from pixeltable.functions.huggingface import (
    sentence_transformer,
    clip,
    detr_for_object_detection,
    vit_for_image_classification,
    text_to_image,
    image_to_image,
    summarization,
    translation,
    text_classification,
    token_classification,
    question_answering,
    text_generation,
    automatic_speech_recognition,
    image_captioning,
    text_to_speech,
    image_to_video
)

# Sentence embeddings
t.add_computed_column(
    embedding=sentence_transformer(t.text, model_id='all-MiniLM-L6-v2')
)

# Image embeddings (CLIP)
t.add_computed_column(
    embedding=clip(t.image, model_id='openai/clip-vit-base-patch32')
)

# Object detection
t.add_computed_column(
    detections=detr_for_object_detection(
        t.image,
        model_id='facebook/detr-resnet-50',
        threshold=0.7
    )
)

# Image classification
t.add_computed_column(
    classification=vit_for_image_classification(
        t.image,
        model_id='google/vit-base-patch16-224',
        top_k=5
    )
)

# Text-to-image (Stable Diffusion)
t.add_computed_column(
    generated=text_to_image(
        t.prompt,
        model_id='stable-diffusion-v1-5/stable-diffusion-v1-5',
        height=512,
        width=512
    )
)

# Image-to-image transformation
t.add_computed_column(
    transformed=image_to_image(
        t.image,
        t.prompt,
        model_id='stable-diffusion-v1-5/stable-diffusion-v1-5',
        model_kwargs={'strength': 0.7, 'num_inference_steps': 30}
    )
)

# Summarization
t.add_computed_column(
    summary=summarization(
        t.long_text,
        model_id='facebook/bart-large-cnn'
    )
)

# Translation
t.add_computed_column(
    french=translation(
        t.english_text,
        model_id='Helsinki-NLP/opus-mt-en-fr'
    )
)

# Text classification (sentiment)
t.add_computed_column(
    sentiment=text_classification(
        t.review,
        model_id='cardiffnlp/twitter-roberta-base-sentiment-latest'
    )
)

# Named entity recognition
t.add_computed_column(
    entities=token_classification(
        t.text,
        model_id='dbmdz/bert-large-cased-finetuned-conll03-english'
    )
)

# Question answering
t.add_computed_column(
    answer=question_answering(
        context=t.document,
        question=t.question,
        model_id='deepset/roberta-base-squad2'
    )
)

# Text generation
t.add_computed_column(
    continuation=text_generation(
        t.prompt,
        model_id='gpt2',
        model_kwargs={'max_length': 100}
    )
)

# Speech recognition
t.add_computed_column(
    transcript=automatic_speech_recognition(
        t.audio,
        model_id='openai/whisper-small'
    )
)

# Image captioning
t.add_computed_column(
    caption=image_captioning(
        t.image,
        model_id='Salesforce/blip-image-captioning-base'
    )
)
```

### Whisper (Local)

```python
from pixeltable.functions.whisper import transcribe

t.add_computed_column(
    transcript=transcribe(t.audio, model='base')
)
```

### Together AI

```python
from pixeltable.functions.together import chat_completions

t.add_computed_column(
    response=chat_completions(
        messages=[{'role': 'user', 'content': t.prompt}],
        model='meta-llama/Llama-3-70b-chat-hf'
    ).choices[0].message.content
)
```

### Ollama (Local)

```python
from pixeltable.functions.ollama import chat

t.add_computed_column(
    response=chat(
        messages=[{'role': 'user', 'content': t.prompt}],
        model='llama3'
    ).message.content
)
```

### Voyage AI

```python
from pixeltable.functions.voyageai import embed

t.add_computed_column(
    embedding=embed(input=t.text, model='voyage-3')
)
```

### Replicate

```python
from pixeltable.functions.replicate import run

t.add_computed_column(
    output=run(
        input={'prompt': t.prompt},
        ref='stability-ai/sdxl:latest'
    )
)
```

---

## Built-in Functions

### String Functions

```python
from pixeltable.functions import string

# Or use as methods on string columns
t.name.lower()
t.name.upper()
t.name.capitalize()
t.name.strip()
t.name.lstrip()
t.name.rstrip()
t.name.replace('old', 'new')
t.name.split(' ')
t.name.len()  # or string.len(t.name)

# String predicates
t.name.contains('alice')
t.name.startswith('A')
t.name.endswith('son')
t.name.isalpha()
t.name.isdigit()
t.name.isalnum()
t.name.islower()
t.name.isupper()

# Regex
t.name.contains_re(r'\d+')
t.name.findall(r'\w+')
t.name.replace_re(r'\s+', ' ')

# Formatting
t.name.center(20)
t.name.ljust(20)
t.name.rjust(20)
t.name.zfill(10)

# Join
string.join(', ', t.tags)  # Join list elements
```

### Image Functions

```python
from pixeltable.functions import image

# Image properties
t.image.width
t.image.height
t.image.mode

# Transformations
t.image.resize((256, 256))
t.image.rotate(90)
t.image.crop((0, 0, 100, 100))
t.image.convert('L')  # Grayscale
t.image.transpose(0)  # Flip

# Operations
t.image.blend(other_image, alpha=0.5)
t.image.composite(overlay, mask)

# Analysis
t.image.histogram()
t.image.entropy()
t.image.getcolors()
t.image.getextrema()

# Metadata
t.image.get_metadata()
```

### Video Functions

```python
from pixeltable.functions import video

# Metadata
t.video.get_metadata()
t.video.get_duration()

# Extract
t.video.extract_frame(timestamp=5.0)
t.video.extract_audio()

# Edit
t.video.clip(start=10.0, end=30.0)
t.video.overlay_text(text='Hello', position='center')

# Scene detection
t.video.scene_detect_content()
t.video.scene_detect_adaptive()
t.video.scene_detect_threshold()
```

### Audio Functions

```python
from pixeltable.functions import audio

# Metadata
t.audio.get_metadata()
```

### Timestamp Functions

```python
from pixeltable.functions import timestamp

# Properties
t.created_at.year
t.created_at.month
t.created_at.day
t.created_at.hour
t.created_at.minute
t.created_at.second
t.created_at.microsecond

# Methods
t.created_at.weekday()
t.created_at.isoweekday()
t.created_at.isocalendar()
t.created_at.isoformat()
t.created_at.strftime('%Y-%m-%d')
t.created_at.astimezone('US/Pacific')

# Create timestamp
timestamp.make_timestamp(year=2024, month=1, day=1)
```

### Math Functions

```python
from pixeltable.functions import math

math.abs(t.value)
math.ceil(t.value)
math.floor(t.value)
math.round(t.value, 2)
math.sqrt(t.value)
math.pow(t.base, t.exp)
math.log(t.value)
math.log10(t.value)
math.exp(t.value)
math.sin(t.angle)
math.cos(t.angle)
math.tan(t.angle)
```

### JSON Functions

```python
from pixeltable.functions import json

# Access nested fields
t.metadata['key']
t.metadata['nested']['field']

# Array access
t.items[0]
t.items[-1]
```

---

## Data Import/Export

### Import

```python
from pixeltable import io

# CSV
rows = io.import_csv('data.csv')
t.insert(rows)

# Or with schema inference
t = pxt.create_table('project.data', io.import_csv('data.csv', infer_schema=True))

# JSON
rows = io.import_json('data.json')
t.insert(rows)

# Parquet
rows = io.import_parquet('data.parquet')
t.insert(rows)

# Excel
rows = io.import_excel('data.xlsx', sheet_name='Sheet1')
t.insert(rows)

# Pandas DataFrame
import pandas as pd
df = pd.read_csv('data.csv')
t.insert(df.to_dict('records'))

# Hugging Face datasets
from datasets import load_dataset
ds = load_dataset('squad', split='train[:100]')
t.insert(ds.to_list())
```

### Export

```python
from pixeltable import io

# To pandas
df = t.collect().to_pandas()

# To Parquet
io.export_parquet(t, 'output.parquet')

# To PyTorch DataLoader
from torch.utils.data import DataLoader
dataset = t.to_pytorch_dataset()
loader = DataLoader(dataset, batch_size=32)

# To COCO format (for object detection)
coco_path = t.to_coco_dataset()
```

### External Storage (S3, etc.)

```python
# Configure global default in config.toml or env vars
# PIXELTABLE_INPUT_MEDIA_DEST="s3://my-bucket/input/"

# Or specify destination for computed columns
t.add_computed_column(
    thumbnail=t.image.resize((128, 128)),
    destination='s3://my-bucket/thumbnails/'
)
```

---

## Configuration

### API Keys

```python
import os

# Environment variables (recommended)
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
os.environ['HF_TOKEN'] = 'hf_...'
os.environ['TOGETHER_API_KEY'] = '...'
os.environ['GOOGLE_API_KEY'] = '...'

# Interactive input
import getpass
if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key: ')
```

### Config File

Create `~/.pixeltable/config.toml`:

```toml
[openai]
api_key = "sk-..."

[anthropic]
api_key = "sk-ant-..."

[together]
api_key = "..."
```

### Logging

```python
pxt.configure_logging(level='DEBUG')
pxt.configure_logging(level='INFO')
pxt.configure_logging(level='WARNING')
```

### Initialization

```python
# Custom database location
pxt.init({'home': '/path/to/pixeltable/data'})
```

---

## Common Patterns

### RAG Pipeline

```python
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.openai import embeddings, chat_completions

# 1. Create document table
docs = pxt.create_table('rag.documents', {'document': pxt.Document})

# 2. Create chunks view
chunks = pxt.create_view(
    'rag.chunks',
    docs,
    iterator=DocumentSplitter.create(
        document=docs.document,
        separators='paragraph',
        limit=500
    )
)

# 3. Add embedding index
chunks.add_embedding_index(
    'chunk_idx',
    column=chunks.text,
    embedding=embeddings(input=chunks.text, model='text-embedding-3-small')
)

# 4. Query function
def ask(question: str, k: int = 5) -> str:
    # Search for relevant chunks
    results = chunks.chunk_idx.similarity_search(question, k=k)
    context = results.select(chunks.text).collect()
    context_text = '\n\n'.join([row['text'] for row in context])
    
    # Generate answer
    response = chat_completions(
        messages=[
            {'role': 'system', 'content': f'Answer based on this context:\n{context_text}'},
            {'role': 'user', 'content': question}
        ],
        model='gpt-4o-mini'
    )
    return response.choices[0].message.content
```

### Iterate-then-Commit Workflow

```python
# 1. Preview with select() - nothing stored
t.select(
    t.image,
    preview=transform(t.image, strength=0.5)
).head(5)

# 2. Adjust and preview again
t.select(
    t.image,
    preview=transform(t.image, strength=0.7)
).head(5)

# 3. Commit when satisfied
t.add_computed_column(
    result=transform(t.image, strength=0.7)
)
```

### Video Analysis Pipeline

```python
from pixeltable.iterators import FrameIterator
from pixeltable.functions.huggingface import detr_for_object_detection

# 1. Create video table
videos = pxt.create_table('project.videos', {'video': pxt.Video})

# 2. Extract frames
frames = pxt.create_view(
    'project.frames',
    videos,
    iterator=FrameIterator.create(video=videos.video, fps=1)
)

# 3. Detect objects in each frame
frames.add_computed_column(
    detections=detr_for_object_detection(
        frames.frame,
        model_id='facebook/detr-resnet-50'
    )
)

# 4. Query results
frames.where(
    frames.detections['label_text'].apply(lambda x: 'person' in x, col_type=pxt.Bool)
).select(frames.frame, frames.pos, frames.detections).collect()
```

### Batch Processing with Seeds (Reproducibility)

```python
# Reproducible image generation
t.add_computed_column(
    generated=text_to_image(
        t.prompt,
        model_id='stable-diffusion-v1-5/stable-diffusion-v1-5',
        seed=42  # Fixed seed for reproducibility
    )
)
```

### Tool Calling with LLMs

```python
# Define tools
@pxt.udf
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

@pxt.udf
def search_web(query: str) -> str:
    """Search the web."""
    return f"Search results for: {query}"

# Create tools
tools = pxt.tools(
    pxt.tool(get_weather, description='Get current weather'),
    pxt.tool(search_web, description='Search the web')
)

# Use with chat
from pixeltable.functions.openai import chat_completions

t.add_computed_column(
    response=chat_completions(
        messages=[{'role': 'user', 'content': t.question}],
        model='gpt-4o',
        tools=tools
    )
)
```

---

## Quick Reference Card

### Essential Commands

| Task | Command |
|------|---------|
| Create directory | `pxt.create_dir('name')` |
| Create table | `pxt.create_table('dir.table', schema)` |
| Get table | `pxt.get_table('dir.table')` |
| Insert data | `t.insert([{'col': 'val'}])` |
| Query all | `t.collect()` |
| Filter | `t.where(t.col == 'val').collect()` |
| Select columns | `t.select(t.col1, t.col2).collect()` |
| Add computed | `t.add_computed_column(new=expr)` |
| Count rows | `t.count()` |
| Drop table | `pxt.drop_table('dir.table')` |

### Column Access

```python
t.column_name      # Column reference
t['column_name']   # Alternative syntax
t.json_col['key']  # JSON field access
```

### Common Filters

```python
t.where(t.col == value)           # Equals
t.where(t.col != value)           # Not equals
t.where(t.col > value)            # Greater than
t.where(t.col.contains('text'))   # Contains substring
t.where(t.col == None)            # Is null
t.where((cond1) & (cond2))        # AND
t.where((cond1) | (cond2))        # OR
```

---

## Resources

- **Documentation**: [docs.pixeltable.com](https://docs.pixeltable.com)
- **GitHub**: [github.com/pixeltable/pixeltable](https://github.com/pixeltable/pixeltable)
- **Discord**: [discord.com/invite/QPyqFYx2UN](https://discord.com/invite/QPyqFYx2UN)
- **Cookbooks**: [docs.pixeltable.com/howto/cookbooks](https://docs.pixeltable.com/howto/cookbooks)
- **API Reference**: [docs.pixeltable.com/sdk/latest](https://docs.pixeltable.com/sdk/latest)

---

*Last updated: December 2025 | Pixeltable v0.5.x*
