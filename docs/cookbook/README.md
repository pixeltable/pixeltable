# Pixeltable cookbook

Quick, practical recipes showing how to solve real problems with Pixeltable.

## Using a recipe

1. **Pick a recipe** that matches your use case
2. **Install Pixeltable**: `pip install pixeltable`
3. **Open the notebook** in Jupyter or your preferred environment
4. **Run the cells** - each recipe is self-contained and ready to use
5. **Adapt to your data** - replace example URLs with your own images, prompts, or data

Most recipes run in under 5 minutes and show working end-to-end examples.

## Available recipes

### Images

Transform and process images:

| Recipe | Description |
|--------|-------------|
| [Transform images with PIL operations](img-pil-transforms.ipynb) | Resize, rotate, flip, crop |
| [Add watermarks to images](img-add-watermarks.ipynb) | Brand or protect your images |

### Vision AI

Analyze images with AI models:

| Recipe | Description |
|--------|-------------|
| [Analyze images in batch](vision-batch-analysis.ipynb) | Run the same prompt on multiple images automatically |
| [Extract structured data from images](vision-structured-output.ipynb) | Get JSON from receipts, forms, documents |
| [Generate captions for images](img-generate-captions.ipynb) | Auto-caption images for accessibility and SEO |
| [Detect objects in images](img-detect-objects.ipynb) | Find and locate objects with YOLOX |
| [Visualize object detections](img-visualize-detections.ipynb) | Draw bounding boxes on images |

### Video Generation

Create videos with AI models:

| Recipe | Description |
|--------|-------------|
| [Generate videos with AI](video-generate-ai.ipynb) | Text-to-video and image-to-video with Veo |

### Audio

Process and transcribe audio:

| Recipe | Description |
|--------|-------------|
| [Transcribe audio files](audio-transcribe.ipynb) | Convert speech to text with Whisper |
| [Transcribe with speaker identification](audio-speaker-diarization.ipynb) | Meeting transcripts with who said what |
| [Summarize podcasts and audio](audio-summarize-podcast.ipynb) | Transcribe and summarize in one pipeline |
| [Extract audio from video](audio-extract-from-video.ipynb) | Get audio tracks for transcription |
| [Convert text to speech](audio-text-to-speech.ipynb) | Generate audio from text with OpenAI TTS |

### Video

Process and analyze videos:

| Recipe | Description |
|--------|-------------|
| [Extract frames from videos](video-extract-frames.ipynb) | Pull frames at intervals or keyframes only |
| [Generate thumbnails from videos](video-generate-thumbnails.ipynb) | Create preview images at specific timestamps |
| [Detect scene changes](video-scene-detection.ipynb) | Find cuts, transitions, and fades |
| [Add text overlays](video-add-text-overlay.ipynb) | Burn captions, watermarks, titles into videos |

### Text

Process text with LLMs:

| Recipe | Description |
|--------|-------------|
| [Summarize text with LLMs](text-summarize.ipynb) | Generate summaries of articles and documents |
| [Translate text between languages](text-translate.ipynb) | Automated multi-language translation |
| [Extract named entities](text-extract-entities.ipynb) | Find people, organizations, locations in text |

### Documents

Process and search documents:

| Recipe | Description |
|--------|-------------|
| [Split documents for RAG](doc-chunk-for-rag.ipynb) | Break PDFs into searchable chunks |

### Search & Embeddings

Build semantic search systems:

| Recipe | Description |
|--------|-------------|
| [Semantic text search](search-semantic-text.ipynb) | Find content by meaning, not keywords |
| [Find similar images](search-similar-images.ipynb) | Visual similarity search with CLIP |
| [Create text embeddings](embed-text-openai.ipynb) | Generate vectors for similarity search |

### Data Import

Load data from external sources:

| Recipe | Description |
|--------|-------------|
| [Import CSV files](data-import-csv.ipynb) | Load data from CSV files |
| [Import Excel files](data-import-excel.ipynb) | Load data from Excel spreadsheets |
| [Import Parquet files](data-import-parquet.ipynb) | Load columnar data from Parquet |
| [Import JSON files](data-import-json.ipynb) | Load data from JSON files or URLs |
| [Import Hugging Face datasets](data-import-huggingface.ipynb) | Load datasets from Hugging Face Hub |
| [Load media from S3/cloud](data-import-s3.ipynb) | Import images, videos from S3, GCS, HTTP |

### Data Export & ML Training

Prepare and export data for machine learning:

| Recipe | Description |
|--------|-------------|
| [Export for PyTorch training](data-export-pytorch.ipynb) | Convert to PyTorch DataLoader |
| [Sample data for training](data-sampling.ipynb) | Random and stratified sampling |

### Queries

Query and combine data:

| Recipe | Description |
|--------|-------------|
| [Join tables](query-join-tables.ipynb) | Combine data from multiple tables |
| [Create custom aggregates (UDAs)](custom-aggregates-uda.ipynb) | Sum of squares, string concat, weighted avg |
| [Create custom iterators](custom-iterators.ipynb) | Build your own data splitters |

### Workflow

Common setup and configuration patterns:

| Recipe | Description |
|--------|-------------|
| [Configure API keys](workflow-api-keys.ipynb) | Set up credentials for AI services |
| [Extract fields from JSON](workflow-json-extraction.ipynb) | Parse LLM response fields |

### LLMs

Work with language models:

| Recipe | Description |
|--------|-------------|
| [Use tool calling with LLMs](llm-tool-calling.ipynb) | Function calling and tool execution |

> **Note:** For local LLM setup with Ollama, see [working-with-ollama.ipynb](../notebooks/integrations/working-with-ollama.ipynb) in integrations.

### Patterns

End-to-end AI application patterns:

| Recipe | Description |
|--------|-------------|
| [Build a RAG pipeline](pattern-rag-pipeline.ipynb) | Question answering with document retrieval |
| [Build an agent with memory](pattern-agent-memory.ipynb) | Semantic memory for AI agents |
| [Look up structured data](pattern-data-lookup.ipynb) | Query tables with retrieval UDFs |

### Iteration

Speed up your development workflow:

| Recipe | Description |
|--------|-------------|
| [Get fast feedback on transformations](dev-iterative-workflow.ipynb) | Test logic before processing full datasets |

## Contributing

Want to write a recipe? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [STYLE_GUIDE.md](STYLE_GUIDE.md) for detailed standards.

## Help and support

- [Documentation](https://docs.pixeltable.com)
- [Discord](https://discord.gg/QPyqFYx2UN)
- [GitHub Issues](https://github.com/pixeltable/pixeltable/issues)
