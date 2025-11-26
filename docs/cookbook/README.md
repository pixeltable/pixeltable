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

Transform and process images using PIL methods and custom UDFs:

| Recipe | Description |
|--------|-------------|
| [Transform images with PIL operations](img-pil-transforms.ipynb) | Resize, rotate, flip, crop |
| [Convert RGB images to grayscale](img-rgb-to-grayscale.ipynb) | Simple and perceptually accurate methods |
| [Apply image filters](img-apply-filters.ipynb) | Blur, sharpen, edge detection |
| [Adjust image brightness and contrast](img-brightness-contrast.ipynb) | Fix lighting and enhance visibility |
| [Add watermarks to images](img-add-watermarks.ipynb) | Brand or protect your images |
| [Adjust image opacity](img-adjust-opacity.ipynb) | Create semi-transparent effects |

### Vision AI

Analyze images with AI models:

| Recipe | Description |
|--------|-------------|
| [Analyze images in batch](vision-batch-analysis.ipynb) | Run the same prompt on multiple images automatically |
| [Extract structured data from images](vision-structured-output.ipynb) | Get JSON from receipts, forms, documents |

### Audio

Process and transcribe audio:

| Recipe | Description |
|--------|-------------|
| [Transcribe audio files](audio-transcribe.ipynb) | Convert speech to text with Whisper |

### Video

Process and analyze videos:

| Recipe | Description |
|--------|-------------|
| [Extract frames from videos](video-extract-frames.ipynb) | Pull frames at intervals or keyframes only |

### Documents

Process and search documents:

| Recipe | Description |
|--------|-------------|
| [Split documents for RAG](doc-chunk-for-rag.ipynb) | Break PDFs into searchable chunks |

### Search

Build semantic search systems:

| Recipe | Description |
|--------|-------------|
| [Semantic text search](search-semantic-text.ipynb) | Find content by meaning, not keywords |
| [Find similar images](search-similar-images.ipynb) | Visual similarity search with CLIP |

### Data Import

Load data from external sources:

| Recipe | Description |
|--------|-------------|
| [Import CSV files](data-import-csv.ipynb) | Load data from CSV and Excel files |

### Workflow

Common setup and configuration patterns:

| Recipe | Description |
|--------|-------------|
| [Configure API keys](workflow-api-keys.ipynb) | Set up credentials for AI services |
| [Extract fields from JSON](workflow-json-extraction.ipynb) | Parse LLM response fields |

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
