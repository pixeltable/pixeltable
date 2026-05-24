# Pixeltable Sample Apps & Templates

## Application Templates

Full-stack vertical apps that replace paid SaaS. Each is one `schema.py` + `pyproject.toml`, scaffolded in one command:

```bash
uvx pixeltable-new --template <name> my-app
cd my-app && uv sync && python schema.py
```

| Template | Description |
|----------|-------------|
| **knowledge-base** | Upload docs, images, video, audio — unified search + RAG Q&A across all media types |
| **chat-agent** | Persistent multimodal agent with durable memory, tool calling, MCP exposure |
| **audio-transcription** | Audio/podcast transcription, summarization, semantic search |
| **full-stack-showcase** | Complete reference app: Gemini + DETR + Whisper, React UI, cross-modal search |
| **video-search** | Declarative video pipeline: frames, scenes, transcription, object detection, temporal search |
| **media-indexing** | Enterprise media processing: ingest from S3, process all modalities, export to your DB |
| **image-dataset** | ML dataset engineering: auto-annotate, curate, version, export to PyTorch |

Templates live in the [Pixeltable Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit/tree/main/templates). Run `uvx pixeltable-new --list` to see all available options.

## Showcase Apps

Standalone applications demonstrating Pixeltable in specific contexts:

| App | Description |
|-----|-------------|
| [cli-media-toolkit](cli-media-toolkit/) | Command-line tool for media processing with Pixeltable |
| [jfk-files-mcp-server](jfk-files-mcp-server/) | MCP server for JFK declassified files with semantic search |
| [context-aware-discord-bot](context-aware-discord-bot/) | Discord bot with persistent context via Pixeltable |
| [ai-based-trading-insight](ai-based-trading-insight-chrome-extension/) | Chrome extension for AI-powered trading insights |
| [prompt-engineering-studio](prompt-engineering-studio-gradio-application/) | Gradio app for prompt engineering and comparison |
| [reddit-agentic-bot](reddit-agentic-bot/) | Reddit bot with tool-calling agent pipeline |

## Learn More

- [Pixeltable Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit) — structural patterns + templates + deployment configs
- [pixeltable-new](https://github.com/pixeltable/pixeltable-new) — `uvx pixeltable-new` scaffolder
- [Cookbooks](https://docs.pixeltable.com/howto/cookbooks) — step-by-step recipes
- [Documentation](https://docs.pixeltable.com/)
