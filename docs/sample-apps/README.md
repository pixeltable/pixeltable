# Sample apps and examples

## Applications in `app.py`

Pixeltable applications declare `TableModel` tables and `FastAPIRouter` routes in `app.py`, with custom UDFs in importable modules. Run `pxt schema update` to create tables and `pxt service update` to start HTTP endpoints:

- **[Awesome Pixeltable Apps (40+ Production-Style Example Apps)](https://github.com/pierrebrunelle/awesome-pixeltable-apps)** ([**Live Interactive Gallery ->**](https://pierrebrunelle.github.io/awesome-pixeltable-apps/)): Curated collection of 40+ self-contained Pixeltable apps across **Image & Vision**, **Video**, **Audio & Speech**, **Documents & Text**, and **Structured & Hybrid** workloads. Every app includes `app.py`, `udfs.py`, `pixeltable.toml` (local + Cloud), `seed.py`, `client_demo.py`, and 1-click GitHub Codespaces support.
- **[Pixeltable Starter Kit (`pixeltable-starter-kit`)](https://github.com/pixeltable/pixeltable-starter-kit)** (`uvx pixeltable-new --list`): Official starter templates:
  - **[Chat Agent (`chat-agent`)](https://github.com/pixeltable/pixeltable-starter-kit/tree/main/chat-agent)** (`uvx pixeltable-new myapp`): Knowledge base chunking, sentence embeddings, conversation memory, and Anthropic tool/chat routes.
  - **[Video Search (`video-search`)](https://github.com/pixeltable/pixeltable-starter-kit/tree/main/video-search)** (`uvx pixeltable-new myapp --video`): Frame extraction, multimodal CLIP embeddings, and cross-modal text/image/video search routes.
- **[Lead Enrichment & Outreach Ledger (`lead-enrichment/`)](lead-enrichment/)**: Lead enrichment sample in this directory (`app.py`, `udfs.py`, `ui.html`, `seed.py`, `client_demo.py`, `pixeltable.toml`): custom `@pxt.udf` functions over [`treg.to`](https://treg.to) for waterfall email discovery, inline mailbox verification, live website extraction, OpenAI cold outreach, Fish Audio voice notes (`pxt.Audio`), paid `/preview` calls without storing rows, and per-row cost tracking with an interactive web UI.

---

## More examples and integrations

See the [starter-kit guide](../release/resources/starter-kit.mdx) and the
[documentation navigation](../release/docs.json) for cookbooks, provider examples,
HTTP serving, and Cloud deployment.

Framework packages: [PydanticAI](https://github.com/pixeltable/pydantic-ai-pixeltable),
[LangGraph](https://github.com/pixeltable/langgraph-store-pixeltable),
[LangChain](https://github.com/pixeltable/langchain-pixeltable),
[LlamaIndex](https://github.com/pixeltable/llama-index-vector-stores-pixeltable), and
[Haystack](https://github.com/pixeltable/haystack-pixeltable).

## Earlier showcases

The remaining directories next to this file are earlier full-stack showcases (predating single-file `app.py` + `pxt schema update` / `pxt service update`):

- **[`multimodal-chat/`](multimodal-chat/)**: Next.js + FastAPI multimodal RAG chat over documents, audio, and video.
- **[`text-and-image-similarity-search-nextjs-fastapi/`](text-and-image-similarity-search-nextjs-fastapi/)**: Next.js + FastAPI cross-modal CLIP text and image search.
- **[`jfk-files-mcp-server/`](jfk-files-mcp-server/)**: Multimodal MCP server indexing the JFK archive (PDFs, audio, images).
- **[`context-aware-discord-bot/`](context-aware-discord-bot/)**: Discord bot with persistent conversation memory and semantic search.
- **[`reddit-agentic-bot/`](reddit-agentic-bot/)**: Reddit agentic Q&A bot.
- **[`intelligence-hub/`](intelligence-hub/)**: Multi-agent workspace for document, audio, and video analysis.
- **[`cli-media-toolkit/`](cli-media-toolkit/)**: Command-line media indexing and search toolkit.
- **[`ai-based-trading-insight-chrome-extension/`](ai-based-trading-insight-chrome-extension/)**: Chrome extension for chart image analysis.
- **[`prompt-engineering-studio-gradio-application/`](prompt-engineering-studio-gradio-application/)**: Gradio studio for prompt experimentation and side-by-side model comparison.
