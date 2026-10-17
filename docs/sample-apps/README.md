# Sample Apps & Examples

## 1. Canonical Onboarding: Single-File `app.py` Starter Kit

Start with the **[Pixeltable Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit)** (`uvx pixeltable-new --list`). Each application is a single declarative `app.py` (`TableModel` + `@pxt.udf` + `FastAPIRouter`) deployed locally or to Pixeltable Cloud with `pxt schema update` and `pxt service update`:

- **[Chat Agent (`chat-agent`)](https://github.com/pixeltable/pixeltable-starter-kit/tree/main/chat-agent)** (`uvx pixeltable-new myapp`) — Knowledge base chunking, sentence embeddings, conversation memory, and Anthropic tool/chat routes.
- **[Video Search (`video-search`)](https://github.com/pixeltable/pixeltable-starter-kit/tree/main/video-search)** (`uvx pixeltable-new myapp --video`) — Frame extraction, multimodal CLIP embeddings, and cross-modal text/image/video search routes.
- **[Lead Enrichment & Outreach (`lead-enrichment/`)](lead-enrichment/)** — Programmable Clay + Lemlist alternative in one `app.py`: custom `@pxt.udf` functions over [`treg.to`](https://treg.to) for waterfall email discovery, inline mailbox verification, live website extraction, OpenAI cold outreach, and Fish Audio voice notes with per-row cost tracking.

---

## 2. All Other Sources of Pixeltable Examples & Integrations

| Resource | What You'll Find |
|---|---|
| **[Documentation & Quickstart](https://docs.pixeltable.com/overview/quick-start)** | Core concepts, CLI reference (`pxt schema`, `pxt service`, `pxt db`), [Cloud deployment](https://docs.pixeltable.com/howto/deployment/cloud), and [HTTP serving](https://docs.pixeltable.com/howto/deployment/serving). |
| **[Interactive Cookbooks (`docs/release/howto/cookbooks/`)](../release/howto/cookbooks/)** | 40+ runnable notebooks covering [Agents & Tool Calling](../release/howto/cookbooks/agents/), [Video](../release/howto/cookbooks/video/), [Audio](../release/howto/cookbooks/audio/), [Images](../release/howto/cookbooks/images/), [Text & Documents](../release/howto/cookbooks/text/), [Semantic Search](../release/howto/cookbooks/search/), [Data Import/Export](../release/howto/cookbooks/data/), and [Core Workflows](../release/howto/cookbooks/core/). Also browseable at [pixeltable.com/notebooks](https://www.pixeltable.com/notebooks). |
| **[Provider Guides (`docs/release/howto/providers/`)](../release/howto/providers/)** | Dedicated notebooks for OpenAI, Anthropic, Google Gemini, RunwayML, vLLM, and local models. |
| **[How-To Hub & Recipes](https://www.pixeltable.com/how-to)** | Copy-pasteable recipes and end-to-end multimodal patterns. |
| **[AI Coding Skill (`pixeltable-skill`)](https://github.com/pixeltable/pixeltable-skill)** | Install with `npx skills add pixeltable/pixeltable-skill` (Claude Code, Cursor, Codex, Windsurf) or point agents at [`https://docs.pixeltable.com/llms-full.txt`](https://docs.pixeltable.com/llms-full.txt). |
| **[MCP Servers](https://github.com/pixeltable/mcp-server-pixeltable-developer)** | [Local Developer MCP](https://github.com/pixeltable/mcp-server-pixeltable-developer) and [Pixeltable Cloud MCP](https://www.pixeltable.com/developers/mcp-cloud). |
| **[Framework Integrations](https://docs.pixeltable.com/integrations/frameworks)** | Official packages for [PydanticAI](https://github.com/pixeltable/pydantic-ai-pixeltable), [LangGraph](https://github.com/pixeltable/langgraph-store-pixeltable), [LangChain](https://github.com/pixeltable/langchain-pixeltable), [LlamaIndex](https://github.com/pixeltable/llama-index-vector-stores-pixeltable), and [Haystack](https://github.com/pixeltable/haystack-pixeltable). |
| **[Open-Source Agent Blueprints](https://github.com/pixeltable)** | [Pixelagent](https://github.com/pixeltable/pixelagent) (agent engineering blueprint) and [Pixelbot](https://github.com/pixeltable/pixelbot) (multimodal infinite-memory chatbot). |

---

## 3. Older Full-Stack & Multi-Service Showcases in This Folder

The remaining directories next to this file are earlier full-stack showcases (predating single-file `app.py` + `pxt schema update` / `pxt service update`):

- **[`multimodal-chat/`](multimodal-chat/)** — Next.js + FastAPI multimodal RAG chat over documents, audio, and video.
- **[`text-and-image-similarity-search-nextjs-fastapi/`](text-and-image-similarity-search-nextjs-fastapi/)** — Next.js + FastAPI cross-modal CLIP text and image search.
- **[`jfk-files-mcp-server/`](jfk-files-mcp-server/)** — Multimodal MCP server indexing the JFK archive (PDFs, audio, images).
- **[`context-aware-discord-bot/`](context-aware-discord-bot/)** — Discord bot with persistent conversation memory and semantic search.
- **[`reddit-agentic-bot/`](reddit-agentic-bot/)** — Reddit agentic Q&A bot.
- **[`intelligence-hub/`](intelligence-hub/)** — Multi-agent workspace for document, audio, and video analysis.
- **[`cli-media-toolkit/`](cli-media-toolkit/)** — Command-line media indexing and search toolkit.
- **[`ai-based-trading-insight-chrome-extension/`](ai-based-trading-insight-chrome-extension/)** — Chrome extension for chart image analysis.
- **[`prompt-engineering-studio-gradio-application/`](prompt-engineering-studio-gradio-application/)** — Gradio studio for prompt experimentation and side-by-side model comparison.
