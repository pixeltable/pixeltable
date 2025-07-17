---
title: "Welcome to Pixeltable"
description: "The declarative data infrastructure for multimodal AI applications"
---

# Welcome to Pixeltable

Pixeltable is a **declarative data infrastructure** that unifies multimodal AI workflows. Think of it as "Snowflake for AI data" - a single platform that replaces 5+ tools while reducing infrastructure code by 70-90%.

:::info Why Pixeltable?
Store, version, index, and orchestrate multimodal data with simple Python APIs
:::

## What makes Pixeltable different?

### 🔧 Declarative
Define what you want, not how to get it. Let Pixeltable handle the infrastructure.

### ⚡ Incremental  
Only recompute what changed. Save 90% on compute costs with smart caching.

### 🖼️ Multimodal Native
First-class support for images, videos, audio, documents, and embeddings.

### 🤖 AI-First
Built-in integrations with OpenAI, Anthropic, Hugging Face, and more.

## Quick Example

Transform messy data workflows into clean, declarative code:

```python
import pixeltable as pxt

# Create a table
videos = pxt.create_table('videos', {'video': pxt.Video})

# Add AI transformations - computed automatically
videos.add_computed_column(
    transcript=whisper.transcribe(videos.video)
)
videos.add_computed_column(
    summary=openai.chat_completions(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': f'Summarize: {videos.transcript}'}]
    )
)

# Insert data and watch the magic happen
videos.insert({'video': '/path/to/video.mp4'})
# ✨ Transcript and summary computed automatically
```

## What you'll accomplish

1. **Install Pixeltable** - Get up and running in under 2 minutes
2. **Create your first table** - Store and transform multimodal data  
3. **Add AI transformations** - Leverage built-in model integrations
4. **Build production workflows** - Scale with incremental computation

## Ready to get started?

- [**Quick Start →**](./quickstart) Build your first multimodal AI pipeline in 5 minutes
- [**Installation →**](./installation) Install Pixeltable and set up your environment

---

**Join the community:** Get help and share ideas in our [Discord](https://discord.com/invite/QPyqFYx2UN)
