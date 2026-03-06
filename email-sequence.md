# Pixeltable Email Sequences

Based on the new positioning ("Collapsing the AI Stack", "Prototype to Production", and multimodal capabilities), here are several email sequences tailored to different audiences and use cases.

---

## Sequence 1: Generic / "The Stack Collapser"
**Target Audience:** CTOs, VP Engineering, Lead AI Engineers, Head of Data
**Goal:** Highlight how Pixeltable eliminates infrastructure complexity and bridges the gap between experimentation and production.

### Email 1: The Hook
**Subject:** Collapsing the multimodal AI stack

Hi {{firstName}},

I'm Pierre, ex-CEO of Noteable (acquired by Confluent). I'm working with the creator of Parquet and Impala on Pixeltable — an open-source Python library that collapses the typical multimodal AI stack into a single import.

Right now, building AI apps means duct-taping together Postgres, Pinecone, S3, Airflow, and LangChain. Pixeltable replaces all of that with a declarative table interface. You define your workflow as computed columns, and Pixeltable automatically handles storage, incremental updates, vector indexing, and caching. 

It natively handles video, audio, images, and documents alongside your structured data.

You can see how it works in our [10-minute tour](https://docs.pixeltable.com/overview/ten-minute-tour) or check out our [GitHub](https://github.com/pixeltable/pixeltable).

Let me know if you're open to a quick chat about your AI infrastructure.

Onward,
Pierre

### Email 2: Prototype to Production
**Subject:** Re: Collapsing the multimodal AI stack

Hi {{firstName}},

One of the biggest pain points we hear from engineering teams is the painful handoff between prototyping (in notebooks/pandas) and production (Airflow/ETL).

With Pixeltable, there is no handoff. You can experiment on a sample of data ephemerally:
`t.sample(5).select(t.text, summary=my_udf(t.text)).collect()`

And when you're happy, you commit it to your production pipeline with one line:
`t.add_computed_column(summary=my_udf(t.text))`

API calls are automatically parallelized, cached, and versioned. No orchestrator needed. If you want to see a production-ready reference architecture, check out our new [Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit).

Thanks,
Pierre 

### Email 3: The Breakup / Traction
**Subject:** Re: Collapsing the multimodal AI stack

Hi {{firstName}}, 

I wanted to send a last note to ensure this didn't get buried.

Teams using Pixeltable are seeing massive reductions in pipeline code (often deleting thousands of lines of Airflow, vector DB sync scripts, and glue code) and zero time lost to dev-prod mismatches. 

All we do right now is help teams leverage the open-source solution as we build out our cloud offering.

If you're dealing with fragmented AI infrastructure, I'd love to show you how Pixeltable can help: https://github.com/pixeltable/pixeltable.

Onward,
Pierre

---

## Sequence 2: Backend for AI Apps & RAG
**Target Audience:** Full Stack AI Developers, Software Engineers building AI features
**Goal:** Position Pixeltable as the ultimate backend for RAG, semantic search, and AI applications.

### Email 1: The Hook
**Subject:** A better backend for your AI apps

Hi {{firstName}},

I'm Pierre, ex-CEO of Noteable (acquired by Confluent). I'm working on Pixeltable, an open-source Python library that acts as the complete backend for AI applications.

Building RAG or semantic search usually requires keeping a database, an embedding model, and a vector store (like Pinecone) perfectly in sync. In Pixeltable, you just add an embedding index to a column:
`table.add_embedding_index('content', string_embed=sentence_transformer.using(...))`

Pixeltable automatically maintains the index whenever data is inserted, updated, or deleted. It also handles document chunking (via built-in iterators) and caches all your LLM API calls automatically.

Check out our [RAG Pipeline Cookbook](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) to see it in action. Open to a quick chat to see if this could simplify your stack?

Onward,
Pierre

### Email 2: Built-in AI Integrations
**Subject:** Re: A better backend for your AI apps

Hi {{firstName}},

Wanted to follow up on my last note. Another reason developers love Pixeltable as an AI backend is that we've abstracted away the boilerplate for over 30+ AI providers (OpenAI, Anthropic, Gemini, Hugging Face, etc.).

You don't need to write custom retry logic, rate-limiting, or caching wrappers. You just define a computed column, and Pixeltable handles the execution and incremental updates as new data flows in. 

If you're building a multimodal app, we also have a [Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit) (FastAPI + React) that gives you a production-ready architecture out of the box.

Thanks,
Pierre

---

## Sequence 3: ML Data Wrangling & Computer Vision
**Target Audience:** Machine Learning Engineers, Computer Vision Engineers, Data Scientists
**Goal:** Highlight Pixeltable's multimodal native types, automatic iteration (frame extraction), and ML export capabilities.

### Email 1: The Hook
**Subject:** Automating multimodal data wrangling

Hi {{firstName}},

I'm Pierre, ex-CEO of Noteable (acquired by Confluent). I'm working with the creator of Parquet and Impala on Pixeltable — an open-source data infrastructure built specifically for multimodal ML workflows.

We noticed CV and ML teams spend way too much time writing boilerplate to extract frames, process audio, and run pre-annotations. Pixeltable natively understands `Image`, `Video`, and `Audio` types. 

You can extract frames from a video dataset into a virtual view with one line:
`frames = pxt.create_view('frames', videos, iterator=frame_iterator(videos.video, fps=1))`

From there, you can add computed columns for object detection (e.g., YOLO) that run incrementally as new videos arrive. 

Check out our [Object Detection tutorial](https://docs.pixeltable.com/howto/use-cases/object-detection-in-videos). Open to a quick chat about your data wrangling workflows?

Onward,
Pierre

### Email 2: Versioning & Exports
**Subject:** Re: Automating multimodal data wrangling

Hi {{firstName}},

One more thing I wanted to share: Pixeltable completely replaces the need for tools like DVC or MLflow for data versioning. 

Every schema change and data insertion is automatically versioned. You get built-in time travel (`table:N`), history tracking, and the ability to revert changes instantly. 

When your dataset is ready for training, you can export it directly to PyTorch (`table.to_pytorch_dataset()`), COCO, or Parquet. We also have a native [Label Studio integration](https://docs.pixeltable.com/howto/using-label-studio-with-pixeltable) for human-in-the-loop workflows.

Thanks,
Pierre

---

## Sequence 4: Agents & MCP
**Target Audience:** AI Agent Builders, AI Researchers
**Goal:** Showcase Pixeltable as the memory and tool-execution layer for agentic workflows.

### Email 1: The Hook
**Subject:** Persistent memory and tool execution for AI Agents

Hi {{firstName}},

I'm Pierre, ex-CEO of Noteable (acquired by Confluent). I'm working on Pixeltable, an open-source library that provides the missing infrastructure for AI agents: persistent memory and declarative tool execution.

Instead of managing conversation history in memory or custom databases, Pixeltable stores interactions in a versioned table. You can give your LLMs access to custom Python functions (`@pxt.udf`) or external APIs, and Pixeltable handles the execution, caching, and dependency tracking automatically.

We recently added full support for the **Model Context Protocol (MCP)**, allowing your agents to seamlessly connect to external tools and datasets.

You can see how we built a multimodal agent in our [Agents & MCP guide](https://docs.pixeltable.com/use-cases/agents-mcp). Open to a quick chat about how you're building agents?

Onward,
Pierre

### Email 2: The Starter Kit
**Subject:** Re: Persistent memory and tool execution for AI Agents

Hi {{firstName}},

To make it even easier to get started, we just released the [Pixeltable Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit). 

It's a production-ready FastAPI + React reference architecture that includes a tool-calling agent wired entirely through Pixeltable computed columns. It handles multimodal uploads, cross-modal search, and agentic workflows out of the box.

Would love to hear what you're building and see if Pixeltable can save you some infrastructure headaches.

Thanks,
Pierre