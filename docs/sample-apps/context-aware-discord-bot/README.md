# 🤖 Context-Aware Discord Bot: Incremental RAG
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/) [![Discord.py Version](https://img.shields.io/badge/discord.py-2.0%2B-blue.svg)](https://github.com/Rapptz/discord.py) [![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)

A Discord bot that remembers and learns from conversations using [Pixeltable](https://github.com/pixeltable/pixeltable) for semantic search and context awareness.

## Command Flow
This architecture provides a way to solve a fundamental challenge in building LLM-based chatbots: maintaining context. Traditional chatbots are limited by fixed context windows and struggle with maintaining long-term memory between conversations.

```mermaid
sequenceDiagram
    participant U as User
    participant B as Discord Bot
    participant F as Message Formatter
    participant P as Pixeltable
    participant MT as Messages Table
    participant MV as Sentences View
    participant CT as Chat Table
    participant H as HuggingFace<br/>E5-large-v2
    participant O as OpenAI<br/>GPT-4

    rect rgb(230, 240, 255)
        Note over U,O: Step 1: Message Storage & Command Processing
        U->>+B: /chat or /search command
        B->>F: Create processing message
        F-->>U: Show processing status
        B->>P: Process command
        rect rgb(250, 251, 248)
            Note right of P: Pixeltable does:<br/>1. Incremetal updates<br/>2. Chunking<br/>3.Versioning/Lineage
            alt Chat Command
                P->>CT: Store question in Chat Table
            else Message Storage
                P->>MT: Store in Messages Table
                MT->>MV: Update Sentences View
            end
        end
    end

    rect rgb(255, 235, 235)
        Note over U,O: Step 2: Semantic Search
        rect rgb(250, 251, 248)
            Note right of P: Pixeltable does:<br/>1. Embedding index management<br/>2. Similarity search<br/>3. Context ranking
            par Embedding Generation
                MV->>H: Get sentence embeddings
                H-->>MV: Return vectors
            and Context Retrieval
                CT->>MV: Find similar sentences
                MV-->>CT: Return relevant context
                Note right of MV: Similarity threshold: 0.3
            end
        end
    end

    rect rgb(235, 255, 235)
        Note over U,O: Step 3: Response Generation
        rect rgb(250, 251, 248)
            Note right of P: Pixeltable does:<br/>1. Context assembly<br/>2. Prompt construction<br/>3. API orchestration
            CT->>P: Build context prompt
            P->>O: Request completion
            O-->>P: Return AI response
        end
    end

    rect rgb(240, 240, 245)
        Note over U,O: Step 4: Response Delivery
        P->>F: Format response & context
        F->>B: Create embed message
        B->>U: Send formatted response
    end
```

## Quick Start

This guide assumes that you have your [Discord server setup](https://github.com/pixeltable/pixeltable/tree/main/examples/context-aware-discord-bot#-discord-setup-guide).

```bash
# 1. Clone and setup
git clone https://github.com/pixeltable/pixeltable.git
cd https://github.com/pixeltable/pixeltable/tree/main/examples/context-aware-discord-bot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables in .env
DISCORD_TOKEN=your-discord-token
OPENAI_API_KEY=your-openai-key

# 4. Run the bot
python bot.py
```
## Key Commands

- `/search [query]`: Find semantically similar messages

- `/chat [question]`: Get AI responses with conversation context

## How It Works?

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%" style="vertical-align: top;">
        <h4>/Search<h4>
        <img src="images/search-command.png" alt="Chat Command Demo" width="100%"/>
        <br>
      </td>
      <td align="center" width="50%" style="vertical-align: top;">
        <h4>/Chat<h4>
        <h6>Pre-Discord Activity<h6>
        <img src="images/initial-discussion.png" alt="Chat Command Demo" width="100%"/>
        <h6>Post-Discord Activity<h6>
        <img src="images/after-discussion.png" alt="Search Command Demo" width="100%"/>
        <br>
      </td>
    </tr>
  </table>
</div>

[Pixeltable](https://github.com/pixeltable/pixeltable) is AI Data infrastructure providing a declarative, incremental approach for multimodal workloads. Transformations, model inference, and custom logic are embedded as computed columns, automatically capturing relationships between data, transformations, and model outputs for full reproducibility.

```python
# 1. Store and index messages at sentence level
messages_view = pxt.create_view(
    'discord_bot.sentences',
    messages_table,
    iterator=StringSplitter.create(
        text=messages_table.content,
        separators='sentence'
    )
)

# 2. Generate embeddings for semantic search
messages_view.add_embedding_index('text', string_embed=get_embeddings)

# 3. Get relevant context for questions
@pxt.query
def get_context(question_text: str):
    sim = messages_view.text.similarity(question_text)
    return messages_view.order_by(sim, asc=False).select(
        text=messages_view.text,
        username=messages_view.username,
        sim=sim
    ).limit(5)

# 4. Generate AI responses with context
chat_table['response'] = openai.chat_completions(
    messages=[
        {
            "role": "system",
            "content": "Answer based on the context provided."
        },
        {
            "role": "user",
            "content": chat_table.prompt
        }
    ],
    model='gpt-4o-mini'
)
```

## Discord Setup Guide

1. Create app at [Discord Developer Portal](https://discord.com/developers/applications)
2. Add bot to your application
3. Enable intents: `Messages`, `Server Members`, `Message Content`
4. Generate invite URL with `bot` and `application.commands` scopes
5. Invite bot to your server

**Required bot permissions:**
- Read Messages/View Channels
- Send Messages & Manage Messages
- Create & Manage Threads
- Add Reactions
- Use Slash Commands

## What else can you do with this?

#### Make this your local AI assistant with multimodal capabilities:

- 🖼️ **Image Understanding**: [Add image search and analysis](https://github.com/pixeltable/pixeltable/tree/main/examples/text-and-image-similarity-search-nextjs-fastapi)
- 🎥 **Video Processing**: [Index and search video content](https://huggingface.co/spaces/Pixeltable/Call-Analysis-AI-Tool)
- 🔊 **Audio Analysis**: [Transcribe and analyze voice messages](https://docs.pixeltable.com/docs/transcribing-and-indexing-audio-and-video)
- 💻 **Local Deployment**: Run entirely on your hardware with [Ollama](https://docs.pixeltable.com/docs/working-with-ollama)/[Llama.cpp](https://docs.pixeltable.com/docs/working-with-llamacpp)

More examples available on our [Hugging Face Spaces](https://huggingface.co/Pixeltable).

#### Enhanced Memory Management

- Split storage into active and archival memory tables
- Add character limits for active memory storage
- Implement importance scoring for messages
- Create automated archival process for old/less relevant messages

#### Improved Search & Retrieval

- Implement cross-table semantic search
- Add relevance scoring for search results
- Create smart context merging from both memory types
- Add result ranking based on recency and relevance

## Support & Resources

- 📚 [Documentation](https://docs.pixeltable.com/)
- 🐛 [Issue Tracker](https://github.com/pixeltable/pixeltable/issues)
- 💬 [Discord Community](https://discord.gg/6MnmFYZJ9N)
- 💡 [Discussions](https://github.com/orgs/pixeltable/discussions)
