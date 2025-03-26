# JFK Files MCP Server: A Pixeltable Tutorial

This MCP Server demonstrates how to build a powerful document search and analysis tool for the JFK files using Pixeltable and MCP (Model Context Protocol).

## Background

On March 18th, 2025, approximately 80,000 pages of documents related to the JFK assassination were released by Executive Order from President Donald Trump. This application shows how to process, index, and search these documents using Pixeltable's multimodal database capabilities.

## What You'll Learn

This tutorial demonstrates how to:
- Extract text from PDFs using Mistral OCR
- Store and index documents in Pixeltable
- Create embeddings for semantic search
- Build a search interface using MCP (Model Context Protocol)

## Project Structure

### Key Files

- **load_data.py**: Core data processing script that:
  - Scrapes PDF links from the National Archives website
  - Sets up a Pixeltable table with appropriate columns
  - Processes PDFs using Mistral OCR to extract text
  - Generates document summaries using Mistral AI
  - Creates embeddings for semantic search

- **server.py**: The entry point of the MCP server that:
  - Initializes the Pixeltable database and kicks off the data loading process
  - Sets up a Starlette web application with SSE (Server-Sent Events)
  - Configures routes and endpoints for the MCP protocol

- **tools.py**: Defines the MCP tools available to clients, including:
  - `query_document`: Performs semantic search on document summaries

- **config.py**: Contains configuration parameters for the application:
  - `DIRECTORY`: Specifies the name ('JFK') of the Pixeltable database directory where all data is stored 
  - `MISTRAL_MODEL`: Defines the Mistral AI model ('mistral-small-latest') used for OCR text extraction and document summarization

- **Dockerfile**: Containerizes the application for easy deployment

## Getting Started

### Prerequisites

You'll need a Mistral API key for OCR and text processing. Please visit [Mistral docs](https://docs.mistral.ai/getting-started/quickstart/).

### Setup

Create a `.env` file in the project directory with your API key:
```
MISTRAL_API_KEY=your-mistral-api-key
```

### Run with Python

1. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Install Spacy Model:
```bash
python -m spacy download en_core_web_sm
```

3. Run the server:
```bash
python server.py
```

4. Connect to your favorite MCP client by adding the following URL:
```
http://localhost:8083/sse
```

### Run with Docker

1. Build the Docker container:
```bash
docker build -t jfk-mcp-server .
```

2. Run the Docker container:
```bash
docker run -d -p 8083:8083 --name jfk-mcp-server jfk-mcp-server
```

3. Connect to your favorite MCP client with:
```
http://localhost:8083/sse
```

## How It Works

1. **Data Collection**: The application scrapes PDF links from the National Archives website.
2. **Data Processing**: Mistral OCR extracts text from the PDFs.
3. **Summarization**: Mistral AI generates concise summaries of each document.
4. **Indexing**: Pixeltable creates embeddings for semantic search.
5. **Search Interface**: The MCP server provides a query interface for searching documents.

## Why Pixeltable?

Pixeltable is a declarative data infrastructure for multimodal AI applications that offers significant advantages:

### Simplified Development
- **One Script Solution**: Within a single script (`load_data.py`), we scrape PDF links, process with Mistral OCR, embed the summaries, and store everything for search
- **Declarative Interface**: Define your data transformations once, and they run automatically on new data
- **Minimal Code**: Achieve complex workflows with significantly less code than traditional approaches

### Powerful Data Management
- **Persistent Storage**: All data and computed results are stored and versioned without extra effort
- **Computed Columns**: Define processing workflows that execute on new data
- **Multimodal Support**: Handle images, PDFs, videos, and text seamlessly in one unified interface
- **Incremental Processing**: Documents are uploaded and processed asynchronously

### AI Integration
- **Built-in AI Services**: Ready-to-use integrations with models like Mistral AI
- **Embedding Indices**: Vector search capabilities included out of the box
- **Automatic Orchestration**: Pixeltable handles model execution, ensuring results are stored, indexed, and accessible

### Enterprise-Ready
- **Scalability**: Auto=scales to handle the complexity of processing and indexing large document collections
- **Production-Ready**: Built to handle real-world data volumes
- **MCP Integration**: Seamlessly connects with Model Context Protocol clients

Pixeltable handles the entire workflow from data ingest to search, allowing you to focus on building your application rather than managing infrastructure.

## Learn More

- [MCP Servers Documentation](https://docs.pixeltable.com/docs/cookbooks/mcp/overview)
- [More Pixeltable MCP servers](https://github.com/pixeltable/pixeltable-mcp-server)
- [Pixeltable Documentation](https://docs.pixeltable.com/)