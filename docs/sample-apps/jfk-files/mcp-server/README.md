# JFK MCP Server

On March 18th, 2025, the JFK files were released by Executive Order from President Donald Trump.

There were approximately 80,000 pages of documents released. Naturally, this inspired us to build a JFK MCP server that preforms search and analysis on this data.

We examined the PDF files and noticed these were not simple to parse. We are using Mistral OCR to extract the text from the PDFs then storing them in Pixeltable Multimodal Index.

Checkout more details on our [MCP Servers](https://docs.pixeltable.com/docs/cookbooks/mcp/overview) documentation.

Steps to use:

Add a .env file to this path:
`pixeltable\docs\sample-apps\jfk-files\mcp-server`

The .env file should contain the following variables:
```
MISTRAL_API_KEY=your-mistral-api-key
```
## Run with Python
1. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2. Run the server:
```bash
python server.py
```
3. Connect to you favorite MCP client by adding the following URL to your client:
`http://localhost:8083/sse`

## Run with Docker
1. Build the docker container:
```bash
docker build -t jfk-mcp-server .
```
2. Run the docker container:
```bash
docker run -d -p 8083:8083 --name jfk-mcp-server jfk-mcp-server
```
3. Connect to you favorite MCP client by adding the following URL to your client:
`http://localhost:8083/sse`