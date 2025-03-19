# JFK MCP Server

On March 28th, 2025, the JFK files were released by Executive Order from President Donald Trump.

There were approximaltely 80,000 pages of documents released. Naturally, this inspired us to build a JFK MCP server that preforms search and analysis on this data.

We examined the PDF files and noticed these were not simple to parse. We are using Mistral OCR to extract the text from the PDFs then storing them in Pixeltable Multimodal Index.

Steps to use:

Add a .env file with the following variables:
```
MISTRAL_API_KEY=your-mistral-api-key
```

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