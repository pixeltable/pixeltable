# Pixeltable Cloud for Cursor

This plugin connects Cursor to Pixeltable's hosted, read-only Cloud MCP server. Sign in through WorkOS AuthKit when Cursor prompts you. Tools can list databases, services, and catalog entries, read table schemas, and fetch at most 25 table rows per call from the Cloud organization selected during sign-in.

The plugin needs no local command or API key. It uses `https://www.pixeltable.com/mcp/cloud` with Cursor's remote MCP OAuth flow. The endpoint must be live before this plugin is submitted to the Cursor Marketplace.

For local catalog inspection, queries, and REPL tools, install the separate [developer MCP](https://github.com/pixeltable/mcp-server-pixeltable-developer). For public documentation search, use [Docs MCP](https://docs.pixeltable.com/mcp). To deploy or change Cloud resources, use the `pxt` CLI.

See the [Cloud MCP setup page](https://pixeltable.com/developers/mcp-cloud) for the endpoint, other clients, privacy policy, terms, and support.
