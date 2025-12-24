===/docs/api-reference===
# REST API Reference

The xAI Enterprise API is a robust, high-performance RESTful interface designed for seamless integration into existing systems.
It offers advanced AI capabilities with full compatibility with the OpenAI REST API.

The base for all routes is at `https://api.x.ai`. For all routes, you have to authenticate with the header `Authorization: Bearer <your xAI API key>`.

***

## POST /v1/chat/completions

API endpoint for POST requests to /v1/chat/completions.

```
Method: POST
Path: /v1/chat/completions
```

***

## POST /v1/responses

API endpoint for POST requests to /v1/responses.

```
Method: POST
Path: /v1/responses
```

***

## GET /v1/responses/\{response\_id}

API endpoint for GET requests to /v1/responses/\{response\_id}.

```
Method: GET
Path: /v1/responses/{response_id}
```

***

## DELETE /v1/responses/\{response\_id}

API endpoint for DELETE requests to /v1/responses/\{response\_id}.

```
Method: DELETE
Path: /v1/responses/{response_id}
```

***

## POST /v1/messages

API endpoint for POST requests to /v1/messages.

```
Method: POST
Path: /v1/messages
```

***

## POST /v1/images/generations

API endpoint for POST requests to /v1/images/generations.

```
Method: POST
Path: /v1/images/generations
```

***

## GET /v1/api-key

API endpoint for GET requests to /v1/api-key.

```
Method: GET
Path: /v1/api-key
```

***

## GET /v1/models

API endpoint for GET requests to /v1/models.

```
Method: GET
Path: /v1/models
```

***

## GET /v1/models/\{model\_id}

API endpoint for GET requests to /v1/models/\{model\_id}.

```
Method: GET
Path: /v1/models/{model_id}
```

***

## GET /v1/language-models

API endpoint for GET requests to /v1/language-models.

```
Method: GET
Path: /v1/language-models
```

***

## GET /v1/language-models/\{model\_id}

API endpoint for GET requests to /v1/language-models/\{model\_id}.

```
Method: GET
Path: /v1/language-models/{model_id}
```

***

## GET /v1/image-generation-models

API endpoint for GET requests to /v1/image-generation-models.

```
Method: GET
Path: /v1/image-generation-models
```

***

## GET /v1/image-generation-models/\{model\_id}

API endpoint for GET requests to /v1/image-generation-models/\{model\_id}.

```
Method: GET
Path: /v1/image-generation-models/{model_id}
```

***

## POST /v1/tokenize-text

API endpoint for POST requests to /v1/tokenize-text.

```
Method: POST
Path: /v1/tokenize-text
```

***

## GET /v1/chat/deferred-completion/\{request\_id}

API endpoint for GET requests to /v1/chat/deferred-completion/\{request\_id}.

```
Method: GET
Path: /v1/chat/deferred-completion/{request_id}
```

***

## POST /v1/completions

API endpoint for POST requests to /v1/completions.

```
Method: POST
Path: /v1/completions
```

***

## POST /v1/complete

API endpoint for POST requests to /v1/complete.

```
Method: POST
Path: /v1/complete
```


===/docs/collections-api/collection===
#### Collections API Reference

# Collection Management

The base URL for `collection` management is shared with [Management API](../management-api) at `https://management-api.x.ai/`.
You have to authenticate using **xAI Management API Key** with the header `Authorization: Bearer <your xAI Management API key>`.

For more details on provisioning xAI Management API key and using Management API, you can visit

***

## POST /v1/collections

API endpoint for POST requests to /v1/collections.

```
Method: POST
Path: /v1/collections
```

***

## GET /v1/collections

API endpoint for GET requests to /v1/collections.

```
Method: GET
Path: /v1/collections
```

***

## GET /v1/collections/\{collection\_id}

API endpoint for GET requests to /v1/collections/\{collection\_id}.

```
Method: GET
Path: /v1/collections/{collection_id}
```

***

## DELETE /v1/collections/\{collection\_id}

API endpoint for DELETE requests to /v1/collections/\{collection\_id}.

```
Method: DELETE
Path: /v1/collections/{collection_id}
```

***

## PUT /v1/collections/\{collection\_id}

API endpoint for PUT requests to /v1/collections/\{collection\_id}.

```
Method: PUT
Path: /v1/collections/{collection_id}
```

***

## POST /v1/collections/\{collection\_id}/documents/\{file\_id}

API endpoint for POST requests to /v1/collections/\{collection\_id}/documents/\{file\_id}.

```
Method: POST
Path: /v1/collections/{collection_id}/documents/{file_id}
```

***

## GET /v1/collections/\{collection\_id}/documents

API endpoint for GET requests to /v1/collections/\{collection\_id}/documents.

```
Method: GET
Path: /v1/collections/{collection_id}/documents
```

***

## GET /v1/collections/\{collection\_id}/documents/\{file\_id}

API endpoint for GET requests to /v1/collections/\{collection\_id}/documents/\{file\_id}.

```
Method: GET
Path: /v1/collections/{collection_id}/documents/{file_id}
```

***

## PATCH /v1/collections/\{collection\_id}/documents/\{file\_id}

API endpoint for PATCH requests to /v1/collections/\{collection\_id}/documents/\{file\_id}.

```
Method: PATCH
Path: /v1/collections/{collection_id}/documents/{file_id}
```

***

## DELETE /v1/collections/\{collection\_id}/documents/\{file\_id}

API endpoint for DELETE requests to /v1/collections/\{collection\_id}/documents/\{file\_id}.

```
Method: DELETE
Path: /v1/collections/{collection_id}/documents/{file_id}
```

***

## GET /v1/collections/\{collection\_id}/documents:batchGet

API endpoint for GET requests to /v1/collections/\{collection\_id}/documents:batchGet.

```
Method: GET
Path: /v1/collections/{collection_id}/documents:batchGet
```


===/docs/collections-api===
# Collections API Reference

The Collections API allows you to manage your Collections `documents` and `collections` programmatically.

The base url for `collection` management is shared with [Management API](management-api) at `https://management-api.x.ai/v1/`. You have to authenticate using **xAI Management API Key** with the header `Authorization: Bearer <your xAI Management API key>`.

For more details on provisioning xAI Management API key and using Management API, you can visit

.

The base url for searching within `collections` is shared with [REST API](api-reference) at `https://api.x.ai`. You have to authenticate using **xAI API Key** with the header `Authorization: Bearer <your xAI API key>`.


===/docs/collections-api/search===
#### Collections API Reference

# Search in Collections

The base url for searching `collections` is shared with [REST API](api-reference) at `https://api.x.ai`. You have to authenticate using **xAI API Key** with the header `Authorization: Bearer <your xAI API key>`.

***

## POST /v1/documents/search

API endpoint for POST requests to /v1/documents/search.

```
Method: POST
Path: /v1/documents/search
```


===/docs/files-api===
#### Files API Reference

# Files API Reference

***

## GET /v1/files

API endpoint for GET requests to /v1/files.

```
Method: GET
Path: /v1/files
```

***

## POST /v1/files

API endpoint for POST requests to /v1/files.

```
Method: POST
Path: /v1/files
```

***

## POST /v1/files/batch\_upload

API endpoint for POST requests to /v1/files/batch\_upload.

```
Method: POST
Path: /v1/files/batch_upload
```

***

## POST /v1/files/batch\_upload/\{batch\_job\_id}:complete

API endpoint for POST requests to /v1/files/batch\_upload/\{batch\_job\_id}:complete.

```
Method: POST
Path: /v1/files/batch_upload/{batch_job_id}:complete
```

***

## GET /v1/files/\{file\_id}

API endpoint for GET requests to /v1/files/\{file\_id}.

```
Method: GET
Path: /v1/files/{file_id}
```

***

## DELETE /v1/files/\{file\_id}

API endpoint for DELETE requests to /v1/files/\{file\_id}.

```
Method: DELETE
Path: /v1/files/{file_id}
```

***

## PUT /v1/files/\{file\_id}

API endpoint for PUT requests to /v1/files/\{file\_id}.

```
Method: PUT
Path: /v1/files/{file_id}
```

***

## POST /v1/files:download

API endpoint for POST requests to /v1/files:download.

```
Method: POST
Path: /v1/files:download
```

***

## POST /v1/files:initialize

API endpoint for POST requests to /v1/files:initialize.

```
Method: POST
Path: /v1/files:initialize
```

***

## POST /v1/files:uploadChunks

API endpoint for POST requests to /v1/files:uploadChunks.

```
Method: POST
Path: /v1/files:uploadChunks
```


===/docs/grpc-reference===
# gRPC Reference

The xAI Enterprise gRPC API is a robust, high-performance gRPC interface designed for seamless integration into existing systems.

The base url for all services is at `api.x.ai`. For all services, you have to authenticate with the header `Authorization: Bearer <your xAI API key>`.

Visit [xAI API Protobuf Definitions](https://github.com/xai-org/xai-proto) to view and download our protobuf definitions.

***

<GrpcDocsSection title="Image" protoFileName={'xai/api/v1/image.proto'} serviceFullName={'xai_api.Image'} />

<GrpcDocsSection title="Auth Service" protoFileName={'xai/api/v1/auth.proto'} serviceFullName={'xai_api.Auth'} />


===/docs/management-api/auth===
## Accounts and Authorization

## POST /auth/teams/\{teamId}/api-keys

API endpoint for POST requests to /auth/teams/\{teamId}/api-keys.

```
Method: POST
Path: /auth/teams/{teamId}/api-keys
```

***

## GET /auth/teams/\{teamId}/api-keys

API endpoint for GET requests to /auth/teams/\{teamId}/api-keys.

```
Method: GET
Path: /auth/teams/{teamId}/api-keys
```

***

## PUT /auth/api-keys/\{api\_key\_id}

API endpoint for PUT requests to /auth/api-keys/\{api\_key\_id}.

```
Method: PUT
Path: /auth/api-keys/{api_key_id}
```

***

## DELETE /auth/api-keys/\{apiKeyId}

API endpoint for DELETE requests to /auth/api-keys/\{apiKeyId}.

```
Method: DELETE
Path: /auth/api-keys/{apiKeyId}
```

***

## GET /auth/api-keys/\{apiKeyId}/propagation

API endpoint for GET requests to /auth/api-keys/\{apiKeyId}/propagation.

```
Method: GET
Path: /auth/api-keys/{apiKeyId}/propagation
```

***

## GET /auth/teams/\{teamId}/models

API endpoint for GET requests to /auth/teams/\{teamId}/models.

```
Method: GET
Path: /auth/teams/{teamId}/models
```

***

## GET /auth/teams/\{teamId}/endpoints

API endpoint for GET requests to /auth/teams/\{teamId}/endpoints.

```
Method: GET
Path: /auth/teams/{teamId}/endpoints
```


===/docs/management-api/billing===
## Billing Management

## GET /v1/billing/teams/\{team\_id}/billing-info

API endpoint for GET requests to /v1/billing/teams/\{team\_id}/billing-info.

```
Method: GET
Path: /v1/billing/teams/{team_id}/billing-info
```

***

## POST /v1/billing/teams/\{team\_id}/billing-info

API endpoint for POST requests to /v1/billing/teams/\{team\_id}/billing-info.

```
Method: POST
Path: /v1/billing/teams/{team_id}/billing-info
```

***

## GET /v1/billing/teams/\{team\_id}/invoices

API endpoint for GET requests to /v1/billing/teams/\{team\_id}/invoices.

```
Method: GET
Path: /v1/billing/teams/{team_id}/invoices
```

***

## GET /v1/billing/teams/\{team\_id}/payment-method

API endpoint for GET requests to /v1/billing/teams/\{team\_id}/payment-method.

```
Method: GET
Path: /v1/billing/teams/{team_id}/payment-method
```

***

## POST /v1/billing/teams/\{team\_id}/payment-method/default

API endpoint for POST requests to /v1/billing/teams/\{team\_id}/payment-method/default.

```
Method: POST
Path: /v1/billing/teams/{team_id}/payment-method/default
```

***

## GET /v1/billing/teams/\{team\_id}/postpaid/invoice/preview

API endpoint for GET requests to /v1/billing/teams/\{team\_id}/postpaid/invoice/preview.

```
Method: GET
Path: /v1/billing/teams/{team_id}/postpaid/invoice/preview
```

***

## GET /v1/billing/teams/\{team\_id}/postpaid/spending-limits

API endpoint for GET requests to /v1/billing/teams/\{team\_id}/postpaid/spending-limits.

```
Method: GET
Path: /v1/billing/teams/{team_id}/postpaid/spending-limits
```

***

## POST /v1/billing/teams/\{team\_id}/postpaid/spending-limits

API endpoint for POST requests to /v1/billing/teams/\{team\_id}/postpaid/spending-limits.

```
Method: POST
Path: /v1/billing/teams/{team_id}/postpaid/spending-limits
```

***

## GET /v1/billing/teams/\{team\_id}/prepaid/balance

API endpoint for GET requests to /v1/billing/teams/\{team\_id}/prepaid/balance.

```
Method: GET
Path: /v1/billing/teams/{team_id}/prepaid/balance
```

***

## POST /v1/billing/teams/\{team\_id}/prepaid/top-up

API endpoint for POST requests to /v1/billing/teams/\{team\_id}/prepaid/top-up.

```
Method: POST
Path: /v1/billing/teams/{team_id}/prepaid/top-up
```

***

## POST /v1/billing/teams/\{team\_id}/usage

API endpoint for POST requests to /v1/billing/teams/\{team\_id}/usage.

```
Method: POST
Path: /v1/billing/teams/{team_id}/usage
```


===/docs/management-api===
## Overview

The Management API serves as a dedicated interface to the xAI platform, empowering developers and teams to
programmatically manage their xAI API teams.

For example, users can provision their API key, handle access controls,
and perform team-level operations like creating, listing, updating, or deleting keys and associated access control lists
(ACLs). This API also facilitates oversight of billing aspects, including monitoring prepaid credit balances and usage
deductions, ensuring seamless scalability and cost transparency for Grok model integrations.

To get started, go to [xAI Console](https://console.x.ai). On users page, make sure your xAI account has
`Management Keys` Read + Write permission, and obtain your Management API key on the settings page. If you don't see
any of these options, please ask your team administrator to enable the appropriate permissions.


===/docs/introduction===
#### Introduction

# What is Grok?

Grok is a family of Large Language Models (LLMs) developed by [xAI](https://x.ai).

Inspired by the Hitchhiker's Guide to the Galaxy, Grok is a maximally truth-seeking AI that provides insightful, unfiltered truths about the universe.

xAI offers an API for developers to programmatically interact with our Grok [models](models). The same models power our consumer facing services such as [Grok.com](https://grok.com), the [iOS](https://apps.apple.com/us/app/grok/id6670324846) and [Android](https://play.google.com/store/apps/details?id=ai.x.grok) apps, as well as [Grok in X experience](https://grok.x.com).

## What is the xAI API? How is it different from Grok in other services?

The xAI API is a toolkit for developers to integrate xAI's Grok models into their own applications, the xAI API provides the building blocks to create new AI experiences.

To get started building with the xAI API, please head to [The Hitchhiker's Guide to Grok](tutorial).

## xAI API vs Grok in other services

Because these are separate offerings, your purchase on X (e.g. X Premium) won't affect your service status on xAI API, and vice versa.

This documentation is intended for users using xAI API.


===/docs/models===
#### Key Information

# Models and Pricing

An overview of our models' capabilities and their associated pricing.

## Model Pricing


When moving from `grok-3`/`grok-3-mini` to `grok-4`, please note the following differences:

## Tools Pricing

Requests which make use of xAI provided [server-side tools](guides/tools/overview) are priced based on two components: **token usage** and **server-side tool invocations**. Since the agent autonomously decides how many tools to call, costs scale with query complexity.

### Token Costs

All standard token types are billed at the [rate](/docs/models#model-pricing) for the model used in the request:

* **Input tokens**: Your query and conversation history
* **Reasoning tokens**: Agent's internal thinking and planning
* **Completion tokens**: The final response
* **Image tokens**: Visual content analysis (when applicable)
* **Cached prompt tokens**: Prompt tokens that were served from cache rather than recomputed

### Tool Invocation Costs

| Tool | Cost per 1,000 calls | Description |
|------|---------------------|-------------|
| **[Web Search](/docs/guides/tools/search-tools)** | $5 | Internet search and page browsing |
| **[X Search](/docs/guides/tools/search-tools)** | $5 | X posts, users, and threads |
| **[Code Execution](/docs/guides/tools/code-execution-tool)** | $5 | Python code execution environment |
| **[Document Search](/docs/guides/files)** | $5 | Search through uploaded files and documents |
| **[View Image](/docs/guides/tools/search-tools#parameter-enable_image_understanding-supported-by-web-search-and-x-search)** | Token-based only | Image analysis within search results |
| **[View X Video](/docs/guides/tools/search-tools#parameter-enable_video_understanding-supported-by-x-search)** | Token-based only | Video analysis within X posts |
| **[Collections Search](/docs/guides/tools/collections-search-tool)** | $2.50 | Knowledge base search using xAI Collections |
| **[Remote MCP Tools](/docs/guides/tools/remote-mcp-tools)** | Token-based only | Custom MCP tools |

For the view image and view x video tools, you will not be charged for the tool invocation itself but will be charged for the image tokens used to process the image or video.

For Remote MCP tools, you will not be charged for the tool invocation but will be charged for any tokens used.

For more information on using Tools, please visit [our guide on Tools](guides/tools/overview).

## Live Search Pricing

The advanced agentic search capabilities powering grok.com are generally available in the new [**agentic tool calling API**](/docs/guides/tools/overview), and the Live Search API but will be deprecated by December 15, 2025.

Live Search costs $25 per 1,000 sources requested, each source used (Web, X, News, RSS) in a request counts toward the usage. That means a search using 4 sources costs $0.10 while a search using 1 source is $0.025. A source (e.g. Web) may return multiple citations, but you will be charged for only one source.

The number of sources used can be found in the `response` object, which contains a field called `response.usage.num_sources_used`.

For more information on using Live Search, visit our [guide on Live Search](guides/live-search) or look for `search_parameters` parameter on [API Reference - Chat Completions](api-reference#chat-completions).

## Documents Search Pricing

For users using our Collections API and Documents Search, the following pricing applies:

## Usage Guidelines Violation Fee

A rare occurrence for most users, when your request is deemed to be in violation of our usage guideline by our system, we will charge a $0.05 per request usage guidelines violation fee.

## Additional Information Regarding Models

* **No access to realtime events without Live Search enabled**
  * Grok has no knowledge of current events or data beyond what was present in its training data.
  * To incorporate realtime data with your request, please use [Live Search](guides/live-search) function, or pass any realtime data as context in your system prompt.
* **Chat models**
  * No role order limitation: You can mix `system`, `user`, or `assistant` roles in any sequence for your conversation context.
* **Image input models**
  * Maximum image size: `20MiB`
  * Maximum number of images: No limit
  * Supported image file types: `jpg/jpeg` or `png`.
  * Any image/text input order is accepted (e.g. text prompt can precede image prompt)

The knowledge cut-off date of Grok 3 and Grok 4 is November, 2024.

## Model Aliases

Some models have aliases to help users automatically migrate to the next version of the same model. In general:

* `<modelname>` is aliased to the latest stable version.
* `<modelname>-latest` is aliased to the latest version. This is suitable for users who want to access the latest features.
* `<modelname>-<date>` refers directly to a specific model release. This will not be updated and is for workflows that demand consistency.

For most users, the aliased `<modelname>` or `<modelname>-latest` are recommended, as you would receive the latest features automatically.

## Billing and Availability

Your model access might vary depending on various factors such as geographical location, account limitations, etc.

For how the **bills are charged**, visit [Manage Billing](key-information/billing) for more information.

For the most up-to-date information on **your team's model availability**, visit [Models Page](https://console.x.ai/team/default/models) on xAI Console.

## Model Input and Output

Each model can have one or multiple input and output capabilities.
The input capabilities refer to which type(s) of prompt can the model accept in the request message body.
The output capabilities refer to which type(s) of completion will the model generate in the response message body.

This is a prompt example for models with `text` input capability:

```json
[
  {
    "role": "system",
    "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
  },
  {
    "role": "user",
    "content": "What is the meaning of life, the universe, and everything?"
  }
]
```

This is a prompt example for models with `text` and `image` input capabilities:

```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,<base64_image_string>",
          "detail": "high"
        }
      },
      {
        "type": "text",
        "text": "Describe what's in this image."
      }
    ]
  }
]
```

This is a prompt example for models with `text` input and `image` output capabilities:

```json
// The entire request body
{
  "model": "grok-4",
  "prompt": "A cat in a tree",
  "n": 4
}
```

## Context Window

The context window determines the maximum amount of tokens accepted by the model in the prompt.

For more information on how token is counted, visit [Consumption and Rate Limits](key-information/consumption-and-rate-limits).

If you are sending the entire conversation history in the prompt for use cases like chat assistant, the sum of all the prompts in your conversation history must be no greater than the context window.

## Cached prompt tokens

Trying to run the same prompt multiple times? You can now use cached prompt tokens to incur less cost on repeated prompts. By reusing stored prompt data, you save on processing expenses for identical requests. Enable caching in your settings and start saving today!

The caching is automatically enabled for all requests without user input. You can view the cached prompt token consumption in [the `"usage"` object](key-information/consumption-and-rate-limits#checking-token-consumption).

For details on the pricing, please refer to the pricing table above, or on [xAI Console](https://console.x.ai).


===/docs/overview===
#### Getting started

# Welcome

Welcome to the xAI developer docs! Our API makes it easy to harness Grok's intelligence in your projects. Grok is our flagship AI model designed to deliver truthful, insightful answers.

## Jump right in

Are you a non-developer or simply looking for our consumer services? Visit [Grok.com](https://grok.com) or download one of the [iOS](https://apps.apple.com/us/app/grok/id6670324846) or [Android](https://play.google.com/store/apps/details?id=ai.x.grok) apps. See our [Comparison Table](introduction#xai-api-vs-grok-in-other-services) for the differences.

## Questions and feedback

If you have any questions or feedback, feel free to email us at support@x.ai.

Happy Grokking! 😎


===/docs/release-notes===
#### What's New?

# Release Notes

Stay up to date with the latest changes to the xAI API.

# November 2025

### Grok 4.1 Fast is available in Enterprise API

You can now use Grok 4.1 Fast in the [xAI Enterprise API](https://x.ai/api). For more details, check out [our blogpost](https://x.ai/news/grok-4-1-fast).

### Agent tools adapt to Grok 4.1 Fast models and tool prices dropped

* You can now use Grok 4.1 Fast models with the agent tools, check out the [documentation of agent tools](/docs/guides/tools/overview) to get started.
* The price of agent tools drops by up to 50% to no more than $5 per 1000 successful calls, see the new prices at [the pricing page](/docs/models#tools-pricing).

### Files API is generally available

You can now upload files and use them in chat conversations with the Files API. For more details, check out [our guide on Files](/docs/guides/files).

### New Tools Available

* **Collections Search Tool**: You can now search through uploaded knowledge bases (collections) in chat conversations via the API. For more details, check out the [docs](/docs/guides/tools/collections-search-tool).
* **Remote MCP Tools**: You can now use tools from remote MCP servers in chat conversations via the API. For more details, check out the [docs](/docs/guides/tools/remote-mcp-tools).
* **Mixing client-side and server-side tools**: You can now mix client-side and server-side tools in the same chat conversation. For more details, check out the [docs](/docs/guides/tools/advanced-usage#mixing-server-side-and-client-side-tools).

# October 2025

### Tools are now generally available

New agentic server-side tools including `web_search`, `x_search` and `code_execution` are available. For more details, check out [our guide on using Tools](/docs/guides/tools/overview).

# September 2025

### Responses API is generally available

You can now use our stateful Responses API to process requests.

# August 2025

### Grok Code Fast 1 is released

We have released our first Code Model to be used with code editors.

### Collections API is released

You can upload files, create embeddings, and use them for inference with our Collections API.

# July 2025

### Grok 4 is released

You can now use Grok 4 via our API or on https://grok.com.

# June 2025

### Management API is released

You can manage your API keys via Management API at
`https://management-api.x.ai`.

# May 2025

### Cached prompt is now available

You can now use cached prompt to save on repeated prompts. For
more info, see [models](models).

### Live Search is available on API

Live search is now available on API. Users can generate
completions with queries on supported data sources.

# April 2025

### Grok 3 models launch on API

Our latest flagship `Grok 3` models are now generally available via
the API. For more info, see [models](models).

# March 2025

### Image Generation Model available on API

The image generation model is available on API. Visit
[Image Generations](/docs/guides/image-generations) for more details on using the model.

# February 2025

### Audit Logs

Team admins can now view audit logs on [console.x.ai](https://console.x.ai).

# January 2025

### Docs Dark Mode Released dark mode support on docs.x.ai

### Status Page Check service statuses across all xAI products at

[status.x.ai](https://status.x.ai/).

# December 2024

### Replit & xAI

Replit Agents can now integrate with xAI! Start empowering your agents with Grok.
Check out the [announcement](https://x.com/Replit/status/1874211039258333643) for more information.

### Tokenizer Playground Understanding tokens can be hard. Check out

[console.x.ai](https://console.x.ai) to get a better understanding of what counts as a token.

### Structured Outputs We're excited to announce that Grok now supports structured outputs. Grok can

now format responses in a predefined, organized format rather than free-form text. 1. Specify the
desired schema

```
{
    "name": "movie_response",
    "schema": {
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "rating": { "type": "number" },
        },
        "required": [ "title", "rating" ],
        "additionalProperties": false
    },
    "strict": true
}
```

2. Get the desired data

```
{
  "title": "Star Wars",
  "rating": 8.6
}
```

Start building more reliable applications. Check out the [docs](guides/structured-outputs#structured-outputs) for more information.

### Released the new grok-2-1212 and grok-2-vision-1212 models A month ago, we launched the public

beta of our enterprise API with grok-beta and grok-vision-beta. We’re adding [grok-2-1212 and
grok-2-vision-1212](https://docs.x.ai/docs/models), offering better accuracy, instruction-following,
and multilingual capabilities.

# November 2024

### LangChain & xAI Our API is now available through LangChain! - Python Docs:

http://python.langchain.com/docs/integrations/providers/xai/ - Javascript Docs:
http://js.langchain.com/docs/integrations/chat/xai/

What are you going to build?

### API Public Beta Released We are happy to announce the immediate availability of our API, which

gives developers programmatic access to our Grok series of foundation models. To get started, head
to [console.x.ai](https://console.x.ai/) and sign up to create an account. We are excited to see
what developers build using Grok.


===/docs/tutorial===
#### Getting Started

# The Hitchhiker's Guide to Grok

Welcome! In this guide, we'll walk you through the basics of using the xAI API.

## Step 1: Create an xAI Account

First, you'll need to create an xAI account to access xAI API. Sign up for an account [here](https://accounts.x.ai/sign-up?redirect=cloud-console).

Once you've created an account, you'll need to load it with credits to start using the API.

## Step 2: Generate an API Key

Create an API key via the [API Keys Page](https://console.x.ai/team/default/api-keys) in the xAI API Console.

After generating an API key, we need to save it somewhere safe! We recommend you export it as an environment variable in your terminal or save it to a `.env` file.

```bash
export XAI_API_KEY="your_api_key"
```

## Step 3: Make your first request

With your xAI API key exported as an environment variable, you're ready to make your first API request.

Let's test out the API using `curl`. Paste the following directly into your terminal.

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "messages": [
        {
            "role": "system",
            "content": "You are Grok, a highly intelligent, helpful AI assistant."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        }
    ],
    "model": "grok-4",
    "stream": false
}'
```

## Step 4: Make a request from Python or Javascript

As well as a native xAI Python SDK, the majority our APIs are fully compatible with the OpenAI and Anthropic SDKs. For example, we can make the same request from Python or Javascript like so:

```pythonXAI
# In your terminal, first run:
# pip install xai-sdk

import os

from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

chat = client.chat.create(model="grok-4")
chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
chat.append(user("What is the meaning of life, the universe, and everything?"))

response = chat.sample()
print(response.content)
```

```pythonOpenAISDK
# In your terminal, first run:

# pip install openai

import os
import httpx
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

completion = client.chat.completions.create(
    model="grok-4",
    messages=[
        {
            "role": "system",
            "content": "You are Grok, a highly intelligent, helpful AI assistant."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        },
    ],
)

print(completion.choices[0].message.content)
```

```javascriptAISDK
// In your terminal, first run:
// pnpm add ai @ai-sdk/xai

import { xai } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
    model: xai('grok-4'),
    system: 'You are Grok, a highly intelligent, helpful AI assistant.',
    prompt: 'What is the meaning of life, the universe, and everything?',
});

console.log(result.text);
```

```javascriptOpenAISDK
// In your terminal, first run:
// npm install openai

import OpenAI from 'openai';

const client = new OpenAI({
    apiKey: "your_api_key",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const completion = await client.chat.completions.create({
    model: "grok-4",
    messages: [
        {
            role: "system",
            content:
            "You are Grok, a highly intelligent, helpful AI assistant.",
        },
        {
            role: "user",
            content:
            "What is the meaning of life, the universe, and everything?",
        },
    ],
});

console.log(completion.choices[0].message.content);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "messages": [
        {
            "role": "system",
            "content": "You are Grok, a highly intelligent, helpful AI assistant."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        }
    ],
    "model": "grok-4"
}'
```

Certain models also support [Structured Outputs](guides/structured-outputs), which allows you to enforce a schema for the LLM output.

For an in-depth guide about using Grok for text responses, check out our [Chat Guide](guides/chat).

## Step 5: Use Grok to analyze images

Certain grok models can accept both text AND images as an input. For example:

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user, image

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

chat = client.chat.create(model="grok-4")
chat.append(
    user(
        "What's in this image?",
        image("https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png")
    )
)

response = chat.sample()
print(response.content)
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
image_url = "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png"

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": "What's in this image?",
            },
        ],
    },
]

completion = client.chat.completions.create(
    model="grok-4",
    messages=messages,
)
print(completion.choices[0].message.content)
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { generateText } from 'ai';

const imageUrl =
'https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png';

const result = await generateText({
    model: xai('grok-4'),
    messages: [
        {
            role: 'user',
            content: [
                { type: 'image', image: imageUrl },
                { text: "What's in this image?", type: 'text' },
            ],
        },
    ],
});

console.log(result.text);
```

```javascriptOpenAISDK
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.XAI_API_KEY,
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const image_url =
"https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png";

const completion = await client.chat.completions.create({
    model: "grok-4",
    messages: [
        {
            role: "user",
            content: [
                {
                    type: "image_url",
                    image_url: {
                        url: image_url,
                        detail: "high",
                    },
                },
                {
                    type: "text",
                    text: "What's in this image?",
                },
            ],
        },
    ],
});

console.log(completion.choices[0].message.content);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image"
                    }
                ]
            }
        ],
        "model": "grok-4"
}'
```

And voila! Grok will tell you exactly what's in the image:

> This image is a photograph of a region in space, specifically a part of the Carina Nebula, captured by the James Webb Space Telescope. It showcases a stunning view of interstellar gas and dust, illuminated by young, hot stars. The bright points of light are stars, and the colorful clouds are composed of various gases and dust particles. The image highlights the intricate details and beauty of star formation within a nebula.

To learn how to use Grok vision for more advanced use cases, check out our [Image Understanding Guide](guides/image-understanding).

## Monitoring usage

As you use your API key, you will be charged for the number of tokens used. For an overview, you can monitor your usage on the [xAI Console Usage Page](https://console.x.ai/team/default/usage).

If you want a more granular, per request usage tracking, the API response includes a usage object that provides detail on prompt (input) and completion (output) token usage.

```json
"usage": {
    "prompt_tokens":37,
    "completion_tokens":530,
    "total_tokens":800,
    "prompt_tokens_details": {
        "text_tokens":37,
        "audio_tokens":0,
        "image_tokens":0,
        "cached_tokens":8
    },
    "completion_tokens_details": {
        "reasoning_tokens":233,
        "audio_tokens":0,
        "accepted_prediction_tokens":0,
        "rejected_prediction_tokens":0
    },
    "num_sources_used":0
}
```

If you send requests too frequently or with long prompts, you might run into rate limits and get an error response. For more information, read [Consumption and Rate Limits](consumption-and-rate-limits).

## Next steps

Now you have learned the basics of making an inference on xAI API. Check out [Models](models) page to start building with one of our latest models.


===/docs/grok-business/grok-user-guide===
#### Grok Business / Enterprise

# Grok.com User Guide

**Grok Business provides dedicated workspaces for personal and team use, with enhanced privacy and sharing controls.** Switch between workspaces to access team-specific features and ensure your conversations are protected under business plan terms.

A team workspace offers:

* Privacy guarantees as outlined in xAI's [terms of service](https://x.ai/legal/terms-of-service-enterprise).
* Full benefits of SuperGrok (or SuperGrok Heavy for upgraded licenses).
* Secure sharing of conversations limited to active team members.

## Workspaces Overview

Grok Business features two types of workspaces:

* **Personal Workspace:** For individual use, available unless disabled by your organization.
* **Team Workspace:** For collaborative work within your team, accessible only with an active license.

To switch between workspaces, use the workspace selector in the bottom left navigation on grok.com. Ensure you are in the correct workspace before starting new conversations.

&#x20;You can only access the team workspace when you have an
active license. If you lack access, contact your team admin.

## Privacy and Benefits

In your team workspace, enjoy enterprise-grade privacy protections as detailed in xAI's [terms of service](https://x.ai/legal/terms-of-service-enterprise). This includes data handling and, for the Enterprise tier, custom retention policies tailored for business use.

Additionally, unlock the full capabilities of SuperGrok, including higher usage quotas and advanced features. If your organization has an upgraded license, you may access SuperGrok Heavy for even more powerful performance.

Some users may not see a personal workspace. This indicates your organization has disabled
personal workspaces via an enterprise license. To enable or disable personal workspaces, reach out
to xAI sales for an Enterprise plan.

## Sharing Conversations

Sharing is restricted to your team for security:

* Share conversations only with team members who have active licenses.
* Share links are only accessible to licensed team members.
* If sent to non-team members or unlicensed team members, the link will not open.

To share a conversation:

1. Open the conversation in your team workspace.
2. Click the share button and select team members.
3. Generate and distribute the secure link.

View all shared conversations in your history at [https://grok.com/history?tab=shared-with-me](https://grok.com/history?tab=shared-with-me).

## Activating Your License

To activate or manage your license:

1. Visit your Grok Business overview at [console.x.ai](https://console.x.ai).
2. Press "Assign license" and select your license type.
3. If you encounter access issues or lack permissions, contact your team admin for assistance.

Once activated, your team workspace will become available on grok.com.

&#x20;For white-glove support and Enterprise features, contact xAI sales at .


===/docs/grok-business/management===
#### Grok Business / Enterprise

# License & User Management

**The Grok Business overview page at [console.x.ai](https://console.x.ai) is your central hub for handling team licenses and user invitations.** As a team admin or user with appropriate permissions, you can buy licenses, invite new members, and manage access to ensure smooth collaboration.

Access this page by logging into [console.x.ai](https://console.x.ai) and navigating to the overview section. Note that actions like purchasing or provisioning require specific permissions—see the [Permissions](#permissions-and-troubleshooting) section for details.

## Purchasing Licenses

Expand your team's capabilities by buying additional licenses directly from the overview page.

Available license types:

* **SuperGrok:** Standard business access with enhanced quotas and features.
* **SuperGrok Heavy:** Upgraded performance for demanding workloads.

To purchase:

1. On the overview page, select the license type and quantity.
2. Enter payment details if prompted <em>(requires billing read-write permissions)</em>.
3. Confirm the purchase—licenses will be added to your available pool for assignment.

Purchased licenses become immediately available for provisioning to users.

&#x20;Ensure your team's billing is set up correctly to avoid
interruptions. Visit [Billing Settings](https://console.x.ai/team/default/billing) for more
details.

## Inviting Users

Invite new team members to join your Grok Business workspace with a simple email invitation process.

To invite:

1. On the overview page, click "Invite users to Grok Business".
2. Enter the users' email addresses.
3. Select a license type to auto-provision upon acceptance <em>(requires team read-write permissions)</em>.
4. Send the invitation—the user will receive an email with a link to activate their account.

Invited users gain access to the team workspace and basic team read permissions. (the latter is to allow for sharing conversations with your team members)

View invited users in the "Pending invitations" list on the overview page. As long as you have unassigned licenses available, they will be automatically provisioned when the user accepts.

## Assigning and Revoking Licenses

Once licenses are purchased or available, assign them to users for full team workspace access.

To assign:

1. From the overview page, select a user from your team list.
2. Choose an available license and assign it—access activates immediately.

To revoke:

1. Click the "..." for the user and choose "Unassign License" from the dropdown.
2. Confirm the action—the license returns to your available pool, and the user's will no longer have access to your team's workspace.

Revocations take effect instantly, but ensure you communicate changes to affected users.

&#x20;Revoking a license removes team workspace access. Users will
retain personal workspace functionality.

## Canceling Licenses

Reduce your team's commitment by canceling unused licenses.

To cancel:

1. On the overview page, select the license type and quantity to cancel.
2. Submit the cancellation request <em>(requires billing read-write permissions)</em>.

Cancellations may take a few days to process, and eligible refunds will be issued to your billing method. Canceled licenses are removed from your pool once processed.

## Permissions and Troubleshooting

Most management actions require specific role-based permissions:

* **Billing Read-Write:** Needed to purchase or cancel licenses.
* **Team Read-Write:** Needed to invite users or assign/revoke licenses.

These are typically granted only to team admins. If you lack permissions:

* Contact your team admin to request actions like license assignment or purchases.
* Admins can adjust permissions via the overview page's role settings.

If you encounter issues, such as invitations not provisioning due to insufficient licenses, purchase more or revoke unused ones first.

&#x20;For white-glove support, Enterprise upgrades, or permission issues, contact xAI sales at .


===/docs/grok-business/organization===
#### Grok Business / Enterprise

# Organization Management

**Organizations provide a higher-level governance structure for enterprise customers, encompassing multiple console teams under unified IT controls.** Available only to Enterprise tier subscribers, organizations enable centralized management of users, teams, and security features like SSO.

Access the organization dashboard by visiting [console.x.ai/organization](https://console.x.ai/organization). This page is restricted to organization admins.

&#x20;Organizations are exclusive to the Enterprise tier. Contact xAI
sales to upgrade if needed.

## Understanding Organizations

An organization acts as an overarching entity that groups related console teams, ideal for large enterprises with multiple business units or departments.

Key features:

* **Domain Association:** Link your organization to a specific email domain (e.g., @yourcompany.com). Any user signing up or logging in with an email from this domain is automatically associated with the organization.
* **User Visibility:** Organization admins can view a comprehensive list of all associated users across teams on the `/organization` page.
* **Team Association:** Teams created by organization members are automatically linked to the organization and displayed in the dashboard for oversight.

This structure supports a multi-team architecture, allowing independent Grok Business or API teams while maintaining centralized governance, such as uniform access controls and auditing.

## Viewing Users and Teams

Monitor your enterprise's activity from a single pane of glass.

To view users:

1. Navigate to [console.x.ai/organization](https://console.x.ai/organization).
2. Scroll to the "Users" section for a list of all domain-associated users, including their team affiliations and access status.

To view teams:

1. In the same dashboard, access the "Teams" section.
2. Review associated console teams, their members, and high-level usage metrics.

Use these views to ensure compliance, spot inactive accounts, or identify growth needs.

## Setting Up SSO

Secure and streamline logins by integrating Single Sign-On (SSO) with your preferred Identity Provider (IdP).

To configure SSO:

1. On the `/organization` page, click "Configure SSO".
2. Choose your IdP from the supported list (e.g., Okta, Azure AD, Google Workspace).
3. Follow the self-guided, IdP-specific instructions provided—each includes step-by-step setup, metadata exchange, and attribute mapping details.
4. Save your configuration and test SSO to confirm the functionality.

SSO setup is straightforward and tailored to common providers, ensuring quick deployment.

## Activating SSO and User Impact

Once configured, SSO will be activated and enforced organization-wide.

Post-activation:

* Users must log in via SSO on their next access.
* If a user selects "Log in with email" and enters a domain-associated address, (e.g., @yourcompany.com) the system automatically detects it and redirects to your IdP for authentication.
* Non-domain emails (e.g., @differentcompany.com) fall back to standard login methods.

This ensures seamless, secure access without disrupting workflows.

&#x20;Notify your users in advance about the SSO rollout to minimize
support queries.

## Need Help?

For assistance with organization setup, SSO troubleshooting, or Enterprise features, contact xAI sales at [x.ai/grok/business/enquire](https://x.ai/grok/business/enquire).


===/docs/guides/async===
#### Guides

# Asynchronous Requests

When working with the xAI API, you may need to process hundreds or even thousands of requests. Sending these requests sequentially can be extremely time-consuming.

To improve efficiency, you can use `AsyncClient` from `xai_sdk` or `AsyncOpenAI` from `openai`, which allows you to send multiple requests concurrently. The example below is a Python script demonstrating how to use `AsyncClient` to batch and process requests asynchronously, significantly reducing the overall execution time:

The xAI API does not currently offer a batch API.

## Rate Limits

Adjust the `max_concurrent` param to control the maximum number of parallel requests.

You are unable to concurrently run your requests beyond the rate limits shown in the API console.

```pythonXAI
import asyncio

from xai_sdk import AsyncClient
from xai_sdk.chat import Response, user

async def main():
    client = AsyncClient(
        api_key=os.getenv("XAI_API_KEY"),
        timeout=3600, # Override default timeout with longer timeout for reasoning models
    )

    model = "grok-4"
    requests = [
        "Tell me a joke",
        "Write a funny haiku",
        "Generate a funny X post",
        "Say something unhinged",
    ]


    # Define a semaphore to limit concurrent requests (e.g., max 2 concurrent requests at a time)
    max_in_flight_requests = 2
    semaphore = asyncio.Semaphore(max_in_flight_requests)

    async def process_request(request) -> Response:
        async with semaphore:
            print(f"Processing request: {request}")
            chat = client.chat.create(model=model, max_tokens=100)
            chat.append(user(request))
            return await chat.sample()

    tasks = []
    for request in requests:
        tasks.append(process_request(request))

    responses = await asyncio.gather(*tasks)
    for i, response in enumerate(responses):
        print(f"Total tokens used for response {i}: {response.usage.total_tokens}")

if **name** == "**main**":
asyncio.run(main())
```

```pythonOpenAISDK
import asyncio
import os
import httpx
from asyncio import Semaphore
from typing import List

from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0) # Override default timeout with longer timeout for reasoning models
)

async def send_request(sem: Semaphore, request: str) -> dict:
"""Send a single request to xAI with semaphore control."""
# The 'async with sem' ensures only a limited number of requests run at once
    async with sem:
        return await client.chat.completions.create(
            model="grok-4",
            messages=[{"role": "user", "content": request}]
        )

async def process_requests(requests: List[str], max_concurrent: int = 2) -> List[dict]:
"""Process multiple requests with controlled concurrency."""
    # Create a semaphore that limits how many requests can run at the same time # Think of it like having only 2 "passes" to make requests simultaneously
    sem = Semaphore(max_concurrent)

    # Create a list of tasks (requests) that will run using the semaphore
    tasks = [send_request(sem, request) for request in requests]

    # asyncio.gather runs all tasks in parallel but respects the semaphore limit
    # It waits for all tasks to complete and returns their results
    return await asyncio.gather(*tasks)

async def main() -> None:
"""Main function to handle requests and display responses."""
    requests = [
        "Tell me a joke",
        "Write a funny haiku",
        "Generate a funny X post",
        "Say something unhinged"
    ]

    # This starts processing all asynchronously, but only 2 at a time
    # Instead of waiting for each request to finish before starting the next,
    # we can have 2 requests running at once, making it faster overall
    responses = await process_requests(requests)

    # Print each response in order
    for i, response in enumerate(responses):
        print(f"# Response {i}:")
        print(response.choices[0].message.content)

if **name** == "**main**":
asyncio.run(main())
```


===/docs/guides/chat===
#### Guides

# Chat

Text in, text out. Chat is the most popular feature on the xAI API, and can be used for anything from summarizing articles, generating creative writing, answering questions, providing customer support, to assisting with coding tasks.

## Prerequisites

* xAI Account: You need an xAI account to access the API.
* API Key: Ensure that your API key has access to the chat endpoint and the chat model is enabled.

If you don't have these and are unsure of how to create one, follow [the Hitchhiker's Guide to Grok](../tutorial).

You can create an API key on the [xAI Console API Keys Page](https://console.x.ai/team/default/api-keys).

Set your API key in your environment:

```bash
export XAI_API_KEY="your_api_key"
```

## A Basic Chat Completions Example

You can also stream the response, which is covered in [Streaming Response](streaming-response).

The user sends a request to the xAI API endpoint. The API processes this and returns a complete response.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

chat = client.chat.create(model="grok-4")
chat.append(system("You are a PhD-level mathematician."))
chat.append(user("What is 2 + 2?"))

response = chat.sample()
print(response.content)
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

completion = client.chat.completions.create(
    model="grok-4",
    messages=[
        {"role": "system", "content": "You are a PhD-level mathematician."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
)

print(completion.choices[0].message)
```

```javascriptOpenAISDK
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const completion = await client.chat.completions.create({
    model: "grok-4",
    messages: [
        {
            role: "system",
            content: "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."
        },
        {
            role: "user",
            content: "What is the meaning of life, the universe, and everything?"
        },
    ],
});
console.log(completion.choices[0].message);
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  system:
    "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy.",
  prompt: 'What is the meaning of life, the universe, and everything?',
});

console.log(result.text);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "messages": [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        }
    ],
    "model": "grok-4",
    "stream": false
}'
```

Response:

```pythonXAI
'2 + 2 equals 4.'
```

```pythonOpenAISDK
ChatCompletionMessage(
  content='2 + 2 equals 4.',
  refusal=None,
  role='assistant',
  audio=None,
  function_call=None,
  tool_calls=None
)
```

```javascriptOpenAISDK
{
  role: 'assistant',
  content: \`Ah, the ultimate question! According to Douglas Adams' "The Hitchhiker's Guide to the Galaxy," the answer to the ultimate question of life, the universe, and everything is **42**. However, the guide also notes that the actual question to which this is the answer is still unknown. Isn't that delightfully perplexing? Now, if you'll excuse me, I'll just go ponder the intricacies of existence.\`
  refusal: null
}
```

```javascriptAISDK
// result object structure
{
  text: "Ah, the ultimate question! As someone...",
  finishReason: "stop",
  usage: {
    inputTokens: 716,
    outputTokens: 126,
    totalTokens: 1009,
    reasoningTokens: 167
  },
  totalUsage: { /* same as usage */ }
}
```

```bash
{
  "id": "0daf962f-a275-4a3c-839a-047854645532",
  "object": "chat.completion",
  "created": 1739301120,
  "model": "grok-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The meaning of life, the universe, and everything is a question that has puzzled philosophers, scientists, and hitchhikers alike. According to the Hitchhiker's Guide to the Galaxy, the answer to this ultimate question is simply \"42\". However, the exact nature of the question itself remains unknown. So, while we may have the answer, the true meaning behind it is still up for debate. In the meantime, perhaps we should all just enjoy the journey and have a good laugh along the way!",
        "refusal": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 41,
    "completion_tokens": 104,
    "total_tokens": 145,
    "prompt_tokens_details": {
      "text_tokens": 41,
      "audio_tokens": 0,
      "image_tokens": 0,
      "cached_tokens": 0
    }
  },
  "system_fingerprint": "fp_84ff176447"
}
```

## Conversations

The xAI API is stateless and does not process new request with the context of your previous request history.

However, you can provide previous chat generation prompts and results to a new chat generation request to let the model process your new request with the context in mind.

An example message:

```json
{
  "role": "system",
  "content": [{ "type": "text", "text": "You are a helpful and funny assistant."}]
}
{
  "role": "user",
  "content": [{ "type": "text", "text": "Why don't eggs tell jokes?" }]
},
{
  "role": "assistant",
  "content": [{ "type": "text", "text": "They'd crack up!" }]
},
{
  "role": "user",
  "content": [{"type": "text", "text": "Can you explain the joke?"}],
}
```

By specifying roles, you can change how the model ingests the content.
The `system` role content should define, in an instructive tone, the way the model should respond to user request.
The `user` role content is usually used for user requests or data sent to the model.
The `assistant` role content is usually either in the model's response, or when sent within the prompt, indicates the model's response as part of conversation history.

## Message role order flexibility

Unlike some models from other providers, one of the unique aspects of xAI API is its flexibility with message role ordering:

* No Order Limitation: You can mix `system`, `user`, or `assistant` roles in any order for your conversation context.

**Example 1 - Multiple System Messages:**

```json
[
  { "role": "system", "content": "..." },
  { "role": "system", "content": "..." },
  { "role": "user", "content": "..." },
  { "role": "user", "content": "..." }
]
```

**Example 2 - User Messages First:**

```json
[
  { "role": "user", "content": "..." },
  { "role": "user", "content": "..." },
  { "role": "system", "content": "..." }
]
```


===/docs/guides/deferred-chat-completions===
#### Guides

# Deferred Chat Completions

Deferred Chat Completions are currently available only via REST requests or xAI SDK.

Deferred Chat Completions allow you to create a chat completion, get a `response_id`, and retrieve the response at a later time. The result would be available to be requested exactly once within 24 hours, after which it would be discarded.

Your deferred completion rate limit is the same as your chat completions rate limit. To view your rate limit, please visit [xAI Console](https://console.x.ai).

After sending the request to the xAI API, the chat completion result will be available at `https://api.x.ai/v1/chat/deferred-completion/{request_id}`. The response body will contain `{'request_id': 'f15c114e-f47d-40ca-8d5c-8c23d656eeb6'}`, and the `request_id` value can be inserted into the `deferred-completion` endpoint path. Then, we send this GET request to retrieve the deferred completion result.

When the completion result is not ready, the request will return `202 Accepted` with an empty response body.

You can access the model's raw thinking trace via the `message.reasoning_content` of the chat completion response.



## Example

A code example is provided below, where we retry retrieving the result until it has been processed:

```pythonXAI
import os
from datetime import timedelta

from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(api_key=os.getenv('XAI_API_KEY'))

chat = client.chat.create(
    model="grok-4",
    messages=[system("You are Zaphod Beeblebrox.")]
)
chat.append(user("126/3=?"))

# Poll the result every 10 seconds for a maximum of 10 minutes

response = chat.defer(
    timeout=timedelta(minutes=10), interval=timedelta(seconds=10)
)

# Print the result when it is ready

print(response.content)
```

```pythonRequests
import json
import os
import requests

from tenacity import retry, wait_exponential

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}

payload = {
    "messages": [
        {"role": "system", "content": "You are Zaphod Beeblebrox."},
        {"role": "user", "content": "126/3=?"}
    ],
    "model": "grok-4",
    "deferred": True
}

response = requests.post(
    "https://api.x.ai/v1/chat/completions",
    headers=headers,
    json=payload
)
request_id = response.json()["request_id"]
print(f"Request ID: {request_id}")

@retry(wait=wait_exponential(multiplier=1, min=1, max=60),)
def get_deferred_completion():
    response = requests.get(f"https://api.x.ai/v1/chat/deferred-completion/{request_id}", headers=headers)
    if response.status_code == 200:
    return response.json()
    elif response.status_code == 202:
    raise Exception("Response not ready yet")
else:
    raise Exception(f"{response.status_code} Error: {response.text}")

completion_data = get_deferred_completion()
print(json.dumps(completion_data, indent=4))
```

```javascriptWithoutSDK
const axios = require('axios');
const retry = require('retry');

const headers = {
    'Content-Type': 'application/json',
    'Authorization': \`Bearer \${process.env.XAI_API_KEY}\`
};

const payload = {
    messages: [
        { role: 'system', content: 'You are Zaphod Beeblebrox.' },
        { role: 'user', content: '126/3=?' }
    ],
    model: 'grok-4',
    deferred: true
};

async function main() {
    const requestId = (await axios.post('https://api.x.ai/v1/chat/completions', payload, { headers })).data.request_id;
    console.log(\`Request ID: \${requestId}\`);

    const operation = retry.operation({
        minTimeout: 1000,
        maxTimeout: 60000,
        factor: 2
    });

    const completion = await new Promise((resolve, reject) => {
        operation.attempt(async () => {
            const res = await axios.get(\`https://api.x.ai/v1/chat/deferred-completion/\${requestId}\`, { headers });
            if (res.status === 200) resolve(res.data);
            else if (res.status === 202) operation.retry(new Error('Not ready'));
            else reject(new Error(\`\${res.status}: \${res.statusText}\`));
        });
    });

    console.log(JSON.stringify(completion, null, 4));
}

main().catch(console.error);
```

```bash
RESPONSE=$(curl -s https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {"role": "system", "content": "You are Zaphod Beeblebrox."},
        {"role": "user", "content": "126/3=?"}
    ],
    "model": "grok-4",
    "deferred": true
}')

REQUEST_ID=$(echo "$RESPONSE" | jq -r '.request_id')
echo "Request ID: $REQUEST_ID"

sleep 10

curl -s https://api.x.ai/v1/chat/deferred-completion/$REQUEST_ID \\
-H "Authorization: Bearer $XAI_API_KEY"
```

The response body will be the same as what you would expect with non-deferred chat completions:

```json
{
  "id": "3f4ddfca-b997-3bd4-80d4-8112278a1508",
  "object": "chat.completion",
  "created": 1752077400,
  "model": "grok-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Whoa, hold onto your improbability drives, kid! This is Zaphod Beeblebrox here, the two-headed, three-armed ex-President of the Galaxy, and you're asking me about 126 divided by 3? Pfft, that's kid stuff for a guy who's stolen starships and outwitted the universe itself.\n\nBut get this\u2014126 slashed by 3 equals... **42**! Yeah, that's right, the Ultimate Answer to Life, the Universe, and Everything! Deep Thought didn't compute that for seven and a half million years just for fun, you know. My left head's grinning like a Vogon poet on happy pills, and my right one's already planning a party. If you need more cosmic math or a lift on the Heart of Gold, just holler. Zaphod out! \ud83d\ude80",
        "refusal": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 26,
    "completion_tokens": 168,
    "total_tokens": 498,
    "prompt_tokens_details": {
      "text_tokens": 26,
      "audio_tokens": 0,
      "image_tokens": 0,
      "cached_tokens": 4
    },
    "completion_tokens_details": {
      "reasoning_tokens": 304,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    },
    "num_sources_used": 0
  },
  "system_fingerprint": "fp_44e53da025"
}
```

For more details, refer to [Chat completions](../api-reference#chat-completions) and [Get deferred chat completions](../api-reference#get-deferred-chat-completions) in our REST API Reference.


===/docs/guides/files/chat-with-files===
#### Guides

# Chat with Files

Once you've uploaded files, you can reference them in conversations using the `file()` helper function in the xAI Python SDK. When files are attached, the system automatically enables document search capabilities, transforming your request into an agentic workflow.

**xAI Python SDK Users**: Version 1.4.0 of the xai-sdk package is required to use the Files API.

## Basic Chat with a Single File

Reference an uploaded file in a conversation to let the model search through it for relevant information.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, file

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload a document
document_content = b"""Quarterly Sales Report - Q4 2024

Revenue Summary:
- Total Revenue: $5.2M
- Year-over-Year Growth: +18%
- Quarter-over-Quarter Growth: +7%

Top Performing Products:
- Product A: $2.1M revenue (+25% YoY)
- Product B: $1.8M revenue (+12% YoY)
- Product C: $1.3M revenue (+15% YoY)
"""

uploaded_file = client.files.upload(document_content, filename="sales_report.txt")

# Create a chat with the file attached
chat = client.chat.create(model="grok-4-fast")
chat.append(user("What was the total revenue in this report?", file(uploaded_file.id)))

# Get the response
response = chat.sample()

print(f"Answer: {response.content}")
print(f"\\nUsage: {response.usage}")

# Clean up
client.files.delete(uploaded_file.id)
```

```pythonOpenAISDK
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# Upload a file
document_content = b"""Quarterly Sales Report - Q4 2024

Revenue Summary:
- Total Revenue: $5.2M
- Year-over-Year Growth: +18%
"""

with open("temp_sales.txt", "wb") as f:
    f.write(document_content)

with open("temp_sales.txt", "rb") as f:
    uploaded_file = client.files.create(file=f, purpose="assistants")

# Create a chat with the file
response = client.responses.create(
    model="grok-4-fast",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What was the total revenue in this report?"},
                {"type": "input_file", "file_id": uploaded_file.id}
            ]
        }
    ]
)

final_answer = response.output[-1].content[0].text

print(f"Answer: {final_answer}")

# Clean up
client.files.delete(uploaded_file.id)
```

```pythonRequests
import os
import requests

api_key = os.getenv("XAI_API_KEY")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Upload file first
upload_url = "https://api.x.ai/v1/files"
files = {"file": ("sales_report.txt", b"Total Revenue: $5.2M")}
data = {"purpose": "assistants"}
upload_response = requests.post(upload_url, headers={"Authorization": f"Bearer {api_key}"}, files=files, data=data)
file_id = upload_response.json()["id"]

# Create chat with file
chat_url = "https://api.x.ai/v1/responses"
payload = {
    "model": "grok-4-fast",
    "input": [
        {
            "role": "user",
            "content": "What was the total revenue in this report?",
            "attachments": [
                {
                    "file_id": file_id,
                    "tools": [{"type": "file_search"}]
                }
            ]
        }
    ]
}
response = requests.post(chat_url, headers=headers, json=payload)
print(response.json())
```

```bash
# First upload the file
FILE_ID=$(curl https://api.x.ai/v1/files \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -F file=@sales_report.txt \\
  -F purpose=assistants | jq -r '.id')

# Then use it in chat
curl -X POST "https://api.x.ai/v1/responses" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d "{
    \\"model\\": \\"grok-4-fast\\",
    \\"input\\": [
      {
        \\"role\\": \\"user\\",
        \\"content\\": [
          {\\"type\\": \\"input_text\\", \\"text\\": \\"What was the total revenue in this report?\\"},
          {\\"type\\": \\"input_file\\", \\"file_id\\": \\"$FILE_ID\\"}
        ]
      }
    ]
  }"
```

## Streaming Chat with Files

Get real-time responses while the model searches through your documents.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, file

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload a document
document_content = b"""Product Specifications:
- Model: XR-2000
- Weight: 2.5 kg
- Dimensions: 30cm x 20cm x 10cm
- Power: 100W
- Features: Wireless connectivity, LCD display, Energy efficient
"""

uploaded_file = client.files.upload(document_content, filename="specs.txt")

# Create chat with streaming
chat = client.chat.create(model="grok-4-fast")
chat.append(user("What is the weight of the XR-2000?", file(uploaded_file.id)))

# Stream the response
is_thinking = True
for response, chunk in chat.stream():
    # Show tool calls as they happen
    for tool_call in chunk.tool_calls:
        print(f"\\nSearching: {tool_call.function.name}")
    
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    
    if chunk.content and is_thinking:
        print("\\n\\nAnswer:")
        is_thinking = False
    
    if chunk.content:
        print(chunk.content, end="", flush=True)

print(f"\\n\\nUsage: {response.usage}")

# Clean up
client.files.delete(uploaded_file.id)
```

## Multiple File Attachments

Query across multiple documents simultaneously.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, file

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload multiple documents
file1_content = b"Document 1: The project started in January 2024."
file2_content = b"Document 2: The project budget is $500,000."
file3_content = b"Document 3: The team consists of 5 engineers and 2 designers."

file1 = client.files.upload(file1_content, filename="timeline.txt")
file2 = client.files.upload(file2_content, filename="budget.txt")
file3 = client.files.upload(file3_content, filename="team.txt")

# Create chat with multiple files
chat = client.chat.create(model="grok-4-fast")
chat.append(
    user(
        "Based on these documents, when did the project start, what is the budget, and how many people are on the team?",
        file(file1.id),
        file(file2.id),
        file(file3.id),
    )
)

response = chat.sample()

print(f"Answer: {response.content}")
print("\\nDocuments searched: 3")
print(f"Usage: {response.usage}")

# Clean up
client.files.delete(file1.id)
client.files.delete(file2.id)
client.files.delete(file3.id)
```

## Multi-Turn Conversations with Files

Maintain context across multiple questions about the same documents. Use encrypted content to preserve file context efficiently across multiple turns.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, file

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload an employee record
document_content = b"""Employee Information:
Name: Alice Johnson
Department: Engineering
Years of Service: 5
Performance Rating: Excellent
Skills: Python, Machine Learning, Cloud Architecture
Current Project: AI Platform Redesign
"""

uploaded_file = client.files.upload(document_content, filename="employee.txt")

# Create a multi-turn conversation with encrypted content
chat = client.chat.create(
    model="grok-4-fast",
    use_encrypted_content=True,  # Enable encrypted content for efficient multi-turn
)

# First turn: Ask about the employee name
chat.append(user("What is the employee's name?", file(uploaded_file.id)))
response1 = chat.sample()
print("Q1: What is the employee's name?")
print(f"A1: {response1.content}\\n")

# Add the response to conversation history
chat.append(response1)

# Second turn: Ask about department (agentic context is retained via encrypted content)
chat.append(user("What department does this employee work in?"))
response2 = chat.sample()
print("Q2: What department does this employee work in?")
print(f"A2: {response2.content}\\n")

# Add the response to conversation history
chat.append(response2)

# Third turn: Ask about skills
chat.append(user("What skills does this employee have?"))
response3 = chat.sample()
print("Q3: What skills does this employee have?")
print(f"A3: {response3.content}\\n")

# Clean up
client.files.delete(uploaded_file.id)
```

## Combining Files with Other Modalities

You can combine file attachments with images and other content types in a single message.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, file, image

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload a text document with cat care information
text_content = b"Cat Care Guide: Cats require daily grooming, especially long-haired breeds. Regular brushing helps prevent matting and reduces shedding."
text_file = client.files.upload(text_content, filename="cat-care.txt")

# Use both file and image in the same message
chat = client.chat.create(model="grok-4-fast")
chat.append(
    user(
        "Based on the attached care guide, do you have any advice about the pictured cat?",
        file(text_file.id),
        image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"),
    )
)

response = chat.sample()

print(f"Analysis: {response.content}")
print(f"\\nUsage: {response.usage}")

# Clean up
client.files.delete(text_file.id)
```

## Combining Files with Code Execution

For data analysis tasks, you can attach data files and enable the code execution tool. This allows Grok to write and run Python code to analyze and process your data.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, file
from xai_sdk.tools import code_execution

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload a CSV data file
csv_content = b"""product,region,revenue,units_sold
Product A,North,245000,1200
Product A,South,189000,950
Product A,East,312000,1500
Product A,West,278000,1350
Product B,North,198000,800
Product B,South,156000,650
Product B,East,234000,950
Product B,West,201000,850
Product C,North,167000,700
Product C,South,134000,550
Product C,East,198000,800
Product C,West,176000,725
"""

data_file = client.files.upload(csv_content, filename="sales_data.csv")

# Create chat with both file attachment and code execution
chat = client.chat.create(
    model="grok-4-fast",
    tools=[code_execution()],  # Enable code execution
)

chat.append(
    user(
        "Analyze this sales data and calculate: 1) Total revenue by product, 2) Average units sold by region, 3) Which product-region combination has the highest revenue",
        file(data_file.id)
    )
)

# Stream the response to see code execution in real-time
is_thinking = True
for response, chunk in chat.stream():
    for tool_call in chunk.tool_calls:
        if tool_call.function.name == "code_execution":
            print("\\n[Executing Code]")
    
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    
    if chunk.content and is_thinking:
        print("\\n\\nAnalysis Results:")
        is_thinking = False
    
    if chunk.content:
        print(chunk.content, end="", flush=True)

print(f"\\n\\nUsage: {response.usage}")

# Clean up
client.files.delete(data_file.id)
```

The model will:

1. Access the attached data file
2. Write Python code to load and analyze the data
3. Execute the code in a sandboxed environment
4. Perform calculations and statistical analysis
5. Return the results and insights in the response

## Limitations and Considerations

### Request Constraints

* **No batch requests**: File attachments with document search are agentic requests and do not support batch mode (`n > 1`)
* **Streaming recommended**: Use streaming mode for better observability of document search process

### Document Complexity

* Highly unstructured or very long documents may require more processing
* Well-organized documents with clear structure are easier to search
* Large documents with many searches can result in higher token usage

### Model Compatibility

* **Recommended models**: `grok-4-fast`, `grok-4` for best document understanding
* **Agentic requirement**: File attachments require [agentic-capable](/docs/guides/tools/overview#model-compatibility) models that support server-side tools.

## Next Steps

Learn more about managing your files:


===/docs/guides/files/managing-files===
#### Guides

# Managing Files

The Files API provides a complete set of operations for managing your files. Before using files in chat conversations, you need to upload them using one of the methods described below.

**xAI Python SDK Users**: Version 1.4.0 of the xai-sdk package is required to use the Files API.

## Uploading Files

You can upload files in several ways: from a file path, raw bytes, BytesIO object, or an open file handle.

### Upload from File Path

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload a file from disk
file = client.files.upload("/path/to/your/document.pdf")

print(f"File ID: {file.id}")
print(f"Filename: {file.filename}")
print(f"Size: {file.size} bytes")
print(f"Created at: {file.created_at}")
```

```pythonOpenAISDK
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# Upload a file
with open("/path/to/your/document.pdf", "rb") as f:
    file = client.files.create(
        file=f,
        purpose="assistants"
    )

print(f"File ID: {file.id}")
print(f"Filename: {file.filename}")
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/files"
headers = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}

with open("/path/to/your/document.pdf", "rb") as f:
    files = {"file": f}
    data = {"purpose": "assistants"}
    response = requests.post(url, headers=headers, files=files, data=data)

file_data = response.json()
print(f"File ID: {file_data['id']}")
print(f"Filename: {file_data['filename']}")
```

```bash
curl https://api.x.ai/v1/files \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -F file=@/path/to/your/document.pdf \\
  -F purpose=assistants
```

### Upload from Bytes

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload file content directly from bytes
content = b"This is my document content.\\nIt can span multiple lines."
file = client.files.upload(content, filename="document.txt")

print(f"File ID: {file.id}")
print(f"Filename: {file.filename}")
```

### Upload from file object

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Upload a file directly from disk
file = client.files.upload(open("document.pdf", "rb"), filename="document.pdf")

print(f"File ID: {file.id}")
print(f"Filename: {file.filename}")
```

## Upload with Progress Tracking

Track upload progress for large files using callbacks or progress bars.

### Custom Progress Callback

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Define a custom progress callback
def progress_callback(bytes_uploaded: int, total_bytes: int):
    percentage = (bytes_uploaded / total_bytes) * 100 if total_bytes else 0
    mb_uploaded = bytes_uploaded / (1024 * 1024)
    mb_total = total_bytes / (1024 * 1024)
    print(f"Progress: {mb_uploaded:.2f}/{mb_total:.2f} MB ({percentage:.1f}%)")

# Upload with progress tracking
file = client.files.upload(
    "/path/to/large-file.pdf",
    on_progress=progress_callback
)

print(f"Successfully uploaded: {file.filename}")
```

### Progress Bar with tqdm

```pythonXAI
import os
from xai_sdk import Client
from tqdm import tqdm

client = Client(api_key=os.getenv("XAI_API_KEY"))

file_path = "/path/to/large-file.pdf"
total_bytes = os.path.getsize(file_path)

# Upload with tqdm progress bar
with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Uploading") as pbar:
    file = client.files.upload(
        file_path,
        on_progress=pbar.update
    )

print(f"Successfully uploaded: {file.filename}")
```

## Listing Files

Retrieve a list of your uploaded files with pagination and sorting options.

### Available Options

* **`limit`**: Maximum number of files to return. If not specified, uses server default of 100.
* **`order`**: Sort order for the files. Either `"asc"` (ascending) or `"desc"` (descending).
* **`sort_by`**: Field to sort by. Options: `"created_at"`, `"filename"`, or `"size"`.
* **`pagination_token`**: Token for fetching the next page of results.

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# List files with pagination and sorting
response = client.files.list(
    limit=10,
    order="desc",
    sort_by="created_at"
)

for file in response.data:
    print(f"File: {file.filename} (ID: {file.id}, Size: {file.size} bytes)")
```

```pythonOpenAISDK
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# List files
files = client.files.list()

for file in files.data:
    print(f"File: {file.filename} (ID: {file.id})")
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/files"
headers = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}

response = requests.get(url, headers=headers)
files = response.json()

for file in files.get("data", []):
    print(f"File: {file['filename']} (ID: {file['id']})")
```

```bash
curl https://api.x.ai/v1/files \\
  -H "Authorization: Bearer $XAI_API_KEY"
```

## Getting File Metadata

Retrieve detailed information about a specific file.

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Get file metadata by ID
file = client.files.get("file-abc123")

print(f"Filename: {file.filename}")
print(f"Size: {file.size} bytes")
print(f"Created: {file.created_at}")
print(f"Team ID: {file.team_id}")
```

```pythonOpenAISDK
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# Get file metadata
file = client.files.retrieve("file-abc123")

print(f"Filename: {file.filename}")
print(f"Size: {file.bytes} bytes")
```

```pythonRequests
import os
import requests

file_id = "file-abc123"
url = f"https://api.x.ai/v1/files/{file_id}"
headers = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}

response = requests.get(url, headers=headers)
file = response.json()

print(f"Filename: {file['filename']}")
print(f"Size: {file['bytes']} bytes")
```

```bash
curl https://api.x.ai/v1/files/file-abc123 \\
  -H "Authorization: Bearer $XAI_API_KEY"
```

## Getting File Content

Download the actual content of a file.

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Get file content
content = client.files.content("file-abc123")

# Content is returned as bytes
print(f"Content length: {len(content)} bytes")
print(f"Content preview: {content[:100]}")
```

```pythonOpenAISDK
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# Get file content
content = client.files.content("file-abc123")

print(f"Content: {content.text}")
```

```pythonRequests
import os
import requests

file_id = "file-abc123"
url = f"https://api.x.ai/v1/files/{file_id}/content"
headers = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}

response = requests.get(url, headers=headers)
content = response.content

print(f"Content length: {len(content)} bytes")
```

```bash
curl https://api.x.ai/v1/files/file-abc123/content \\
  -H "Authorization: Bearer $XAI_API_KEY"
```

## Deleting Files

Remove files when they're no longer needed.

```pythonXAI
import os
from xai_sdk import Client

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Delete a file
delete_response = client.files.delete("file-abc123")

print(f"Deleted: {delete_response.deleted}")
print(f"File ID: {delete_response.id}")
```

```pythonOpenAISDK
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# Delete a file
delete_response = client.files.delete("file-abc123")

print(f"Deleted: {delete_response.deleted}")
print(f"File ID: {delete_response.id}")
```

```pythonRequests
import os
import requests

file_id = "file-abc123"
url = f"https://api.x.ai/v1/files/{file_id}"
headers = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}

response = requests.delete(url, headers=headers)
result = response.json()

print(f"Deleted: {result['deleted']}")
print(f"File ID: {result['id']}")
```

```bash
curl -X DELETE https://api.x.ai/v1/files/file-abc123 \\
  -H "Authorization: Bearer $XAI_API_KEY"
```

## Limitations and Considerations

### File Size Limits

* **Maximum file size**: 48 MB per file
* **Processing time**: Larger files may take longer to process

### File Retention

* **Cleanup**: Delete files when no longer needed to manage storage
* **Access**: Files are scoped to your team/organization

### Supported Formats

While many text-based formats are supported, the system works best with:

* Structured documents (with clear sections, headings)
* Plain text and markdown
* Documents with clear information hierarchy

Supported file types include:

* Plain text files (.txt)
* Markdown files (.md)
* Code files (.py, .js, .java, etc.)
* CSV files (.csv)
* JSON files (.json)
* PDF documents (.pdf)
* And many other text-based formats

## Next Steps

Now that you know how to manage files, learn how to use them in chat conversations:


===/docs/guides/files===
#### Guides

# Files

The Files API enables you to upload documents and use them in chat conversations with Grok. When you attach files to a chat message, the system automatically activates the `document_search` tool, transforming your request into an agentic workflow where Grok can intelligently search through and reason over your documents to answer questions.

You can view more information at [Files API Reference](/docs/files-api).

**xAI Python SDK Users**: Version 1.4.0 of the xai-sdk package is required to use the Files API.

## How Files Work with Chat

Behind the scenes, when you attach files to a chat message, the xAI API implicitly adds the `document_search` server-side tool to your request. This means:

1. **Automatic Agentic Behavior**: Your chat request becomes an agentic request, where Grok autonomously searches through your documents
2. **Intelligent Document Analysis**: The model can reason over document content, extract relevant information, and synthesize answers
3. **Multi-Document Support**: You can attach multiple files, and Grok will search across all of them

This seamless integration allows you to simply attach files and ask questions—the complexity of document search and retrieval is handled automatically by the agentic workflow.

## Understanding Document Search

When you attach files to a chat message, the xAI API automatically activates the `document_search` [server-side tool](/docs/guides/tools/overview). This transforms your request into an [agentic workflow](/docs/guides/tools/overview#agentic-tool-calling) where Grok:

1. **Analyzes your query** to understand what information you're seeking
2. **Searches the documents** intelligently, finding relevant sections across all attached files
3. **Extracts and synthesizes information** from multiple sources if needed
4. **Provides a comprehensive answer** with the context from your documents

### Agentic Workflow

Just like other agentic tools (web search, X search, code execution), document search operates autonomously:

* **Multiple searches**: The model may search documents multiple times with different queries to find comprehensive information
* **Reasoning**: The model uses its reasoning capabilities to decide what to search for and how to interpret the results
* **Streaming visibility**: In streaming mode, you can see when the model is searching your documents via tool call notifications

### Token Usage with Files

File-based chats follow similar token patterns to other agentic requests:

* **Prompt tokens**: Include the conversation history and internal processing. Document content is processed efficiently
* **Reasoning tokens**: Used for planning searches and analyzing document content
* **Completion tokens**: The final answer text
* **Cached tokens**: Repeated document content benefits from prompt caching for efficiency

The actual document content is processed by the server-side tool and doesn't directly appear in the message history, keeping token usage optimized.

### Pricing

Document search is billed at **$10 per 1,000 tool invocations**, in addition to standard token costs. Each time the model searches your documents, it counts as one tool invocation. For complete pricing details, see the [Models and Pricing](/docs/models#tools-pricing) page.

## Getting Started

To use files with Grok, you'll need to:

1. **[Upload and manage files](/docs/guides/files/managing-files)** - Learn how to upload, list, retrieve, and delete files using the Files API
2. **[Chat with files](/docs/guides/files/chat-with-files)** - Discover how to attach files to chat messages and ask questions about your documents

## Quick Example

Here's a quick example of the complete workflow:

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, file

client = Client(api_key=os.getenv("XAI_API_KEY"))

# 1. Upload a document
document_content = b"""Quarterly Sales Report - Q4 2024
Total Revenue: $5.2M
Growth: +18% YoY
"""

uploaded_file = client.files.upload(document_content, filename="sales.txt")

# 2. Chat with the file
chat = client.chat.create(model="grok-4-fast")
chat.append(user("What was the total revenue?", file(uploaded_file.id)))

# 3. Get the answer
response = chat.sample()
print(response.content)  # "The total revenue was $5.2M"

# 4. Clean up
client.files.delete(uploaded_file.id)
```

## Key Features

### Multiple File Support

Attach [multiple documents](/docs/guides/files/chat-with-files#multiple-file-attachments) to a single query and Grok will search across all of them to find relevant information.

### Multi-Turn Conversations

File context persists across [conversation turns](/docs/guides/files/chat-with-files#multi-turn-conversations-with-files), allowing you to ask follow-up questions without re-attaching files.

### Code Execution Integration

Combine files with the [code execution tool](/docs/guides/files/chat-with-files#combining-files-with-code-execution) to perform advanced data analysis, statistical computations, and transformations on your uploaded data. The model can write and execute Python code that processes your files directly.

## Limitations

* **File size**: Maximum 48 MB per file
* **No batch requests**: File attachments with document search are agentic requests and do not support batch mode (`n > 1`)
* **Agentic models only**: Requires models that support agentic tool calling (e.g., `grok-4-fast`, `grok-4`)
* **Supported file formats**:
  * Plain text files (.txt)
  * Markdown files (.md)
  * Code files (.py, .js, .java, etc.)
  * CSV files (.csv)
  * JSON files (.json)
  * PDF documents (.pdf)
  * And many other text-based formats

## Next Steps


===/docs/guides/fingerprint===
#### Guides

# Fingerprint

For each request to the xAI API, the response body will include a unique `system_fingerprint` value. This fingerprint serves as an identifier for the current state of the backend system's configuration.

Example:

```bash
curl https://api.x.ai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $XAI_API_KEY" \
  -d '{
        "messages": [
          {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
          },
          {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
          }
        ],
        "model": "grok-4",
        "stream": false,
        "temperature": 0
      }'
```

Response:

```json
{..., "system_fingerprint":"fp_6ca29cf396"}
```

You can automate your system to keep track of the `system_fingerprint` along with token consumption and other metrics.

## Usage of fingerprint

* **Monitoring System Changes:** The system fingerprint acts as a version control for the backend configuration. If any part of the backend system—such as model parameters, server settings, or even the underlying infrastructure—changes, the fingerprint will also change. This allows developers to track when and how the system has evolved over time. This is crucial for debugging, performance optimization, and ensuring consistency in API responses.
* **Security and Integrity:** The fingerprint can be used to ensure the integrity of the response. If a response's fingerprint matches the expected one based on a recent system configuration, it helps in verifying that the data hasn't been tampered with during transmission or that the service hasn't been compromised. **The fingerprint will change over time and it is expected.**
* **Compliance and Auditing:** For regulated environments, this fingerprint can serve as part of an audit trail, showing when specific configurations were in use for compliance purposes.


===/docs/guides/function-calling===
#### Guides

# Function calling

Connect the xAI models to external tools and systems to build AI assistants and various integrations.

With stream response, the function call will be returned in whole in a single chunk, instead of
being streamed across chunks.

## Introduction

Function calling enables language models to use external tools, which can intimately connect models to digital and physical worlds.

This is a powerful capability that can be used to enable a wide range of use cases.

* Calling public APIs for actions ranging from looking up football game results to getting real-time satellite positioning data
* Analyzing internal databases
* Browsing web pages
* Executing code
* Interacting with the physical world (e.g. booking a flight ticket, opening your tesla car door, controlling robot arms)

You can call a maximum of 200 tools with function calling.

## Walkthrough

The request/response flow for function calling can be demonstrated in the following illustration.

You can think of it as the LLM initiating [RPCs (Remote Procedure Calls)](https://en.wikipedia.org/wiki/Remote_procedure_call) to user system. From the LLM's perspective, the "2. Response" is an RPC request from LLM to user system, and the "3. Request" is an RPC response with information that LLM needs.

One simple example of a local computer/server, where the computer/server determines if the response from Grok contains a `tool_call`, and calls the locally-defined functions to perform user-defined actions:

The whole process looks like this in pseudocode:

```pseudocode
// ... Define tool calls and their names

messages = []

/* Step 1: Send a new user request */

messages += {<new user request message>}
response = send_request_to_grok(message)

messages += response.choices[0].message  // Append assistant response

while (true) {
    /* Step 2: Run tool call and add tool call result to messages */
    if (response contains tool_call) {
        // Grok asks for tool call

        for (tool in tool_calls) {
            tool_call_result = tool(arguments provided in response) // Perform tool call
            messages += tool_call_result  // Add result to message
        }
    }

    read(user_request)

    if (user_request) {
        messages += {<new user request message>}
    }

    /* Step 3: Send request with tool call result to Grok*/
    response = send_request_to_grok(message)

    print(response)
}

```

We will demonstrate the function calling in the following Python script. First, let's create an API client:

```pythonXAI
import os
import json

from xai_sdk import Client
from xai_sdk.chat import tool, tool_result, user

client = Client(api_key=os.getenv('XAI_API_KEY'))
chat = client.chat.create(model="grok-4")
```

```pythonOpenAISDK
import os
import json
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)
```

### Preparation - Define tool functions and function mapping

Define tool functions as callback functions to be called when model requests them in response.

Normally, these functions would either retrieve data from a database, or call another API endpoint, or perform some actions.
For demonstration purposes, we hardcode to return 59° Fahrenheit/15° Celsius as the temperature, and 15,000 feet as the cloud ceiling.

The parameters definition will be sent in the initial request to Grok, so Grok knows what tools and parameters are available to be called.

To reduce human error, you can define the tools partially using Pydantic.

Function definition using Pydantic:

```pythonXAI
from typing import Literal

from pydantic import BaseModel, Field

class TemperatureRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Literal["celsius", "fahrenheit"] = Field(
        "fahrenheit", description="Temperature unit"
    )

class CeilingRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

def get_current_temperature(request: TemperatureRequest):
    temperature = 59 if request.unit.lower() == "fahrenheit" else 15
    return {
        "location": request.location,
        "temperature": temperature,
        "unit": request.unit,
    }

def get_current_ceiling(request: CeilingRequest):
    return {
        "location": request.location,
        "ceiling": 15000,
        "ceiling_type": "broken",
        "unit": "ft",
    }

# Generate the JSON schema from the Pydantic models

get_current_temperature_schema = TemperatureRequest.model_json_schema()
get_current_ceiling_schema = CeilingRequest.model_json_schema()

# Definition of parameters with Pydantic JSON schema

tool_definitions = [
    tool(
        name="get_current_temperature",
        description="Get the current temperature in a given location",
        parameters=get_current_temperature_schema,
    ),
    tool(
        name="get_current_ceiling",
        description="Get the current cloud ceiling in a given location",
        parameters=get_current_ceiling_schema,
    ),
]
```

```pythonOpenAISDK
from typing import Literal

from pydantic import BaseModel, Field

class TemperatureRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Literal["celsius", "fahrenheit"] = Field(
        "fahrenheit", description="Temperature unit"
    )

class CeilingRequest(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

def get_current_temperature(request: TemperatureRequest):
    temperature = 59 if request.unit.lower() == "fahrenheit" else 15
    return {
        "location": request.location,
        "temperature": temperature,
        "unit": request.unit,
    }

def get_current_ceiling(request: CeilingRequest):
    return {
        "location": request.location,
        "ceiling": 15000,
        "ceiling_type": "broken",
        "unit": "ft",
    }

# Generate the JSON schema from the Pydantic models

get_current_temperature_schema = TemperatureRequest.model_json_schema()
get_current_ceiling_schema = CeilingRequest.model_json_schema()

# Definition of parameters with Pydantic JSON schema

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature in a given location",
            "parameters": get_current_temperature_schema,
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_ceiling",
            "description": "Get the current cloud ceiling in a given location",
            "parameters": get_current_ceiling_schema,
        }
    },
]
```

Function definition using raw dictionary:

```pythonXAI
from typing import Literal

def get_current_temperature(location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit"):
    temperature = 59 if unit == "fahrenheit" else 15
    return {
        "location": location,
        "temperature": temperature,
        "unit": unit,
    }

def get_current_ceiling(location: str):
    return {
        "location": location,
        "ceiling": 15000,
        "ceiling_type": "broken",
        "unit": "ft",
    }

# Raw dictionary definition of parameters

tool_definitions = [
    tool(
        name="get_current_temperature",
        description="Get the current temperature in a given location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "fahrenheit",
                },
            },
            "required": ["location"],
        },
    ),
    tool(
        name="get_current_ceiling",
        description="Get the current cloud ceiling in a given location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    ),
]
```

```pythonOpenAISDK
from typing import Literal

def get_current_temperature(location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit"):
    temperature = 59 if unit == "fahrenheit" else 15
    return {
        "location": location,
        "temperature": temperature,
        "unit": unit,
    }

def get_current_ceiling(location: str):
    return {
        "location": location,
        "ceiling": 15000,
        "ceiling_type": "broken",
        "unit": "ft",
    }

# Raw dictionary definition of parameters

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "fahrenheit"
                    }
                },
            "required": ["location"]
        }
    }
},
{
    "type": "function",
    "function": {
    "name": "get_current_ceiling",
    "description": "Get the current cloud ceiling in a given location",
    "parameters": {
    "type": "object",
    "properties": {
    "location": {
    "type": "string",
    "description": "The city and state, e.g. San Francisco, CA"
    }
    },
    "required": ["location"]
    }
    }
}
]
```

Create a string -> function mapping, so we can call the function when model sends it's name. e.g.

```pythonWithoutSDK
tools_map = {
    "get_current_temperature": get_current_temperature,
    "get_current_ceiling": get_current_ceiling,
}
```

### 1. Send initial message

With all the functions defined, it's time to send our API request to Grok!

Now before we send it over, let's look at how the generic request body for a new task looks like.

Here we assume a previous tool call has Note how the tool call is referenced three times:

* By `id` and `name` in "Mesage History" assistant's first response
* By `tool_call_id` in "Message History" tool's content
* In the `tools` field of the request body

Now we compose the request messages in the request body and send it over to Grok. Grok should return a response that asks us for a tool call.

```pythonXAI
chat = client.chat.create(
    model="grok-4",
    tools=tool_definitions,
    tool_choice="auto",
)
chat.append(user("What's the temperature like in San Francisco?"))
response = chat.sample()

# You can inspect the response tool calls which contains a tool call

print(response.tool_calls)
```

```pythonOpenAISDK
messages = [{"role": "user", "content": "What's the temperature like in San Francisco?"}]
response = client.chat.completions.create(
    model="grok-4",
    messages=messages,
    tools=tool_definitions, # The dictionary of our functions and their parameters
    tool_choice="auto",
)

# You can inspect the response which contains a tool call

print(response.choices[0].message)
```

### 2. Run tool functions if Grok asks for tool call and append function returns to message

We retrieve the tool function names and arguments that Grok wants to call, run the functions, and add the result to messages.

At this point, you can choose to **only respond to tool call with results** or **add a new user message request**.

The `tool` message would contain the following:

```json
{
    "role": "tool",
    "content": <json string of tool function's returned object>,
    "tool_call_id": <tool_call.id included in the tool call response by Grok>,
}
```

The request body that we try to assemble and send back to Grok. Note it looks slightly different from the new task request body:

The corresponding code to append messages:

```pythonXAI
# Append assistant message including tool calls to messages
chat.append(response)

# Check if there is any tool calls in response body

# You can also wrap this in a function to make the code cleaner

if response.tool_calls:
    for tool_call in response.tool_calls:

        # Get the tool function name and arguments Grok wants to call
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        # Call one of the tool function defined earlier with arguments
        result = tools_map[function_name](**function_args)

        # Append the result from tool function call to the chat message history
        chat.append(tool_result(result))
```

```pythonOpenAISDK
# Append assistant message including tool calls to messages

messages.append(response.choices[0].message)

# Check if there is any tool calls in response body

# You can also wrap this in a function to make the code cleaner

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:

        # Get the tool function name and arguments Grok wants to call
        function_name = tool_call.function.name
        if function_name not in tools_map:
            messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": f"Function {function_name} not found"}),
                    "tool_call_id": tool_call.id
                })
            continue
        function_args = json.loads(tool_call.function.arguments)

        # Call one of the tool function defined earlier with arguments
        result = tools_map[function_name](**function_args)

        # Append the result from tool function call to the chat message history,
        # with "role": "tool"
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id  # tool_call.id supplied in Grok's response
            })
```

### 3. Send the tool function returns back to the model to get the response

```pythonXAI
response = chat.sample()
print(response.content)
```

```pythonOpenAISDK
response = client.chat.completions.create(
    model="grok-4",
    messages=messages,
    tools=tool_definitions,
    tool_choice="auto"
    )
print(response.choices[0].message.content)
```

### 4. (Optional) Continue the conversation

You can continue the conversation following [Step 2](#2-run-tool-functions-if-grok-asks-for-tool-call-and-append-function-returns-to-message). Otherwise you can terminate.

## Function calling modes

By default, the model will automatically decide whether a function call is necessary and select which functions to call, as determined by the `tool_choice: "auto"` setting.

We offer three ways to customize the default behavior:

1. To force the model to always call one or more functions, you can set `tool_choice: "required"`. The model will then always call function. Note this could force the model to hallucinate parameters.
2. To force the model to call a specific function, you can set `tool_choice: {"type": "function", "function": {"name": "my_function"}}`.
3. To disable function calling and force the model to only generate a user-facing message, you can either provide no tools, or set `tool_choice: "none"`.

## Parallel function calling

By default, parallel function calling is enabled, so you can process multiple function calls in one request/response cycle.
When two or more tool calls are required, all of the tool call requests will be included in the response body. You can disable it by setting `parallel_function_calling : "false"`.

## Complete Example with Vercel AI SDK

The Vercel AI SDK simplifies function calling by handling tool definition, mapping, and execution automatically. Here's a complete example:

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { streamText, tool, stepCountIs } from 'ai';
import { z } from 'zod';

const result = streamText({
  model: xai('grok-4'),
  tools: {
    getCurrentTemperature: tool({
      description: 'Get the current temperature in a given location',
      inputSchema: z.object({
        location: z
          .string()
          .describe('The city and state, e.g. San Francisco, CA'),
        unit: z
          .enum(['celsius', 'fahrenheit'])
          .default('fahrenheit')
          .describe('Temperature unit'),
      }),
      execute: async ({ location, unit }) => {
        const temperature = unit === 'fahrenheit' ? 59 : 15;
        return {
          location,
          temperature,
          unit,
        };
      },
    }),
    getCurrentCeiling: tool({
      description: 'Get the current cloud ceiling in a given location',
      inputSchema: z.object({
        location: z
          .string()
          .describe('The city and state, e.g. San Francisco, CA'),
      }),
      execute: async ({ location }) => {
        return {
          location,
          ceiling: 15000,
          ceiling_type: 'broken',
          unit: 'ft',
        };
      },
    }),
  },
  stopWhen: stepCountIs(5),
  prompt: "What's the temperature like in San Francisco?",
});

for await (const chunk of result.fullStream) {
  switch (chunk.type) {
    case 'text-delta':
      process.stdout.write(chunk.text);
      break;
    case 'tool-call':
      console.log(\`Tool call: \${chunk.toolName}\`, chunk.input);
      break;
    case 'tool-result':
      console.log(\`Tool response: \${chunk.toolName}\`, chunk.output);
      break;
  }
}
```

With the Vercel AI SDK, you don't need to manually:

* Map tool names to functions
* Parse tool call arguments
* Append tool results back to messages
* Handle the request/response cycle

The SDK automatically handles all of these steps, making function calling much simpler.


===/docs/guides/grok-code-prompt-engineering===
#### Guides

# Prompt Engineering for Grok Code Fast 1

## For developers using agentic coding tools

`grok-code-fast-1` is a lightweight agentic model which is designed to excel as your pair-programmer inside most common coding tools. To optimize your experience, we present a few guidelines so that you can fly through your day-to-day coding tasks.

### Provide the necessary context

Most coding tools will gather the necessary context for you on their own. However, it is oftentimes better to be specific by selecting the specific code you want to use as context. This allows `grok-code-fast-1` to focus on your task and prevent unnecessary deviations. Try to specify relevant file paths, project structures, or dependencies and avoid providing irrelevant context.

* No-context prompt to avoid
  > Make error handling better
* Good prompt with specified context
  > My error codes are defined in @errors.ts, can you use that as reference to add proper error handling and error codes to @sql.ts where I am making queries

### Set explicit goals and requirements

Clearly define your goals and the specific problem you want `grok-code-fast-1` to solve. Detailed and concrete queries can lead to better performance. Try to avoid vague or underspecified prompts, as they can result in suboptimal results.

* Vague prompt to avoid
  > Create a food tracker
* Good, detailed prompt
  > Create a food tracker which shows the breakdown of calorie consumption per day divided by different nutrients when I enter a food item. Make it such that I can see an overview as well as get high level trends.

### Continually refine your prompts

`grok-code-fast-1` is a highly efficient model, delivering up to 4x the speed and 1/10th the cost of other leading agentic models. This enables you to test your complex ideas at an unprecedented speed and affordability. Even if the initial output isn’t perfect, we strongly suggest taking advantage of the uniquely rapid and cost-effective iteration to refine your query—using the suggestions above (e.g., adding more context) or by referencing the specific failures from the first attempt.

* Good prompt example with refinement
  > The previous approach didn’t consider the IO heavy process which can block the main thread, we might want to run it in its own threadloop such that it does not block the event loop instead of just using the async lib version

### Assign agentic tasks

We encourage users to try `grok-code-fast-1` for agentic-style tasks rather than one-shot queries. Our Grok 4 models are more suited for one-shot Q\&A while `grok-code-fast-1` is your ideal companion for navigating large mountains of code with tools to deliver you precise answers.

A good way to think about this is:

* `grok-code-fast-1` is great at working quickly and tirelessly to find you the answer or implement the required change.
* Grok 4 is best for diving deep into complex concepts and tough debugging when you provide all the necessary context upfront.

## For developers building coding agents via the xAI API

With `grok-code-fast-1`, we wanted to bring an agentic coding model into the hands of developers. Outside of our launch partners, we welcome all developers to try out `grok-code-fast-1` in tool-call-heavy domains as the fast speed and low cost makes it both efficient and affordable for using many tools to figure out the correct answer.

As mentioned in the blog post, `grok-code-fast-1` is a reasoning model with interleaved tool-calling during its thinking. We also send summarized thinking via the OpenAI-compatible API for better UX support. More API details can be found at https://docs.x.ai/docs/guides/function-calling.

### Reasoning content

`grok-code-fast-1` is a reasoning model, and we expose its thinking trace via `chunk.choices[0].delta.reasoning_content`. Please note that the thinking traces are only accessible when using streaming mode.

### Use native tool calling

`grok-code-fast-1` offers first-party support for native tool-calling and was specifically designed with native tool-calling in mind. We encourage you to use it instead of XML-based tool-call outputs, which may hurt performance.

### Give a detailed system prompt

Be thorough and give many details in your system prompt. A well-written system prompt which describes the task, expectations, and edge-cases the model should be aware of can make a night-and-day difference. For more inspiration, refer to the User Best Practices above.

### Introduce context to the model

`grok-code-fast-1` is accustomed to seeing a lot of context in the initial user prompt. We recommend developers to use XML tags or Markdown-formatted content to mark various sections of the context and to add clarity to certain sections. Descriptive Markdown headings/XML tags and their corresponding definitions will allow `grok-code-fast-1` to use the context more effectively.

### Optimize for cache hits

Our cache hits are a big contributor to `grok-code-fast-1`’s fast inference speed. In agentic tasks where the model uses multiple tools in sequence, most of the prefix remains the same and thus is automatically retrieved from the cache to speed up inference. We recommend against changing or augmenting the prompt history, as that could lead to cache misses and therefore significantly slower inference speeds.


===/docs/guides/image-generations===
#### Guides

# Image Generations

Some of the models can provide image generation capabilities. You can provide some descriptions of the image you would like to generate, and let the model generate one or multiple pictures in the output.

If you're used to interacting with the chat/image-understanding models, the image generation is a bit different from them.
You only need to send a prompt text in the request, instead of a list of messages with system/user/assistant roles.
When you sent the prompt for image generation, your prompt will be revised by a chat model, and then sent to the image generation model.

## Parameters

* `n`: Number of image(s) to generate (1-10, default to 1)
* `response_format`: `"url"` or `"b64_json"`. If `"url"` is specified, the response will return a url to the image(s) in `data[index].url`; if "b64\_json" is specified, the response will return the image(s) in base64 encoded format in `data[index].b64_json`.

> Note: `quality`, `size` or `style` are not supported by xAI API at the moment.

## Generate an image

The image generation is offered at a different endpoint `https://api.x.ai/v1/images/generations` from the chat and image-understanding models that share `https://api.x.ai/v1/chat/completions`.
The endpoint is **compatible with OpenAI SDK** (but **not with Anthropic SDK**), so you can keep using the same `base_url` of `https://api.x.ai/v1`.

You can set `"model": "grok-2-image"` in the request body to use the model. The generated image will be in `jpg` format.

```pythonXAI
import os

from xai_sdk import Client

client = Client(api_key=os.getenv('XAI_API_KEY'))

response = client.image.sample(
    model="grok-2-image",
    prompt="A cat in a tree",
    image_format="url"
)

print(response.url)
```

```pythonOpenAISDK
import os
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

response = client.images.generate(
    model="grok-2-image",
    prompt="A cat in a tree"
)

print(response.data[0].url)
```

```javascriptOpenAISDK
import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
});

const response = await openai.images.generate({
    model: "grok-2-image",
    prompt: "A cat in a tree",
});
console.log(response.data[0].url);
```

```bash
curl -X 'POST' https://api.x.ai/v1/images/generations \\
-H 'accept: application/json' \\
-H 'Authorization: Bearer <API_KEY>' \\
-H 'Content-Type: application/json' \\
-d '{
    "model": "grok-2-image",
    "prompt": "A cat in a tree"
}'
```

The Python and JavaScript examples will print out url of the image on xAI managed storage.

This is an example image generated from the above prompt:

### Base 64 JSON Output

Instead of getting an image url by default, you can choose to get a base64 encoded image instead.
To do so, you need to specify the `response_format` parameter to `"b64_json"`.

```pythonXAI
import os

from xai_sdk import Client

client = Client(api_key=os.getenv('XAI_API_KEY'))

response = client.image.sample(
    model="grok-2-image",
    prompt="A cat in a tree",
    image_format="base64"
)

print(response.image) # returns the raw image bytes
```

```pythonOpenAISDK
import os

from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

response = client.images.generate(
    model="grok-2-image",
    prompt="A cat in a tree",
    response_format="b64_json"
)

print(response.data[0].b64_json)
```

```javascriptOpenAISDK
import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
});

const response = await openai.images.generate({
    model: "grok-2-image",
    prompt: "A cat in a tree",
    response_format: "b64_json"
});
console.log(response.data[0].b64_json);
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { experimental_generateImage as generateImage } from 'ai';

const result = await generateImage({
  model: xai.image('grok-2-image'),
  prompt: 'A cat in a tree',
});

console.log(result.image.base64Data);
```

```bash
curl -X 'POST' https://api.x.ai/v1/images/generations \\
-H 'accept: application/json' \\
-H 'Authorization: Bearer <API_KEY>' \\
-H 'Content-Type: application/json' \\
-d '{
    "model": "grok-2-image",
    "prompt": "A cat in a tree",
    "response_format": "b64_json"
}'
```

You will get a `b64_json` field instead of `url` in the response image object.

### Generating multiple images

You can generate up to 10 images in one request by adding a parameter `n` in your request body. For example, to generate four images:

```pythonXAI
import os

from xai_sdk import Client

client = Client(api_key=os.getenv('XAI_API_KEY'))

response = client.image.sample_batch(
    model="grok-2-image",
    prompt="A cat in a tree",
    n=4
    image_format="url",
)

for image in response:
    print(response.url)
```

```pythonOpenAISDK
import os

from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

responses = client.images.generate(
    model="grok-2-image",
    prompt="A cat in a tree"
    n=4
)
for response in responses:
    print(response.url)
```

```javascriptOpenAISDK
import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
});

const response = await openai.images.generate({
    model: "grok-2-image",
    prompt: "A cat in a tree",
    n: 4
});
response.data.forEach((image) => {
    console.log(image.url);
});
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { experimental_generateImage as generateImage } from 'ai';

const result = await generateImage({
  model: xai.image('grok-2-image'),
  prompt: 'A cat in a tree',
  n: 4,
});

console.log(result.images);
```

```bash
curl -X 'POST' https://api.x.ai/v1/images/generations \\
-H 'accept: application/json' \\
-H 'Authorization: Bearer <API_KEY>' \\
-H 'Content-Type: application/json' \\
-d '{
    "model": "grok-2-image",
    "prompt": "A cat in a tree",
    "n": 4
}'
```

## Revised prompt

If you inspect the response object, you can see something similar to this:

```json
{
  "data": [
    {
      "b64_json": "data:image/png;base64,...",
      "revised_prompt": "..."
    }
  ]
}
```

Before sending the prompt to the image generation model, the prompt will be revised by a chat model. The revised prompt from chat model will be used by image generation model to create the image, and returned in `revised_prompt` to the user.

To see the revised prompt with SDK:

```pythonXAI
# ... Steps to make image generation request

print(response.prompt)
```

```pythonOpenAISDK
# ... Steps to make image generation request

print(response.data[0].revised_prompt)
```

```javascriptOpenAISDK
// ... Steps to make image generation request

console.log(response.data[0].revised_prompt);
```

For example:
| Input/Output | Example |
|------------------------------------ | -------- |
| prompt (in request body) | A cat in a tree |
| revised\_prompt (in response body) | 3D render of a gray cat with green eyes perched on a thick branch of a leafy tree, set in a suburban backyard during the day. The cat's fur is slightly ruffled by a gentle breeze, and it is looking directly at the viewer. The background features a sunny sky with a few clouds and other trees, creating a natural and serene environment. The scene is focused on the cat, with no distracting foreground elements, ensuring the cat remains the central subject of the image. |


===/docs/guides/image-understanding===
#### Guides

# Image Understanding

The vision model can receive both text and image inputs. You can pass images into the model in one of two ways: base64 encoded strings or web URLs.

Under the hood, image understanding shares the same API route and the same message body schema consisted of `system`/`user`/`assistant` messages. The difference is having image in the message content body instead of text.

As the knowledge in this guide is built upon understanding of the chat capability. It is suggested that you familiarize yourself with the [chat](chat) capability before following this guide.

## Prerequisites

* xAI Account: You need an xAI account to access the API.
* API Key: Ensure that your API key has access to the vision endpoint and a model supporting image input is enabled.

If you don't have these and are unsure of how to create one, follow [the Hitchhiker's Guide to Grok](../tutorial).

Set your API key in your environment:

```bash
export XAI_API_KEY="your_api_key"
```

## Reminder on image understanding model general limitations

It might be easier to run into model limit with these models than chat models:

* Maximum image size: `20MiB`
* Maximum number of images: No limit
* Supported image file types: `jpg/jpeg` or `png`.
* Any image/text input order is accepted (e.g. text prompt can precede image prompt)

## Constructing the Message Body - Difference from Chat

The request message to image understanding is similar to chat. The main difference is that instead of text input:

```json
[
  {
    "role": "user",
    "content": "What is in this image ?"
  }
]
```

We send in `content` as a list of objects:

```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,<base64_image_string>",
          "detail": "high"
        }
      },
      {
        "type": "text",
        "text": "What is in this image?"
      }
    ]
  }
]
```

The `image_url.url` can also be the image's url on the Internet.

You can use the text prompt to ask questions about the image(s), or discuss topics with the image as context to the discussion, etc.

## Image Detail Levels

The `"detail"` field controls the level of pre-processing applied to the image that will be provided to the model. It is optional and determines the resolution at which the image is processed. The possible values for `"detail"` are:

* **`"auto"`**: The system will automatically determine the image resolution to use. This is the default setting, balancing speed and detail based on the model's assessment.
* **`"low"`**: The system will process a low-resolution version of the image. This option is faster and consumes fewer tokens, making it more cost-effective, though it may miss finer details.
* **`"high"`**: The system will process a high-resolution version of the image. This option is slower and more expensive in terms of token usage, but it allows the model to attend to more nuanced details in the image.

## Web URL input

The model supports web URL as inputs for images. The API will fetch the image from the public URL and handle it as part of the chat. Integrating with URLs is as simple as:

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user, image

client = Client(api_key=os.getenv('XAI_API_KEY'))

image_url = "https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png"

chat = client.chat.create(model="grok-4")
chat.append(
    user(
        "What's in this image?",
        image(image_url=image_url, detail="high"),
    )
)

response = chat.sample()
print(response.content)
```

```pythonOpenAISDK
import os
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
image_url = (
"https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png"
)

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": "What's in this image?",
            },
        ],
    },
]

completion = client.chat.completions.create(
    model="grok-4",
    messages=messages,
)

print(completion.choices[0].message.content)
```

```javascriptOpenAISDK
import OpenAI from "openai";
const openai = new OpenAI({
    apiKey: process.env.XAI_API_KEY,
    baseURL: "https://api.x.ai/v1",
});
const image_url =
"https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png";

const completion = await openai.chat.completions.create({
    model: "grok-4",
    messages: [
        {
            role: "user",
            content: [
                {
                    type: "image_url",
                    image_url: {
                        url: image_url,
                        detail: "high",
                    },
                },
                {
                    type: "text",
                    text: "What's in this image?",
                },
            ],
        },
    ],
});

console.log(completion.choices[0].message.content);
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: "grok-4",
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'image',
          image: new URL(
            'https://science.nasa.gov/wp-content/uploads/2023/09/web-first-images-release.png',
          ),
        },
        {
          type: 'text',
          text: "What's in this image?",
        },
      ],
    },
  ],
});

console.log(result.text);
```

## Base64 string input

You will need to pass in base64 encoded image directly in the request, in the user messages.

Here is an example of how you can load a local image, encode it in Base64 and use it as part of your conversation:

```pythonXAI
import os
import base64

from xai_sdk import Client
from xai_sdk.chat import user, image

client = Client(api_key=os.getenv('XAI_API_KEY'))
image_path = "..."

chat = client.chat.create(model="grok-4")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Getting the base64 string

base64_image = encode_image(image_path)

# assumes jpeg image, update image format in the url accordingly

chat.append(
    user(
        "What's in this image?",
        image(image_url=f"data:image/jpeg;base64,{base64_image}", detail="high"),
    )
)

response = chat.sample()
print(response.content)
```

```pythonOpenAISDK
import os
from openai import OpenAI
import base64

XAI_API_KEY = os.getenv("XAI_API_KEY")
image_path = "..."

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# Getting the base64 string

base64_image = encode_image(image_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": "What's in this image?",
            },
        ],
    },
]

completion = client.chat.completions.create(
    model="grok-4",
    messages=messages,
)

print(completion.choices[0].message.content)
```

```javascriptOpenAISDK
import { promises as fs } from "fs";
import OpenAI from "openai";

const openai = new OpenAI({
    apiKey: process.env.XAI_API_KEY,
    baseURL: "https://api.x.ai/v1",
});
const image_path =
"...";

async function getBase64(filePath) {
try {
        const buffer = await fs.readFile(filePath);
        let base64 = buffer.toString("base64");
        while (base64.length % 4 > 0) {
            base64 += "=";
        }
        return base64;
    } catch (error) {
        throw error;
    }
}

const base64_image = await getBase64(image_path);

const completion = await openai.chat.completions.create({
    model: "grok-4",
    messages: [
        {
            role: "user",
            content: [
                {
                    type: "image_url",
                    image_url: {
                        url: \`data:image/jpeg;base64,$\{base64_image\}\`,
                        detail: "high",
                    },
                },
                {
                    type: "text",
                    text: "What's in this image?",
                },
            ],
        },
    ],
});

console.log(completion.choices[0].message.content);
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { generateText } from 'ai';
import fs from 'fs';

const result = await generateText({
  model: xai('grok-4'),
  messages: [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Describe the image in detail.' },
        { type: 'image', image: fs.readFileSync('./data/comic-cat.png') },
      ],
    },
  ],
});

console.log(result.text);
```

## Multiple images input

You can send multiple images in the prompt, for example:

```pythonXAI
chat.append(
    user(
        "What are in these images?",
        image(image_url=f"data:image/jpeg;base64,{base64_image1}", detail="high"),
        image(image_url=f"data:image/jpeg;base64,{base64_image2}", detail="high")
    )
)
```

```pythonOpenAISDK
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image1}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": "What are in these images?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image2}",
                    "detail": "high",
                }
            }
        ],
    },
]
```

```javascriptOpenAISDK
messages: [
        {
            role: "user",
            content: [
                {
                    type: "image_url",
                    image_url: {
                        url: \`data:image/jpeg;base64,$\{base64_image1\}\`,
                        detail: "high",
                    },
                },
                {
                    type: "text",
                    text: "What are in these images?",
                },
                {
                    type: "image_url",
                    image_url: {
                        url: \`data:image/jpeg;base64,$\{base64_image2\}\`,
                        detail: "high",
                    },
                },
            ]
        }
    ],
```

```javascriptAISDK
messages: [
  {
    role: 'user',
    content: [
      {
        type: 'image',
        image: \`data:image/jpeg;base64,$\{base64Image1\}\`,
      },
      {
        type: 'text',
        text: 'What are in these images?',
      },
      {
        type: 'image',
        image: \`data:image/jpeg;base64,$\{base64Image2\}\`,
      },
    ],
  },
],
```

The image prompts can interleave with text prompts in any order.

## Image token usage

The prompt image token usage is provided in the API response. Each image will be automatically broken down into tiles of 448x448 pixels, and each tile will consume 256 tokens. The final generation will include an extra tile, so each image would consume `(# of tiles + 1) * 256` tokens. There is a maximum limit of 6 tiles, so your input would consume less than 1,792 tokens per image.

```pythonXAI
print(response.usage.prompt_image_tokens)
```

```pythonOpenAISDK
# Stream response
print(next(stream).usage.prompt_tokens_details.image_tokens)

# Non-stream response

print(response.usage.prompt_tokens_details.image_tokens)
```


===/docs/guides/live-search===
#### Guides

# Live Search

The advanced agentic search capabilities powering grok.com are generally available in the new [**agentic tool calling API**](/docs/guides/tools/overview), and the Live Search API will be deprecated by December 15, 2025.

The chat completion endpoint supports querying live data and considering those in generating responses. With this
functionality, instead of orchestrating web search and LLM tool calls yourself, you can get chat responses with
live data directly from the API.

Live search is available via the chat completions endpoint. It is turned off by default. Customers have control over the
content they access, and we are not liable for any resulting damages or liabilities.

For more details, refer to `search_parameters` in [API Reference - Chat completions](../api-reference#chat-completions).

For examples on search sources, jump to [Data Sources and Parameters](#data-sources-and-parameters).

## Live Search Pricing

Live Search costs **$25 per 1,000 sources used**. That means each source costs $0.025.

The number of sources used can be found in the `response` object, which contains a field called `response.usage.num_sources_used`.

## Enabling Search

To enable search, you need to specify in your chat completions request an additional field
`search_parameters`, with `"mode"` from one of `"auto"`, `"on"`, `"off"`.

If you want to use Live Search with default values, you still need to specify an empty `search_parameters`.

```json
"search_parameters": {}
```

Or if using xAI Python SDK:

```pythonWithoutSDK
search_parameters=SearchParameters(),
```

The `"mode"` field sets the preference of data source: - `"off"`: Disables search and uses the model without accessing additional information from data sources. - `"auto"` (default): Live search is available to the model, but the model automatically decides whether to perform
live search. - `"on"`: Enables live search.

The model decides which data source to use within the provided data sources, via the `"sources"` field in
`"search_parameters"`. If no `"sources"` is provided, live search will default to making web and X data available to
the model.

For example, you can send the following request, where the model will decide whether to search in data:

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters

client = Client(api_key=os.getenv("XAI_API_KEY"))

chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(mode="auto"),
)

chat.append(user("Provide me a digest of world news of the week before July 9, 2025."))

response = chat.sample()
print(response.content)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Provide me a digest of world news of the week before July 9, 2025."
        }
    ],
    "search_parameters": {
        "mode": "auto"
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'Provide me a digest of world news of the week before July 9, 2025.',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "Provide me a digest of world news of the week before July 9, 2025."
        }
    ],
    "search_parameters": {
        "mode": "auto"
    },
    "model": "grok-4"
}'
```

## Returning citations

The live search endpoint supports returning citations to the data sources used in the response in the form of a list of URLs. To enable this, you can set `"return_citations": true` in your search parameters. This field defaults to `true`.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters

client = Client(api_key=os.getenv("XAI_API_KEY"))

chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto",
        return_citations=True,
    ),
)
chat.append(user("Provide me a digest of world news on July 9, 2025."))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Provide me a digest of world news on July 9, 2025."
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "return_citations": True
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'Provide me a digest of world news on July 9, 2025.',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        returnCitations: true,
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "Provide me a digest of world news on July 9, 2025."
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "return_citations": true
    },
    "model": "grok-4"
}'
```

### Streaming behavior with citations

During streaming, you would get the chat response chunks as usual. The citations will be returned as a list of url
strings in the field `"citations"` only in the last chunk. This is similar to how the usage data is returned with
streaming.

## Set date range of the search data

You can restrict the date range of search data used by specifying `"from_date"` and `"to_date"`. This limits the
data to the period from `"from_date"` to `"to_date"`, including both dates.

Both fields need to be in ISO8601 format, e.g. "YYYY-MM-DD". If you're using the xAI Python SDK, the `from_date` and `to_date` fields can be passed as `datetime.datetime` objects to the `SearchParameters` class.

The fields can also be independently used. With only `"from_date"` specified, the data used will be from the
`"from_date"` to today, and with only `"to_date"` specified, the data used will be all data till the `"to_date"`.

```pythonXAI
import os
from datetime import datetime

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters

client = Client(api_key=os.getenv('XAI_API_KEY'))

chat = client.chat.create(
    model="grok-4",
    search_parameters = SearchParameters(
        mode="auto",
        from_date=datetime(2022, 1, 1),
        to_date=datetime(2022, 12, 31)
    )
)
chat.append(user("What is the most viral meme in 2022?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What is the most viral meme in 2022?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "from_date": "2022-01-01",
        "to_date": "2022-12-31"
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'What is the most viral meme in 2022?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        fromDate: '2022-01-01',
        toDate: '2022-12-31',
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "What is the most viral meme in 2022?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "from_date": "2022-01-01",
        "to_date": "2022-12-31"
    },
    "model": "grok-4"
}'
```

## Limit the maximum amount of data sources

You can set a limit on how many data sources will be considered in the query via `"max_search_results"`.
The default limit is 20.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters

client = Client(api_key=os.getenv("XAI_API_KEY"))

chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto",
        max_search_results=10,
    ),
)
chat.append(user("Can you recommend the top 10 burger places in London?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Can you recommend the top 10 burger places in London?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "max_search_results": 10
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'Can you recommend the top 10 burger places in London?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        maxSearchResults: 10,
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "Can you recommend the top 10 burger places in London?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "max_search_results": 10
    },
    "model": "grok-4"
}'
```

## Data sources and parameters

In `"sources"` of `"search_parameters"`, you can add a list of sources to be potentially used in search. Each source is
an object with source name and parameters for that source, with the name of the source in the `"type"` field.

If nothing is specified, the sources to be used will default to `"web"`, `"news"` and `"x"`.

For example, the following enables web, X search, news and rss:

```json
"sources": [
  {"type": "web"},
  {"type": "x"},
  {"type": "news"},
  {"type": "rss"}
]
```

### Overview of data sources and supported parameters

| Data Source | Description                                 | Supported Parameters                                                                         |
| ----------- | ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `"web"`     | Searching on websites.                      | `"country"`, `"excluded_websites"`, `"allowed_websites"`, `"safe_search"`                    |
| `"x"`       | Searching X posts.                          | `"included_x_handles"`, `"excluded_x_handles"`, `"post_favorite_count"`, `"post_view_count"` |
| `"news"`    | Searching from news sources.                | `"country"`, `"excluded_websites"`, `"safe_search"`                                          |
| `"rss"`     | Retrieving data from the RSS feed provided. | `"links"`                                                                                    |

### Parameter `"country"` (Supported by Web and News)

Sometimes you might want to include data from a specific country/region. To do so, you can add an ISO alpha-2 code of
the country to `"country"` in `"web"` or `"news"` of the `"sources"`.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters, web_source

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto",
        sources=[web_source(country="CH")],
    ),
)
chat.append(user("Where is the best place to go skiing this year?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Where is the best place to go skiing this year?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [{ "type": "web", "country": "CH" }]
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'Where is the best place to go skiing this year?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        sources: [
          {
            type: 'web',
            country: 'CH',
          },
        ],
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "Where is the best place to go skiing this year?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [{ "type": "web", "country": "CH" }]
    },
    "model": "grok-4"
}'
```

### Parameter `"excluded_websites"` (Supported by Web and News)

Use `"excluded_websites"`to exclude websites from the query. You can exclude a maximum of five websites.

This cannot be used with `"allowed_websites"` on the same search source.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters, news_source, web_source

client = Client(api_key=os.getenv("XAI_API_KEY"))

chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto",
        sources=[
            web_source(excluded_websites=["wikipedia.org"]),
            news_source(excluded_websites=["bbc.co.uk"]),
        ],
    ),
)
chat.append(user("What are some recently discovered alternative DNA shapes"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are some recently discovered alternative DNA shapes?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [
            { "type": "web", "excluded_websites": ["wikipedia.org"] },
            { "type": "news", "excluded_websites": ["bbc.co.uk"] }
        ]
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'What are some recently discovered alternative DNA shapes',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        sources: [
          {
            type: 'web',
            excludedWebsites: ['wikipedia.org'],
          },
          {
            type: 'news',
            excludedWebsites: ['bbc.co.uk'],
          },
        ],
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "What are some recently discovered alternative DNA shapes?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [
            { "type": "web", "excluded_websites": ["wikipedia.org"] },
            { "type": "news", "excluded_websites": ["bbc.co.uk"] }
        ]
    },
    "model": "grok-4"
}'
```

### Parameter `"allowed_websites"` (Supported by Web)

Use `"allowed_websites"`to allow only searching on these websites for the query. You can include a
maximum of five websites.

This cannot be used with `"excluded_websites"` on the same search source.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters, web_source

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
    mode="auto",
    sources=[web_source(allowed_websites=["x.ai"])],
),
)
chat.append(user("What are the latest releases at xAI?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are the latest releases at xAI?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [
            { "type": "web", "allowed_websites": ["x.ai"] },
        ]
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'What are the latest releases at xAI?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        sources: [
          {
            type: 'web',
            allowedWebsites: ['x.ai'],
          },
        ],
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
            {
                "role": "user",
                "content": "What are the latest releases at xAI?"
            }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [
            { "type": "web", "allowed_websites": ["x.ai"] },
        ]
    },
    "model": "grok-4"
}'
```

### Parameter `"included_x_handles"` (Supported by X)

Use `"included_x_handles"` to consider X posts only from a given list of X handles. The maximum number of handles you can include is 10.

This parameter cannot be set together with `"excluded_x_handles"`.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters, x_source

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto",
        sources=[x_source(included_x_handles=["xai"])],
    ),
)
chat.append(user("What are the latest updates from xAI?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are the latest updates from xAI?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [{ "type": "x", "included_x_handles": ["xai"] }]
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'What are the latest updates from xAI?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        sources: [
          {
            type: 'x',
            includedXHandles: ['xai'],
          },
        ],
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "What are the latest updates from xAI?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [{ "type": "x", "included_x_handles": ["xai"] }]
    },
    "model": "grok-4"
}'
```

### Parameter `"excluded_x_handles"` (Supported by X)

Use `"excluded_x_handles"` to exclude X posts from a given list of X handles. The maximum number of handles you can exclude is 10.

This parameter cannot be set together with `"included_x_handles"`.

To prevent the model from citing itself in its responses, the `"grok"` handle is automatically
excluded by default. If you want to include posts from `"grok"` in your search, you must pass it
explicitly in the `"included_x_handles"` parameter.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters, x_source

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto",
        sources=[x_source(excluded_x_handles=["xai"])],
    ),
)
chat.append(user("What are people saying about xAI?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are people saying about xAI?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [{ "type": "x", "excluded_x_handles": ["xai"] }]
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'What are people saying about xAI?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        sources: [
          {
            type: 'x',
            excludedXHandles: ['xai'],
          },
        ],
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "What are people saying about xAI?"
        }
    ],
    "search_parameters": {
        "mode": "auto",
        "sources": [{ "type": "x", "excluded_x_handles": ["xai"] }]
    },
    "model": "grok-4"
}'
```

### Parameters `"post_favorite_count"` and `"post_view_count"` (Supported by X)

Use `"post_favorite_count"` and `"post_view_count"` to filter X posts by the number of favorites and views they have. Only posts with at least the specified number of favorites **and** views will be considered.

You can set both parameters to consider posts with at least the specified number of favorites **and** views.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters, x_source

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto", # Only consider posts with at least 1000 favorites and 20000 views
        sources=[x_source(post_favorite_count=1000, post_view_count=20000)],
    ),
)
chat.append(user("What are the most popular X posts?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are people saying about xAI?"
        }
    ],
    "search_parameters": {
        "mode": "auto", # Only consider posts with at least 1000 favorites and 20000 views
        "sources": [{ "type": "x", "post_favorite_count": 1000, "post_view_count": 20000 }]
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'What are the most popular X posts?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        sources: [
          {
            type: 'x',
            postFavoriteCount: 1000,
            postViewCount: 20000,
          },
        ],
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "What are people saying about xAI?"
        }
    ],
    "search_parameters": {
        "mode": "auto", # Only consider posts with at least 1000 favorites and 20000 views
        "sources": [{ "type": "x", "post_favorite_count": 1000, "post_view_count": 20000 }]
    },
    "model": "grok-4"
}'
```

### Parameter `"link"` (Supported by RSS)

You can also fetch data from a list of RSS feed urls via `{ "links": ... }`. You can only add one RSS
link at the moment.

For example:

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.search import SearchParameters, rss_source

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4",
    search_parameters=SearchParameters(
        mode="auto",
        sources=[rss_source(links=["https://status.x.ai/feed.xml"])],
    ),
)
chat.append(user("What are the latest updates on Grok?"))

response = chat.sample()
print(response.content)
print(response.citations)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "What are the latest updates on Grok?"
        }
    ],
    "search_parameters": {
        "mode": "on",
        "sources": [{"type": "rss", "links": ["https://status.x.ai/feed.xml"]}]
    },
    "model": "grok-4"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```javascriptAISDK
import { xai, XaiProviderOptions } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  prompt: 'What are the latest updates on Grok?',
  providerOptions: {
    xai: {
      searchParameters: {
        mode: 'auto',
        sources: [
          {
            type: 'rss',
            links: ['https://status.x.ai/feed.xml'],
          },
        ],
      },
    } satisfies XaiProviderOptions,
  },
});

console.log(result.text);
console.log(result.sources);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
    "messages": [
        {
            "role": "user",
            "content": "What are the latest updates on Grok?"
        }
    ],
    "search_parameters": {
        "mode": "on",
        "sources": [{ "type": "rss", "links": ["https://status.x.ai/feed.xml"] }]
    },
    "model": "grok-4"
}'
```

### Parameter `"safe_search"` (Supported by Web and News)

Safe search is on by default. You can disable safe search for `"web"` and `"news"` via
`"sources": [{..., "safe_search": false }]`.


===/docs/guides/migration===
#### Guides

# Migration from Other Providers

Some of Grok users might have migrated from other LLM providers. xAI API is designed to be compatible with both OpenAI and Anthropic SDKs, except certain capabilities not offered by respective SDK.
If you can use either SDKs, we recommend using OpenAI SDK for better stability.

In two steps:

1. At API client object construction, you need to set the "base url" to `https://api.x.ai/v1` and "API key" to your xAI API key (obtained from [xAI Console](https://console.x.ai)).
2. When sending message for inference, set "model" to be one of the Grok [model](../models) names.

If you use third-party tools such as LangChain ([JavaScript](https://js.langchain.com/docs/integrations/chat/xai/)/[Python](https://python.langchain.com/docs/integrations/providers/xai/)) and [Continue](https://docs.continue.dev/customize/model-providers/xai),
they usually have a common base class for LLM providers. You only need to change the provider and API keys. You can refer to their documentations for case-by-case instrcutions.

Examples using OpenAI and Anthropic SDKs:

**OpenAI SDK**

```pythonOpenAISDK
from openai import OpenAI

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# ...

completion = client.chat.completions.create(
    model="grok-4",
# ...
)
```

```javascriptOpenAISDK
import OpenAI from "openai";

const openai = new OpenAI({
    apiKey: $XAI_API_KEY,
    baseURL: "https://api.x.ai/v1",
});

// ...

const completion = await openai.chat.completions.create({
    model: "grok-4",
    // ...
```

**Anthropic SDK**

```pythonAnthropicSDK
from anthropic import Anthropic

client = Anthropic(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai",
)

# ...

message = client.messages.create(
    model="grok-4",
# ...
)
```

```javascriptAnthropicSDK
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
    apiKey: $XAI_API_KEY,
    baseURL: "https://api.x.ai/",
});

// ...

const msg = await anthropic.messages.create({
    model: "grok-4",
// ...
```


===/docs/guides/reasoning===
#### Guides

# Reasoning

`grok-4-fast-non-reasoning` variant is based on `grok-4-fast-reasoning` with reasoning disabled.

`presencePenalty`, `frequencyPenalty` and `stop` parameters are not supported by reasoning models.
Adding them in the request would result in an error.

## Key Features

* **Think Before Responding**: Thinks through problems step-by-step before delivering an answer.
* **Math & Quantitative Strength**: Excels at numerical challenges and logic puzzles.
* **Reasoning Trace**: The model's thoughts are available via the `reasoning_content` or `encrypted_content` field in the response completion object (see example below).

You can access the model's raw thinking trace via the `message.reasoning_content` of the chat completion response. Only `grok-3-mini` returns `reasoning_content`.

`grok-3`, `grok-4` and `grok-4-fast-reasoning` do not return `reasoning_content`. It may optionally return [encrypted reasoning content](#encrypted-reasoning-content) instead.

### Encrypted Reasoning Content

For `grok-4`, the reasoning content is encrypted by us and sent back if `use_encrypted_content` is set to `true`. You can send the encrypted content back to provide more context to a previous conversation. See [Stateful Response with Responses API](responses-api) for more details on how to use the content.

## Control how hard the model thinks

`reasoning_effort` is not supported by `grok-3`, `grok-4` and `grok-4-fast-reasoning`. Specifying `reasoning_effort` parameter will get
an error response. Only `grok-3-mini` supports `reasoning_effort`.

The `reasoning_effort` parameter controls how much time the model spends thinking before responding. It must be set to one of these values:

* **`low`**: Minimal thinking time, using fewer tokens for quick responses.
* **`high`**: Maximum thinking time, leveraging more tokens for complex problems.

Choosing the right level depends on your task: use `low` for simple queries that should complete quickly, and `high` for harder problems where response latency is less important.

## Usage Example

Here’s a simple example using `grok-3-mini` to multiply 101 by 3.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import system, user

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

chat = client.chat.create(
    model="grok-3-mini",
    reasoning_effort="high",
    messages=[system("You are a highly intelligent AI assistant.")],
)
chat.append(user("What is 101\*3?"))

response = chat.sample()

print("Final Response:")
print(response.content)

print("Number of completion tokens:")
print(response.usage.completion_tokens)

print("Number of reasoning tokens:")
print(response.usage.reasoning_tokens)
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

messages = [
{
    "role": "system",
    "content": "You are a highly intelligent AI assistant.",
},
{
    "role": "user",
    "content": "What is 101*3?",
},
]

client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.getenv("XAI_API_KEY"),
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

completion = client.chat.completions.create(
    model="grok-3-mini",
    reasoning_effort="high",
    messages=messages,
)

print("Final Response:")
print(completion.choices[0].message.content)

print("Number of completion tokens:")
print(completion.usage.completion_tokens)

print("Number of reasoning tokens:")
print(completion.usage.completion_tokens_details.reasoning_tokens)
```

```javascriptOpenAISDK
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const completion = await client.chat.completions.create({
    model: "grok-3-mini",
    reasoning_effort: "high",
    messages: [
        {
            "role": "system",
            "content": "You are a highly intelligent AI assistant.",
        },
        {
            "role": "user",
            "content": "What is 101*3?",
        },
    ],
});


console.log("\\nFinal Response:", completion.choices[0].message.content);

console.log("\\nNumber of completion tokens (input):", completion.usage.completion_tokens);

console.log("\\nNumber of reasoning tokens (input):", completion.usage.completion_tokens_details.reasoning_tokens);
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { generateText } from 'ai';

const result = await generateText({
  model: xai('grok-4'),
  system: 'You are a highly intelligent AI assistant.',
  prompt: 'What is 101*3?',
});

console.log('Final Response:', result.text);
console.log('Number of completion tokens:', result.totalUsage.completionTokens);
console.log('Number of reasoning tokens:', result.totalUsage.reasoningTokens);
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "messages": [
        {
            "role": "system",
            "content": "You are a highly intelligent AI assistant."
        },
        {
            "role": "user",
            "content": "What is 101*3?"
        }
    ],
    "model": "grok-3-mini",
    "reasoning_effort": "high",
    "stream": false
}'
```

### Sample Output

```output

Final Response:
The result of 101 multiplied by 3 is 303.

Number of completion tokens:
14

Number of reasoning tokens:
310
```

## Notes on Consumption

When you use a reasoning model, the reasoning tokens are also added to your final consumption amount. The reasoning token consumption will likely increase when you use a higher `reasoning_effort` setting.


===/docs/guides/responses-api===
#### Guides

# Stateful Response with Responses API

Responses API is a new way of interacting with our models via API. It allows a **stateful interaction** with our models,
where **previous input prompts, reasoning content and model responses are saved by us**. A user can continue the interaction by appending new
prompt messages, rather than sending all of the previous messages.

Although you don't need to enter the conversation history in the request body, you will still be
billed for the entire conversation history when using Responses API. The cost might be reduced as
the conversation history might be [automatically cached](../models#cached-prompt-tokens).

**The responses will be stored for 30 days, after which they will be removed.** If you want to continue a response after 30 days, please store your responses history as well as the encrypted thinking content to create a new response. The encrypted thinking content can then be sent in the request body to give you a better result. See [Returning encrypted thinking content](#returning-encrypted-thinking-content) for more information on retrieving encrypted content.

&#x20;The Responses API is not yet supported in the Vercel AI SDK. Please use the xAI SDK or OpenAI SDK for this functionality.

## Prerequisites

* xAI Account: You need an xAI account to access the API.
* API Key: Ensure that your API key has access to the chat endpoint and the chat model is enabled.

If you don't have these and are unsure of how to create one, follow [the Hitchhiker's Guide to Grok](../tutorial).

You can create an API key on the [xAI Console API Keys Page](https://console.x.ai/team/default/api-keys).

Set your API key in your environment:

```bash
export XAI_API_KEY="your_api_key"
```

## Creating a new model response

The first step in using Responses API is analogous to using Chat Completions API. You will create a new response with prompts.

`instructions` parameter is currently not supported. The API will return an error if it is specified.

When sending images, it is advised to set `store` parameters to `false`. Otherwise the request may fail.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600,
)

chat = client.chat.create(model="grok-4", store_messages=True)
chat.append(system("You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."))
chat.append(user("What is the meaning of life, the universe, and everything?"))
response = chat.sample()

print(response)

# The response id that can be used to continue the conversation later

print(response.id)
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

response = client.responses.create(
    model="grok-4",
    input=[
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ],
)

print(response)

# The response id that can be used to continue the conversation later

print(response.id)
```

```javascriptOpenAISDK
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const response = await client.responses.create({
    model: "grok-4",
    input: [
        {
            role: "system",
            content: "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."
        },
        {
            role: "user",
            content: "What is the meaning of life, the universe, and everything?"
        },
    ],
});

console.log(response);

// The response id that can be used to recall the conversation later
console.log(response.id);
```

```bash
curl https://api.x.ai/v1/responses \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "model": "grok-4",
    "input": [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        }
    ]
}'
```

If no system prompt is desired, for non-xAI SDK users, the request's input parameter can be simplified as a string user prompt:

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600,
)

chat = client.chat.create(model="grok-4", store_messages=True)
chat.append(user("What is 101\*3"))
response = chat.sample()

print(response)

# The response id that can be used to continue the conversation later

print(response.id)
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

response = client.responses.create(
    model="grok-4",
    input="What is 101\*3?",
)

print(response)

# The response id that can be used to continue the conversation later

print(response.id)
```

```javascriptWithoutSDK
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const response = await client.responses.create({
    model: "grok-4",
    input: "What is 101\*3?",
});

console.log(response);

// The response id that can be used to recall the conversation later
console.log(response.id);
```

```bash
curl https://api.x.ai/v1/responses \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "model": "grok-4",
    "input": "What is 101\*3?"
}'
```

### Returning encrypted thinking content

If you want to return the encrypted thinking traces, you need to specify `use_encrypted_content=True` in xAI SDK or gRPC request message, or `include: ["reasoning.encrypted_content"]` in the request body.

Modify the steps to create a chat client (xAI SDK) or change the request body as following:

```pythonXAI
chat = client.chat.create(model="grok-4",
        store_messages=True,
        use_encrypted_content=True)
```

```pythonOpenAISDK
response = client.responses.create(
    model="grok-4",
    input=[
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ],
    include=["reasoning.encrypted_content"]
)
```

```javascriptWithoutSDK
const response = await client.responses.create({
    model: "grok-4",
    input: [
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ],
    include: ["reasoning.encrypted_content"],
});
```

```bash
curl https://api.x.ai/v1/responses \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "model": "grok-4",
    "input": [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        }
    ],
    "include": ["reasoning.encrypted_content"]
}'
```

See [Adding encrypted thinking content](#adding-encrypted-thinking-content) on how to use the returned encrypted thinking content.

## Chaining the conversation

We now have the `id` of the first response. With Chat Completions API, we typically send a stateless new request with all the previous messages.

With Responses API, we can send the `id` of the previous response, and the new messages to append to it.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600,
)

chat = client.chat.create(model="grok-4", store_messages=True)
chat.append(system("You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."))
chat.append(user("What is the meaning of life, the universe, and everything?"))
response = chat.sample()

print(response)

# The response id that can be used to continue the conversation later

print(response.id)

# New steps

chat = client.chat.create(
    model="grok-4",
    previous_response_id=response.id,
    store_messages=True,
)
chat.append(user("What is the meaning of 42?"))
second_response = chat.sample()

print(second_response)

# The response id that can be used to continue the conversation later

print(second_response.id)
```

```pythonOpenAISDK
# Previous steps
import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

response = client.responses.create(
    model="grok-4",
    input=[
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ],
)

print(response)

# The response id that can be used to continue the conversation later

print(response.id)

# New steps

second_response = client.responses.create(
    model="grok-4",
    previous_response_id=response.id,
    input=[
        {"role": "user", "content": "What is the meaning of 42?"},
    ],
)

print(second_response)

# The response id that can be used to continue the conversation later

print(second_response.id)
```

```javascriptWithoutSDK
// Previous steps
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const response = await client.responses.create({
    model: "grok-4",
    input: [
        {
            role: "system",
            content: "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."
        },
        {
            role: "user",
            content: "What is the meaning of life, the universe, and everything?"
        },
    ],
});

console.log(response);

// The response id that can be used to recall the conversation later
console.log(response.id);

const secondResponse = await client.responses.create({
    model: "grok-4",
    previous_response_id: response.id,
    input: [
        {"role": "user", "content": "What is the meaning of 42?"},
    ],
});

console.log(secondResponse);

// The response id that can be used to recall the conversation later
console.log(secondResponse.id);
```

```bash
curl https://api.x.ai/v1/responses \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "model": "grok-4",
    "previous_response_id": "The previous response id",
    "input": [
        {
            "role": "user",
            "content": "What is the meaning of 42?"
        }
    ]
}'
```

### Adding encrypted thinking content

After returning the encrypted thinking content, you can also add it to a new response's input:

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600,
)

chat = client.chat.create(model="grok-4", store_messages=True, use_encrypted_content=True)
chat.append(system("You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."))
chat.append(user("What is the meaning of life, the universe, and everything?"))
response = chat.sample()

print(response)

# The response id that can be used to continue the conversation later

print(response.id)

# New steps

chat.append(response)  ## Append the response and the SDK will automatically add the outputs from response to message history

chat.append(user("What is the meaning of 42?"))
second_response = chat.sample()

print(second_response)

# The response id that can be used to continue the conversation later

print(second_response.id)
```

```pythonOpenAISDK
# Previous steps
import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

response = client.responses.create(
    model="grok-4",
    input=[
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ],
    include=["reasoning.encrypted_content"]
)

print(response)

# The response id that can be used to continue the conversation later

print(response.id)

# New steps

second_response = client.responses.create(
    model="grok-4",
    input=[
        *response.output,  # Use response.output instead of the stored response
        {"role": "user", "content": "What is the meaning of 42?"},
    ],
)

print(second_response)

# The response id that can be used to continue the conversation later

print(second_response.id)
```

```javascriptWithoutSDK
// Previous steps
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const response = await client.responses.create({
    model: "grok-4",
    input: [
        {
            role: "system",
            content: "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."
        },
        {
            role: "user",
            content: "What is the meaning of life, the universe, and everything?"
        },
    ],
    include: ["reasoning.encrypted_content"],
});

console.log(response);

// The response id that can be used to recall the conversation later
console.log(response.id);

const secondResponse = await client.responses.create({
    model: "grok-4",
    input: [
        ...response.output,  // Use response.output instead of the stored response
        {"role": "user", "content": "What is the meaning of 42?"},
    ],
});

console.log(secondResponse);

// The response id that can be used to recall the conversation later
console.log(secondResponse.id);
```

```bash
curl https://api.x.ai/v1/responses \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "model": "grok-4",
    "input": [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        },
        {
            "id": "rs_51abe1aa-599b-80b6-57c8-dddc6263362f_us-east-1",
            "summary": [],
            "type": "reasoning",
            "status": "completed",
            "encrypted_content": "bvV88j99ILvgfHRTHCUSJtw+ISji6txJzPdZNbcSVuDk4OMG2Z9r5wOBBwjd3u3Hhm9XtpCWJO1YgTOlpgbn+g7DZX+pOagYYrCFUpQ19XkWz6Je8bHG9JcSDoGDqNgRbDbAUO8at6RCyqgPupJj5ArBDCt73fGQLTC4G3S0JMK9LsPiWz6GPj6qyzYoRzkj4R6bntRm74E4h8Y+z6u6B7+ixPSv8s1EFs8c+NUAB8TNKZZpXZquj2LXfx1xAie85Syl7qLqxLNtDG1dNBhBnHpYoE4gQzwyXqywf5pF2Q2imzPNzGQhurK+6gaNWgZbxRmjhdsW6TnzO5Kk6pzb5qpfgfcEScQeYHSj5GpD+yDUCNlhdbzhhWnEErH+wuBPpTG6UQhiC7m7yrJ7IY2E8K/BeUPlUvkhMaMwb4dA279pWMJdchNJ+TAxca+JVc80pXMG/PmrQUNJU9qdXRLbNmQbRadBNwV2qkPfgggL3q0yNd7Un9P+atmP3B9keBILif3ufsBDtVUobEniiyGV7YVDvQ/fQRVs7XDxJiOKkogjjQySyHgpjseO8iG5xtb9mrz6B3mDvv2aAuyDL6MHZRM7QDVPjUbgNMzDm5Sm3J7IhtzfR+3eMDws3qeTsxOt1KOslu983Btv1Wx37b5HJqX1pQU1dae/kOSJ7MifFd6wMkQtQBDgVoG3ka9wq5Vxq9Ki8bDOOMcwA2kUXhCcY3TZCXJfDWSKPTcCoNCYIv5LT2NFVdamiSfLIyeOjBNz459BfMvAoOZShFViQyc5YwjnReUQPQ8a18jcz8GoAK1O99e0h91oYxIgDV52EfS+IYrzqvJOEQbKQinB+LJwkPbBEp7ZtgAtiNBzm985hNgLfiBaVFWcRYwI3tNBCT1vkw2YI0NEEG0yOF29x+u64XzqyP1CX1pU6sGXEFn3RPdfYibf6bt/Y1BRqBL5l0CrXWsgDw02SqIFta8OvJ7Iwmq40/4acE/Ew6eWO/z2MHkWgqSpwGNjn7MfeKkTi44foZjfNqN9QOFQt6VG2tY+biKZDo0h9DAftae8Q2Xs2UDvsBYOm7YEahVkput6/uKzxljpXlz269qHk6ckvdN9hKLbaTO3/IZPCCPQ5a/a/sWn/1VOJj72sDk+23RNjBf0FL6bJMXZI5aQdtxbF1zij9mWcP9nJ9FHhj53ytuf1NiKl5xU8ZsaoKmCAJcXUz1n2FZvyWlqvgPYiszc7R8Y5dF6QbW2mlKnXzVy6qRMHNeQqGhCEncyT5nPNSdK5QlUwLokAIg"
        },
        {
            "content": [
                {
                    "type": "output_text",
                    "text": "42\n\nThis is, of course, the iconic answer from Douglas Adams' *The Hitchhiker's Guide to the Galaxy*, where a supercomputer named Deep Thought spends 7.5 million years computing the \"Answer to the Ultimate Question of Life, the Universe, and Everything\"—only to reveal it's 42. (The real challenge, it turns out, is figuring out what the actual *question* was.)\n\nIf you're asking in a more literal or philosophical sense, the universe doesn't have a single tidy answer—it's full of mysteries like quantum mechanics, dark matter, and why cats knock things off tables. But 42? That's as good a starting point as any. What's your take on it?",
                    "logprobs": null,
                    "annotations": []
                }
            ],
            "id": "msg_c2f68a9b-87cd-4f85-a9e9-b6047213a3ce_us-east-1",
            "role": "assistant",
            "type": "message",
            "status": "completed"
        },
        {
            "role": "user",
            "content": "What is the meaning of 42?"
        }
    ],
    "include": [
        "reasoning.encrypted_content"
    ]
}'
```

## Retrieving a previous model response

If you have a previous response's ID, you can retrieve the content of the response.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600,
)

response = client.chat.get_stored_completion("<The previous response's id>")

print(response)
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

response = client.responses.retrieve("<The previous response's id>")

print(response)
```

```javascriptOpenAISDK
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const response = await client.responses.retrieve("<The previous response's id>");

console.log(response);
```

```bash
curl https://api.x.ai/v1/responses/{response_id} \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600
```

## Delete a model response

If you no longer want to store the previous model response, you can delete it.

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600,
)

response = client.chat.delete_stored_completion("<The previous response's id>")
print(response)
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
)

response = client.responses.delete("<The previous response's id>")

print(response)
```

```javascriptOpenAISDK
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Override default timeout with longer timeout for reasoning models
});

const response = await client.responses.delete("<The previous response's id>");

console.log(response);
```

```bash
curl -X DELETE https://api.x.ai/v1/responses/{response_id} \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600
```


===/docs/guides/streaming-response===
#### Guides

# Streaming Response

Streaming outputs is **supported by all models with text output capability** (Chat, Image Understanding, etc.). It is **not supported by models with image output capability** (Image Generation).

Streaming outputs uses [Server-Sent Events (SSE)](https://en.wikipedia.org/wiki/Server-sent_events) that let the server send back the delta of content in event streams.

Streaming responses are beneficial for providing real-time feedback, enhancing user interaction by allowing text to be displayed as it's generated.

To enable streaming, you must set `"stream": true` in your request.

When using streaming output with reasoning models, you might want to **manually override request
timeout** to avoid prematurely closing connection.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user, system

client = Client(
    api_key=os.getenv('XAI_API_KEY'),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

chat = client.chat.create(model="grok-4")
chat.append(
    system("You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."),
)
chat.append(
    user("What is the meaning of life, the universe, and everything?")
)

for response, chunk in chat.stream():
    print(chunk.content, end="", flush=True) # Each chunk's content
    print(response.content, end="", flush=True) # The response object auto-accumulates the chunks

print(response.content) # The full response
```

```pythonOpenAISDK
import os
import httpx
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0) # Timeout after 3600s for reasoning models
)

stream = client.chat.completions.create(
    model="grok-4",
    messages=[
        {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
        {"role": "user", "content": "What is the meaning of life, the universe, and everything?"},
    ],
    stream=True # Set streaming here
)

for chunk in stream:
print(chunk.choices[0].delta.content, end="", flush=True)
```

```javascriptOpenAISDK
import OpenAI from "openai";
const openai = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
    timeout: 360000, // Timeout after 3600s for reasoning models
});

const stream = await openai.chat.completions.create({
    model: "grok-4",
    messages: [
        { role: "system", content: "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy." },
        {
            role: "user",
            content: "What is the meaning of life, the universe, and everything?",
        }
    ],
    stream: true
});

for await (const chunk of stream) {
    console.log(chunk.choices[0].delta.content);
}
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { streamText } from 'ai';

const result = streamText({
  model: xai('grok-4'),
  system:
    "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy.",
  prompt: 'What is the meaning of life, the universe, and everything?',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

```bash
curl https://api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-m 3600 \\
-d '{
    "messages": [
        {
            "role": "system",
            "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
        },
        {
            "role": "user",
            "content": "What is the meaning of life, the universe, and everything?"
        }
    ],
    "model": "grok-4",
    "stream": true
}'
```

You'll get the event streams like these:

```bash
data: {
    "id":"<completion_id>","object":"chat.completion.chunk","created":<creation_time>,
    "model":"grok-4",
    "choices":[{"index":0,"delta":{"content":"Ah","role":"assistant"}}],
    "usage":{"prompt_tokens":41,"completion_tokens":1,"total_tokens":42,
    "prompt_tokens_details":{"text_tokens":41,"audio_tokens":0,"image_tokens":0,"cached_tokens":0}},
    "system_fingerprint":"fp_xxxxxxxxxx"
}

data: {
    "id":"<completion_id>","object":"chat.completion.chunk","created":<creation_time>,
    "model":"grok-4",
    "choices":[{"index":0,"delta":{"content":",","role":"assistant"}}],
    "usage":{"prompt_tokens":41,"completion_tokens":2,"total_tokens":43,
    "prompt_tokens_details":{"text_tokens":41,"audio_tokens":0,"image_tokens":0,"cached_tokens":0}},
    "system_fingerprint":"fp_xxxxxxxxxx"
}

data: [DONE]
```

It is recommended that you use a client SDK to parse the event stream.

Example streaming responses in Python/Javascript:

```
Ah, the ultimate question! According to Douglas Adams, the answer is **42**. However, the trick lies in figuring out what the actual question is. If you're looking for a bit more context or a different perspective:

- **Philosophically**: The meaning of life might be to seek purpose, happiness, or to fulfill one's potential.
- **Biologically**: It could be about survival, reproduction, and passing on genes.
- **Existentially**: You create your own meaning through your experiences and choices.

But let's not forget, the journey to find this meaning might just be as important as the answer itself! Keep exploring, questioning, and enjoying the ride through the universe. And remember, don't panic!
```


===/docs/guides/structured-outputs===
#### Guides

# Structured Outputs

Structured Outputs is a feature that lets the API return responses in a specific, organized format, like JSON or other schemas you define. Instead of getting free-form text, you receive data that's consistent and easy to parse.

Ideal for tasks like document parsing, entity extraction, or report generation, it lets you define schemas using tools like
[Pydantic](https://pydantic.dev/) or [Zod](https://zod.dev/) to enforce data types, constraints, and structure.

When using structured outputs, the LLM's response is **guaranteed** to match your input schema.

## Supported models

Structured outputs is supported by all language models later than `grok-2-1212` and `grok-2-vision-1212`.

## Supported schemas

For structured output, the following types are supported for structured output:

* string
  * `minLength` and `maxLength` properties are not supported
* number
  * integer
  * float
* object
* array
  * `minItems` and `maxItem` properties are not supported
  * `maxContains` and `minContains` properties are not supported
* boolean
* enum
* anyOf

`allOf` is not supported at the moment.

## Example: Invoice Parsing

A common use case for Structured Outputs is parsing raw documents. For example, invoices contain structured data like vendor details, amounts, and dates, but extracting this data from raw text can be error-prone. Structured Outputs ensure the extracted data matches a predefined schema.

Let's say you want to extract the following data from an invoice:

* Vendor name and address
* Invoice number and date
* Line items (description, quantity, price)
* Total amount and currency

We'll use structured outputs to have Grok generate a strongly-typed JSON for this.

### Step 1: Defining the Schema

You can use [Pydantic](https://pydantic.dev/) or [Zod](https://zod.dev/) to define your schema.

```pythonWithoutSDK
from datetime import date
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"

class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit", ge=0)

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City")
    postal_code: str = Field(description="Postal/ZIP code")
    country: str = Field(description="Country")

class Invoice(BaseModel):
    vendor_name: str = Field(description="Name of the vendor")
    vendor_address: Address = Field(description="Vendor's address")
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: date = Field(description="Date the invoice was issued")
    line_items: List[LineItem] = Field(description="List of purchased items/services")
    total_amount: float = Field(description="Total amount due", ge=0)
    currency: Currency = Field(description="Currency of the invoice")
```

```javascriptWithoutSDK
import { z } from "zod";

const CurrencyEnum = z.enum(["USD", "EUR", "GBP"]);

const LineItemSchema = z.object({
    description: z.string().describe("Description of the item or service"),
    quantity: z.number().int().min(1).describe("Number of units"),
    unit_price: z.number().min(0).describe("Price per unit"),
});

const AddressSchema = z.object({
    street: z.string().describe("Street address"),
    city: z.string().describe("City"),
    postal_code: z.string().describe("Postal/ZIP code"),
    country: z.string().describe("Country"),
});

const InvoiceSchema = z.object({
    vendor_name: z.string().describe("Name of the vendor"),
    vendor_address: AddressSchema.describe("Vendor's address"),
    invoice_number: z.string().describe("Unique invoice identifier"),
    invoice_date: z.string().date().describe("Date the invoice was issued"),
    line_items: z.array(LineItemSchema).describe("List of purchased items/services"),
    total_amount: z.number().min(0).describe("Total amount due"),
    currency: CurrencyEnum.describe("Currency of the invoice"),
});
```

### Step 2: Prepare The Prompts

### System Prompt

The system prompt instructs the model to extract invoice data from text. Since the schema is defined separately, the prompt can focus on the task without explicitly specifying the required fields in the output JSON.

```text
Given a raw invoice, carefully analyze the text and extract the relevant invoice data into JSON format.
```

### Example Invoice Text

```text
Vendor: Acme Corp, 123 Main St, Springfield, IL 62704
Invoice Number: INV-2025-001
Date: 2025-02-10
Items:
- Widget A, 5 units, $10.00 each
- Widget B, 2 units, $15.00 each
Total: $80.00 USD
```

### Step 3: The Final Code

Use the structured outputs feature of the the SDK to parse the invoice.

```pythonXAI
import os
from datetime import date
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from xai_sdk import Client
from xai_sdk.chat import system, user

# Pydantic Schemas

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"

class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit", ge=0)

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City")
    postal_code: str = Field(description="Postal/ZIP code")
    country: str = Field(description="Country")

class Invoice(BaseModel):
    vendor_name: str = Field(description="Name of the vendor")
    vendor_address: Address = Field(description="Vendor's address")
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: date = Field(description="Date the invoice was issued")
    line_items: List[LineItem] = Field(description="List of purchased items/services")
    total_amount: float = Field(description="Total amount due", ge=0)
    currency: Currency = Field(description="Currency of the invoice")

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(model="grok-4")

chat.append(system("Given a raw invoice, carefully analyze the text and extract the invoice data into JSON format."))
chat.append(
user("""
Vendor: Acme Corp, 123 Main St, Springfield, IL 62704
Invoice Number: INV-2025-001
Date: 2025-02-10
Items: - Widget A, 5 units, $10.00 each - Widget B, 2 units, $15.00 each
Total: $80.00 USD
""")
)

# The parse method returns a tuple of the full response object as well as the parsed pydantic object.

response, invoice = chat.parse(Invoice)
assert isinstance(invoice, Invoice)

# Can access fields of the parsed invoice object directly

print(invoice.vendor_name)
print(invoice.invoice_number)
print(invoice.invoice_date)
print(invoice.line_items)
print(invoice.total_amount)
print(invoice.currency)

# Can also access fields from the raw response object such as the content.

# In this case, the content is the JSON schema representation of the parsed invoice object

print(response.content)
```

```pythonOpenAISDK
from openai import OpenAI

from pydantic import BaseModel, Field
from datetime import date
from enum import Enum
from typing import List

# Pydantic Schemas

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"

class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit", ge=0)

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City")
    postal_code: str = Field(description="Postal/ZIP code")
    country: str = Field(description="Country")

class Invoice(BaseModel):
    vendor_name: str = Field(description="Name of the vendor")
    vendor_address: Address = Field(description="Vendor's address")
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: date = Field(description="Date the invoice was issued")
    line_items: List[LineItem] = Field(description="List of purchased items/services")
    total_amount: float = Field(description="Total amount due", ge=0)
    currency: Currency = Field(description="Currency of the invoice")

client = OpenAI(
    api_key="<YOUR_XAI_API_KEY_HERE>",
    base_url="https://api.x.ai/v1",
)

completion = client.beta.chat.completions.parse(
    model="grok-4",
    messages=[
    {"role": "system", "content": "Given a raw invoice, carefully analyze the text and extract the invoice data into JSON format."},
    {"role": "user", "content": """
    Vendor: Acme Corp, 123 Main St, Springfield, IL 62704
    Invoice Number: INV-2025-001
    Date: 2025-02-10
    Items:

    - Widget A, 5 units, $10.00 each
    - Widget B, 2 units, $15.00 each
      Total: $80.00 USD
      """}
      ],
      response_format=Invoice,
  )

invoice = completion.choices[0].message.parsed
print(invoice)
```

```javascriptOpenAISDK
import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import { z } from "zod";

const CurrencyEnum = z.enum(["USD", "EUR", "GBP"]);

const LineItemSchema = z.object({
    description: z.string().describe("Description of the item or service"),
    quantity: z.number().int().min(1).describe("Number of units"),
    unit_price: z.number().min(0).describe("Price per unit"),
});

const AddressSchema = z.object({
    street: z.string().describe("Street address"),
    city: z.string().describe("City"),
    postal_code: z.string().describe("Postal/ZIP code"),
    country: z.string().describe("Country"),
});

const InvoiceSchema = z.object({
    vendor_name: z.string().describe("Name of the vendor"),
    vendor_address: AddressSchema.describe("Vendor's address"),
    invoice_number: z.string().describe("Unique invoice identifier"),
    invoice_date: z.string().date().describe("Date the invoice was issued"),
    line_items: z.array(LineItemSchema).describe("List of purchased items/services"),
    total_amount: z.number().min(0).describe("Total amount due"),
    currency: CurrencyEnum.describe("Currency of the invoice"),
});

const client = new OpenAI({
    apiKey: "<api key>",
    baseURL: "https://api.x.ai/v1",
});

const completion = await client.beta.chat.completions.parse({
    model: "grok-4",
    messages: [
    { role: "system", content: "Given a raw invoice, carefully analyze the text and extract the invoice data into JSON format." },
    { role: "user", content: \`
    Vendor: Acme Corp, 123 Main St, Springfield, IL 62704
    Invoice Number: INV-2025-001
    Date: 2025-02-10
    Items:

    - Widget A, 5 units, $10.00 each
    - Widget B, 2 units, $15.00 each
      Total: $80.00 USD
      \` },
    ],
    response_format: zodResponseFormat(InvoiceSchema, "invoice"),
});

const invoice = completion.choices[0].message.parsed;
console.log(invoice);
```

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { generateObject } from 'ai';
import { z } from 'zod';

const CurrencyEnum = z.enum(['USD', 'EUR', 'GBP']);

const LineItemSchema = z.object({
  description: z.string().describe('Description of the item or service'),
  quantity: z.number().int().min(1).describe('Number of units'),
  unit_price: z.number().min(0).describe('Price per unit'),
});

const AddressSchema = z.object({
  street: z.string().describe('Street address'),
  city: z.string().describe('City'),
  postal_code: z.string().describe('Postal/ZIP code'),
  country: z.string().describe('Country'),
});

const InvoiceSchema = z.object({
  vendor_name: z.string().describe('Name of the vendor'),
  vendor_address: AddressSchema.describe("Vendor's address"),
  invoice_number: z.string().describe('Unique invoice identifier'),
  invoice_date: z.string().date().describe('Date the invoice was issued'),
  line_items: z
    .array(LineItemSchema)
    .describe('List of purchased items/services'),
  total_amount: z.number().min(0).describe('Total amount due'),
  currency: CurrencyEnum.describe('Currency of the invoice'),
});

const result = await generateObject({
  model: xai('grok-4'),
  schema: InvoiceSchema,
  system:
    'Given a raw invoice, carefully analyze the text and extract the invoice data into JSON format.',
  prompt: \`
  Vendor: Acme Corp, 123 Main St, Springfield, IL 62704
  Invoice Number: INV-2025-001
  Date: 2025-02-10
  Items:

  - Widget A, 5 units, $10.00 each
  - Widget B, 2 units, $15.00 each
    Total: $80.00 USD
    \`,
});

console.log(result.object);
```

### Step 4: Type-safe Output

The output will **always** be type-safe and respect the input schema.

```json
{
  "vendor_name": "Acme Corp",
  "vendor_address": {
    "street": "123 Main St",
    "city": "Springfield",
    "postal_code": "62704",
    "country": "IL"
  },
  "invoice_number": "INV-2025-001",
  "invoice_date": "2025-02-10",
  "line_items": [
    { "description": "Widget A", "quantity": 5, "unit_price": 10.0 },
    { "description": "Widget B", "quantity": 2, "unit_price": 15.0 }
  ],
  "total_amount": 80.0,
  "currency": "USD"
}
```


===/docs/guides/tools/advanced-usage===
#### Guides

# Advanced Usage

In this section, we explore advanced usage patterns for agentic tool calling, including:

* **[Use Client-side Tools](#mixing-server-side-and-client-side-tools)** - Combine server-side agentic tools with your own client-side tools for specialized functionality that requires local execution.
* **[Multi-turn Conversations](#multi-turn-conversations-with-preservation-of-agentic-state)** - Maintain context across multiple turns in agentic tool-enabled conversations, allowing the model to build upon previous research and tool results for more complex, iterative problem-solving
* **[Requests with Multiple Active Tools](#tool-combinations)** - Send requests with multiple server-side tools active simultaneously, enabling comprehensive analysis with web search, X search, and code execution tools working together
* **[Image Integration](#using-images-in-the-context)** - Include images in your tool-enabled conversations for visual analysis and context-aware searches

**xAI Python SDK Users**: Version **1.4.0** of the xai-sdk package is required to use some advanced capabilities in the agentic tool calling API, for example, the client-side tools.

&#x20;Advanced tool usage patterns are not yet supported in the Vercel AI SDK. Please use the xAI SDK or OpenAI SDK for this functionality.

## Mixing Server-Side and Client-Side Tools

You can combine server-side agentic tools (like web search and code execution) with custom client-side tools to create powerful hybrid workflows. This approach lets you leverage the model's reasoning capabilities with server-side tools while adding specialized functionality that runs locally in your application.

### How It Works

The key difference when mixing server-side and client-side tools is that **server-side tools are executed automatically by xAI**, while **client-side tools require developer intervention**:

1. Define your client-side tools using [standard function calling patterns](/docs/guides/function-calling)
2. Include both server-side and client-side tools in your request
3. **xAI automatically executes any server-side tools** the model decides to use (web search, code execution, etc.)
4. **When the model calls client-side tools, execution pauses** - xAI returns the tool calls to you instead of executing them
5. **Detect and execute client-side tool calls yourself**, then append the results back to continue the conversation
6. **Repeat this process** until the model generates a final response with no additional client-side tool calls

### Practical Example

Given a local client-side function `get_weather` to get the weather of a specified city, the model can use this client-side tool and the web-search tool to determine the weather in the base city of the 2025 NBA champion.

### Using the xAI SDK

You can determine whether a tool call is a client-side tool call by using `xai_sdk.tools.get_tool_call_type` against a tool call from the `response.tool_calls` list.
For more details, check [this](/docs/guides/tools/overview#server-side-tool-call-and-client-side-tool-call) out.

1. Import the dependencies, and define the client-side tool.

   ```pythonXAI
   import os
   import json

   from xai_sdk import Client
   from xai_sdk.chat import user, tool, tool_result
   from xai_sdk.tools import web_search, get_tool_call_type

   client = Client(api_key=os.getenv("XAI_API_KEY"))

   # Define client-side tool
   def get_weather(city: str) -> str:
       """Get the weather for a given city."""
       # In a real app, this would query your database
       return f"The weather in {city} is sunny."

   # Tools array with both server-side and client-side tools
   tools = [
       web_search(),
       tool(
           name="get_weather",
           description="Get the weather for a given city.",
           parameters={
               "type": "object",
               "properties": {
                   "city": {
                       "type": "string",
                       "description": "The name of the city",
                   }
               },
               "required": ["city"]
           },
       ),
   ]

   model = "grok-4-1-fast"
   ```

2. Perform the tool loop with conversation continuation:
   * You can either use `previous_response_id` to continue the conversation from the last response.

     ```pythonXAI
     # Create chat with both server-side and client-side tools
     chat = client.chat.create(
         model=model,
         tools=tools,
         store_messages=True,
     )
     chat.append(
         user(
             "What is the weather in the base city of the team that won the "
             "2025 NBA championship?"
         )
     )

     while True:
         client_side_tool_calls = []
         for response, chunk in chat.stream():
             for tool_call in chunk.tool_calls:
                 if get_tool_call_type(tool_call) == "client_side_tool":
                     client_side_tool_calls.append(tool_call)
                 else:
                     print(
                         f"Server-side tool call: {tool_call.function.name} "
                         f"with arguments: {tool_call.function.arguments}"
                     )

         if not client_side_tool_calls:
             break

         chat = client.chat.create(
             model=model,
             tools=tools,
             store_messages=True,
             previous_response_id=response.id,
         )

         for tool_call in client_side_tool_calls:
             print(
                 f"Client-side tool call: {tool_call.function.name} "
                 f"with arguments: {tool_call.function.arguments}"
             )
             args = json.loads(tool_call.function.arguments)
             result = get_weather(args["city"])
             chat.append(tool_result(result))

     print(f"Final response: {response.content}")
     ```

   * Alternatively, you can use the encrypted content to continue the conversation.

     ```pythonXAI
     # Create chat with both server-side and client-side tools
     chat = client.chat.create(
         model=model,
         tools=tools,
         use_encrypted_content=True,
     )
     chat.append(
         user(
             "What is the weather in the base city of the team that won the "
             "2025 NBA championship?"
         )
     )

     while True:
         client_side_tool_calls = []
         for response, chunk in chat.stream():
             for tool_call in chunk.tool_calls:
                 if get_tool_call_type(tool_call) == "client_side_tool":
                     client_side_tool_calls.append(tool_call)
                 else:
                     print(
                         f"Server-side tool call: {tool_call.function.name} "
                         f"with arguments: {tool_call.function.arguments}"
                     )

         chat.append(response)

         if not client_side_tool_calls:
             break

         for tool_call in client_side_tool_calls:
             print(
                 f"Client-side tool call: {tool_call.function.name} "
                 f"with arguments: {tool_call.function.arguments}"
             )
             args = json.loads(tool_call.function.arguments)
             result = get_weather(args["city"])
             chat.append(tool_result(result))

     print(f"Final response: {response.content}")
     ```

You will see an output similar to the following:

```
Server-side tool call: web_search with arguments: {"query":"Who won the 2025 NBA championship?","num_results":5}
Client-side tool call: get_weather with arguments: {"city":"Oklahoma City"}
Final response: The Oklahoma City Thunder won the 2025 NBA championship. The current weather in Oklahoma City is sunny.
```

### Using the OpenAI SDK

You can determine whether a tool call is a client-side tool call by checking the `type` field of an output entry from the `response.output` list.
For more details, please check [this](/docs/guides/tools/overview#identifying-the-client-side-tool-call) out.

1. Import the dependencies, and define the client-side tool.

   ```pythonOpenAISDK
   import os
   import json

   from openai import OpenAI

   client = OpenAI(
       api_key=os.getenv("XAI_API_KEY"),
       base_url="https://api.x.ai/v1",
   )

   # Define client-side tool
   def get_weather(city: str) -> str:
       """Get the weather for a given city."""
       # In a real app, this would query your database
       return f"The weather in {city} is sunny."

   model = "grok-4-1-fast"
   tools = [
       {
           "type": "function",
           "name": "get_weather",
           "description": "Get the weather for a given city.",
           "parameters": {
               "type": "object",
               "properties": {
                   "city": {
                       "type": "string",
                       "description": "The name of the city",
                   },
               },
               "required": ["city"],
           },
       },
       {
           "type": "web_search",
       },
   ]
   ```

2. Perform the tool loop:

   * You can either use `previous_response_id`.

     ```pythonOpenAISDK
     response = client.responses.create(
         model=model,
         input=(
             "What is the weather in the base city of the team that won the "
             "2025 NBA championship?"
         ),
         tools=tools,
     )

     while True:
         tool_outputs = []
         for item in response.output:
             if item.type == "function_call":
                 print(f"Client-side tool call: {item.name} with arguments: {item.arguments}")
                 args = json.loads(item.arguments)
                 weather = get_weather(args["city"])
                 tool_outputs.append(
                     {
                         "type": "function_call_output",
                         "call_id": item.call_id,
                         "output": weather,
                     }
                 )
             elif item.type in (
                 "web_search_call",
                 "x_search_call", 
                 "code_interpreter_call",
                 "file_search_call",
                 "mcp_call"
             ):
                 print(
                     f"Server-side tool call: {item.name} with arguments: {item.arguments}"
                 )

         if not tool_outputs:
             break

         response = client.responses.create(
             model=model,
             tools=tools,
             input=tool_outputs,
             previous_response_id=response.id,
         )

     print("Final response:", response.output[-1].content[0].text)
     ```

   * or using the encrypted content

     ```pythonOpenAISDK
     input_list = [
         {
             "role": "user",
             "content": (
                 "What is the weather in the base city of the team that won the "
                 "2025 NBA championship?"
             ),
         }
     ]

     response = client.responses.create(
         model=model,
         input=input_list,
         tools=tools,
         include=["reasoning.encrypted_content"],
     )

     while True:
         input_list.extend(response.output)
         tool_outputs = []
         for item in response.output:
             if item.type == "function_call":
                 print(f"Client-side tool call: {item.name} with arguments: {item.arguments}")
                 args = json.loads(item.arguments)
                 weather = get_weather(args["city"])
                 tool_outputs.append(
                     {
                         "type": "function_call_output",
                         "call_id": item.call_id,
                         "output": weather,
                     }
                 )
             elif item.type in (
                 "web_search_call",
                 "x_search_call", 
                 "code_interpreter_call",
                 "file_search_call",
                 "mcp_call"
             ):
                 print(
                     f"Server-side tool call: {item.name} with arguments: {item.arguments}"
                 )

         if not tool_outputs:
             break

         input_list.extend(tool_outputs)
         response = client.responses.create(
             model=model,
             input=input_list,
             tools=tools,
             include=["reasoning.encrypted_content"],
         )

     print("Final response:", response.output[-1].content[0].text)
     ```

## Multi-turn Conversations with Preservation of Agentic State

When using agentic tools, you may want to have multi-turn conversations where follow-up prompts maintain all agentic state, including the full history of reasoning, tool calls, and tool responses. This is possible using the stateful API, which provides seamless integration for preserving conversation context across multiple interactions. There are two options to achieve this outlined below.

### Store the Conversation History Remotely

You can choose to store the conversation history remotely on the xAI server, and every time you want to continue the conversation, you can pick up from the last response where you want to resume from.

There are only 2 extra steps:

1. Add the parameter `store_messages=True` when making the first agentic request. This tells the service to store the entire conversation history on xAI servers, including the model's reasoning, server-side tool calls, and corresponding responses.
2. Pass `previous_response_id=response.id` when creating the follow-up conversation, where `response` is the response returned by `chat.sample()` or `chat.stream()` from the conversation that you wish to continue.

Note that the follow-up conversation does not need to use the same tools, model parameters, or any other configuration as the initial conversation—it will still be fully hydrated with the complete agentic state from the previous interaction.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search, x_search
client = Client(api_key=os.getenv("XAI_API_KEY"))
# First turn.
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[web_search(), x_search()],
    store_messages=True,
)
chat.append(user("What is xAI?"))
print("\\n\\n##### First turn #####\\n")
for response, chunk in chat.stream():
    print(chunk.content, end="", flush=True)
print("\\n\\nUsage for first turn:", response.server_side_tool_usage)

# Second turn.
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[web_search(), x_search()],
    # pass the response id of the first turn to continue the conversation
    previous_response_id=response.id,
)

chat.append(user("What is its latest mission?"))
print("\\n\\n##### Second turn #####\\n")
for response, chunk in chat.stream():
    print(chunk.content, end="", flush=True)
print("\\n\\nUsage for second turn:", response.server_side_tool_usage)
```

### Append the Encrypted Agentic Tool Calling States

There is another option for the ZDR (Zero Data Retention) users, or the users who don't want to use the above option, that is to let the xAI server also return
the encrypted reasoning and the encrypted tool output besides the final content to the client side, and those encrypted contents can be included as a part of the context
in the next turn conversation.

Here are the extra steps you need to take for this option:

1. Add the parameter `use_encrypted_content=True` when making the first agentic request. This tells the service to return the entire conversation history to the client side, including the model's reasoning (encrypted), server-side tool calls, and corresponding responses (encrypted).
2. Append the response to the conversation you wish to continue before making the call to `chat.sample()` or `chat.stream()`.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search, x_search
client = Client(api_key=os.getenv("XAI_API_KEY"))
# First turn.
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[web_search(), x_search()],
    use_encrypted_content=True,
)
chat.append(user("What is xAI?"))
print("\\n\\n##### First turn #####\\n")
for response, chunk in chat.stream():
    print(chunk.content, end="", flush=True)
print("\\n\\nUsage for first turn:", response.server_side_tool_usage)

chat.append(response)

print("\\n\\n##### Second turn #####\\n")
chat.append(user("What is its latest mission?"))
# Second turn.
for response, chunk in chat.stream():
    print(chunk.content, end="", flush=True)
print("\\n\\nUsage for second turn:", response.server_side_tool_usage)
```

For more details about stateful responses, please check out [this guide](/docs/guides/responses-api).

## Tool Combinations

Equipping your requests with multiple tools is straightforward—simply include the tools you want to activate in the `tools` array of your request. The model will intelligently orchestrate between them based on the task at hand.

### Suggested Tool Combinations

Here are some common patterns for combining tools, depending on your use case:

| If you're trying to... | Consider activating... | Because... |
|------------------------|----------------------|------------|
| **Research & analyze data** | Web Search + Code Execution | Web search gathers information, code execution analyzes and visualizes it |
| **Aggregate news & social media** | Web Search + X Search | Get comprehensive coverage from both traditional web and social platforms |
| **Extract insights from multiple sources** | Web Search + X Search + Code Execution | Collect data from various sources then compute correlations and trends |
| **Monitor real-time discussions** | X Search + Web Search | Track social sentiment alongside authoritative information |

```pythonXAI
from xai_sdk.tools import web_search, x_search, code_execution

# Example tool combinations for different scenarios
research_setup = [web_search(), code_execution()]
news_setup = [web_search(), x_search()]
comprehensive_setup = [web_search(), x_search(), code_execution()]
```

```pythonWithoutSDK
research_setup = {
  "tools": [
    {"type": "web_search"},
    {"type": "code_interpreter"}
  ]
}

news_setup = {
  "tools": [
    {"type": "web_search"},
    {"type": "x_search"}
  ]
}

comprehensive_setup = {
  "tools": [
    {"type": "web_search"},
    {"type": "x_search"},
    {"type": "code_interpreter"}
  ]
}
```

### Using Tool Combinations in Different Scenarios

1. When you want to search for news on the Internet, you can activate all search tools:
   * Web search tool
   * X search tool

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search, x_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        web_search(),
        x_search(),
    ],
)

chat.append(user("what is the latest update from xAI?"))

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nFinal Response:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nCitations:")
print(response.citations)
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "what is the latest update from xAI?",
        },
    ],
    tools=[
        {
            "type": "web_search",
        },
        {
            "type": "x_search",
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "what is the latest update from xAI?"
        }
    ],
    "tools": [
        {
            "type": "web_search",
        },
        {
            "type": "x_search",
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is the latest update from xAI?"
    }
  ],
  "tools": [
    {
      "type": "web_search"
    },
    {
      "type": "x_search"
    }
  ]
}'
```

2. When you want to collect up-to-date data from the Internet and perform calculations based on the Internet data, you can choose to activate:
   * Web search tool
   * Code execution tool

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search, code_execution

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    # research_tools
    tools=[
        web_search(),
        code_execution(),
    ],
)

chat.append(user("What is the average market cap of the companies with the top 5 market cap in the US stock market today?"))

# sample or stream the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is the average market cap of the companies with the top 5 market cap in the US stock market today?",
        },
    ],
    # research_tools
    tools=[
        {
            "type": "web_search",
        },
        {
            "type": "code_interpreter",
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is the average market cap of the companies with the top 5 market cap in the US stock market today?"
        }
    ],
    # research_tools
    "tools": [
        {
            "type": "web_search",
        },
        {
            "type": "code_interpreter",
        },
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is the average market cap of the companies with the top 5 market cap in the US stock market today?"
    }
  ],
  "tools": [
    {
      "type": "web_search"
    },
    {
      "type": "code_interpreter"
    }
  ]
}'
```

## Using Images in the Context

You can bootstrap your requests with an initial conversation context that includes images.

In the code sample below, we pass an image into the context of the conversation before initiating an agentic request.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import image, user
from xai_sdk.tools import web_search, x_search

# Create the client and define the server-side tools to use
client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[web_search(), x_search()],
)

# Add an image to the conversation
chat.append(
    user(
        "Search the internet and tell me what kind of dog is in the image below.",
        "And what is the typical lifespan of this dog breed?",
        image(
            "https://pbs.twimg.com/media/G3B7SweXsAAgv5N?format=jpg&name=900x900"
        ),
    )
)

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nFinal Response:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nCitations:")
print(response.citations)
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```


===/docs/guides/tools/code-execution-tool===
#### Guides

# Code Execution Tool

The code execution tool enables Grok to write and execute Python code in real-time, dramatically expanding its capabilities beyond text generation. This powerful feature allows Grok to perform precise calculations, complex data analysis, statistical computations, and solve mathematical problems that would be impossible through text alone.

**xAI Python SDK Users**: Version 1.3.1 of the xai-sdk package is required to use the agentic tool calling API.

&#x20;The code execution tool is not yet supported in the Vercel AI SDK. Please use the xAI SDK or OpenAI SDK for this functionality.

## Key Capabilities

* **Mathematical Computations**: Solve complex equations, perform statistical analysis, and handle numerical calculations with precision
* **Data Analysis**: Process datasets, and extract insights from the prompt
* **Financial Modeling**: Build financial models, calculate risk metrics, and perform quantitative analysis
* **Scientific Computing**: Handle scientific calculations, simulations, and data transformations
* **Code Generation & Testing**: Write, test, and debug Python code snippets in real-time

## When to Use Code Execution

The code execution tool is particularly valuable for:

* **Numerical Problems**: When you need exact calculations rather than approximations
* **Data Processing**: Analyzing complex data from the prompt
* **Complex Logic**: Multi-step calculations that require intermediate results
* **Verification**: Double-checking mathematical results or validating assumptions

## SDK Support

The code execution tool is available across multiple SDKs and APIs with different naming conventions:

| SDK/API | Tool Name | Description |
|---------|-----------|-------------|
| xAI SDK | `code_execution` | Native xAI SDK implementation |
| OpenAI Responses API | `code_interpreter` | Compatible with OpenAI's API format |

## Implementation Example

Below are comprehensive examples showing how to integrate the code execution tool across different platforms and use cases.

### Basic Calculations

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import code_execution

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[code_execution()],
)

# Ask for a mathematical calculation
chat.append(user("Calculate the compound interest for $10,000 at 5% annually for 10 years"))

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nFinal Response:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nCitations:")
print(response.citations)
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "Calculate the compound interest for $10,000 at 5% annually for 10 years",
        },
    ],
    tools=[
        {
            "type": "code_interpreter",
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "Calculate the compound interest for $10,000 at 5% annually for 10 years"
        }
    ],
    "tools": [
        {
            "type": "code_interpreter",
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "Calculate the compound interest for $10,000 at 5% annually for 10 years"
    }
  ],
  "tools": [
    {
      "type": "code_interpreter"
    }
  ]
}'
```

### Data Analysis

```pythonXAI
import os
from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import code_execution

client = Client(api_key=os.getenv("XAI_API_KEY"))

# Multi-turn conversation with data analysis
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[code_execution()],
)

# Step 1: Load and analyze data
chat.append(user("""
I have sales data for Q1-Q4: [120000, 135000, 98000, 156000].
Please analyze this data and create a visualization showing:
1. Quarterly trends
2. Growth rates
3. Statistical summary
"""))

print("##### Step 1: Data Analysis #####\\n")

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nAnalysis Results:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nCitations:")
print(response.citations)
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)

chat.append(response)

# Step 2: Follow-up analysis
chat.append(user("Now predict Q1 next year using linear regression"))

print("\\n\\n##### Step 2: Prediction Analysis #####\\n")

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nPrediction Results:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nCitations:")
print(response.citations)
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```

## Best Practices

### 1. **Be Specific in Requests**

Provide clear, detailed instructions about what you want the code to accomplish:

```pythonWithoutSDK
# Good: Specific and clear
"Calculate the correlation matrix for these variables and highlight correlations above 0.7"

# Avoid: Vague requests  
"Analyze this data"
```

### 2. **Provide Context and Data Format**

Always specify the data format and any constraints on the data, and provide as much context as possible:

```pythonWithoutSDK
# Good: Includes data format and requirements
"""
Here's my CSV data with columns: date, revenue, costs
Please calculate monthly profit margins and identify the best-performing month.
Data: [['2024-01', 50000, 35000], ['2024-02', 55000, 38000], ...]
"""
```

### 3. **Use Appropriate Model Settings**

* **Temperature**: Use lower values (0.0-0.3) for mathematical calculations
* **Model**: Use reasoning models like `grok-4-1-fast` for better code generation

## Common Use Cases

### Financial Analysis

```pythonWithoutSDK
# Portfolio optimization, risk calculations, option pricing
"Calculate the Sharpe ratio for a portfolio with returns [0.12, 0.08, -0.03, 0.15] and risk-free rate 0.02"
```

### Statistical Analysis

```pythonWithoutSDK
# Hypothesis testing, regression analysis, probability distributions
"Perform a t-test to compare these two groups and interpret the p-value: Group A: [23, 25, 28, 30], Group B: [20, 22, 24, 26]"
```

### Scientific Computing

```pythonWithoutSDK
# Simulations, numerical methods, equation solving
"Solve this differential equation using numerical methods: dy/dx = x^2 + y, with initial condition y(0) = 1"
```

## Limitations and Considerations

* **Execution Environment**: Code runs in a sandboxed Python environment with common libraries pre-installed
* **Time Limits**: Complex computations may have execution time constraints
* **Memory Usage**: Large datasets might hit memory limitations
* **Package Availability**: Most popular Python packages (NumPy, Pandas, Matplotlib, SciPy) are available
* **File I/O**: Limited file system access for security reasons

## Security Notes

* Code execution happens in a secure, isolated environment
* No access to external networks or file systems
* Temporary execution context that doesn't persist between requests
* All computations are stateless and secure


===/docs/guides/tools/collections-search-tool===
#### Guides

# Collections Search Tool

The collections search tool enables Grok to search through your uploaded knowledge bases (collections), allowing it to retrieve relevant information from your documents to provide more accurate and contextually relevant responses. This tool is particularly powerful for analyzing complex documents like financial reports, legal contracts, or technical documentation, where Grok can autonomously search through multiple documents and synthesize information to answer sophisticated analytical questions.

For an introduction to Collections, please check out the [Collections documentation](/docs/key-information/collections).

**xAI Python SDK Users**: Version 1.4.0 of the xai-sdk package is required to use this collections-search tool in the agentic tool calling API.

## Key Capabilities

* **Document Retrieval**: Search across uploaded files and collections to find relevant information
* **Semantic Search**: Find documents based on meaning and context, not just keywords
* **Knowledge Base Integration**: Seamlessly integrate your proprietary data with Grok's reasoning
* **RAG Applications**: Power retrieval-augmented generation workflows
* **Multi-format Support**: Search across PDFs, text files, CSVs, and other supported formats

## When to Use Collections Search

The collections search tool is particularly valuable for:

* **Enterprise Knowledge Bases**: When you need Grok to reference internal documents and policies
* **Financial Analysis**: Analyzing SEC filings, earnings reports, and financial statements across multiple documents
* **Customer Support**: Building chatbots that can answer questions based on your product documentation
* **Research & Due Diligence**: Synthesizing information from academic papers, technical reports, or industry analyses
* **Compliance & Legal**: Ensuring responses are grounded in your official guidelines and regulations
* **Personal Knowledge Management**: Organizing and querying your personal document collections

## SDK Support

The collections search tool is available across multiple SDKs and APIs with different naming conventions:

| SDK/API | Tool Name | Description |
|---------|-----------|-------------|
| xAI SDK | `collections_search` | Native xAI SDK implementation |
| OpenAI Responses API | `file_search` | Compatible with OpenAI's API format |

## Implementation Example

### End-to-End Financial Analysis Example

This comprehensive example demonstrates analyzing Tesla's SEC filings using the collections search tool. It covers:

1. Creating a collection for document storage
2. Uploading multiple financial documents concurrently (10-Q and 10-K filings)
3. Using Grok with collections search to analyze and synthesize information across documents in an agentic manner
4. Enabling code execution to allow the model to perform calculations and mathematical analysis effectively should it be needed.
5. Receiving cited responses and tool usage information

This pattern is applicable to any document analysis workflow where you need to search through and reason over multiple documents.

```pythonXAI
import asyncio
import os

import httpx

from xai_sdk import AsyncClient
from xai_sdk.chat import user
from xai_sdk.proto import collections_pb2
from xai_sdk.tools import code_execution, collections_search

TESLA_10_Q_PDF_URL = "https://ir.tesla.com/_flysystem/s3/sec/000162828025045968/tsla-20250930-gen.pdf"
TESLA_10_K_PDF_URL = "https://ir.tesla.com/_flysystem/s3/sec/000162828025003063/tsla-20241231-gen.pdf"


async def main():
    client = AsyncClient(api_key=os.getenv("XAI_API_KEY"))

    # Step 1: Create a collection for Tesla SEC filings
    response = await client.collections.create("tesla-sec-filings")
    print(f"Created collection: {response.collection_id}")

    # Step 2: Upload documents to the collection concurrently
    async def upload_document(
        url: str, name: str, collection_id: str, http_client: httpx.AsyncClient
    ) -> None:
        pdf_response = await http_client.get(url, timeout=30.0)
        pdf_content = pdf_response.content

        print(f"Uploading {name} document to collection")
        response = await client.collections.upload_document(
            collection_id=collection_id,
            name=name,
            data=pdf_content,
            content_type="application/pdf",
        )

        # Poll until document is processed and ready for search
        response = await client.collections.get_document(response.file_metadata.file_id, collection_id)
        print(f"Waiting for document {name} to be processed")
        while response.status != collections_pb2.DOCUMENT_STATUS_PROCESSED:
            await asyncio.sleep(3)
            response = await client.collections.get_document(response.file_metadata.file_id, collection_id)

        print(f"Document {name} processed")

    # Upload both documents concurrently
    async with httpx.AsyncClient() as http_client:
        await asyncio.gather(
            upload_document(TESLA_10_Q_PDF_URL, "tesla-10-Q-2024.pdf", response.collection_id, http_client),
            upload_document(TESLA_10_K_PDF_URL, "tesla-10-K-2024.pdf", response.collection_id, http_client),
        )

    # Step 3: Create a chat with collections search enabled
    chat = client.chat.create(
        model="grok-4-1-fast",  # Use a reasoning model for better analysis
        tools=[
            collections_search(
                collection_ids=[response.collection_id],
            ),
            code_execution(),
        ],
    )

    # Step 4: Ask a complex analytical question that requires searching multiple documents
    chat.append(
        user(
            "How many consumer vehicles did Tesla produce in total in 2024 and 2025? "
            "Show your working and cite your sources."
        )
    )

    # Step 5: Stream the response and display reasoning progress
    is_thinking = True
    async for response, chunk in chat.stream():
        # View server-side tool calls as they happen
        for tool_call in chunk.tool_calls:
            print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
        if response.usage.reasoning_tokens and is_thinking:
            print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
        if chunk.content and is_thinking:
            print("\\n\\nFinal Response:")
            is_thinking = False
        if chunk.content and not is_thinking:
            print(chunk.content, end="", flush=True)
        latest_response = response

    # Step 6: Review citations and tool usage
    print("\\n\\nCitations:")
    print(latest_response.citations)
    print("\\n\\nUsage:")
    print(latest_response.usage)
    print(latest_response.server_side_tool_usage)
    print("\\n\\nTool Calls:")
    print(latest_response.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())
```

```pythonOpenAISDK
import os
from openai import OpenAI

# Using OpenAI SDK with xAI API (requires pre-created collection)
api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

# Note: You must create the collection and upload documents first using either the xAI console (console.x.ai) or the xAI SDK
# The collection_id below should be replaced with your actual collection ID
response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "How many consumer vehicles did Tesla produce in total in 2024 and 2025? Show your working and cite your sources.",
        },
    ],
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["your_collection_id_here"],  # Replace with actual collection ID
            "max_num_results": 10
        },
        {"type": "code_interpreter"},  # Enable code execution for calculations
    ],
)

print(response)
```

```pythonRequests
import os
import requests

# Using raw requests (requires pre-created collection)
url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "How many consumer vehicles did Tesla produce in total in 2024 and 2025? Show your working and cite your sources."
        }
    ],
    "tools": [
        {
            "type": "file_search",
            "vector_store_ids": ["your_collection_id_here"],  # Replace with actual collection ID
            "max_num_results": 10,
        },
        {"type": "code_interpreter"}  # Enable code execution for calculations
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
# Using curl (requires pre-created collection)
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "How many consumer vehicles did Tesla produce in total in 2024 and 2025? Show your working and cite your sources."
    }
  ],
  "tools": [
    {
      "type": "file_search",
      "vector_store_ids": ["your_collection_id_here"],
      "max_num_results": 10
    },
    {
      "type": "code_interpreter"
    }
  ]
}'
```

## Example Output

When you run the Python xAI SDK example above, you'll see output like this showing the complete workflow from collection creation to the final analyzed response:

```output
Created collection: collection_3be0eec8-ee8e-4a18-a9d4-fb70a3150d64
Uploading tesla-10-K-2024.pdf document to collection
Uploading tesla-10-Q-2024.pdf document to collection
Waiting for document tesla-10-K-2024.pdf to be processed
Waiting for document tesla-10-Q-2024.pdf to be processed
Document tesla-10-Q-2024.pdf processed
Document tesla-10-K-2024.pdf processed

Thinking... (199 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla total vehicle production 2024","limit":10}
Thinking... (370 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla total vehicles produced 2024","limit":5}
Thinking... (370 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla vehicles produced 2025 Q1 Q2 Q3","limit":5}
Thinking... (477 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla production and deliveries 2024 all quarters","limit":10}
Thinking... (477 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla production and deliveries 2025 Q1 Q2 Q3","limit":10}
Thinking... (569 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla total production 2024 full year consumer vehicles","limit":5}
Thinking... (776 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla annual vehicle production 2024 total","limit":5}
Thinking... (1148 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla Q4 2024 production and deliveries","limit":5}
Thinking... (1240 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla Q1 2024 vehicle production","limit":3}
Thinking... (1240 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla Q2 2024 vehicle production","limit":3}
Thinking... (1240 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla Q3 2024 vehicle production","limit":3}
Thinking... (1240 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla Q4 2024 vehicle production","limit":3}
Thinking... (2804 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla vehicle production nine months 2024","limit":5}
Thinking... (3352 tokens)

Final Response:
### Working

To answer this question, I used the collections_search tool to query Tesla's official 
financial filings (10-K and 10-Q reports) for production numbers. Tesla reports vehicle 
production and deliveries quarterly, and consumer vehicles refer to passenger vehicles 
like Model 3, Model Y, Model S, Model X, and Cybertruck (excluding Tesla Semi or other 
non-consumer products).

#### Step 1: 2024 Production
Based on Tesla's official quarterly production and delivery reports (aggregated from SEC 
filings and press releases referenced in the collections), Tesla produced **1,773,443 
consumer vehicles in 2024**.
  - Q1 2024: 433,371 produced
  - Q2 2024: 410,831 produced
  - Q3 2024: 469,796 produced
  - Q4 2024: 459,445 produced

#### Step 2: 2025 Production
The Q3 2025 10-Q filing explicitly states: "In 2025, we produced approximately 1,220,000 
consumer vehicles [...] through the third quarter."
  - This is the sum of Q1, Q2, and Q3 2025 production
  - Q4 2025 data is not available as of November 13, 2025

#### Step 3: Total for 2024 and 2025
- 2024 full year: 1,773,443
- 2025 (through Q3): 1,220,000
- **Total: 2,993,443 consumer vehicles**

Citations:
['collections://collection_3be0eec8-ee8e-4a18-a9d4-fb70a3150d64/files/file_d4d1a968-9037-4caa-8eca-47a1563f28ab', 
 'collections://collection_3be0eec8-ee8e-4a18-a9d4-fb70a3150d64/files/file_ff41a42e-6cdc-4ca1-918a-160644d52704']

Usage:
completion_tokens: 1306
prompt_tokens: 383265
total_tokens: 387923
prompt_text_tokens: 383265
reasoning_tokens: 3352
cached_prompt_text_tokens: 177518

{'SERVER_SIDE_TOOL_COLLECTIONS_SEARCH': 13}


Tool Calls:
... (omitted for brevity)
```

### Understanding Collections Citations

When using the collections search tool, citations follow a special URI format that uniquely identifies the source documents:

```
collections://collection_id/files/file_id
```

For example:

```
collections://collection_3be0eec8-ee8e-4a18-a9d4-fb70a3150d64/files/file_d4d1a968-9037-4caa-8eca-47a1563f28ab
```

**Format Breakdown:**

* **`collections://`**: Protocol identifier indicating this is a collection-based citation
* **`collection_id`**: The unique identifier of the collection that was searched (e.g., `collection_3be0eec8-ee8e-4a18-a9d4-fb70a3150d64`)
* **`files/`**: Path segment indicating file-level reference
* **`file_id`**: The unique identifier of the specific document file that was referenced (e.g., `file_d4d1a968-9037-4caa-8eca-47a1563f28ab`)

These citations represent all the documents from your collections that Grok referenced during its search and analysis. Each citation points to a specific file within a collection, allowing you to trace back exactly which uploaded documents contributed to the final response.

### Key Observations

1. **Autonomous Search Strategy**: Grok autonomously performs 13 different searches across the documents, progressively refining queries to find specific quarterly and annual production data.

2. **Reasoning Process**: The output shows reasoning tokens accumulating (199 → 3,352 tokens), demonstrating how the model thinks through the problem before generating the final response.

3. **Cited Sources**: All information is grounded in the uploaded documents with specific file citations, ensuring transparency and verifiability.

4. **Structured Analysis**: The final response breaks down the methodology, shows calculations, and clearly states assumptions and limitations (e.g., Q4 2025 data not yet available).

5. **Token Efficiency**: Notice the high number of cached prompt tokens (177,518) - this demonstrates how the collections search tool efficiently reuses context across multiple queries.

## Combining Collections Search with Web Search/X-Search

One of the most powerful patterns is combining the collections search tool with web search/x-search to answer questions that require both your internal knowledge base and real-time external information. This enables sophisticated analysis that grounds responses in your proprietary data while incorporating current market intelligence, news, and public sentiment.

### Example: Internal Data + Market Intelligence

Building on the Tesla example above, let's analyze how market analysts view Tesla's performance based on the production numbers from our internal documents:

```pythonXAI
import asyncio

import httpx

from xai_sdk import AsyncClient
from xai_sdk.chat import user
from xai_sdk.proto import collections_pb2
from xai_sdk.tools import code_execution, collections_search, web_search, x_search

# ... (collection creation and document upload same as before)

async def hybrid_analysis(client: AsyncClient, collection_id: str, model: str) -> None:
    # Enable collections search, web search, and code execution
    chat = client.chat.create(
        model=model,
        tools=[
            collections_search(
                collection_ids=[collection_id],
            ),
            web_search(),  # Enable web search for external data
            x_search(),  # Enable x-search for external data
            code_execution(),  # Enable code execution for calculations
        ],
    )

    # Ask a question that requires both internal and external information
    chat.append(
        user(
            "Based on Tesla's actual production figures in my documents (collection), what is the "
            "current market and analyst sentiment on their 2024-2025 vehicle production performance?"
        )
    )

    is_thinking = True
    async for response, chunk in chat.stream():
        for tool_call in chunk.tool_calls:
            print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
        if response.usage.reasoning_tokens and is_thinking:
            print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
        if chunk.content and is_thinking:
            print("\\n\\nFinal Response:")
            is_thinking = False
        if chunk.content and not is_thinking:
            print(chunk.content, end="", flush=True)
        latest_response = response

    print("\\n\\nCitations:")
    print(latest_response.citations)
    print("\\n\\nTool Usage:")
    print(latest_response.server_side_tool_usage)
```

### How It Works

When you provide both `collections_search()` and `web_search()`/`x_search()` tools, Grok autonomously determines the optimal search strategy:

1. **Internal Analysis First**: Searches your uploaded Tesla SEC filings to extract actual production numbers
2. **External Context Gathering**: Performs web/x-search searches to find analyst reports, market sentiment, and production expectations
3. **Synthesis**: Combines both data sources to provide a comprehensive analysis comparing actual performance against market expectations
4. **Cited Sources**: Returns citations from both your internal documents (using `collections://` URIs) and external web sources (using `https://` URLs)

### Example Output Pattern

```output
Thinking... (201 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla vehicle production figures 2024 2025","limit":20}
Thinking... (498 tokens)
Calling tool: collections_search with arguments: {"query":"Tesla quarterly vehicle production and deliveries 2024 2025","limit":20}
Thinking... (738 tokens)
Calling tool: web_search with arguments: {"query":"Tesla quarterly vehicle production and deliveries 2024 2025","num_results":10}
Thinking... (738 tokens)
Calling tool: web_search with arguments: {"query":"market and analyst sentiment Tesla vehicle production performance 2024 2025","num_results":10}
Thinking... (1280 tokens)

Final Response 
... (omitted for brevity)
```

### Use Cases for Hybrid Search

This pattern is valuable for:

* **Market Analysis**: Compare internal financial data with external market sentiment and competitor performance
* **Competitive Intelligence**: Analyze your product performance against industry reports and competitor announcements
* **Compliance Verification**: Cross-reference internal policies with current regulatory requirements and industry standards
* **Strategic Planning**: Ground business decisions in both proprietary data and real-time market conditions
* **Customer Research**: Combine internal customer data with external reviews, social sentiment, and market trends


===/docs/guides/tools/overview===
#### Guides

# Overview

The xAI API supports **agentic server-side tool calling** which enables the model to autonomously explore, search, and execute code to solve complex queries. Unlike traditional tool-calling where clients must handle each tool invocation themselves, xAI's agentic API manages the entire reasoning and tool-execution loop on the server side.

**xAI Python SDK Users**: Version 1.3.1 of the xai-sdk package is required to use the agentic tool calling API.

## Tools Pricing

Agentic requests are priced based on two components: **token usage** and **tool invocations**. Since the agent autonomously decides how many tools to call, costs scale with query complexity.

For more details on Tools pricing, please check out [the pricing page](/docs/models#tools-pricing).

## Agentic Tool Calling

When you provide server-side tools to a request, the xAI server orchestrates an autonomous reasoning loop rather than returning tool calls for you to execute. This creates a seamless experience where the model acts as an intelligent agent that researches, analyzes, and responds automatically.

Behind the scenes, the model follows an iterative reasoning process:

1. **Analyzes the query** and current context to determine what information is needed
2. **Decides what to do next**: Either make a tool call to gather more information or provide a final answer
3. **If making a tool call**: Selects the appropriate tool and parameters based on the reasoning
4. **Executes the tool** in real-time on the server and receives the results
5. **Processes the tool response** and integrates it with previous context and reasoning
6. **Repeats the loop**: Uses the new information to decide whether more research is needed or if a final answer can be provided
7. **Returns the final response** once the agent determines it has sufficient information to answer comprehensively

This autonomous orchestration enables complex multi-step research and analysis to happen automatically, with clients seeing the final result as well as optional real-time progress indicators like tool call notifications during streaming.

## Core Capabilities

* **[Web Search](/docs/guides/tools/search-tools)**: Real-time search across the internet with the ability to both search the web and browse web pages.
* **[X Search](/docs/guides/tools/search-tools)**: Semantic and keyword search across X posts, users, and threads.
* **[Code Execution](/docs/guides/tools/code-execution-tool)**: The model can write and execute Python code for calculations, data analysis, and complex computations.
* **[Image/Video Understanding](/docs/guides/tools/search-tools#parameter-enable_image_understanding-supported-by-web-search-and-x-search)**: Optional visual content understanding and analysis for search results encountered (video understanding is only available for X posts).
* **[Collections Search](/docs/guides/tools/collections-search-tool)**: The model can search through your uploaded knowledge bases and collections to retrieve relevant information.
* **[Remote MCP Tools](/docs/guides/tools/remote-mcp-tools)**: Connect to external MCP servers to access custom tools.
* **[Document Search](/docs/guides/files)**: Upload files and chat with them using intelligent document search. This tool is automatically enabled when you attach files to a chat message.

## Quick Start

We strongly recommend using the xAI Python SDK in streaming mode when using agentic tool calling. Doing so grants you the full feature set of the API, including the ability to get real-time observability and immediate feedback during potentially long-running requests.

Here is a quick start example of using the agentic tool calling API.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search, x_search, code_execution

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    # All server-side tools active
    tools=[
        web_search(),
        x_search(),
        code_execution(),
    ],
)

# Feel free to change the query here to a question of your liking
chat.append(user("What are the latest updates from xAI?"))

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nFinal Response:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nCitations:")
print(response.citations)
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```

You will be able to see output like:

```output
Thinking... (270 tokens)
Calling tool: x_user_search with arguments: {"query":"xAI official","count":1}
Thinking... (348 tokens)
Calling tool: x_user_search with arguments: {"query":"xAI","count":5}
Thinking... (410 tokens)
Calling tool: x_keyword_search with arguments: {"query":"from:xai","limit":10,"mode":"Latest"}
Thinking... (667 tokens)
Calling tool: web_search with arguments: {"query":"xAI latest updates site:x.ai","num_results":5}
Thinking... (850 tokens)
Calling tool: browse_page with arguments: {"url": "https://x.ai/news"}
Thinking... (1215 tokens)

Final Response:
### Latest Updates from xAI (as of October 12, 2025)

xAI primarily shares real-time updates via their official X (Twitter) account (@xai), with more formal announcements on their website (x.ai). Below is a summary of the most recent developments...

... full response omitted for brevity

Citations:
[
'https://x.com/i/user/1912644073896206336',
'https://x.com/i/user/1019237602585645057',
'https://x.com/i/status/1975607901571199086',
'https://x.com/i/status/1975608122845896765',
'https://x.com/i/status/1975608070245175592',
'https://x.com/i/user/1603826710016819209',
'https://x.com/i/status/1975608007250829383',
'https://status.x.ai/',
'https://x.com/i/user/150543432',
'https://x.com/i/status/1975608184711880816',
'https://x.com/i/status/1971245659660718431',
'https://x.com/i/status/1975608132530544900',
'https://x.com/i/user/1661523610111193088',
'https://x.com/i/status/1977121515587223679',
'https://x.ai/news/grok-4-fast',
'https://x.com/i/status/1975608017396867282',
'https://x.ai/',
'https://x.com/i/status/1975607953391755740',
'https://x.com/i/user/1875560944044273665',
'https://x.ai/news',
'https://docs.x.ai/docs/release-notes'
]


Usage:
completion_tokens: 1216
prompt_tokens: 29137
total_tokens: 31568
prompt_text_tokens: 29137
reasoning_tokens: 1215
cached_prompt_text_tokens: 22565
server_side_tools_used: SERVER_SIDE_TOOL_X_SEARCH
server_side_tools_used: SERVER_SIDE_TOOL_X_SEARCH
server_side_tools_used: SERVER_SIDE_TOOL_X_SEARCH
server_side_tools_used: SERVER_SIDE_TOOL_WEB_SEARCH
server_side_tools_used: SERVER_SIDE_TOOL_WEB_SEARCH

{'SERVER_SIDE_TOOL_X_SEARCH': 3, 'SERVER_SIDE_TOOL_WEB_SEARCH': 2}


Server Side Tool Calls:
[id: "call_51132959"
function {
  name: "x_user_search"
  arguments: "{\"query\":\"xAI official\",\"count\":1}"
}
, id: "call_00956753"
function {
  name: "x_user_search"
  arguments: "{\"query\":\"xAI\",\"count\":5}"
}
, id: "call_07881908"
function {
  name: "x_keyword_search"
  arguments: "{\"query\":\"from:xai\",\"limit\":10,\"mode\":\"Latest\"}"
}
, id: "call_43296276"
function {
  name: "web_search"
  arguments: "{\"query\":\"xAI latest updates site:x.ai\",\"num_results\":5}"
}
, id: "call_70310550"
function {
  name: "browse_page"
  arguments: "{\"url\": \"https://x.ai/news\"}"
}
]
```

## Understanding the Agentic Tool Calling Response

The agentic tool calling API provides rich observability into the autonomous research process. This section dives deep into the original code snippet above, covering key ways to effectively use the API and understand both real-time streaming responses and final results:

### Real-time server-side tool calls

When executing agentic requests using streaming, you can observe **every tool call decision** the model makes in real-time via the `tool_calls` attribute on the `chunk` object. This shows the exact parameters the agent chose for each tool invocation, giving you visibility into its search strategy. Occasionally the model may decide to invoke multiple tools in parallel during a single turn, in which case each entry in the list of `tool_calls` would represent one of those parallel tool calls; otherwise, only a single entry would be present in `tool_calls`.

**Note**: Only the tool call invocations themselves are shown - **server-side tool call outputs are not returned** in the API response. The agent uses these outputs internally to formulate its final response, but they are not exposed to the user.

When using the xAI Python SDK in streaming mode, it will automatically accumulate the `tool_calls` into the `response` object for you, letting you access a final list of all the server-side tool calls made during the agentic loop. This is demonstrated in the [section below](#server-side-tool-calls-vs-tool-usage).

```pythonWithoutSDK
for tool_call in chunk.tool_calls:
    print(f"\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
```

```output
Calling tool: x_user_search with arguments: {"query":"xAI official","count":1}
Calling tool: x_user_search with arguments: {"query":"xAI","count":5}
Calling tool: x_keyword_search with arguments: {"query":"from:xai","limit":10,"mode":"Latest"}
Calling tool: web_search with arguments: {"query":"xAI latest updates site:x.ai","num_results":5}
Calling tool: browse_page with arguments: {"url": "https://x.ai/news"}
```

### Citations

The `citations` attribute on the `response` object provides a comprehensive list of URLs for all sources the agent encountered during its search process. They are **only returned when the agentic request completes** and are **not available in real-time** during streaming. Citations are automatically collected from successful tool executions and provide full traceability of the agent's information sources.

Note that not every URL here will necessarily be relevant to the final answer, as the agent may examine a particular source and determine it is not sufficiently relevant to the user's original query.

```pythonWithoutSDK
response.citations
```

```output
[
'https://x.com/i/user/1912644073896206336',
'https://x.com/i/status/1975607901571199086',
'https://x.ai/news',
'https://docs.x.ai/docs/release-notes',
...
]
```

### Server-side Tool Calls vs Tool Usage

The API provides two related but distinct metrics for server-side tool executions:

`tool_calls` - All Attempted Calls

```pythonWithoutSDK
response.tool_calls
```

Returns a list of all **attempted** tool calls made during the agentic process. Each entry is a [ToolCall](https://github.com/xai-org/xai-proto/blob/736b835b0c0dd93698664732daad49f87a2fbc6f/proto/xai/api/v1/chat.proto#L474) object containing:

* `id`: Unique identifier for the tool call
* `function.name`: The name of the specific server-side tool called
* `function.arguments`: The parameters passed to the server-side tool

This includes **every tool call attempt**, even if some fail.

```output
[id: "call_51132959"
function {
  name: "x_user_search"
  arguments: "{\"query\":\"xAI official\",\"count\":1}"
}
, id: "call_07881908"
function {
  name: "x_keyword_search"
  arguments: "{\"query\":\"from:xai\",\"limit\":10,\"mode\":\"Latest\"}"
}
, id: "call_43296276"
function {
  name: "web_search"
  arguments: "{\"query\":\"xAI latest updates site:x.ai\",\"num_results\":5}"
}
]
```

`server_side_tool_usage` - Successful Calls (Billable)

```pythonWithoutSDK
response.server_side_tool_usage
```

Returns a map of successfully executed tools and their invocation counts. This represents only the tool calls that returned meaningful responses and is what determines your billing.

```output
{'SERVER_SIDE_TOOL_X_SEARCH': 3, 'SERVER_SIDE_TOOL_WEB_SEARCH': 2}
```

### Tool Call Function Names vs Usage Categories

The function names in `tool_calls` represent the precise/exact name of the tool invoked by the model, while the entries in `server_side_tool_usage` provide a more high-level categorization that aligns with the original tool passed in the `tools` array of the request.

**Function Name to Usage Category Mapping:**

| Usage Category | Function Name(s) |
|----------------|------------------|
| `SERVER_SIDE_TOOL_WEB_SEARCH` | `web_search`, `web_search_with_snippets`, `browse_page` |
| `SERVER_SIDE_TOOL_X_SEARCH` | `x_user_search`, `x_keyword_search`, `x_semantic_search`, `x_thread_fetch` |
| `SERVER_SIDE_TOOL_CODE_EXECUTION` | `code_execution` |
| `SERVER_SIDE_TOOL_VIEW_X_VIDEO` | `view_x_video` |
| `SERVER_SIDE_TOOL_VIEW_IMAGE` | `view_image` |
| `SERVER_SIDE_TOOL_COLLECTIONS_SEARCH` | `collections_search` |
| `SERVER_SIDE_TOOL_MCP` | `{server_label}.{tool_name}` if `server_label` provided, otherwise `{tool_name}` |

### When Tool Calls and Usage Differ

In most cases, `tool_calls` and `server_side_tool_usage` will show the same tools. However, they can differ when:

* **Failed tool executions**: The model attempts to browse a non-existent webpage, fetch a deleted X post, or encounters other execution errors
* **Invalid parameters**: Tool calls with malformed arguments that can't be processed
* **Network or service issues**: Temporary failures in the tool execution pipeline

The agentic system is robust enough to handle these failures gracefully, updating its trajectory and continuing with alternative approaches when needed.

**Billing Note**: Only successful tool executions (`server_side_tool_usage`) are billed. Failed attempts are not charged.

### Server-side Tool Call and Client-side Tool Call

Agentic tool calling supports mixing server-side tools and client-side tools, which enables more use cases when some private tools and data are needed during the agentic tool calling process.

To determine whether the received tool calls need to be executed by the client side, you can simply check the type of the tool call.

For xAI Python SDK users, you can use the provided `get_tool_call_type` function to get the type of the tool calls.

For a full guide into requests that mix server-side and client-side tools, please check out the [advanced usage](/docs/guides/tools/advanced-usage) page.

**xAI Python SDK Users**: Version 1.4.0 of the xai-sdk package is the minimum requirement to use the `get_tool_call_type` function.

```pythonXAI
# ...
response = chat.sample()

from xai_sdk.tools import get_tool_call_type

for tool_call in response.tool_calls:
    print(get_tool_call_type(tool_call))
```

The available tool call types are listed below:

| Tool call types | Description |
|---------------|-------------|
| `"client_side_tool"` | Indicates this tool call is a **client-side tool** call, and an invocation to this function on the client side is required and the tool output needs to be appended to the chat |
| `"web_search_tool"` | Indicates this tool call is a **web-search tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"x_search_tool"` | Indicates this tool call is an **x-search tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"code_execution_tool"` | Indicates this tool call is a **code-execution tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"collections_search_tool"` | Indicates this tool call is a **collections-search tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"mcp_tool"` | Indicates this tool call is an **MCP tool** call, which is performed by xAI server, **NO** action needed from the client side |

### Understanding Token Usage

Agentic requests have unique token usage patterns compared to standard chat completions. Here's how each token field in the usage object is calculated:

#### `completion_tokens`

Represents **only the final text output** of the model - the comprehensive answer returned to the user. This is typically much smaller than you might expect for such rich, research-driven responses, as the agent performs all its intermediate reasoning and tool orchestration internally.

#### `prompt_tokens`

Represents the **cumulative input tokens** across all inference requests made during the agentic process. Since agentic workflows involve multiple reasoning steps with tool calls, the model makes several inference requests throughout the research process. Each request includes the full conversation history up to that point, which grows as the agent progresses through its research.

While this can result in higher `prompt_tokens` counts, agentic requests benefit significantly from **prompt caching**. The majority of the prompt (the conversation prefix) remains unchanged between inference steps, allowing for efficient caching of the shared context. This means that while the total `prompt_tokens` may appear high, much of the computation is optimized through intelligent caching of the stable conversation history, leading to better cost efficiency overall.

#### `reasoning_tokens`

Represents the tokens used for the model's internal reasoning process during agentic workflows. This includes the computational work the agent performs to plan tool calls, analyze results, and formulate responses, but excludes the final output tokens.

#### `cached_prompt_text_tokens`

Indicates how many prompt tokens were served from cache rather than recomputed. This shows the efficiency gains from prompt caching - higher values indicate better cache utilization and lower costs.

#### `prompt_image_tokens`

Represents the tokens derived from visual content that the agent processes during the request. These tokens are produced when visual understanding is enabled and the agent views images (e.g., via web browsing) or analyzes video frames on X. They are counted separately from text tokens and reflect the cost of ingesting visual features alongside the textual context. If no images or videos are processed, this value will be zero.

#### `prompt_text_tokens` and `total_tokens`

`prompt_text_tokens` reflects the actual text tokens in prompts (excluding any special tokens), while `total_tokens` is the sum of all token types used in the request.

## Synchronous Agentic Requests (Non-streaming)

Although not typically recommended, for simpler use cases or when you want to wait for the complete agentic workflow to finish before processing the response, you can use synchronous requests:

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import code_execution, web_search, x_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        web_search(),
        x_search(),
        code_execution(),
    ],
)

chat.append(user("What is the latest update from xAI?"))

# Get the final response in one go once it's ready
response = chat.sample()

print("\\n\\nFinal Response:")
print(response.content)

# Access the citations of the final response
print("\\n\\nCitations:")
print(response.citations)

# Access the usage details from the entire search process
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)

# Access the server side tool calls of the final response
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```

Synchronous requests will wait for the entire agentic process to complete before returning the response. This is simpler for basic use cases but provides less visibility into the intermediate steps compared to streaming.

## Using Tools with OpenAI Responses API

We also support using the OpenAI Responses API in both streaming and non-streaming modes.

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "what is the latest update from xAI?",
        },
    ],
    tools=[
        {
            "type": "web_search",
        },
        {
            "type": "x_search",
        },
    ],
)

print(response)
```

```pythonWithoutSDK
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "what is the latest update from xAI?"
        }
    ],
    "tools": [
        {
            "type": "web_search"
        },
        {
            "type": "x_search"
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "what is the latest update from xAI?"
    }
  ],
  "tools": [
    {
      "type": "web_search"
    },
    {
      "type": "x_search"
    }
  ]
}'
```

### Identifying the Client-side Tool Call

A critical step in mixing server-side tools and client-side tools is to identify whether a returned tool call is a client-side tool that needs to be executed locally on the client side.

Similar to the way in xAI Python SDK, you can identify the client-side tool call by checking the `type` of the output entries (`response.output[].type`) in the response of OpenAI Responses API.

| Types | Description |
|---------------|-------------|
| `"function_call"` | Indicates this tool call is a **client-side tool** call, and an invocation to this function on the client side is required and the tool output needs to be appended to the chat |
| `"web_search_call"` | Indicates this tool call is a **web-search tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"x_search_call"` | Indicates this tool call is an **x-search tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"code_interpreter_call"` | Indicates this tool call is a **code-execution tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"file_search_call"` | Indicates this tool call is a **collections-search tool** call, which is performed by xAI server, **NO** action needed from the client side |
| `"mcp_call"` | Indicates this tool call is an **MCP tool** call, which is performed by xAI server, **NO** action needed from the client side |

## Agentic Tool Calling Requirements and Limitations

### Model Compatibility

* **Supported Models**: `grok-4`, `grok-4-fast`, `grok-4-fast-non-reasoning`, `grok-4-1-fast`, `grok-4-1-fast-non-reasoning`
* **Strongly Recommended**: `grok-4-1-fast` - specifically trained to excel at agentic tool calling

### Request Constraints

* **No batch requests**: `n > 1` not supported
* **No response format**: Structured output not yet available with agentic tool calling
* **Limited sampling params**: Only `temperature` and `top_p` are respected

**Note**: These constraints may be relaxed in future releases based on user feedback.

## FAQ and Troubleshooting

### I'm seeing empty or incorrect content when using agentic tool calling with the xAI Python SDK

Please make sure to upgrade to the latest version of the xAI SDK. Agentic tool calling requires version `1.3.1` or above.


===/docs/guides/tools/remote-mcp-tools===
#### Guides

# Remote MCP Tools

Remote MCP Tools allow Grok to connect to external MCP (Model Context Protocol) servers, extending its capabilities with custom tools from third parties or your own implementations. Simply specify a server URL and optional configuration - xAI manages the MCP server connection and interaction on your behalf.

**xAI Python SDK Users**: Version 1.4.0 of the xai-sdk package is required to use Remote MCP Tools.

## SDK Support

Remote MCP tools are supported in the xAI native SDK and the OpenAI compatible Responses API.

The `require_approval` and `connector_id` parameters in the OpenAI Responses API are not currently supported.

## Configuration

To use remote MCP tools, you need to configure the connection to your MCP server in the tools array of your request.

| Parameter | Required | Description |
|-----------|-------------------|-------------|
| `server_url` | Yes | The URL of the MCP server to connect to. Only Streaming HTTP and SSE transports are supported. |
| `server_label` | No | A label to identify the server (used for tool call prefixing) |
| `server_description` | No | A description of what the server provides |
| `allowed_tool_names` | No | List of specific tool names to allow (empty allows all) |
| `authorization` | No | A token that will be set in the Authorization header on requests to the MCP server |
| `extra_headers` | No | Additional headers to include in requests |

### Basic MCP Tool Usage

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import mcp

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",
    tools=[
        mcp(server_url="https://mcp.deepwiki.com/mcp"),
    ],
)

chat.append(user("What can you do with https://github.com/xai-org/xai-sdk-python?"))

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nFinal Response:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What can you do with https://github.com/xai-org/xai-sdk-python?",
        },
    ],
    tools=[
        {
            "type": "mcp",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "server_label": "deepwiki",
        }
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What can you do with https://github.com/xai-org/xai-sdk-python?"
        }
    ],
    "tools": [
        {
            "type": "mcp",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "server_label": "deepwiki",
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What can you do with https://github.com/xai-org/xai-sdk-python?"
    }
  ],
  "tools": [
    {
        "type": "mcp",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "server_label": "deepwiki"
    }
  ]
}'
```

## Tool Enablement and Access Control

When you configure a Remote MCP Tool without specifying `allowed_tool_names`, all tool definitions exposed by the MCP server are automatically injected into the model's context. This means the model gains access to every tool that the MCP server provides, allowing it to use any of them during the conversation.

For example, if an MCP server exposes 10 different tools and you don't specify `allowed_tool_names`, all 10 tool definitions will be available to the model. The model can then choose to call any of these tools based on the user's request and the tool descriptions.

Use the `allowed_tool_names` parameter to selectively enable only specific tools from an MCP server. This can give you several key benefits:

* **Better Performance**: Reduce context overhead by limiting tool definitions the model needs to consider
* **Reduced Risk**: For example, restrict access to tools that only perform read-only operations to prevent the model from modifying data

```pythonXAI
# Enable only specific tools from a server with many available tools
mcp(
    server_url="https://comprehensive-tools.example.com/mcp",
    allowed_tool_names=["search_database", "format_data"]
)
```

Instead of giving the model access to every tool the server offers, this approach keeps Grok focused and efficient while ensuring it has exactly the capabilities it needs.

## Multi-Server Support

Enable multiple MCP servers simultaneously to create a rich ecosystem of specialized tools:

```pythonXAI
chat = client.chat.create(
    model="grok-4-1-fast",
    tools=[
        mcp(server_url="https://mcp.deepwiki.com/mcp", server_label="deepwiki"),
        mcp(server_url="https://your-custom-tools.com/mcp", server_label="custom"),
        mcp(server_url="https://api.example.com/tools", server_label="api-tools"),
    ],
)
```

Each server can provide different capabilities - documentation tools, API integrations, custom business logic, or specialized data processing - all accessible within a single conversation.

## Best Practices

* **Provide clear server metadata**: Use descriptive `server_label` and `server_description` when configuring multiple MCP servers to help the model understand each server's purpose and select the right tools
* **Filter tools appropriately**: Use `allowed_tool_names` to restrict access to only necessary tools, especially when servers have many tools since the model must keep all available tool definitions in context
* **Use secure connections**: Always use HTTPS URLs and implement proper authentication mechanisms on your MCP server
* **Provide Examples**: While the model can generally figure out what tools to use based on the tool descriptions and the user request it may help to provide examples in the prompt


===/docs/guides/tools/search-tools===
#### Guides

# Search Tools

Agentic search represents one of the most compelling applications of agentic tool calling, with `grok-4-1-fast` specifically trained to excel in this domain. Leveraging its speed and reasoning capabilities, the model iteratively calls search tools—analyzing responses and making follow-up queries as needed—to seamlessly navigate web pages and X posts, uncovering difficult-to-find information or insights that would otherwise require extensive human analysis.

**xAI Python SDK Users**: Version 1.3.1 of the xai-sdk package is required to use the agentic tool calling API.

## Available Search Tools

You can use the following server-side search tools in your request:

* **Web Search** - allows the agent to search the web and browse pages
* **X Search** - allows the agent to perform keyword search, semantic search, user search, and thread fetch on X

You can customize which tools are enabled in a given request by listing the needed tools in the `tools` parameter in the request.

| Tool | xAI SDK | OpenAI Responses API |
|------|---------|----------------------|
| Web Search | `web_search` | `web_search` |
| X Search | `x_search` | `x_search` |

## Retrieving Citations

Citations provide traceability for sources used during agentic search. Access them from the response object:

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[web_search()],
)

chat.append(user("What is xAI?"))

is_thinking = True
for response, chunk in chat.stream():
    # View the server-side tool calls as they are being made in real-time
    for tool_call in chunk.tool_calls:
        print(f"\\nCalling tool: {tool_call.function.name} with arguments: {tool_call.function.arguments}")
    if response.usage.reasoning_tokens and is_thinking:
        print(f"\\rThinking... ({response.usage.reasoning_tokens} tokens)", end="", flush=True)
    if chunk.content and is_thinking:
        print("\\n\\nFinal Response:")
        is_thinking = False
    if chunk.content and not is_thinking:
        print(chunk.content, end="", flush=True)

print("\\n\\nCitations:")
print(response.citations)
print("\\n\\nUsage:")
print(response.usage)
print(response.server_side_tool_usage)
print("\\n\\nServer Side Tool Calls:")
print(response.tool_calls)
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is xAI?",
        },
    ],
    tools=[
        {
            "type": "web_search",
        },
    ],
)

# Access the response
print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is xAI?"
        }
    ],
    "tools": [
        {
            "type": "web_search",
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)

# Access the citations of the final response
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is xAI?"
    }
  ],
  "tools": [
    {
      "type": "web_search"
    }
  ]
}'
```

As mentioned in the [overview page](/docs/guides/tools/overview#citations), the citations array contains the URLs of all sources the agent encountered during its search process, meaning that not every URL here will necessarily be relevant to the final answer, as the agent may examine a particular source and determine it is not sufficiently relevant to the user's original query.

For complete details on citations, including when they're available and usage notes, see the [overview page](/docs/guides/tools/overview#citations).

## Applying Search Filters to Control Agentic Search

Each search tool supports a set of optional search parameters to help you narrow down the search space and limit the sources/information the agent is exposed to during its search process.

| Tool | Supported Filter Parameters |
|------|-----------------------------|
| Web Search | `allowed_domains`, `excluded_domains`, `enable_image_understanding` |
| X Search | `allowed_x_handles`, `excluded_x_handles`, `from_date`, `to_date`, `enable_image_understanding`, `enable_video_understanding`|

### Web Search Parameters

##### Only Search in Specific Domains

Use `allowed_domains` to make the web search **only** perform the search and web browsing on web pages that fall within the specified domains.

`allowed_domains` can include a maximum of five domains.

`allowed_domains` cannot be set together with `excluded_domains` in the same request.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        web_search(allowed_domains=["wikipedia.org"]),
    ],
)

chat.append(user("What is xAI?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is xAI?",
        },
    ],
    tools=[
        {
            "type": "web_search",
            "filters": {"allowed_domains": ["wikipedia.org"]},
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is xAI?"
        }
    ],
    "tools": [
        {
            "type": "web_search",
            "filters": {"allowed_domains": ["wikipedia.org"]},
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is xAI?"
    }
  ],
  "tools": [
    {
      "type": "web_search",
      "filters": {"allowed_domains": ["wikipedia.org"]}
    }
  ]
}'
```

##### Exclude Specific Domains

Use `excluded_domains` to prevent the model from including the specified domains in any web search tool invocations and from browsing any pages on those domains.

`excluded_domains` can include a maximum of five domains.

`excluded_domains` cannot be set together with `allowed_domains` in the same request.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        web_search(excluded_domains=["wikipedia.org"]),
    ],
)

chat.append(user("What is xAI?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is xAI?",
        },
    ],
    tools=[
        {
            "type": "web_search",
            "filters": {"excluded_domains": ["wikipedia.org"]},
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is xAI?"
        }
    ],
    "tools": [
        {
            "type": "web_search",
            "filters": {"excluded_domains": ["wikipedia.org"]},
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is xAI?"
    }
  ],
  "tools": [
    {
      "type": "web_search",
      "filters": {"excluded_domains": ["wikipedia.org"]}
    }
  ]
}'
```

##### Enable Image Understanding

Setting `enable_image_understanding` to true equips the agent with access to the `view_image` tool, allowing it to invoke this tool on any image URLs encountered during the search process. The model can then interpret and analyze image contents, incorporating this visual information into its context to potentially influence the trajectory of follow-up tool calls.

When the model invokes this tool, you will see it as an entry in `chunk.tool_calls` and `response.tool_calls` with the `image_url` as a parameter. Additionally, `SERVER_SIDE_TOOL_VIEW_IMAGE` will appear in `response.server_side_tool_usage` along with the number of times it was called when using the xAI Python SDK.

Note that enabling this feature increases token usage, as images are processed and represented as image tokens in the model's context.

Enabling this parameter for Web Search will also enable the image understanding for X Search tool if it's also included in the request.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import web_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        web_search(enable_image_understanding=True),
    ],
)

chat.append(user("What is included in the image in xAI's official website?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is included in the image in xAI's official website?",
        },
    ],
    tools=[
        {
            "type": "web_search",
            "enable_image_understanding": True,
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is included in the image in xAI's official website?"
        }
    ],
    "tools": [
        {
            "type": "web_search",
            "enable_image_understanding": True,
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is included in the image in xAI's official website?"
    }
  ],
  "tools": [
    {
      "type": "web_search",
      "enable_image_understanding": true
    }
  ]
}'
```

### X Search Parameters

##### Only Consider X Posts from Specific Handles

Use `allowed_x_handles` to consider X posts only from a given list of X handles. The maximum number of handles you can include is 10.

`allowed_x_handles` cannot be set together with `excluded_x_handles` in the same request.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import x_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        x_search(allowed_x_handles=["elonmusk"]),
    ],
)

chat.append(user("What is the current status of xAI?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is the current status of xAI?",
        },
    ],
    tools=[
        {
            "type": "x_search",
            "allowed_x_handles": ["elonmusk"],
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is the current status of xAI?"
        }
    ],
    "tools": [
        {
            "type": "x_search",
            "allowed_x_handles": ["elonmusk"],
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is the current status of xAI?"
    }
  ],
  "tools": [
    {
      "type": "x_search",
      "allowed_x_handles": ["elonmusk"]
    }
  ]
}'
```

##### Exclude X Posts from Specific Handles

Use `excluded_x_handles` to prevent the model from including X posts from the specified handles in any X search tool invocations. The maximum number of handles you can exclude is 10.

`excluded_x_handles` cannot be set together with `allowed_x_handles` in the same request.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import x_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        x_search(excluded_x_handles=["elonmusk"]),
    ],
)

chat.append(user("What is the current status of xAI?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is the current status of xAI?",
        },
    ],
    tools=[
        {
            "type": "x_search",
            "excluded_x_handles": ["elonmusk"],
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is the current status of xAI?"
        }
    ],
    "tools": [
        {
            "type": "x_search",
            "excluded_x_handles": ["elonmusk"],
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is the current status of xAI?"
    }
  ],
  "tools": [
    {
      "type": "x_search",
      "excluded_x_handles": ["elonmusk"]
    }
  ]
}'
```

##### Date Range

You can restrict the date range of search data used by specifying `from_date` and `to_date`. This limits the data to the period from
`from_date` to `to_date`, including both dates.

Both fields need to be in ISO8601 format, e.g., "YYYY-MM-DD". If you're using the xAI Python SDK, the
`from_date` and `to_date` fields can be passed as `datetime.datetime` objects.

The fields can also be used independently. With only `from_date` specified, the data used will be from the
`from_date` to today, and with only `to_date` specified, the data used will be all data until the `to_date`.

```pythonXAI
import os
from datetime import datetime

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import x_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        x_search(
            from_date=datetime(2025, 10, 1),
            to_date=datetime(2025, 10, 10),
        ),
    ],
)

chat.append(user("What is the current status of xAI?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is the current status of xAI?",
        },
    ],
    tools=[
        {
            "type": "x_search",
            "from_date": "2025-10-01",
            "to_date": "2025-10-10",
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is the current status of xAI?"
        }
    ],
    "tools": [
        {
            "type": "x_search",
            "from_date": "2025-10-01",
            "to_date": "2025-10-10",
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is the current status of xAI?"
    }
  ],
  "tools": [
    {
      "type": "x_search",
      "from_date": "2025-10-01",
      "to_date": "2025-10-10"
    }
  ]
}'
```

##### Enable Image Understanding

Setting `enable_image_understanding` to true equips the agent with access to the `view_image` tool, allowing it to invoke this tool on any image URLs encountered during the search process. The model can then interpret and analyze image contents, incorporating this visual information into its context to potentially influence the trajectory of follow-up tool calls.

When the model invokes this tool, you will see it as an entry in `chunk.tool_calls` and `response.tool_calls` with the `image_url` as a parameter. Additionally, `SERVER_SIDE_TOOL_VIEW_IMAGE` will appear in `response.server_side_tool_usage` along with the number of times it was called when using the xAI Python SDK.

Note that enabling this feature increases token usage, as images are processed and represented as image tokens in the model's context.

Enabling this parameter for X Search will also enable the image understanding for Web Search tool if it's also included in the request.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import x_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        x_search(enable_image_understanding=True),
    ],
)

chat.append(user("What images are being shared in recent xAI posts?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What images are being shared in recent xAI posts?",
        },
    ],
    tools=[
        {
            "type": "x_search",
            "enable_image_understanding": True,
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What images are being shared in recent xAI posts?"
        }
    ],
    "tools": [
        {
            "type": "x_search",
            "enable_image_understanding": True,
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What images are being shared in recent xAI posts?"
    }
  ],
  "tools": [
    {
      "type": "x_search",
      "enable_image_understanding": true
    }
  ]
}'
```

##### Enable Video Understanding

Setting `enable_video_understanding` to true equips the agent with access to the `view_x_video` tool, allowing it to invoke this tool on any video URLs encountered in X posts during the search process. The model can then analyze video content, incorporating this information into its context to potentially influence the trajectory of follow-up tool calls.

When the model invokes this tool, you will see it as an entry in `chunk.tool_calls` and `response.tool_calls` with the `video_url` as a parameter. Additionally, `SERVER_SIDE_TOOL_VIEW_X_VIDEO` will appear in `response.server_side_tool_usage` along with the number of times it was called when using the xAI Python SDK.

Note that enabling this feature increases token usage, as video content is processed and represented as tokens in the model's context.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import x_search

client = Client(api_key=os.getenv("XAI_API_KEY"))
chat = client.chat.create(
    model="grok-4-1-fast",  # reasoning model
    tools=[
        x_search(enable_video_understanding=True),
    ],
)

chat.append(user("What is the latest video talking about from the xAI official X account?"))

# stream or sample the response...
```

```pythonOpenAISDK
import os
from openai import OpenAI

api_key = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1",
)

response = client.responses.create(
    model="grok-4-1-fast",
    input=[
        {
            "role": "user",
            "content": "What is the latest video talking about from the xAI official X account?",
        },
    ],
    tools=[
        {
            "type": "x_search",
            "enable_video_understanding": True,
        },
    ],
)

print(response)
```

```pythonRequests
import os
import requests

url = "https://api.x.ai/v1/responses"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
}
payload = {
    "model": "grok-4-1-fast",
    "input": [
        {
            "role": "user",
            "content": "What is the latest video talking about from the xAI official X account?"
        }
    ],
    "tools": [
        {
            "type": "x_search",
            "enable_video_understanding": True,
        }
    ]
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

```bash
curl https://api.x.ai/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
  "model": "grok-4-1-fast",
  "input": [
    {
      "role": "user",
      "content": "What is the latest video talking about from the xAI official X account?"
    }
  ],
  "tools": [
    {
      "type": "x_search",
      "enable_video_understanding": true
    }
  ]
}'
```


===/docs/guides/use-with-code-editors===
# Use with Code Editors

You can use Grok with coding assistant plugins to help you code. Our Code models are specifically optimized for this task, which would provide you a smoother experience.

For pricing and limits of Code models, check out [Models and Pricing](../models).

## Using Grok Code models with Cline

To use Grok with Cline, first download Cline from VSCode marketplace. Once you have installed Cline in VSCode, open Cline.

Click on "Use your own API key".

Then, you can save your xAI API key to Cline.

After setting up your xAI API key with Cline, you can set to use a coding model. Go to Cline settings -> API Configuration and you can choose `{{LATEST_CODE_MODEL_NAME}}` as the model.

## Using Grok Code models with Cursor

You can also use Grok with Cursor to help you code.

After installing Cursor, head to Cursor Settings -> Models.

Open API Keys settings, enter your xAI API key and set Override OpenAI Base URL to `https://api.x.ai/v1`

In the "Add or search model" input box, enter a coding model such as `{{LATEST_CODE_MODEL_NAME}}`. Then click on "Add Custom Model".

## Other code assistants supporting Grok Code models

Besides Cline and Cursor, you can also use our code model with [GitHub Copilot](https://github.com/features/copilot), [opencode](https://opencode.ai/), [Kilo Code](https://kilocode.ai/), [Roo Code](https://roocode.com/) and [Windsurf](https://windsurf.com/).


===/docs/guides/using-collections===
#### Guides

# Using Collections

In this guide, we will walk through the basics of:

* Creating a `collection`
* Adding a `document` to the `collection`
* Searching for relevant `documents` within the `collection`
* Deleting `documents` and `collections`

For an overview of what Collections is, please see [Collections](../key-information/collections).

You can upload a maximum of 100,000 files per collection.

## Creating a new collection

You can create a `collection` in the [xAI Console](https://console.x.ai) and navigate to the **Collections** tab. Make sure you are in the correct team.

Click on "Create new collection" to create a new `collection`.

You can choose to enable generate embeddings on document upload or not. We recommend leaving the generate embeddings setting to on.

Alternatively, you can create the collection with code:

```pythonXAI
import os
from xai_sdk import Client
client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    management_api_key=os.getenv("XAI_MANAGEMENT_API_KEY"),
    timeout=3600, # Override default timeout with longer timeout for reasoning models
)

collection = client.collections.create(
    name="SEC Filings", # You can optionally add in model_name and/or chunk_configuration
)

print(collection)
```

```bash
curl https://management-api.x.ai/v1/collections \\
  -X POST \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_MANAGEMENT_API_KEY" \\
  -d '{"collection_name": "SEC Filings"}'
```

## List available Collections

After adding a new collection, we can either see it in xAI Console, or list it via an API request. This example lists all collections available in the team.

```pythonXAI
# ... Create client
collections = client.collections.list()
print(collections)
```

```bash
curl https://management-api.x.ai/v1/collections \\
  -H "Authorization: Bearer $XAI_MANAGEMENT_API_KEY"
```

## View and update the configuration of a Collection

You can view and edit the Collection's configuration by clicking on Edit Collection.

This opens up the following modal where you can view the configuration and make changes.

To view the collection's configuration with code:

```pythonXAI
# ... Create client
collection = client.collections.get("collection_dbc087b1-6c99-493d-86c6-b401fee34a9d")

print(collection)
```

```bash
curl https://management-api.x.ai/v1/collections/collection_dbc087b1-6c99-493d-86c6-b401fee34a9d \\
  -H "Authorization: Bearer $XAI_MANAGEMENT_API_KEY"
```

To update the collection's configuration:

```pythonXAI
# ... Create client
collection = client.collections.update(
    "collection_dbc087b1-6c99-493d-86c6-b401fee34a9d",
    name="SEC Filings (New)"
)

print(collection)
```

```bash
curl https://management-api.x.ai/v1/collections/collection_dbc087b1-6c99-493d-86c6-b401fee34a9d \\
  -X PUT \\
  -H "Authorization: Bearer $XAI_MANAGEMENT_API_KEY" \\
  -d '{"collection_name": "SEC Filings (New)"}'
```

## Adding a document to the collection in xAI Console

Once you have created the new `collection`. You can click on it in the collections table to view the `documents` included in the `collection`.

Click on "Upload document" to upload a new `document`.

Once the upload has completed, each document is given a File ID. You can view the File ID, Collection ID and hash of the
`document` by clicking on the `document` in the documents table.

You can also upload documents via code:

```pythonXAI
# ... Create client
with open("tesla-20241231.html", "rb") as file:
    file_data = file.read()

document = client.collections.upload_document(
    collection_id="collection_dbc087b1-6c99-493d-86c6-b401fee34a9d", # The collection ID of the collection we want to upload to
    name="tesla-20241231.html", # The name that you want to use
    data=file_data, # The data payload
    content_type="text/html",
)
print(document)
```

## Searching for relevant documents within the collection

To search for relevant `documents` within one or multiple `collections`, obtain the Collection ID(s) of the collections that you want to search within first. Then, you can follow this example:

```pythonXAI
# ... Create client
response = client.collections.search(
    query="What were the key revenue drivers based on the SEC filings?",
    collection_ids=["collection_dbc087b1-6c99-493d-86c6-b401fee34a9d"],
)
print(response)
```

```bash
curl https://api.x.ai/v1/documents/search \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
      "query": "What were the key revenue drivers based on the SEC filings?",
      "source": {
          "collection_ids": [
              "collection_dbc087b1-6c99-493d-86c6-b401fee34a9d"
          ]
      }
}'
```

There are three search methods available:

* **Keyword search**
* **Semantic search**
* **Hybrid search** (combines both keyword and semantic methods)

By default, the system uses **hybrid search**, which generally delivers the best and most comprehensive results.

### Comparison of search modes

| Mode      | Description                                                                 | Best for                                      | Drawbacks                  |
|-----------|-----------------------------------------------------------------------------|-----------------------------------------------|----------------------------|
| **Keyword**   | Searches for exact matches of specified words, phrases, or numbers         | Precise terms (e.g., account numbers, dates, specific financial figures) | May miss contextually relevant content |
| **Semantic**  | Understands meaning and context to find conceptually related content       | Discovering general ideas, topics, or intent even when exact words differ | Less precise for specific terms |
| **Hybrid**    | Combines keyword and semantic search for broader and more accurate results | Most real-world use cases                     | Slightly higher latency    |

The hybrid approach balances precision and recall, making it the recommended default for the majority of queries.

An example to set hybrid mode:

```bash
curl https://api.x.ai/v1/documents/search \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $XAI_API_KEY" \\
  -d '{
      "query": "What were the key revenue drivers based on the SEC filings?",
      "source": {
          "collection_ids": [
              "collection_dbc087b1-6c99-493d-86c6-b401fee34a9d"
          ]
      },
      "retrieval_mode": {"type": "hybrid"}
}'
```

You can set `"retrieval_mode": {"type": "keyword"}` for keyword search and `"retrieval_mode": {"type": "semantic"}` for semantic search.

## Deleting documents and collections

You can delete the `documents` and `collections` on [xAI Console](https://console.x.ai) by clicking on the more button
on the right side of the collections or documents table.

To remove a document via code:

```pythonXAI
# ... Create client

client.collections.remove_document(
    collection_id="collection_dbc087b1-6c99-493d-86c6-b401fee34a9d",
    file_id="file_55a709d4-8edc-4f83-84d9-9f04fe49f832",
)
```

```bash
curl https://management-api.x.ai/v1/collections/collection_dbc087b1-6c99-493d-86c6-b401fee34a9d/documents/file_55a709d4-8edc-4f83-84d9-9f04fe49f832 \\
  -X DELETE \\
  -H "Authorization: Bearer $XAI_MANAGEMENT_API_KEY"
```

To remove the collection:

```pythonXAI
# ... Create client

client.collections.delete(collection_id="collection_dbc087b1-6c99-493d-86c6-b401fee34a9d")
```

```bash
curl https://management-api.x.ai/v1/collections/collection_dbc087b1-6c99-493d-86c6-b401fee34a9d \\
  -X DELETE \\
  -H "Authorization: Bearer $XAI_MANAGEMENT_API_KEY"
```


===/docs/key-information/billing===
#### Key Information

# Manage Billing

\*\*Ensure you are in the desired team before changing billing information. When you save the billing information or make a purchase for the first time, the billing information is saved to the team you are in and shared with its members.

There are two ways of billing:

* **Prepaid credits:** You can pre-purchase credits for your team. Your API consumption will be deducted from remaining prepaid credits available.
* **Monthly invoiced billing:** xAI will generate a monthly invoice based on your API consumption, when you don't have available prepaid credits. xAI will charge your default payment method with the invoiced amount at the end of each month.

**Monthly invoiced billing is disabled by default, with default Invoiced Spending Limit of $0.** This will introduce service disruption when you have consumed all of your prepaid credits. To enable monthly invoiced billing, set a higher than $0 Invoiced Spending Limit at [Billing -> API Credits](https://console.x.ai/team/default/billing) on xAI Console.

Your API consumption will be accounted for in the following order:

* Free/Promotional credits
* Prepaid credits
* Monthly invoiced billing (if Invoiced Spending Limit > $0)

**Any prepaid credits and added payment method will be made available to the team you made the purchase in.**

## Prepaid credits



You can only purchase prepaid credits with Guest Checkout at the moment, due to regulatory
requirements.

This is the most common way to build with xAI API. Before using API, you purchase a given amount of credits. When you use the API, xAI will track your consumption and deduct the amount from the credits available in your account.

You can add prepaid credits on the xAI Console [Billing -> API Credits](https://console.x.ai/team/default/billing) page.

On the same page, you can view the remaining prepaid credits, enter promo code, as well as any free credits granted by xAI team.

Note: When you make the purchase via bank transfer instead of credit card, the payment will take 2-3 business days to process. You will be granted with credits after the process has completed.

## Monthly invoiced billing and invoiced billing limit

Enterprise customers might find it beneficial to enroll in monthly invoiced billing to avoid disruption to their services.

When you have set a **$0 invoiced billing limit** (default), xAI will only use your available prepaid credits. **Your API requests will be automatically rejected once your prepaid credits are depleted.**

If you want to use monthly billing, you can **increase your invoiced billing limit** on [Billing -> API Credits](https://console.x.ai/team/default/billing) page. xAI will attempt to use your prepaid credits first, and the remaining amount will be charged to your default payment method at the end of the month. This ensures you won't experience interruption while consuming the API.

Once your monthly invoiced billing amount has reached the invoiced billing limit, you won't be able to get response until you have raised the invoiced billing limit.

## Saving payment method

When you make a purchase, we automatically keep it on file to make your next purchase easier. You can also manually add payment method on xAI Console [Billing -> Billing details -> Add Payment Information](https://console.x.ai/team/default/billing).

Currently we don't allow user to remove the last payment method on file. There might be changes in the future.

## Invoices

You can view your invoices for prepaid credits and monthly invoices on [Billing -> Invoices](https://console.x.ai/team/default/billing/invoices).

## Billing address and tax information

Enter your billing information carefully, as it will appear on your invoices. We are not able to
regenerate the invoices at the moment.

Your billing address and tax information will be displayed on the invoice. On [Billing -> Payment](https://console.x.ai/team/default/billing), you can also add/change your billing address. When you add/change billing address, you can optionally add your organization's tax information.


===/docs/key-information/collections===
#### Key Information

# Collections

Collections offers xAI API users a robust set of tools and methods to seamlessly integrate their enterprise requirements and internal knowledge bases with the xAI API. This feature enables efficient management, retrieval, and utilization of documents to enhance AI-driven workflows and applications.

There are two entities that users can create within the Collections service:

* `file`
  * A `file` is a single entity of a user-uploaded file.
* `collection`
  * A `collection` is a group of `files` linked together, with an embedding index for efficient retrieval of each `file`.
  * When you create a `collection` you have the option to automatically generate embeddings for any files uploaded to that `collection`. You can then perform semantic search across files in multiple `collections`.
  * A single `file` can belong to multiple `collections` but must be part of at least one `collection`.

## File storage and retrieval

Visit the **Collections** tab on the [xAI Console](https://console.x.ai) to create a new `collection`. Once created, you can add `files` to the `collection`. You can also add
`files` without adding them to a `collection` using our [Files API](/docs/guides/files/managing-files).

All your `collections` and their associated `files` can be viewed in the **Collections** tab.

Your `files` and their embedding index are securely encrypted and stored on our servers. The index enables efficient retrieval of `files` during a relevance search.

## Usage limits

Users can upload a maximum of 100,000 files per collection. We do not place any limits on the file size, etc.

## Data Privacy

We do not use user data stored on Collections for model training purposes by default, unless the user has given consent.


===/docs/key-information/consumption-and-rate-limits===
#### Key Information

# Consumption and Rate Limits

The cost of using our API is based on token consumptions. We charge different prices based on token category: - **Prompt text, audio and image tokens** - Charged at prompt token price - **Cached prompt tokens** - Charged at cached prompt token price - **Completion tokens** - Charged at completion token price - **Reasoning tokens** - Charged at completion token price

Visit [Models and Pricing](../models) for general pricing, or [xAI Console](https://console.x.ai) for pricing applicable to your team.

Each `grok` model has different rate limits. To check your team's rate limits, you can visit [xAI Console Models Page](https://console.x.ai/team/default/models).

## Basic unit to calculate consumption — Tokens

Token is the base unit of prompt size for model inference and pricing purposes. It consists of one or more character(s)/symbol(s).

When a Grok model handles your request, an input prompt will be decomposed into a list of tokens through a tokenizer.
The model will then make inference based on the prompt tokens, and generate completion tokens.
After the inference is completed, the completion tokens will be aggregated into a completion response sent back to you.

Our system will add additional formatting tokens to the input/output token, and if you selected a reasoning model, additional reasoning tokens will be added into the total token consumption as well.
Your actual consumption would be reflected either in the `usage` object returned in the API response, or in Usage Explorer on the [xAI Console](https://console.x.ai).

You can use [Tokenizer](https://console.x.ai/team/default/tokenizer) on xAI Console to visualize tokens a given text prompt, or use [Tokenize text](../api-reference#tokenize-text) endpoint on the API.

### Text tokens

Tokens can be either of a whole word, or smaller chunks of character combinations. The more common a word is, the more likely it would be a whole token.

For example, Flint is broken down into two tokens, while Michigan is a whole token.

In another example, most words are tokens by themselves, but "drafter" is broken down into "dra" and "fter", and "postmaster" is broken down into "post" and "master".

For a given text/image/etc. prompt or completion sequence, different tokenizers may break it down into different lengths of lists.

Different Grok models may also share or use different tokenizers. Therefore, **the same prompt/completion sequence may not have the same amount of tokens across different models.**

The token count in a prompt/completion sequence should be approximately linear to the sequence length.

### Image prompt tokens

Each image prompt will take between 256 to 1792 tokens, depending on the size of the image. The image + text token count must be less than the overall context window of the model.

### Estimating consumption with tokenizer on xAI Console or through API

The tokenizer page or API might display less token count than the actual token consumption. The
inference endpoints would automatically add pre-defined tokens to help our system process the
request.

On xAI Console, you can use the [tokenizer page](https://console.x.ai/team/default/tokenizer) to estimate how many tokens your text prompt will consume. For example, the following message would consume 5 tokens (the actual consumption may vary because of additional special tokens added by the system).

Message body:

```json
[
  {
    "role": "user",
    "content": "How is the weather today?"
  }
]
```

Tokenize result on Tokenizer page:

You can also utilize the [Tokenize Text](../api-reference#tokenize-text) API endpoint to tokenize the text, and count the output token array length.

### Cached prompt tokens

When you send the same prompt multiple times, we may cache your prompt tokens. This would result in reduced cost for these tokens at the cached token rate, and a quicker response.

### Reasoning tokens

The model may use reasoning to process your request. The reasoning content is returned in the response's `reasoning_content` field. The reasoning token consumption will be counted separately from `completion_tokens`, but will be counted in the `total_tokens`.

The reasoning tokens will be charged at the same price as `completion_tokens`.

`grok-4` does not return `reasoning_content`

## Hitting rate limits

To request a higher rate limit, please email support@x.ai with your anticipated volume.

For each tier, there is a maximum amount of requests per minute and tokens per minute. This is to ensure fair usage by all users of the system.

Once your request frequency has reached the rate limit, you will receive error code `429` in response.

You can either:

* Upgrade your team to higher tiers
* Change your consumption pattern to send fewer requests

## Checking token consumption

In each completion response, there is a `usage` object detailing your prompt and completion token count. You might find it helpful to keep track of it, in order to avoid hitting rate limits or having cost surprises.

```json
"usage": {
    "prompt_tokens":37,
    "completion_tokens":530,
    "total_tokens":800,
    "prompt_tokens_details": {
        "text_tokens":37,
        "audio_tokens":0,
        "image_tokens":0,
        "cached_tokens":8
    },
    "completion_tokens_details": {
        "reasoning_tokens":233,
        "audio_tokens":0,
        "accepted_prediction_tokens":0,
        "rejected_prediction_tokens":0
    },
    "num_sources_used":0
}
```

You can also check with the xAI, OpenAI or Anthropic SDKs.

```pythonXAI
import os

from xai_sdk import Client
from xai_sdk.chat import system, user

client = Client(api_key=os.getenv("XAI_API_KEY"))

chat = client.chat.create(
model="grok-4",
messages=[system("You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy.")]
)
chat.append(user("What is the meaning of life, the universe, and everything?"))

response = chat.sample()
print(response.usage)
```

```pythonOpenAISDK
import os
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

completion = client.chat.completions.create(
model="grok-4",
messages=[
{
"role": "system",
"content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy.",
},
{
"role": "user",
"content": "What is the meaning of life, the universe, and everything?",
},
],
)

if completion.usage:
print(completion.usage.to_json())
```

```javascriptOpenAISDK
import OpenAI from "openai";
const openai = new OpenAI({
apiKey: "<api key>",
baseURL: "https://api.x.ai/v1",
});

const completion = await openai.chat.completions.create({
model: "grok-4",
messages: [
{
role: "system",
content:
"You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy.",
},
{
role: "user",
content:
"What is the meaning of life, the universe, and everything?",
},
],
});

console.log(completion.usage);
```


===/docs/key-information/debugging===
#### Getting Started

# Debugging Errors

When you send a request, you would normally get a `200 OK` response from the server with the expected response body.
If there has been an error with your request, or error with our service, the API endpoint will typically return an error code with error message.

If there is an ongoing service disruption, you can visit
[https://status.x.ai](https://status.x.ai) for the latest updates. The status is also available
via RSS at [https://status.x.ai/feed.xml](https://status.x.ai/feed.xml).

The service status is also indicated in the navigation bar of this site.

Most of the errors will be accompanied by an error message that is self-explanatory. For typical status codes of each endpoint, visit [API Reference](api-reference) or view our [OpenAPI Document](https://docs.x.ai/openapi.json).

## Status Codes

Here is a list of potential errors and statuses arranged by status codes.

### 4XX Status Codes

| Status Code                    | Endpoints                              | Cause                                                                                                                                                                       | Solution                                                                                                                                         |
| ------------------------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 400Bad Request            | All Endpoints                          | - A `POST` method request body specified an invalid argument, or a `GET` method with dynamic route has an invalid param in the URL.- An incorrect API key is supplied. | - Please check your request body or request URL.                                                                                                 |
| 401Unauthorized           | All Endpoints                          | - No authorization header or an invalid authorization token is provided.                                                                                                    | - Supply an `Authorization: Bearer Token <XAI_API_KEY>` in the request header. You can get a new API key on [xAI Console](https://console.x.ai). |
| 403Forbidden              | All Endpoints                          | - Your API key/team doesn't have permission to perform the action.- Your API key/team is blocked.                                                                     | - Ask your team admin for permission.                                                                                                            |
| 404Not Found              | All Endpoints                          | - A model specified in a `POST` method request body is not found.- Trying to reach an invalid endpoint URL. (Misspelled URL)                                           | - Check your request body and endpoint URL with our [API Reference](api-reference).                                                              |
| 405Method Not Allowed     | All Endpoints                          | - The request method is not allowed. For example, sending a `POST` request to an endpoint supporting only `GET`.                                                            | - Check your request method with our [API Reference](api-reference).                                                                             |
| 415Unsupported Media Type | All Endpoints Supporting `POST` Method | - An empty request body in `POST` requests.- Not specifying `Content-Type: application/json` header.                                                                  | - Add a valid request body. - Ensure `Content-Type: application/json` header is present in the request header.                             |
| 422Unprocessable Entity   | All Endpoints Supporting `POST` Method | - An invalid format for a field in the `POST` request body.                                                                                                                 | - Check your request body is valid. You can find more information from [API Reference](api-reference).                                           |
| 429Too Many Requests      | All Inference Endpoints                | - You are sending requests too frequently and reaching rate limit                                                                                                           | - Reduce your request rate or increase your rate limit. You can find your current rate limit on [xAI Console](https://console.x.ai).             |

### 2XX Error Codes

| Status Code      | Endpoints                                   | Cause                                                                                                    | Solution                       |
| ---------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 202Accepted | `/v1/chat/deferred-completion/{request_id}` | - Your deferred chat completion request is queued for processing, but the response is not available yet. | - Wait for request processing. |

## Bug Report

If you believe you have encountered a bug and would like to contribute to our development process, [email API Bug Report](mailto:support@x.ai?subject=API%20Bug%20Report) to support@x.ai with your API request and response and relevant logs.

You can also chat in the `#help` channel of our [xAI API Developer Discord](https://discord.gg/x-ai).


===/docs/key-information/migrating-to-new-models===
#### Key Information

# Migrating to New Models

As we release newer, more advanced models, we are focusing resources on supporting customers with these models and will
be phasing out older versions.

You will see `deprecated` tag by the deprecated model names on [xAI Console](https://console.x.ai) models page. You
should consider moving to a newer model when the model of your choice is being deprecated.

We may transition a `deprecated` model to `obsolete` and discontinue serving the model across our services.
An `obsolete` model will be removed from our [Models and Pricing](../models) page as well as from [xAI Console](https://console.x.ai).

## Moving from an older generation model

When you move from an older model generation to a newer one, you usually won't need to make significant changes to
how you use the API. In your request body, you can switch the `"model"` field from the deprecating model to a current
model on [xAI Console](https://console.x.ai) models page.

The newer models are more performant, but you might want to check if your prompts and other parameters can work with the
new model and modify if necessary.

## Moving to the latest endpoints

When you are setting up to use new models, it might also be a good idea to ensure you're using the latest endpoints. The
latest endpoints have more stable supports for the model functionalities. Endpoints that are marked with `legacy`
might not receive any updates that support newer functionalities.

In general, the following endpoints are recommended: - Text and image input and text output: [Chat Completions](../api-reference#chat-completions) - `/v1/chat/completions` - Text input and image output: [Image Generations](../api-reference#image-generations) - `/v1/image/generations` - Tokenization: [Tokenize Text](../api-reference#tokenize-text) - `/v1/tokenize-text`


===/docs/key-information/regions===
#### Key Information

# Regional Endpoints

By default, you can access our API at `https://api.x.ai`. This is the most suitable endpoint for most customers,
as the request will be automatically routed by us to be processed in the region with lowest latency for your request.

For example, if you are based in US East Coast and send your request to `https://api.x.ai`, your request will be forwarded
to our `us-east-1` region and we will try to process it there first. If there is not enough computing resource in `us-east-1`,
we will send your request to other regions that are geographically closest to you and can handle the request.

## Using a regional endpoint

If you have specific data privacy requirements that would require the request to be processed within a specified region,
you can leverage our regional endpoint.

You can send your request to `https://<region-name>.api.x.ai`. For the same example, to send request from US East Coast to `us-east-1`,
you will now send the request to `https://us-east-1.api.x.ai`. If for some reason, we cannot handle your request in `us-east-1`, the request will fail.

## Example of using regional endpoints

If you want to use a regional endpoint, you need to specify the endpoint url when making request with SDK. In xAI SDK, this is specified through the `api_host` parameter.

For example, to send request to `us-east-1`:

```pythonWithoutSDK
import os

from xai_sdk import Client
from xai_sdk.chat import user

client = Client(
api_key=os.getenv("XAI_API_KEY"),
api_host="us-east-1.api.x.ai" # Without the https://
)

chat = client.chat.create(model="grok-4")
chat.append(user("What is the meaning of life?"))

completion = chat.sample()
```

```pythonOpenAISDK
from openai import OpenAI

client = OpenAI(
api_key=XAI_API_KEY,
base_url="https://us-east-1.api.x.ai/v1",
)

completion = client.chat.completions.create(
model="grok-4",
messages=[
{"role": "user", "content": "What is the meaning of life?"}
]
)
```

```javascriptOpenAISDK
import OpenAI from "openai";

const client = new OpenAI({
apiKey: XAI_API_KEY,
baseURL: "https://us-east-1.api.x.ai/v1",
});

const completion = await client.chat.completions.create({
model: "grok-4",
messages: [
{ role: "user", content: "What is the meaning of life?" }
]
});
```

```bash
curl https://us-east-1.api.x.ai/v1/chat/completions \\
-H "Content-Type: application/json" \\
-H "Authorization: Bearer $XAI_API_KEY" \\
-d '{
"messages": [
{
"role": "user",
"content": "What is the meaning of life, the universe, and everything?"
}
],
"model": "grok-4",
"stream": false
}'
```

## Model availability across regions

While we strive to make every model available across all regions, there could be occasions where some models are not
available in some regions.

By using the global `https://api.x.ai` endpoint, you would have access to all models available to your team, since we
route your request automatically. If you're using a regional endpoint, please refer to [xAI Console](https://console.x.ai)
for the available models to your team in each region, or [Models and Pricing](../models) for the publicly available models.


===/docs/key-information/usage-explorer===
#### Key Information

# Usage Explorer

Sometimes as a team admin, you might want to monitor the API consumption, either to track spending, or to detect anomalies. xAI Console provides an easy-to-use [Usage Explorer](https://console.x.ai/team/default/usage) for team admins to track API usage across API keys, models, etc.

## Basic usage

[Usage Explorer](https://console.x.ai/team/default/usage) page provides intuitive dropdown menus for you to customize how you want to view the consumptions.

For example, you can view your daily credit consumption with `Granularity: Daily`:

By default, the usage is calculated by cost in US Dollar. You can select Dimension -> Tokens or Dimension -> Billing items to change the dimension to token count or billing item count.

You can also see the usage with grouping. This way, you can easily compare the consumption across groups. In this case, we are trying to compare consumptions across test and production API keys, so we select `Group by: API Key`:

## Filters

The basic usage should suffice if you are only viewing general information. However, you can also use filters to conditionally display information.

The filters dropdown gives you the options to filter by a particular API key, a model, a request IP, a cluster, or the token type.


===/docs/key-information/using-management-api===
#### Key Information

# Using Management API

Some enterprise users may prefer to manage their account details programmatically rather than manually through the xAI Console. For this reason, we have developed a Management API to enable enterprise users to efficiently manage their team details.

You can read the endpoint specifications and descriptions at [Management API Reference](../management-api).

You need to get a management key, which is separate from your API key, to use the management API. The management key can be obtained at [xAI Console](https://console.x.ai) -> Settings -> Management Keys.

The base URL is at `https://management-api.x.ai`, which is also different from the inference API.

## Operations related to API Keys

You can create, list, update and delete API keys via the management API.

You can also manage the access control lists (ACLs) associated with the API keys.

The available ACL types are:

* `api-key:model`
* `api-key:endpoint`

To enable all models and endpoints available to your team, use:

* `api-key:model:*`
* `api-key:endpoint:*`

Or if you need to specify the particular endpoint available to the API:

* `api-key:endpoint:chat` for chat and vision models
* `api-key:endpoint:image` for image generation models

And to specify models the API key has access to:

* `api-key:model:<model name such as grok-4>`

### Create an API key

An example to create an API key with all models and endpoints enabled, limiting requests to 5 queries per second and 100 queries per minute, without token number restrictions.

```bash
curl https://management-api.x.ai/auth/teams/{teamId}/api-keys \\
    -X POST \\
    -H "Authorization: Bearer <Your Management API Key>" \\
    -d '{
            "name": "My API key",
            "acls": ["api-key:model:*", "api-key:endpoint:*"],
            "qps": 5,
            "qpm": 100,
            "tpm": null
        }'
```

Specify `tpm` to any integer string to limit the number of tokens produced/consumed per minute. When the token rate limit is triggered, new requests will be rejected and in-flight requests will continue processing.

The newly-created API key will be returned in the `"apiKey"` field of the response object. The API Key ID is returned as `"apiKeyId"` in the response body as well, which is useful for updating and deleting operations.

### List API keys

To retrieve a list of API keys from a team, you can run the following:

```bash
curl https://management-api.x.ai/auth/teams/{teamId}/api-keys?pageSize=10&paginationToken= \\
    -H "Authorization: Bearer <Your Management API Key>"
```

You can customize the query parameters such as `pageSize` and `paginationToken`.

### Update an API key

You can update an API key after it has been created. For example, to update the `qpm` of an API key:

```bash
curl https://management-api.x.ai/auth/teams/{teamId}/api-keys \\
    -X PUT \\
    -d '{
            "apiKey": "<The apiKey Object with updated qpm>",
            "fieldMask": "qpm",
        }'
```

Or to update the `name` of an API key:

```bash
curl https://management-api.x.ai/auth/teams/{teamId}/api-keys \\
    -X PUT \\
    -d '{
            "apiKey": "<The apiKey Object with updated name>",
            "fieldMask": "name",
        }'
```

### Delete an API key

You can also delete an API key with the following:

```bash
curl https://management-api.x.ai/auth/api-keys/{apiKeyId} \\
    -X DELETE \\
    -H "Authorization: Bearer <Your Management API Key>"
```

### Check propagation status of API key across clusters

There could be a slight delay between creating an API key, and the API key being available for use across all clusters.

You can check the propagation status of the API key via API.

```bash
curl https://management-api.x.ai/auth/api-keys/{apiKeyId}/propagation \\
    -H "Authorization: Bearer <Your Management API Key>"
```

### List all models available for the team

You can list all the available models for a team with our management API as well.

The model names in the output can be used with setting ACL string on an API key as `api-key:model:<model-name>`

```bash
curl https://management-api.x.ai/auth/teams/{teamId}/models \\
    -H "Authorization: Bearer <Your Management API Key>"
```

## Access Control List (ACL) management

We also offer endpoint to list possible ACLs for a team. You can then apply the endpoint ACL strings to your API keys.

To view possible endpoint ACLs for a team's API keys:

```bash
curl https://management-api.x.ai/auth/teams/{teamId}/endpoints \\
    -H "Authorization: Bearer <Your Management API Key>"
```


===/docs/resources/community-integrations===
#### Resources

# Community Integrations

Grok is also accessible via your favorite community integrations, enabling you to connect Grok to other parts of your system easily.

## Third-party SDK/frameworks

### LiteLLM

LiteLLM provides a simple SDK or proxy server for calling different LLM providers. If you're using LiteLLM, integrating xAI as your provider is straightforward—just swap out the model name and API key to xAI's Grok model in your configuration.

For latest information and more examples, visit [LiteLLM xAI Provider Documentation](https://docs.litellm.ai/docs/providers/xai).

As a quick start, you can use LiteLLM in the following fashion:

```pythonWithoutSDK
from litellm import completion
import os

os.environ['XAI_API_KEY'] = ""
response = completion(
model="xai/grok-4",
messages=[
{
"role": "user",
"content": "What's the weather like in Boston today in Fahrenheit?",
}
],
max_tokens=10,
response_format={ "type": "json_object" },
seed=123,
stop=["\n\n"],
temperature=0.2,
top_p=0.9,
tool_choice="auto",
tools=[],
user="user",
)
print(response)
```

### Vercel AI SDK

[Vercel's AI SDK](https://sdk.vercel.ai/) supports a [xAI Grok Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/xai) for integrating with xAI API.

By default it uses your xAI API key in `XAI_API_KEY` variable.

To generate text use the `generateText` function:

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { generateText } from 'ai';

const { text } = await generateText({
model: xai('grok-4'),
prompt: 'Write a vegetarian lasagna recipe for 4 people.',
});
```

You can also customize the setup like the following:

```javascriptAISDK
import { createXai } from '@ai-sdk/xai';

const xai = createXai({
apiKey: 'your-api-key',
});
```

You can also generate images with the `generateImage` function:

```javascriptAISDK
import { xai } from '@ai-sdk/xai';
import { experimental_generateImage as generateImage } from 'ai';

const { image } = await generateImage({
model: xai.image('grok-2-image'),
prompt: 'A cat in a tree',
});
```

## Coding assistants

### Continue

You can use Continue extension in VSCode or JetBrains with xAI's models.

To start using xAI models with Continue, you can add the following in Continue's config file `~/.continue/config.json`(MacOS and Linux)/`%USERPROFILE%\.continue\config.json`(Windows).

```json
 "models": [
   {
     "title": "grok-4",
     "provider": "xAI",
     "model": "grok-4",
     "apiKey": "[XAI_API_KEY]"
   }
 ]
```

Visit [Continue's Documentation](https://docs.continue.dev/chat/model-setup#grok-2-from-xai) for more details.


===/docs/resources/faq-api/accounts===
#### Resources / FAQ - xAI API

# Accounts

## How do I create an account for the API?

You can create an account at https://accounts.x.ai, or https://console.x.ai. To link your X account automatically to
your xAI account, choose to sign up with X account.

You can create multiple accounts of different sign-in methods with the same email.

When you sign-up with a sign-in method and with the same email, we will prompt you whether you
want to create a new account, or link to the existing account. We will not be able to merge the
content, subscriptions, etc. of different accounts.

## How do I update my xAI account email?

You can visit [xAI Accounts](https://accounts.x.ai). On the Account page, you can update your email.

## How do I add other sign-in methods?

Once you have signed-up for an account, you can add additional sign-in methods by going to [xAI Accounts](https://accounts.x.ai).

## I've forgotten my Multi-Factor Authentication (MFA) method, can you remove it?

You can generate your recovery codes at [xAI Accounts](https://accounts.x.ai) Security page.

We can't remove or reset your MFA method unless you have recovery codes due to security considerations. Please reach out to support@x.ai if you would like to delete the account instead.

## If I already have an account for Grok, can I use the same account for API access?

Yes, the account is shared between Grok and xAI API. You can manage the sign-in details at https://accounts.x.ai.

However, the billing is separate for Grok and xAI API. You can manage your billing for xAI API on [xAI Console](https://console.x.ai).
To manage billing for Grok, visit https://grok.com -> Settings -> Billing, or directly with Apple/Google if you made the
purchase via Apple App Store or Google Play.

## How do I manage my account?

You can visit [xAI Accounts](https://accounts.x.ai) to manage your account.

Please note the xAI account is different from the X account, and xAI cannot assist you with X account issues. Please
contact X via [X Help Center](https://help.x.com/) or Premium Support if you encounters any issues with your X account.

## How do I delete my xAI account?

We are sorry to see you go!

You can visit [xAI Accounts](https://accounts.x.ai/account) to delete your account. You can restore your account after log in again and confirming restoration within 30 days.

You can cancel the deletion within 30 days by logging in again to any xAI websites and follow the prompt to confirm restoring the account.

For privacy requests, please go to: https://privacy.x.ai.


===/docs/resources/faq-api/billing===
#### Resources / FAQ - xAI Console

# Billing

## I'm having payment issues with an Indian payment card

Unfortunately we cannot process Indian payment cards for our API service. We are working toward supporting it but you might want to consider using a third-party API in the meantime. As Grok Website and Apps' payments are handled differently, those are not affected.

## When will I be charged?

* Prepaid Credits: If you choose to use prepaid credits, you’ll be charged when you buy them. These credits will be assigned to the team you select during purchase.

* Monthly Invoiced Billing: If you set your [invoiced spending limit](billing#monthly-invoiced-billing-and-invoiced-billing-limit) above $0, any usage beyond your prepaid credits will be charged at the end of the month.

* API Usage: When you make API requests, the cost is calculated immediately. The amount is either deducted from your available prepaid credits or added to your monthly invoice if credits are exhausted.

If you change your [invoiced spending limit](billing#monthly-invoiced-billing-and-invoiced-billing-limit) to be greater than $0, you will be charged at the end of the month for any extra consumption after your prepaid credit on the team has run out.

Your API consumption will be calculated when making the requests, and the corresponding amount will be deducted from your remaining credits or added to your monthly invoice.

Check out [Billing](billing) for more information.

## Can you retroactively generate an invoice with new billing information?

We are unable to retroactively generate an invoice. Please ensure your billing information is correct on [xAI Console](https://console.x.ai) Billing -> Payment.

## Can prepaid API credits be refunded?

Unfortunately, we are not able to offer refunds on any prepaid credit purchase unless in regions required by law. For details, please visit https://x.ai/legal/terms-of-service-enterprise.

### My prompt token consumption from the API is different from the token count I get from xAI Console Tokenizer or tokenize text endpoint

The inference endpoints add pre-defined tokens to help us process the request. Therefore, these tokens would be added to the total prompt token consumption. For more information, see:
[Estimating consumption with tokenizer on xAI Console or Estimating consumption with tokenizer on xAI Console or through API](consumption-and-rate-limits#estimating-consumption-with-tokenizer-on-xai-console-or-through-api).


===/docs/resources/faq-api===
#### Resources

# FAQ - xAI Console

Frequently asked questions on using the [xAI Console](https://console.x.ai), including creating teams, managing roles, and configuring settings.

You can find details on the following topics:


===/docs/resources/faq-api/security===
#### Resources / FAQ - xAI API

# Security

## Does xAI train on customers' API requests?

xAI never trains on your API inputs or outputs without your explicit permission.

API requests and responses are temporarily stored on our servers for 30 days in case they need to be audited for potential abuse or misuse. This data is automatically deleted after 30 days.

## Is the xAI API HIPAA compliant?

To inquire about a Business Associate Agreement (BAA), please complete our [BAA Questionnaire](https://forms.gle/YAEdX3XUp6MvdEXW9). A member of our team will review your responses and reach out with next steps.

## Is xAI GDPR and SOC II compliant?

We are SOC 2 Type 2 compliant. Customers with a signed NDA can refer to our [Trust Center](https://trust.x.ai/) for up-to-date information on our certifications and data governance.

## Do you have Audit Logs?

Team admins are able to view an audit log of user interactions. This lists all of the user interactions with our API server. You can view it at [xAI Console -> Audit Log](https://console.x.ai/team/default/audit).

The admin can also search by Event ID, Description or User to filter the results shown. For example, this is to filter by description matching `ListApiKeys`:

You can also view the audit log across a range of dates with the time filter:

## How can I securely manage my API keys?

Treat your xAI API keys as sensitive information, like passwords or credit card details. Do not share keys between teammates to avoid unauthorized access. Store keys securely using environment variables or secret management tools. Avoid committing keys to public repositories or source code.

Rotate keys regularly for added security. If you suspect a compromise, log into the xAI console first. Ensure you are viewing the correct team, as API keys are tied to specific teams. Navigate to the "API Keys" section via the sidebar. In the API Keys table, click the vertical ellipsis (three dots) next to the key. Select "Disable key" to deactivate it temporarily or "Delete key" to remove it permanently. Then, click the "Create API Key" button to generate a new one and update your applications.

xAI partners with GitHub's Secret Scanning program to detect leaked keys. If a leak is found, we disable the key and notify you via email. Monitor your account for unusual activity to stay protected.


===/docs/resources/faq-api/team-management===
#### Resources / FAQ - xAI Console

# Team Management

## What are teams?

Teams are the level at which xAI tracks API usage, processes billing, and issues invoices.

* If you’re the team creator and don’t need a new team, you can rename your Personal Team and add members instead of creating a new one.
* Each team has **roles**:
  * **Admin**: Can modify team name, billing details, and manage members.
  * **Member**: Cannot make these changes.
  * The team creator is automatically an Admin.

## Which team am I on?

When you sign up for xAI, you’re automatically assigned to a **Personal Team**, which you can view the top bar of [xAI Console](https://console.x.ai).

## How can I manage teams and team members?

### Create a Team

1. Click the dropdown menu in the xAI Console.
2. Select **+ Create Team**.
3. Follow the on-screen instructions. You can edit these details later.

### Rename or Describe a Team

Admins can update the team name and description on the [Settings page](https://console.x.ai/team/default/settings).

### Manage Team Members

Admins can add or remove members by email on the [Users page](https://console.x.ai/team/default/users).

* Assign members as **Admin** or **Member**.
* If a user is removed, their API keys remain with the team.

### Delete a Team

Deleting a team removes its prepaid credits.

To permanently delete a team:

1. Go to the [Settings page](https://console.x.ai/team/default/settings).
2. Follow the instructions under **Delete Team**.

## How to automatically add users to team with my organization's email domain?

Admins can enable automatic team joining for users with a shared email domain:

1. Go to the [Settings page](https://console.x.ai/team/default/settings).
2. Add the domain under **Verified Domains**.
3. Add a `domain-verification` key to your domain’s DNS TXT record to verify ownership.

Users signing up with a verified domain email will automatically join the team.


===/docs/resources/faq-general===
#### Getting Started

# Frequently Asked Questions - General

Frequently asked questions by our customers.

For product-specific questions, visit  or .

### Does the xAI API provide access to live data?

Yes! With the [LiveSearch feature](/docs/guides/live-search), Grok can search through realtime data from X posts, the internet, news, and RSS feeds.

### How do I contact Sales?

For customers with bespoke needs or to request custom pricing, please fill out our [Grok for Business form](https://x.ai/grok/business). A member of our team will reach out with next steps. You can also email us at [sales@x.ai](mailto:sales@x.ai).

### Where are your Terms of Service and Privacy Policy?

Please refer to our [Legal Resources](https://x.ai/legal) for our Enterprise Terms of Service and Data Processing Addendum.

### Does xAI sell crypto tokens?

xAI is not affiliated with any cryptocurrency. We are aware of several scam websites that unlawfully use our name and logo.

### I have issues using X, can I reach out to xAI for help?

While xAI provides the Grok in X service on X.com and X apps, it does not have operational oversight of X's service. You can contact X via their [Help Center](https://help.x.com/) or message [@premium on X](https://x.com/premium).

### How do I add/remove other sign-in methods or link my X subscription?

You can add/remove your sign-in methods at https://accounts.x.ai. Your account must have at least one sign-in method.

Linking or signing up with X account will automatically link your X account subscription status with xAI, which can be used on https://grok.com.

### I signed-up to Grok / xAI API with my X account, why is xAI still asking for my email?

When you sign up with X, you will be prompted with the following:

As X does not provide the email address, you can have different emails on your X account and xAI account.

### I received an email of someone logging into my xAI account

xAI will send an email to you when someone logs into your xAI account. The login location is an approximation based on your IP address, which is dependent on your network setup and ISP and might not reflect exactly where the login happened.

If you think the login is not you, please [reset your password](https://accounts.x.ai/request-reset-password) and [clear your login sessions](https://accounts.x.ai/sessions). We also recommend all users to [add a multi-factor authentication method](https://accounts.x.ai/security).


===/docs/resources/faq-grok===
#### Resources

# FAQ - Grok Website / Apps

While the documentation is mainly meant for our API users, you can find some commonly asked questions here for our consumer-facing website/apps.

## How can I link my X account sign-in/subscription to my xAI account?

On [Grok Website](https://grok.com), go to Settings -> Account. Click on Connect your X Account button. This will take you to X's SSO page to add X account as a sign-in method for xAI.

xAI will be able to retrieve your X subscription status and grant relevant benefits after linking.

You can manage your sign-in methods at https://accounts.x.ai.

## How can I delete the account?

Your xAI account can be deleted by following the steps here: [How do I delete my account?](../faq-grok#how-can-i-delete-the-account) If you are using the same account to access our API, your API access will be removed as well.

## How do I unsubscribe?

If you have subscribed to SuperGrok, you can go to https://grok.com -> Settings -> Billing to manage your subscription (purchased from Grok Website), [Request a refund for app](https://support.apple.com/118223) (purchased from Apple App Store), or [Cancel, pause or change a subscription on Google Play](https://support.google.com/googleplay/answer/7018481) (purchased from Google Play).

If you have subscribed to X Premium, X (not xAI) would be responsible for processing refund where required by law. You can [submit a refund request from X](https://help.x.com/forms/x-refund-request). See more details regarding X Premium subscriptions on [X Help Center](https://help.x.com/using-x/x-premium).


===/docs/customer-support-agents===
# Customer Support Agents API Guide

## Overview

The Customer Support Agents API provides a specialized chat service for handling customer inquiries, leveraging RAG for knowledge bases and tools for actions like replies or escalations. This is an optimized replacement for `/v1/chat/completions`, with support-specific parameters.

The API is currently stateless, so you need to provide the full conversation history on each request.
Provisioning (API keys, agents) is handled in the console UI, while this doc focuses on integration.

For API keys, visit [https://console.x.ai/team/default/support-agents/api-keys](https://console.x.ai/team/default/support-agents/api-keys).

To view and create Support Agents, visit [https://console.x.ai/team/default/support-agents/agents](https://console.x.ai/team/default/support-agents/agents).

Key endpoints:

* `/v1/support-agent/chat`: Non-streaming responses.
* `/v1/support-agent/chat-stream`: Streaming responses.

## Quick Setup

1. **Create API Key**: Visit the [console](https://console.x.ai/team/default/support-agents/api-keys) to create a new Support Agent API Key and save it securely.

2. **Find Agent ID**: Go to the [agents page](https://console.x.ai/team/default/support-agents/agents) to view or create your Support Agent, and note the `support_agent_id`.

3. **Send Request**: Use the following curl example to make a real API call. You can generate any `conversation_id` value on your end and reference it later if conversation grows longer.

```bash
curl --location 'https://api.x.ai/v1/support-agent/chat' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer API_KEY_HERE' \
--data '{
  "support_agent_id": "support_agent_d5cdd745-0dc9-479c-9ff1-f14bee21e7ce",  # Found in the agents console
  "deployment_id": "LATEST",
  "conversation_id": "conv-uuid-67890",  # Generate a unique identifier for each conversation
  "messages": [ 
    {
      "content": [
        {
          "text": "Where is my order?"
        }
      ],
      "role": "ROLE_USER"
    }
  ],
  "environment": "ENVIRONMENT_PROD"  # Or "ENVIRONMENT_STAGING" for testing
}'
```

This will return a response from the agent. For subsequent messages, append to the `messages` array and reuse `conversation_id`.

## Using Collections as a Knowledge Base

To enhance your Support Agent using knowledge-based Retrieval-Augmented Generation (RAG)—which allows the agent to retrieve and use relevant information from your documents—we recommend our Collections API.

You can create a Collection directly [in the console](https://console.x.ai/team/default/collections) and upload your documents, such as FAQs and policies.

Once your Collection is created, you'll need to attach it to your agent in the console.

* Full docs: [Collections API Reference](https://docs.x.ai/docs/guides/function-calling).
* Limit: One Collection per agent currently.

Once configured, the agent uses RAG to retrieve and include only relevant document chunks from the Collection in its context for each response.
This adds relevant chunks to the agent's context before responding.

&#x20;Updates to documents in the Collection (e.g., adding new pages) propagate after a short delay **without reconfiguring the agent**.

## Tools Integration

To give your Support Agent access to external actions, you can define Tools (also known as function calling) when deploying it in the console. Function calling lets the agent decide when to invoke specific functions based on the conversation.

Then, when handling queries, your Support Agent will suggest function names it thinks it should call. Your implementation can then loop to execute these functions locally and feed the results back to the agent for further processing.

For official documentation on Function Calling, visit our [Function Calling Reference](https://docs.x.ai/docs/guides/function-calling).

To ensure conversations are handled effectively and reach proper end states (e.g., resolving the issue or escalating), define tools like CLOSE (to end the conversation) and ESCALATE (to hand off to human support). These tools let the agent suggest actions that your implementation can then execute in your messaging system, such as closing a ticket in Intercom or Slack.

### Defining Tools

* Define `tools` when configuring your agent:
  ```json
  "tools": [
    {
      "function": {
        "name": "USER_INFO_LOOKUP",
        "description": "Lookup user information by calling your internal service",
        "parameters": {
          "type": "object",
          "properties": {
            "user_id": {"type": "string", "description": "The unique ID of the user to look up in your system"}
          },
          "required": ["user_id"]
        }
      }
    },
    {
      "function": {
        "name": "ESCALATE",
        "description": "Escalate the conversation to human support by creating a ticket or notifying a team.",
        "parameters": {
          "type": "object",
          "properties": {
            "reason": {"type": "string", "description": "Brief reason for escalation"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Escalation priority"}
          },
          "required": ["reason"]
        }
      }
    },
    {
      "function": {
        "name": "CLOSE",
        "description": "Close the conversation, marking it as resolved.",
        "parameters": {
          "type": "object",
          "properties": {
            "resolution_summary": {"type": "string", "description": "Summary of how the issue was resolved"}
          },
          "required": ["resolution_summary"]
        }
      }
    }
  ]
  ```

### Handling Tool Calls

After the agent suggests a tool (via function calling), implement a loop to process it:

1. **Check for tool calls**: Look for messages with `role: "ROLE_TOOL"` (with `tool_calls` array) or legacy `role: "ROLE_FUNCTION"` (with single `function`). Extract `name` and `arguments` (e.g., `{"user_id": "user123"}` for `USER_INFO_LOOKUP`).

2. **Execute the tool**: Call your internal services with the arguments. For `USER_INFO_LOOKUP({"user_id": "user123"})`, fetch from your database (e.g., return `{"name": "John Doe", "account_status": "active"}`).

3. **Append results and re-query**: Add a `role: "ROLE_TOOL"` message with the JSON result in `content` (e.g., `[{"text": "{\"name\": \"John Doe\", \"account_status\": \"active\"}"}]`) and `tool_call_id`. Send the updated history back to the API.

**Example Tool Result Message**:

```json
{
  "role": "ROLE_TOOL",
  "content": [{"text": "{\"name\": \"John Doe\", \"account_status\": \"active\"}"}],
  "tool_call_id": "id"
}
```

Repeat the loop until an end-state tool (like CLOSE) is called or no more tools are suggested—then reply to the user with the latest agent response.

### End-State Tools

Example CLOSE and ESCALATE logic to resolve conversations:

* **CLOSE**: Extract the `resolution_summary` from the agent's tool call arguments, use it to update your system (e.g., mark ticket resolved), and end the interaction.

* **ESCALATE**: Extract `reason` and `priority` from the agent's tool call arguments, then flag in your CRM (e.g., Intercom: add 'escalate' tag, create admin note with the reason and priority, assign to human agent).

&#x20;Always loop until resolution—re-query after each tool to let the agent build context and conclude naturally.

## Guardrails (Optional)

Guardrails filter inputs before the agent processes them. You can define them in the console when deploying an agent.

* **What is a guardrail?**: A pre-check to block inappropriate queries (e.g., threats, PII). If triggered, returns custom text (e.g., "Escalating to human support") instead of processing.
* Example:
  ```json
  "guardrails": [
    {
      "guardrail_id": "guard-123",
      "type": "GUARDRAIL_TYPE_INPUT",
      "instructions": "Block if query contains threats or inappropriate info.",
      "triggered_text": "This query is blocked. Contact support.",
      "model_name": "grok-3",
      "enabled": true
    }
  ]
  ```

## Chat Endpoint

POST to `/v1/support-agent/chat` for responses.

### Request

Required fields:

* `support_agent_id`: ID of your Support Agent. You can find your ID here [https://console.x.ai/team/default/support-agents/agents](https://console.x.ai/team/default/support-agents/agents).
* `conversation_id`: Choose a unique ID for your conversation. The Chat API is currently stateless.
* `messages`: Array of history (full each time). Roles: `ROLE_USER`, `ROLE_ASSISTANT`, `ROLE_SYSTEM`, `ROLE_FUNCTION`, `ROLE_TOOL`. Content: `[{"text": "query"}]` or image URLs.
* `environment`: `"ENVIRONMENT_PROD"` or `"ENVIRONMENT_STAGING"`.

Schema:

```json
{
  "support_agent_id": "support_agent_d5cdd745-0dc9-479c-9ff1-f14bee21e7ce",
  "conversation_id": "conv-uuid-67890",
  "deployment_id": "LATEST",
  "messages": [
    {
      "content": [{"text": "Where is my order?"}],
      "role": "ROLE_USER"
    }
  ],
  "environment": "ENVIRONMENT_PROD"
}
```

### Response

```json
{
  "id": "string",
  "message": { "role": "ROLE_ASSISTANT", "content": [{"text": "response"}] },
  "system_fingerprint": "string",
  "usage": { "prompt_tokens": int, "completion_tokens": int, "total_tokens": int },
  "created": "timestamp",
  "rag_results": [ { "text": "chunk", "score": float, "collection_id": "string" } ]
}
```

### Curl Example

```bash
curl --location 'https://api.x.ai/v1/support-agent/chat' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer YOUR_API_KEY' \
--data '{
    "support_agent_id": "support_agent_d5cdd745-0dc9-479c-9ff1-f14bee21e7ce",
    "conversation_id": "conv-uuid-67890",
    "deployment_id": "LATEST",
    "messages": [
        {
            "content": [{"text": "Where is my order?"}],
            "role": "ROLE_USER"
        }
    ],
    "environment": "ENVIRONMENT_PROD"
}'
```

## End-to-End Python Example

This example demonstrates a batch script integrating the xAI Support Agents API with Intercom. It polls open conversations from Intercom, processes each (via API calls and tool handling), and handles resolutions or escalations. Suitable for testing.

### Prerequisites

* **Install**: `pip install aiohttp intercom-python`.
* **Support Agent credentials**: API key from [console](https://console.x.ai/team/default/support-agents/api-keys), agent ID from [agents](https://console.x.ai/team/default/support-agents/agents).
* **Intercom setup**: App ID and API key from [Intercom dashboard](https://developers.intercom.com/installing-intercom/docs/intercom-api-keys).

### Configuration

Define your credentials and limits.

```pythonWithoutSDK
import asyncio
import json
import logging
from typing import List, Dict, Any

import aiohttp
from intercom.client import Client as IntercomClient  # pip install intercom-python

# Configuration (EDIT THESE with your values)
XAI_API_URL = "https://api.x.ai/v1/support-agent/chat"
XAI_API_KEY = "YOUR_XAI_API_KEY"
SUPPORT_AGENT_ID = "YOUR_SUPPORT_AGENT_ID"
DEPLOYMENT_ID = "LATEST"
ENVIRONMENT = "ENVIRONMENT_PROD"

INTERCOM_APP_ID = "YOUR_INTERCOM_APP_ID"
INTERCOM_API_KEY = "YOUR_INTERCOM_API_KEY"
MAX_TOOL_LOOPS = 5
MAX_CONVERSATIONS = 5  # Limit for batch processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

### Helper Functions

Implement message conversion, Intercom polling, and tool execution.

**Convert messages to Support Agent API format**:

```pythonWithoutSDK
def convert_to_agent_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to correct format."""
    output = []
    for msg in messages:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        if isinstance(content, str):
            content = [{"text": content}]
        output.append({"role": f"ROLE_{role}", "content": content})
    return output
```

**Poll open conversations from Intercom**:

```pythonWithoutSDK
async def fetch_intercom_conversations(intercom: IntercomClient) -> List[Dict]:
    """Poll open conversations from Intercom."""
    try:
        conversations = intercom.conversations.list(open=True, per_page=10)
        return [conv.to_dict() for conv in conversations]
    except Exception as e:
        logger.error(f"Intercom fetch error: {e}")
        return []
```

**Define a tool handler** (customize based on your needs, e.g., USER\_INFO\_LOOKUP, CLOSE, ESCALATE):

```pythonWithoutSDK
async def execute_tool(function_name: str, arguments: Dict, intercom: IntercomClient, conversation_id: str, session: aiohttp.ClientSession) -> str:
    """Execute tool actions (customize as needed)."""
    if function_name == "USER_INFO_LOOKUP":
        # Placeholder: Replace with real DB query
        user_id = arguments.get("user_id", "")
        return json.dumps({"name": "John Doe", "status": "active", "user_id": user_id})
    elif function_name == "CLOSE":
        summary = arguments.get("resolution_summary", "")
        try:
            intercom.conversations.update(id=conversation_id, open=False)
            return f"Closed: {summary}"
        except Exception as e:
            return f"Failed to close: {e}"
    elif function_name == "ESCALATE":
        reason = arguments.get("reason", "[No reason provided]")
        priority = arguments.get("priority", "general")
        try:
            # Flag for human review
            intercom.conversations.update(
                id=conversation_id,
                tags={"add": [{"name": "escalate"}]}
            )
            # Add admin note
            intercom.admin_notes.create(
                conversation_id=conversation_id,
                body=f"AI Escalation: {reason} (priority: {priority}). Human review needed."
            )
            # Assign to human admin (EDIT: Set YOUR_DEFAULT_HUMAN_ADMIN_ID)
            intercom.conversations.update(
                id=conversation_id,
                admin_assignee_id="YOUR_DEFAULT_HUMAN_ADMIN_ID"
            )
            logger.info(f"ESCALATE {conversation_id}: {reason}")
            return f"Flagged for human review: {reason} (priority: {priority})"
        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            return f"Escalation failed: {e}"
    return f"Unknown tool: {function_name}"
```

### Support Agent Message Loop

Process a single conversation: Call the Support Agent API, check for tool calls, execute tools, append results, and loop until resolution or max loops. If no tools, reply with the agent response.

```pythonWithoutSDK
async def process_message(session: aiohttp.ClientSession, intercom: IntercomClient, messages: List[Dict[str, Any]], conversation_id: str) -> Dict[str, Any]:
    """Process conversation: Call xAI API and handle tools."""
    xai_messages = convert_to_agent_format(messages)

    loop_count = 0
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {XAI_API_KEY}"}
    timeout = aiohttp.ClientTimeout(total=300)
    last_action = None

    while loop_count < MAX_TOOL_LOOPS:
        payload = {
            "support_agent_id": SUPPORT_AGENT_ID,
            "deployment_id": DEPLOYMENT_ID,
            "conversation_id": conversation_id,
            "messages": xai_messages,
            "environment": ENVIRONMENT
        }

        try:
            async with session.post(XAI_API_URL, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    error = await resp.json()
                    return {"error": f"API failed: {error}"}
                response = await resp.json()
        except Exception as e:
            return {"error": str(e)}

        # Parse response
        agent_msg = response.get("message", {})
        content = agent_msg.get("content")
        response_text = content if isinstance(content, str) else (content[0].get("text", "") if content else "")
        tool_calls = agent_msg.get("tool_calls", [])

        should_close = False
        if not tool_calls and response_text:
            should_close = True
        elif last_action == "CLOSE":
            should_close = True

        if not tool_calls:
            logger.info(f"{conversation_id} complete: {response_text}")
            return {"final_response": response_text, "conversation_id": conversation_id, "full_response": response, "should_close": should_close}

        # Handle tools sequentially
        for tool_call in tool_calls:
            func_name = tool_call.get("function", {}).get("name", "")
            args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            tool_result = await execute_tool(func_name, args, intercom, conversation_id, session)

            last_action = func_name

            # Append tool result
            xai_messages.append({
                "role": "ROLE_TOOL",
                "content": [{"text": tool_result}],
                "tool_call_id": tool_call.get("id")
            })

            # Break for end-state tools
            if func_name in ["CLOSE", "ESCALATE"]:
                break

        loop_count += 1

    should_close = (last_action == "CLOSE") or (not last_action and response_text)
    if loop_count >= MAX_TOOL_LOOPS:
        logger.warning(f"{conversation_id} max loops exceeded")
    return {"error": "Max tool loops" if loop_count >= MAX_TOOL_LOOPS else None, "conversation_id": conversation_id, "should_close": should_close}
```

### Main Processing Example

Try fetching conversations from Intercom, process each, and update Intercom (close and tag tickets).

```pythonWithoutSDK
async def main():
    """Batch process recent conversations."""
    intercom = IntercomClient(app_id=INTERCOM_APP_ID, api_key=INTERCOM_API_KEY)

    logger.info("Starting batch Support Agent processing...")
    async with aiohttp.ClientSession() as session:
        convos = await fetch_intercom_conversations(intercom)
        processed = 0
        for convo in convos[:MAX_CONVERSATIONS]:
            convo_id = convo.get("id")
            parts = convo.get('parts', [])
            messages = []
            for part in parts[-10:]:
                author_type = part.get('author', {}).get('type', 'unknown')
                body = part.get('body', '') or part.get('plain', {}).get('body', {}).get('text', '')
                if body:
                    role = 'user' if author_type == 'user' else 'assistant'
                    messages.append({"role": role, "content": body})

            if not messages:
                continue

            result = await process_message(session, intercom, messages, str(convo_id))
            logger.info(f"Processed {convo_id}: {result.get('final_response', result.get('error'))}")

            # Update Intercom based on outcome
            try:
                if result.get("should_close", False):
                    intercom.conversations.update(id=convo_id, open=False)
                    logger.info(f"Closed {convo_id} after resolution")
                else:
                    intercom.conversations.update(
                        id=convo_id,
                        tags={"add": [{"name": "ai_processed"}]}
                    )
                    logger.info(f"Flagged {convo_id} as AI-processed (kept open)")
            except Exception as e:
                logger.error(f"Failed to update {convo_id}: {e}")

            processed += 1

        logger.info(f"Batch complete: Processed {processed} conversations")

if __name__ == "__main__":
    asyncio.run(main())
```

**Customization Notes**:

* **Tools**: Expand `execute_tool` (e.g., real user lookup, custom escalations).
* **Batch Size**: Adjust `MAX_CONVERSATIONS` for more/fewer items per run.

## Chat Streaming

Same request as non-streaming, but POST to `/v1/support-agent/chat-stream`. Response streams deltas (like `/v1/chat/completions` streaming).

&#x20;Streaming is ideal for real-time chat UIs.

