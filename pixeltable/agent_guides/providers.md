# Providers

Calling model providers in computed columns: keys, request options, replies, embeddings, throttling and retries.

## Modules and keys

The provider modules under `pixeltable.functions` are `anthropic`, `bedrock`, `bfl`, `deepseek`, `fabric`, `fal`,
`fireworks`, `gemini`, `groq`, `huggingface`, `jina`, `llama_cpp`, `mistralai`, `nebius`, `ollama`, `openai`,
`openllm`, `openrouter`, `parallel`, `replicate`, `runwayml`, `together`, `twelvelabs`, `vllm`, `vlmrun`,
`vlmrun_gateway`, `voyageai`, `whisper`, `whisperx` and `yolox`. A module's functions need the provider's SDK
package; importing the module does not.

- Pixeltable reads a provider's key from an environment variable or from `$PIXELTABLE_HOME/config.toml`:
  `OPENAI_API_KEY` or `api_key` in the `[openai]` section, `ANTHROPIC_API_KEY` or `api_key` in the `[anthropic]`
  section, and similarly for the other providers.
- The environment variable takes precedence over `$PIXELTABLE_HOME/config.toml`.
- The variable and section names differ from the module name in four cases: `voyageai` reads `VOYAGE_API_KEY`,
  `mistralai` reads `MISTRAL_API_KEY`, `replicate` reads `REPLICATE_API_TOKEN` and `runwayml` reads
  `RUNWAYML_API_SECRET`.
- Without `GEMINI_API_KEY`, `gemini` leaves authentication to the Google GenAI client: `GOOGLE_API_KEY`, or Vertex AI
  with `GOOGLE_GENAI_USE_VERTEXAI=true` and `GOOGLE_CLOUD_PROJECT`.
- With `GEMINI_API_KEY`, or `api_key` in the `[gemini]` section, `gemini` calls the Gemini Developer API, even when
  `GOOGLE_GENAI_USE_VERTEXAI=true`.
- Without `BEDROCK_API_KEY`, `bedrock` uses boto3's default AWS credentials.
- `BEDROCK_REGION_NAME` sets the region for `bedrock`.
- `ollama` connects to `OLLAMA_HOST` if it is set, and to the `ollama` package's default host otherwise.
- Without `PARALLEL_API_KEY`, `parallel` calls Parallel's free MCP endpoint, which needs the `mcp` package.
- `OPENAI_BASE_URL`, or `base_url` in the `[openai]` section, points the `openai` functions at another
  OpenAI-compatible server, such as a LiteLLM proxy.
- `DEEPSEEK_BASE_URL`, or `base_url` in the `[deepseek]` section, does the same for `deepseek`.
- `openllm` calls the local OpenLLM daemon at `http://127.0.0.1:8787/v1` and needs no key; `OPENLLM_BASE_URL`, or
  `base_url` in the `[openllm]` section, points it at another server.
- A missing key fails the call, not the column definition: `` `openai` client not initialized: parameter `api_key`
  is not configured``.
- The `pxt` daemon reads the environment when it starts; `pxt daemon restart` applies a changed variable.
- `pxt config --section openai` shows the source of each value.

## Calling a model

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()

embed = pxtf.openai.embeddings.using(model='text-embedding-3-small')


class Notes(TableModel, name='notes'):
    id = pxt.Column(type=pxt.Int, primary_key=True)
    body: pxt.String
    photo: pxt.Image
    reply = pxtf.openai.chat_completions(
        messages=[{'role': 'user', 'content': body}], model='gpt-4o-mini', model_kwargs={'temperature': 0.2}
    )
    summary = reply.choices[0].message.content
    caption = pxtf.openai.chat_completions(
        messages=[
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Describe the photo.'}, {'type': 'image_url', 'image_url': photo}],
            }
        ],
        model='gpt-4o-mini',
    )
    critique = pxtf.anthropic.messages(
        messages=[{'role': 'user', 'content': body}], model='claude-sonnet-5-5', max_tokens=1024
    )
    critique_text = critique.content.filter(lambda b: b.type == 'text')[0].text
    __indexes__ = [pxt.EmbeddingIndex(body, embedding=embed, name='body_idx')]
```

- Most functions take the model as `model=`. The exceptions are `model_id=` (`bedrock`, `huggingface`, `yolox`),
  `model_name=` (`twelvelabs`), `model_path=` or `repo_id=` with `repo_filename=` (`llama_cpp`), `app=` (`fal`) and
  `ref=` (`replicate`).
- Request options go in one dict: `model_kwargs=` for `openai`, `anthropic` and the OpenAI-compatible providers,
  `config=` for `gemini`, `inference_config=` for `bedrock.converse()`, `options=` for `ollama`, and
  `sampling_params=` for `vllm`.
- Passing a keyword argument outside the function's signature fails when the column is defined: `got an unexpected
  keyword argument 'temperature'`.
- `anthropic.messages()` requires `max_tokens=`: `missing a required argument: 'max_tokens'`.
- An OpenAI message accepts an image column as `{'type': 'image_url', 'image_url': photo}` or as
  `'image_url': {'url': photo}`.
- Pixeltable sends an image column to OpenAI as a PNG data URL.
- OpenAI itself fetches a URL string such as `photo.fileurl`, so the URL has to be publicly reachable.
- `openai.responses()` takes an image as `{'type': 'input_image', 'image_url': photo}`.
- `openai.vision()` is deprecated and warns with `PixeltableDeprecationWarning`; `chat_completions()` and
  `responses()` take images as above.

## Replies

- `openai.chat_completions()`, `openai.responses()` and `anthropic.messages()` return typed Json with the fields of
  the SDK's `ChatCompletion`, `Response` and `Message`.
- A path into a typed reply has the field's type: `.choices[0].message.content` is `String | None`, so string
  functions and `+` apply to it directly.
- A key missing from the reply's type, such as a misspelled one, gives untyped `Json | None` and a null value;
  it does not raise.
- An existing column of untyped Json replies stays untyped Json.
- The other chat functions, except those of `vllm`, return the provider's reply as untyped Json, and every path into
  it is untyped Json.
- String functions and `+` need a `String`: `.astype(pxt.String | None)` converts the text to `String | None`.
- `astype(pxt.String)` raises on a null.

These paths select the text of a reply:

| Function | Path to the text |
|---|---|
| `openai.chat_completions()` | `.choices[0].message.content`, a `String \| None` |
| `openai.responses()` | `.output_text`, a `String \| None` |
| `anthropic.messages()` | `.content.filter(lambda b: b.type == 'text')[0].text`, a `String \| None` |
| `chat_completions()` of `deepseek`, `fabric`, `fireworks`, `groq`, `mistralai`, `nebius`, `openllm`, `openrouter`, `together`, `vlmrun` and `vlmrun_gateway`; `llama_cpp.create_chat_completion()` | `.choices[0].message.content` |
| `gemini.generate_content()` | `.candidates[0].content.parts[0].text` |
| `bedrock.converse()` | `.output.message.content[0].text` |
| `ollama.chat()`, `ollama.generate()` | `.message.content`, `.response` |
| `vllm.chat_completions()`, `vllm.generate()` | `.outputs[0].text` (typed `String`) |

## Embeddings

- An embedding function takes its model arguments through `.using()`, as `embed` above does.
- `.using()` rejects an unknown argument with `Unknown parameter`.
- An embedding index needs a vector of fixed length.
- `openai.embeddings()` knows the embedding's length for `text-embedding-3-small`, `text-embedding-3-large` and
  `text-embedding-ada-002`.
- For another model, `openai.embeddings()` needs `model_kwargs={'dimensions': N}`; without it, the index fails with
  `it must return a 1-dimensional array of a specific length`.
- In `.using()` or in a call, binding `model_id` of `huggingface.sentence_transformer()` or `huggingface.clip()`
  loads the model to find the embedding's length, and downloads the model if needed.

## Throttling and retries

- `openai.chat_completions()`, `openai.responses()`, `openai.embeddings()` and `anthropic.messages()` throttle
  requests by the rate-limit headers of the provider's replies, with no configuration.
- These four functions retry a failed request up to 10 times when the error is retryable, such as a rate limit or a
  timeout.
- Most other API functions send at most `rate_limit` requests per minute, 600 by default.
- The provider's environment variable or config section sets `rate_limit`: `GROQ_RATE_LIMIT` or `rate_limit` in
  `[groq]`, for example.
- For `bedrock` and the other `openai` functions, the limit is per model, with the model id as the key in
  `[bedrock.rate_limits]` or `[openai.rate_limits]`.
- For `together`, the limit is per kind of request, with the key `chat`, `embeddings` or `images` in
  `[together.rate_limits]`.
- The functions with a `rate_limit` retry 408, 429, 503 and 504 errors and throttling errors up to 3 times, after
  the delay in a `Retry-After` header if there is one.
- The local functions, such as those of `huggingface`, `whisper`, `ollama` and `llama_cpp`, are neither throttled
  nor retried.
- A request that fails after its retries fails the insert; with `on_error='ignore'`, the row is stored and the
  cell records the error.

```bash
pxt config --section openai
pxt daemon restart
pxt errors my_app/notes --col reply
pxt recompute my_app/notes reply --errors-only -f
```
