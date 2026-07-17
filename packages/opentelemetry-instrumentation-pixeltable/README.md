# OpenTelemetry instrumentation for Pixeltable

Translates [Pixeltable](https://pixeltable.com/)'s instrumentation hooks into OpenTelemetry spans and
metrics. Instrumentation is opt-in: Pixeltable itself never imports OpenTelemetry or activates telemetry on
its own.

```bash
pip install opentelemetry-instrumentation-pixeltable
# or, to also pull in the OpenTelemetry SDK and OTLP exporters that init() needs:
pip install 'pixeltable[otel]'
```

## Attach to an existing OpenTelemetry SDK

When your application already owns an OpenTelemetry SDK, attach Pixeltable to it:

```python
from opentelemetry.instrumentation.pixeltable import PixeltableInstrumentor

PixeltableInstrumentor().instrument()  # uses the global providers
# or: PixeltableInstrumentor().instrument(tracer_provider=my_tp, meter_provider=my_mp)
```

The standard `opentelemetry-instrument` CLI discovers and activates it automatically via its entry point.

## Let Pixeltable configure the SDK

To build providers and an OTLP exporter from Pixeltable's `[otel]` config (or the standard
`OTEL_EXPORTER_OTLP_*` environment variables), call `init()` once at startup (requires the
`pixeltable[otel]` extra):

```python
import opentelemetry.instrumentation.pixeltable as pxt_otel

pxt_otel.init(endpoint='http://localhost:4318')
```

Configuration is read from the `[otel]` section of Pixeltable's config (`~/.pixeltable/config.toml`),
overridden by the standard `OTEL_EXPORTER_OTLP_*` environment variables.

## Per-provider LLM spans

This package emits Pixeltable's own operation, execution, and UDF spans. It does not capture per-provider
token usage or cost. To add rich provider spans (OpenAI, Anthropic, etc.), install a dedicated GenAI
instrumentor alongside this one:

```python
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.instrumentation.pixeltable import PixeltableInstrumentor

PixeltableInstrumentor().instrument()
OpenAIInstrumentor().instrument()
```

Because Pixeltable establishes the ambient OpenTelemetry context for every UDF call (including across its
thread pools and async scheduler), the provider instrumentor's spans nest correctly under Pixeltable's
`udf.<name>` span. Prefer the bare `opentelemetry-instrumentation-*` packages (or OpenInference), which
attach to the global provider; avoid all-in-one initializers that install their own `TracerProvider` and
would clobber the one configured here.
