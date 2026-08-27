<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c5ab123e-806c-49bf-93e7-151353719b16">
  <img alt="Pixeltable Logo" src="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131" width="40%">
</picture>

<div>
<br>
</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)

[**Quick Start**](https://docs.pixeltable.com/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**CLI**](https://docs.pixeltable.com/platform/cli) |
[**Dashboard**](https://docs.pixeltable.com/platform/dashboard) |
[**llms-full.txt**](https://docs.pixeltable.com/llms-full.txt) |
[**Starter Kit**](https://github.com/pixeltable/pixeltable-starter-kit) |
[**AI Coding Skill**](https://github.com/pixeltable/pixeltable-skill) |
[**Discord**](https://discord.gg/QPyqFYx2UN)

## Make Building Multimodal AI Data Apps Dead Simple

The unified multimodal backend for AI data apps. Declare tables, computed columns, and HTTP routes in one Python file. Apply it locally, serve it, then publish the same file to [Pixeltable Cloud](https://www.pixeltable.com/). Python 3.11+ on Linux, macOS, or Windows.

```bash
pip install 'pixeltable[serve]'
pxt service example --out app.py
pxt schema update app.py my_app
pxt service update app.py my_app
```

`pxt service example` writes this file (`from __future__ import annotations` is required on Python 3.14+):

```python
from __future__ import annotations  # required to declare a model on Python 3.14+

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    title: pxt.String                         # a stored column
    body: pxt.String | None                   # a stored column that may be null
    title_upper = pxtf.string.upper(title)    # a computed column: an assignment, not an annotation


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.title, Docs.body], outputs=[Docs.title, Docs.title_upper]
)
```

Insert a row and the computed column runs. `pxt service list` prints the URL (the port is assigned):

```bash
pxt service list
# ingest  http://127.0.0.1:<port>  ...
curl -X POST http://127.0.0.1:<port>/docs \
  -H 'Content-Type: application/json' \
  -d '{"title": "Hello", "body": "world"}'
```

Same file against a hosted database (`PIXELTABLE_API_KEY` required):

```bash
pxt db create pxt://org:mydb
pxt schema update app.py pxt://org:mydb
```

In a notebook or test, `pxt.create_table()` is still the interactive API. An app’s contract is the application file.

```bash
npx skills add pixeltable/pixeltable-skill
```

Project templates: [pixeltable-starter-kit](https://github.com/pixeltable/pixeltable-starter-kit).

## License

Apache 2.0. [Contributing](CONTRIBUTING.md) · [Discord](https://discord.gg/QPyqFYx2UN)
