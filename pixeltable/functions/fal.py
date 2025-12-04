"""
Pixeltable UDFs
that wrap various endpoints from the fal.ai API. In order to use them, you must
first `pip install fal-client` and configure your fal.ai credentials, as described in
the [Working with fal.ai](https://docs.pixeltable.com/notebooks/integrations/working-with-fal) tutorial.
"""

from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import fal_client


@register_client('fal')
def _(api_key: str) -> 'fal_client.AsyncClient':
    import fal_client

    return fal_client.AsyncClient(key=api_key)


def _fal_client() -> 'fal_client.AsyncClient':
    return Env.get().get_client('fal')


@pxt.udf(resource_pool='request-rate:fal')
async def run(input: dict[str, Any], *, app: str) -> pxt.Json:
    """
    Run a model on fal.ai.

    Uses fal's queue-based subscribe mechanism for reliable execution.
    For additional details, see: <https://fal.ai/docs>

    Request throttling:
    Applies the rate limit set in the config (section `fal`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install fal-client`

    Args:
        input: The input parameters for the model.
        app: The name or ID of the fal.ai application to run (e.g., 'fal-ai/flux/schnell').

    Returns:
        The output of the model as a JSON object.

    Examples:
        Add a computed column that applies the model `fal-ai/flux/schnell`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> input = {'prompt': tbl.prompt}
        ... tbl.add_computed_column(response=run(input, app='fal-ai/flux/schnell'))

        Add a computed column that uses the model `fal-ai/fast-sdxl`
        to generate images from an existing Pixeltable column `tbl.prompt`:

        >>> input = {'prompt': tbl.prompt, 'image_size': 'square', 'num_inference_steps': 25}
        ... tbl.add_computed_column(response=run(input, app='fal-ai/fast-sdxl'))
        ... tbl.add_computed_column(image=tbl.response['images'][0]['url'].astype(pxt.Image))
    """
    Env.get().require_package('fal_client')
    client = _fal_client()
    result = await client.subscribe(app, arguments=input)
    return result


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
