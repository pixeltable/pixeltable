"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Replicate API. In order to use them, you must
first `pip install replicate` and configure your Replicate credentials, as described in
the [Working with Replicate](https://pixeltable.readme.io/docs/working-with-replicate) tutorial.
"""

from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import replicate  # type: ignore[import-untyped]


@register_client('replicate')
def _(api_token: str) -> 'replicate.Client':
    import replicate
    return replicate.Client(api_token=api_token)


def _replicate_client() -> 'replicate.Client':
    return Env.get().get_client('replicate')


@pxt.udf
def run(
    input: dict[str, Any],
    *,
    ref: str,
) -> dict[str, Any]:
    """
    Run a model on Replicate.

    For additional details, see: <https://replicate.com/docs/topics/models/run-a-model>

    __Requirements:__

    - `pip install replicate`

    Args:
        input: The input parameters for the model.
        ref: The name of the model to run.

    Returns:
        The output of the model.

    Examples:
        Add a computed column that applies the model `meta/meta-llama-3-8b-instruct`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> input = {'system_prompt': 'You are a helpful assistant.', 'prompt': tbl.prompt}
        ... tbl['response'] = run(input, ref='meta/meta-llama-3-8b-instruct')

        Add a computed column that uses the model `black-forest-labs/flux-schnell`
        to generate images from an existing Pixeltable column `tbl.prompt`:

        >>> input = {'prompt': tbl.prompt, 'go_fast': True, 'megapixels': '1'}
        ... tbl['response'] = run(input, ref='black-forest-labs/flux-schnell')
        ... tbl['image'] = tbl.response.output[0].astype(pxt.Image)
    """
    Env.get().require_package('replicate')
    return _replicate_client().run(ref, input, use_file_output=False)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
