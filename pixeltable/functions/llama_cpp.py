from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import llama_cpp


@pxt.udf
def create_chat_completion(
    messages: list[dict],
    *,
    model_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    repo_filename: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Generate a chat completion from a list of messages.

    The model can be specified either as a local path, or as a repo_id and repo_filename that reference a pretrained
    model on the Hugging Face model hub. Exactly one of `model_path` or `repo_id` must be provided; if `model_path`
    is provided, then an optional `repo_filename` can also be specified.

    For additional details, see the
    [llama_cpp create_chat_completions documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).

    Args:
        messages: A list of messages to generate a response for.
        model_path: Path to the model (if using a local model).
        repo_id: The Hugging Face model repo id (if using a pretrained model).
        repo_filename: A filename or glob pattern to match the model file in the repo (optional, if using a
            pretrained model).
        model_kwargs: Additional keyword args for the llama_cpp `create_chat_completions` API, such as `max_tokens`,
            `temperature`, `top_p`, and `top_k`. For details, see the
            [llama_cpp create_chat_completions documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion).
    """
    Env.get().require_package('llama_cpp', min_version=[0, 3, 1])

    if model_kwargs is None:
        model_kwargs = {}

    if (model_path is None) == (repo_id is None):
        raise excs.Error('Exactly one of `model_path` or `repo_id` must be provided.')
    if (repo_id is None) and (repo_filename is not None):
        raise excs.Error('`repo_filename` can only be provided along with `repo_id`.')

    n_gpu_layers = -1 if _is_gpu_available() else 0  # 0 = CPU only, -1 = offload all layers to GPU

    if model_path is not None:
        llm = _lookup_local_model(model_path, n_gpu_layers)
    else:
        Env.get().require_package('huggingface_hub')
        llm = _lookup_pretrained_model(repo_id, repo_filename, n_gpu_layers)
    return llm.create_chat_completion(messages, **model_kwargs)  # type: ignore


def _is_gpu_available() -> bool:
    import llama_cpp

    global _IS_GPU_AVAILABLE  # noqa: PLW0603
    if _IS_GPU_AVAILABLE is None:
        llama_cpp_path = Path(llama_cpp.__file__).parent
        lib = llama_cpp.llama_cpp.load_shared_library('llama', llama_cpp_path / 'lib')
        _IS_GPU_AVAILABLE = bool(lib.llama_supports_gpu_offload())

    return _IS_GPU_AVAILABLE


def _lookup_local_model(model_path: str, n_gpu_layers: int) -> 'llama_cpp.Llama':
    import llama_cpp

    key = (model_path, None, n_gpu_layers)
    if key not in _model_cache:
        llm = llama_cpp.Llama(model_path, n_gpu_layers=n_gpu_layers, verbose=False)
        _model_cache[key] = llm
    return _model_cache[key]


def _lookup_pretrained_model(repo_id: str, filename: Optional[str], n_gpu_layers: int) -> 'llama_cpp.Llama':
    import llama_cpp

    key = (repo_id, filename, n_gpu_layers)
    if key not in _model_cache:
        llm = llama_cpp.Llama.from_pretrained(
            repo_id=repo_id, filename=filename, n_gpu_layers=n_gpu_layers, verbose=False
        )
        _model_cache[key] = llm
    return _model_cache[key]


_model_cache: dict[tuple[str, str, int], Any] = {}
_IS_GPU_AVAILABLE: Optional[bool] = None


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
