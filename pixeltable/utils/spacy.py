from typing import TYPE_CHECKING

from pixeltable import exceptions as excs
from pixeltable.env import Env

if TYPE_CHECKING:
    import spacy


_MODEL_CACHE: dict[str, 'spacy.Language'] = {}


def get_spacy_model(model_name: str) -> 'spacy.Language':
    """Get a spaCy model, loading it if necessary.

    Args:
        model_name: Name of the spaCy model to load.

    Returns:
        The loaded spaCy Language model.
    """
    Env.get().require_package('spacy')
    import spacy

    if model_name not in _MODEL_CACHE:
        try:
            model = spacy.load(model_name)
        except OSError as e:
            raise excs.Error(
                f'Failed to locate spaCy model {model_name!r}. To install it, run:\n'
                f'    python -m spacy download {model_name}'
            ) from e
        _MODEL_CACHE[model_name] = model

    return _MODEL_CACHE[model_name]
