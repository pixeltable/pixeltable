from typing import TYPE_CHECKING, Any, Iterator

from deprecated import deprecated

from pixeltable import exceptions as excs, type_system as ts
from pixeltable.iterators.base import ComponentIterator
from pixeltable.utils.spacy import get_spacy_model

if TYPE_CHECKING:
    import spacy


class StringSplitter(ComponentIterator):
    _text: str
    _spacy_model: 'spacy.Language'
    doc: Any  # spacy doc
    iter: Iterator[dict[str, Any]]

    def __init__(self, text: str, *, separators: str, spacy_model: str = 'en_core_web_sm') -> None:
        if separators != 'sentence':
            raise excs.Error('Only `sentence` separators are currently supported.')
        self._text = text
        self._spacy_model = get_spacy_model(spacy_model)
        self.doc = self._spacy_model(self._text)
        self.iter = self._iter()

    def _iter(self) -> Iterator[dict[str, Any]]:
        for sentence in self.doc.sents:
            yield {'text': sentence.text}

    def __next__(self) -> dict[str, Any]:
        return next(self.iter)

    def close(self) -> None:
        pass

    @classmethod
    def input_schema(cls, *args: Any, **kwargs: Any) -> dict[str, ts.ColumnType]:
        return {'text': ts.StringType(), 'separators': ts.StringType()}

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        if kwargs.get('separators') != 'sentence':
            raise excs.Error('Only `sentence` separators are currently supported.')

        # Validate spaCy model
        _ = get_spacy_model(kwargs.get('spacy_model', 'en_core_web_sm'))

        return {'text': ts.StringType()}, []

    @classmethod
    @deprecated('create() is deprecated; use `pixeltable.functions.string.string_splitter` instead', version='0.5.6')
    def create(cls, **kwargs: Any) -> tuple[type[ComponentIterator], dict[str, Any]]:
        return super()._create(**kwargs)
