from typing import Any, Iterator

from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env
from pixeltable.iterators.base import ComponentIterator


class StringSplitter(ComponentIterator):
    # TODO(aaron-siegel): Merge this with `DocumentSplitter` in order to provide additional capabilities.
    def __init__(self, text: str, *, separators: str):
        if separators != 'sentence':
            raise excs.Error('Only `sentence` separators are currently supported.')
        self._text = text
        self.doc = Env.get().spacy_nlp(self._text)
        self.iter = self._iter()

    def _iter(self) -> Iterator[dict[str, Any]]:
        for sentence in self.doc.sents:
            yield {'text': sentence.text}

    def __next__(self) -> dict[str, Any]:
        return next(self.iter)

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        pass

    @classmethod
    def input_schema(cls, *args: Any, **kwargs: Any) -> dict[str, ts.ColumnType]:
        return {'text': ts.StringType(), 'separators': ts.StringType()}

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {'text': ts.StringType()}, []
