from typing import Iterator, Any

import spacy

from pixeltable.iterators.base import ComponentIterator
import pixeltable.type_system as ts


class SentenceSplitter(ComponentIterator):
    def __init__(self, text: str):
        self._text = text
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.doc = self.spacy_nlp(self._text)
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
        return {'text': ts.StringType()}

    @classmethod
    def output_schema(cls,  *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {'text': ts.StringType()}, []
