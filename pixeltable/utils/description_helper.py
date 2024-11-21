import dataclasses
from typing import Optional, Union

import pandas as pd
from pandas.io.formats.style import Styler


@dataclasses.dataclass
class _Descriptor:
    body: Union[str, pd.DataFrame]
    show_index: bool
    show_header: bool
    styler: Optional[Styler] = None


class DescriptionHelper:
    __descriptors: list[_Descriptor]

    def __init__(self) -> None:
        self.__descriptors = []

    def append(
        self,
        descriptor: Union[str, pd.DataFrame],
        show_index: bool = False,
        show_header: bool = True,
        styler: Optional[Styler] = None,
    ) -> None:
        self.__descriptors.append(_Descriptor(descriptor, show_index, show_header, styler))

    def to_string(self) -> str:
        blocks = [
            descriptor.body if isinstance(descriptor.body, str)
            else descriptor.body.to_string(index=descriptor.show_index, header=descriptor.show_header, max_colwidth=49)
            for descriptor in self.__descriptors
        ]
        return '\n\n'.join(blocks)

    def to_html(self) -> str:
        html_blocks = [self.__apply_styles(descriptor).to_html() for descriptor in self.__descriptors]
        return '\n'.join(html_blocks)

    @classmethod
    def __apply_styles(self, descriptor: _Descriptor) -> Styler:
        if isinstance(descriptor.body, str):
            return (
                pd.DataFrame([descriptor.body]).style
                .set_properties(None, **{'white-space': 'pre-wrap', 'text-align': 'left', 'font-weight': 'bold'})
                .hide(axis='index').hide(axis='columns')
            )
        else:
            styler = descriptor.styler
            if styler is None:
                styler = descriptor.body.style
            styler = (
                styler
                .set_properties(None, **{'white-space': 'pre-wrap', 'text-align': 'left'})
                .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            )
            if not descriptor.show_header:
                styler = styler.hide(axis='columns')
            if not descriptor.show_index:
                styler = styler.hide(axis='index')
            return styler
