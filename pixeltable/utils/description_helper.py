from typing import Union

import pandas as pd
from pandas.io.formats.style import Styler


class DescriptionHelper:
    __descriptors: list[Union[str, pd.DataFrame, Styler]]

    def __init__(self) -> None:
        self.__descriptors = []

    def append(self, descriptor: Union[str, pd.DataFrame, Styler]) -> None:
        self.__descriptors.append(descriptor)

    def to_string(self) -> str:
        blocks = [str(descriptor) for descriptor in self.__descriptors]
        return '\n\n'.join(blocks)

    def to_html(self) -> str:
        html_blocks = [self.__apply_styles(descriptor).to_html() for descriptor in self.__descriptors]
        return '\n'.join(html_blocks)

    @classmethod
    def __apply_styles(self, descriptor: Union[str, pd.DataFrame, Styler]) -> Styler:
        if isinstance(descriptor, str):
            return (
                pd.DataFrame([descriptor]).style
                .set_properties(None, **{'white-space': 'pre-wrap', 'text-align': 'left', 'font-weight': 'bold'})
                .hide(axis='index').hide(axis='columns')
            )
        elif isinstance(descriptor, pd.DataFrame):
            return (
                descriptor.style
                .set_properties(None, **{'white-space': 'pre-wrap', 'text-align': 'left'})
                .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
                .hide(axis='index')
            )
        else:
            # Already styled
            return descriptor
