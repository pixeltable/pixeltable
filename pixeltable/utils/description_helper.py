import dataclasses
from typing import Optional, Union

import pandas as pd
from pandas.io.formats.style import Styler


@dataclasses.dataclass
class _Descriptor:
    body: Union[str, pd.DataFrame]
    # The remaining fields only affect the behavior if `body` is a pd.DataFrame.
    show_index: bool
    show_header: bool
    styler: Optional[Styler] = None


class DescriptionHelper:
    """
    Helper class for rendering long-form descriptions of Pixeltable objects.

    The output is specified as a list of "descriptors", each of which can be either a string or a Pandas DataFrame,
    in any combination. The descriptors will be rendered in sequence. This is useful for long-form descriptions that
    include tables with differing schemas or formatting, and/or a combination of tables and text.

    DescriptionHelper can convert a list of descriptors into either HTML or plaintext and do something reasonable
    in each case.
    """

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
        blocks = [self.__render_text(descriptor) for descriptor in self.__descriptors]
        return '\n\n'.join(blocks)

    def to_html(self) -> str:
        html_blocks = [self.__apply_styles(descriptor).to_html() for descriptor in self.__descriptors]
        return '\n'.join(html_blocks)

    @classmethod
    def __render_text(cls, descriptor: _Descriptor) -> str:
        if isinstance(descriptor.body, str):
            return descriptor.body
        else:
            # If `show_index=False`, we get cleaner output (better intercolumn spacing) by setting the index to a
            # list of empty strings than by setting `index=False` in the call to `df.to_string()`. It's pretty silly
            # that `index=False` has side effects in Pandas that go beyond simply not displaying the index, but it
            # is what it is.
            df = descriptor.body
            if not descriptor.show_index:
                df = df.copy()
                df.index = [''] * len(df)  # type: ignore[assignment]
            # max_colwidth=50 is the identical default that Pandas uses for a DataFrame's __repr__() output.
            return df.to_string(header=descriptor.show_header, max_colwidth=50)

    @classmethod
    def __apply_styles(cls, descriptor: _Descriptor) -> Styler:
        if isinstance(descriptor.body, str):
            return (
                # Render the string as a single-cell DataFrame. This will ensure a consistent style of output in
                # cases where strings appear alongside DataFrames in the same DescriptionHelper.
                pd.DataFrame([descriptor.body])
                .style.set_properties(None, **{'white-space': 'pre-wrap', 'text-align': 'left', 'font-weight': 'bold'})
                .hide(axis='index')
                .hide(axis='columns')
            )
        else:
            styler = descriptor.styler
            if styler is None:
                styler = descriptor.body.style
            styler = styler.set_properties(None, **{'white-space': 'pre-wrap', 'text-align': 'left'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'left')]}]
            )
            if not descriptor.show_header:
                styler = styler.hide(axis='columns')
            if not descriptor.show_index:
                styler = styler.hide(axis='index')
            return styler
