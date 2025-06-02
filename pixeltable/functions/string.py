"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `StringType`.
It closely follows the Pandas `pandas.Series.str` API.

Example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(t.str_col.capitalize()).collect()
```
"""

import builtins
import re
import textwrap
from string import whitespace
from typing import Any, Optional

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf(is_method=True)
def capitalize(self: str) -> str:
    """
    Return string with its first character capitalized and the rest lowercased.

    Equivalent to [`str.capitalize()`](https://docs.python.org/3/library/stdtypes.html#str.capitalize).
    """
    return self.capitalize()


@capitalize.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.concat(sql.func.upper(sql.func.left(self, 1)), sql.func.lower(sql.func.right(self, -1)))


@pxt.udf(is_method=True)
def casefold(self: str) -> str:
    """
    Return a casefolded copy of string.

    Equivalent to [`str.casefold()`](https://docs.python.org/3/library/stdtypes.html#str.casefold).
    """
    return self.casefold()


@pxt.udf(is_method=True)
def center(self: str, width: int, fillchar: str = ' ') -> str:
    """
    Return a centered string of length `width`.

    Equivalent to [`str.center()`](https://docs.python.org/3/library/stdtypes.html#str.center).

    Args:
        width: Total width of the resulting string.
        fillchar: Character used for padding.
    """
    return self.center(width, fillchar)


@pxt.udf(is_method=True)
def contains(self: str, substr: str, case: bool = True) -> bool:
    """
    Test if string contains a substring.

    Args:
        substr: string literal or regular expression
        case: if False, ignore case
    """
    if case:
        return substr in self
    else:
        return substr.lower() in self.lower()


@contains.to_sql
def _(
    self: sql.ColumnElement, substr: sql.ColumnElement, case: Optional[sql.ColumnElement] = None
) -> sql.ColumnElement:
    # Replace all occurrences of `%`, `_`, and `\` with escaped versions
    escaped_substr = sql.func.regexp_replace(substr, r'(%|_|\\)', r'\\\1', 'g')
    if case is None:
        # Default `case` is True, so we do a case-sensitive comparison
        return self.like(sql.func.concat('%', escaped_substr, '%'))
    else:
        # Toggle case-sensitivity based on the value of `case`
        return sql.case(
            (case, self.like(sql.func.concat('%', escaped_substr, '%'))),
            else_=sql.func.lower(self).like(sql.func.concat('%', sql.func.lower(escaped_substr), '%')),
        )


@pxt.udf(is_method=True)
def contains_re(self: str, pattern: str, flags: int = 0) -> bool:
    """
    Test if string contains a regular expression pattern.

    Args:
        pattern: regular expression pattern
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    return bool(re.search(pattern, self, flags))


@pxt.udf(is_method=True)
def count(self: str, pattern: str, flags: int = 0) -> int:
    """
    Count occurrences of pattern or regex.

    Args:
        pattern: string literal or regular expression
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    return builtins.len(re.findall(pattern, self, flags))


@pxt.udf(is_method=True)
def endswith(self: str, substr: str) -> bool:
    """
    Return `True` if the string ends with the specified suffix, otherwise return `False`.

    Equivalent to [`str.endswith()`](https://docs.python.org/3/library/stdtypes.html#str.endswith).

    Args:
        substr: string literal
    """
    return self.endswith(substr)


@endswith.to_sql
def _(self: sql.ColumnElement, substr: sql.ColumnElement) -> sql.ColumnElement:
    # Replace all occurrences of `%`, `_`, and `\` with escaped versions
    escaped_substr = sql.func.regexp_replace(substr, r'(%|_|\\)', r'\\\1', 'g')
    return self.like(sql.func.concat('%', escaped_substr))


@pxt.udf(is_method=True)
def fill(self: str, width: int, **kwargs: Any) -> str:
    """
    Wraps the single paragraph in string, and returns a single string containing the wrapped paragraph.

    Equivalent to [`textwrap.fill()`](https://docs.python.org/3/library/textwrap.html#textwrap.fill).

    Args:
        width: Maximum line width.
        kwargs: Additional keyword arguments to pass to `textwrap.fill()`.
    """
    return textwrap.fill(self, width, **kwargs)


@pxt.udf(is_method=True)
def find(self: str, substr: str, start: int = 0, end: Optional[int] = None) -> int:
    """
    Return the lowest index in string where `substr` is found within the slice `s[start:end]`.

    Equivalent to [`str.find()`](https://docs.python.org/3/library/stdtypes.html#str.find).

    Args:
        substr: substring to search for
        start: slice start
        end: slice end
    """
    return self.find(substr, start, end)


@find.to_sql
def _(
    self: sql.ColumnElement,
    substr: sql.ColumnElement,
    start: sql.ColumnElement,
    end: Optional[sql.ColumnElement] = None,
) -> sql.ColumnElement:
    sl = pxt.functions.string.slice._to_sql(self, start, end)
    if sl is None:
        return None

    strpos = sql.func.strpos(sl, substr)
    return sql.case(
        (strpos == 0, -1), (start >= 0, strpos + start - 1), else_=strpos + sql.func.char_length(self) + start - 1
    )


@pxt.udf(is_method=True)
def findall(self: str, pattern: str, flags: int = 0) -> list:
    """
    Find all occurrences of a regular expression pattern in string.

    Equivalent to [`re.findall()`](https://docs.python.org/3/library/re.html#re.findall).

    Args:
        pattern: regular expression pattern
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    return re.findall(pattern, self, flags)


@pxt.udf(is_method=True)
def format(self: str, *args: Any, **kwargs: Any) -> str:
    """
    Perform string formatting.

    Equivalent to [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format).
    """
    return self.format(*args, **kwargs)


@pxt.udf(is_method=True)
def fullmatch(self: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
    """
    Determine if string fully matches a regular expression.

    Equivalent to [`re.fullmatch()`](https://docs.python.org/3/library/re.html#re.fullmatch).

    Args:
        pattern: regular expression pattern
        case: if False, ignore case
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    if not case:
        flags |= re.IGNORECASE
    _ = bool(re.fullmatch(pattern, self, flags))
    return bool(re.fullmatch(pattern, self, flags))


@pxt.udf(is_method=True)
def index(self: str, substr: str, start: int = 0, end: Optional[int] = None) -> int:
    """
    Return the lowest index in string where `substr` is found within the slice `[start:end]`.
    Raises ValueError if `substr` is not found.

    Equivalent to [`str.index()`](https://docs.python.org/3/library/stdtypes.html#str.index).

    Args:
        substr: substring to search for
        start: slice start
        end: slice end
    """
    return self.index(substr, start, end)


@pxt.udf(is_method=True)
def isalnum(self: str) -> bool:
    """
    Return `True` if all characters in the string are alphanumeric and there is at least one character, `False`
    otherwise.

    Equivalent to [`str.isalnum()`](https://docs.python.org/3/library/stdtypes.html#str.isalnum
    """
    return self.isalnum()


@pxt.udf(is_method=True)
def isalpha(self: str) -> bool:
    """
    Return `True` if all characters in the string are alphabetic and there is at least one character, `False` otherwise.

    Equivalent to [`str.isalpha()`](https://docs.python.org/3/library/stdtypes.html#str.isalpha).
    """
    return self.isalpha()


@pxt.udf(is_method=True)
def isascii(self: str) -> bool:
    """
    Return `True` if the string is empty or all characters in the string are ASCII, `False` otherwise.

    Equivalent to [`str.isascii()`](https://docs.python.org/3/library/stdtypes.html#str.isascii).
    """
    return self.isascii()


@pxt.udf(is_method=True)
def isdecimal(self: str) -> bool:
    """
    Return `True` if all characters in the string are decimal characters and there is at least one character, `False`
    otherwise.

    Equivalent to [`str.isdecimal()`](https://docs.python.org/3/library/stdtypes.html#str.isdecimal).
    """
    return self.isdecimal()


@pxt.udf(is_method=True)
def isdigit(self: str) -> bool:
    """
    Return `True` if all characters in the string are digits and there is at least one character, `False` otherwise.

    Equivalent to [`str.isdigit()`](https://docs.python.org/3/library/stdtypes.html#str.isdigit).
    """
    return self.isdigit()


@pxt.udf(is_method=True)
def isidentifier(self: str) -> bool:
    """
    Return `True` if the string is a valid identifier according to the language definition, `False` otherwise.

    Equivalent to [`str.isidentifier()`](https://docs.python.org/3/library/stdtypes.html#str.isidentifier)
    """
    return self.isidentifier()


@pxt.udf(is_method=True)
def islower(self: str) -> bool:
    """
    Return `True` if all cased characters in the string are lowercase and there is at least one cased character,
    `False` otherwise.

    Equivalent to [`str.islower()`](https://docs.python.org/3/library/stdtypes.html#str.islower)
    """
    return self.islower()


@pxt.udf(is_method=True)
def isnumeric(self: str) -> bool:
    """
    Return `True` if all characters in the string are numeric characters, `False` otherwise.

    Equivalent to [`str.isnumeric()`](https://docs.python.org/3/library/stdtypes.html#str.isnumeric)
    """
    return self.isnumeric()


@pxt.udf(is_method=True)
def isupper(self: str) -> bool:
    """
    Return `True` if all cased characters in the string are uppercase and there is at least one cased character,
    `False` otherwise.

    Equivalent to [`str.isupper()`](https://docs.python.org/3/library/stdtypes.html#str.isupper)
    """
    return self.isupper()


@pxt.udf(is_method=True)
def istitle(self: str) -> bool:
    """
    Return `True` if the string is a titlecased string and there is at least one character, `False` otherwise.

    Equivalent to [`str.istitle()`](https://docs.python.org/3/library/stdtypes.html#str.istitle)
    """
    return self.istitle()


@pxt.udf(is_method=True)
def isspace(self: str) -> bool:
    """
    Return `True` if there are only whitespace characters in the string and there is at least one character,
    `False` otherwise.

    Equivalent to [`str.isspace()`](https://docs.python.org/3/library/stdtypes.html#str.isspace)
    """
    return self.isspace()


@pxt.udf
def join(sep: str, elements: list) -> str:
    """
    Return a string which is the concatenation of the strings in `elements`.

    Equivalent to [`str.join()`](https://docs.python.org/3/library/stdtypes.html#str.join)
    """
    return sep.join(elements)


@pxt.udf(is_method=True)
def len(self: str) -> int:
    """
    Return the number of characters in the string.

    Equivalent to [`len(str)`](https://docs.python.org/3/library/functions.html#len)
    """
    return builtins.len(self)


@len.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.char_length(self)


@pxt.udf(is_method=True)
def ljust(self: str, width: int, fillchar: str = ' ') -> str:
    """
    Return the string left-justified in a string of length `width`.

    Equivalent to [`str.ljust()`](https://docs.python.org/3/library/stdtypes.html#str.ljust)

    Args:
        width: Minimum width of resulting string; additional characters will be filled with character defined in
            `fillchar`.
        fillchar: Additional character for filling.
    """
    return self.ljust(width, fillchar)


@pxt.udf(is_method=True)
def lower(self: str) -> str:
    """
    Return a copy of the string with all the cased characters converted to lowercase.

    Equivalent to [`str.lower()`](https://docs.python.org/3/library/stdtypes.html#str.lower)
    """
    return self.lower()


@lower.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.lower(self)


@pxt.udf(is_method=True)
def lstrip(self: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of the string with leading characters removed. The `chars` argument is a string specifying the set of
    characters to be removed. If omitted or `None`, whitespace characters are removed.

    Equivalent to [`str.lstrip()`](https://docs.python.org/3/library/stdtypes.html#str.lstrip)

    Args:
        chars: The set of characters to be removed.
    """
    return self.lstrip(chars)


@lstrip.to_sql
def _(self: sql.ColumnElement, chars: Optional[sql.ColumnElement] = None) -> sql.ColumnElement:
    return sql.func.ltrim(self, chars if chars is not None else whitespace)


@pxt.udf(is_method=True)
def match(self: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
    """
    Determine if string starts with a match of a regular expression

    Args:
        pattern: regular expression pattern
        case: if False, ignore case
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    if not case:
        flags |= re.IGNORECASE
    return bool(re.match(pattern, self, flags))


@pxt.udf(is_method=True)
def normalize(self: str, form: str) -> str:
    """
    Return the Unicode normal form.

    Equivalent to [`unicodedata.normalize()`](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)

    Args:
        form: Unicode normal form (`'NFC'`, `'NFKC'`, `'NFD'`, `'NFKD'`)
    """
    import unicodedata

    return unicodedata.normalize(form, self)  # type: ignore[arg-type]


@pxt.udf(is_method=True)
def pad(self: str, width: int, side: str = 'left', fillchar: str = ' ') -> str:
    """
    Pad string up to width

    Args:
        width: Minimum width of resulting string; additional characters will be filled with character defined in
            `fillchar`.
        side: Side from which to fill resulting string (`'left'`, `'right'`, `'both'`)
        fillchar: Additional character for filling
    """
    if side == 'left':
        return self.ljust(width, fillchar)
    elif side == 'right':
        return self.rjust(width, fillchar)
    elif side == 'both':
        return self.center(width, fillchar)
    else:
        raise ValueError(f'Invalid side: {side}')


@pxt.udf(is_method=True)
def partition(self: str, sep: str = ' ') -> list:
    """
    Splits string at the first occurrence of `sep`, and returns 3 elements containing the part before the
    separator, the separator itself, and the part after the separator. If the separator is not found, return 3 elements
    containing string itself, followed by two empty strings.
    """
    idx = self.find(sep)
    if idx == -1:
        return [self, '', '']
    return [self[:idx], sep, self[idx + builtins.len(sep) :]]


@pxt.udf(is_method=True)
def removeprefix(self: str, prefix: str) -> str:
    """
    Remove prefix. If the prefix is not present, returns string.
    """
    return self.removeprefix(prefix)


@removeprefix.to_sql
def _(self: sql.ColumnElement, prefix: sql.ColumnElement) -> sql.ColumnElement:
    return sql.case((startswith._to_sql(self, prefix), sql.func.right(self, -sql.func.char_length(prefix))), else_=self)


@pxt.udf(is_method=True)
def removesuffix(self: str, suffix: str) -> str:
    """
    Remove suffix. If the suffix is not present, returns string.
    """
    return self.removesuffix(suffix)


@removesuffix.to_sql
def _(self: sql.ColumnElement, suffix: sql.ColumnElement) -> sql.ColumnElement:
    return sql.case((endswith._to_sql(self, suffix), sql.func.left(self, -sql.func.char_length(suffix))), else_=self)


@pxt.udf(is_method=True)
def repeat(self: str, n: int) -> str:
    """
    Repeat string `n` times.
    """
    return self * n


@repeat.to_sql
def _(self: sql.ColumnElement, n: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.repeat(self, n.cast(sql.types.INT))


@pxt.udf(is_method=True)
def replace(self: str, substr: str, repl: str, n: Optional[int] = None) -> str:
    """
    Replace occurrences of `substr` with `repl`.

    Equivalent to [`str.replace()`](https://docs.python.org/3/library/stdtypes.html#str.replace).

    Args:
        substr: string literal
        repl: replacement string
        n: number of replacements to make (if `None`, replace all occurrences)
    """
    return self.replace(substr, repl, n or -1)


@replace.to_sql
def _(
    self: sql.ColumnElement, substr: sql.ColumnElement, repl: sql.ColumnElement, n: Optional[sql.ColumnElement] = None
) -> sql.ColumnElement:
    if n is not None:
        return None  # SQL does not support bounding the number of replacements

    return sql.func.replace(self, substr, repl)


@pxt.udf(is_method=True)
def replace_re(self: str, pattern: str, repl: str, n: Optional[int] = None, flags: int = 0) -> str:
    """
    Replace occurrences of a regular expression pattern with `repl`.

    Equivalent to [`re.sub()`](https://docs.python.org/3/library/re.html#re.sub).

    Args:
        pattern: regular expression pattern
        repl: replacement string
        n: number of replacements to make (if `None`, replace all occurrences)
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    return re.sub(pattern, repl, self, count=(n or 0), flags=flags)


@pxt.udf(is_method=True)
def reverse(self: str) -> str:
    """
    Return a reversed copy of the string.

    Equivalent to `str[::-1]`.
    """
    return self[::-1]


@reverse.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.reverse(self)


@pxt.udf(is_method=True)
def rfind(self: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the highest index where `substr` is found, such that `substr` is contained within `[start:end]`.

    Equivalent to [`str.rfind()`](https://docs.python.org/3/library/stdtypes.html#str.rfind).

    Args:
        substr: substring to search for
        start: slice start
        end: slice end
    """
    return self.rfind(substr, start, end)


@pxt.udf(is_method=True)
def rindex(self: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the highest index where `substr` is found, such that `substr` is contained within `[start:end]`.
    Raises ValueError if `substr` is not found.

    Equivalent to [`str.rindex()`](https://docs.python.org/3/library/stdtypes.html#str.rindex).
    """
    return self.rindex(substr, start, end)


@pxt.udf(is_method=True)
def rjust(self: str, width: int, fillchar: str = ' ') -> str:
    """
    Return the string right-justified in a string of length `width`.

    Equivalent to [`str.rjust()`](https://docs.python.org/3/library/stdtypes.html#str.rjust).

    Args:
        width: Minimum width of resulting string.
        fillchar: Additional character for filling.
    """
    return self.rjust(width, fillchar)


@pxt.udf(is_method=True)
def rpartition(self: str, sep: str = ' ') -> list:
    """
    This method splits string at the last occurrence of `sep`, and returns a list containing the part before the
    separator, the separator itself, and the part after the separator.
    """
    idx = self.rfind(sep)
    if idx == -1:
        return [self, '', '']
    return [self[:idx], sep, self[idx + builtins.len(sep) :]]


@pxt.udf(is_method=True)
def rstrip(self: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of string with trailing characters removed.

    Equivalent to [`str.rstrip()`](https://docs.python.org/3/library/stdtypes.html#str.rstrip).

    Args:
        chars: The set of characters to be removed. If omitted or `None`, whitespace characters are removed.
    """
    return self.rstrip(chars)


@rstrip.to_sql
def _(self: sql.ColumnElement, chars: Optional[sql.ColumnElement] = None) -> sql.ColumnElement:
    return sql.func.rtrim(self, chars if chars is not None else whitespace)


@pxt.udf(is_method=True)
def slice(self: str, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> str:
    """
    Return a slice.

    Args:
        start: slice start
        stop: slice end
        step: slice step
    """
    return self[start:stop:step]


@slice.to_sql
def _(
    self: sql.ColumnElement,
    start: Optional[sql.ColumnElement] = None,
    stop: Optional[sql.ColumnElement] = None,
    step: Optional[sql.ColumnElement] = None,
) -> sql.ColumnElement:
    if step is not None:
        return None

    if start is not None:
        start = start.cast(sql.types.INT)  # Postgres won't accept a BIGINT
        start = sql.case(
            (start >= 0, start + 1),  # SQL is 1-based, Python is 0-based
            else_=sql.func.char_length(self) + start + 1,  # negative index
        )
        start = sql.func.greatest(start, 1)

    if stop is not None:
        stop = stop.cast(sql.types.INT)  # Postgres won't accept a BIGINT
        stop = sql.case(
            (stop >= 0, stop + 1),  # SQL is 1-based, Python is 0-based
            else_=sql.func.char_length(self) + stop + 1,  # negative index
        )
        stop = sql.func.greatest(stop, 0)

    if start is None:
        if stop is None:
            return self
        return sql.func.substr(self, 1, stop)
    if stop is None:
        return sql.func.substr(self, start)
    return sql.func.substr(self, start, sql.func.greatest(stop - start, 0))


@pxt.udf(is_method=True)
def slice_replace(
    self: str, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None
) -> str:
    """
    Replace a positional slice with another value.

    Args:
        start: slice start
        stop: slice end
        repl: replacement value
    """
    return self[:start] + repl + self[stop:]


@pxt.udf(is_method=True)
def startswith(self: str, substr: str) -> int:
    """
    Return `True` if string starts with `substr`, otherwise return `False`.

    Equivalent to [`str.startswith()`](https://docs.python.org/3/library/stdtypes.html#str.startswith).

    Args:
        substr: string literal
    """
    return self.startswith(substr)


@startswith.to_sql
def _(self: sql.ColumnElement, substr: sql.ColumnElement) -> sql.ColumnElement:
    # Replace all occurrences of `%`, `_`, and `\` with escaped versions
    escaped_substr = sql.func.regexp_replace(substr, r'(%|_|\\)', r'\\\1', 'g')
    return self.like(sql.func.concat(escaped_substr, '%'))


@pxt.udf(is_method=True)
def strip(self: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of string with leading and trailing characters removed.

    Equivalent to [`str.strip()`](https://docs.python.org/3/library/stdtypes.html#str.strip).

    Args:
        chars: The set of characters to be removed. If omitted or `None`, whitespace characters are removed.
    """
    return self.strip(chars)


@strip.to_sql
def _(self: sql.ColumnElement, chars: Optional[sql.ColumnElement] = None) -> sql.ColumnElement:
    return sql.func.trim(self, chars if chars is not None else whitespace)


@pxt.udf(is_method=True)
def swapcase(self: str) -> str:
    """
    Return a copy of string with uppercase characters converted to lowercase and vice versa.

    Equivalent to [`str.swapcase()`](https://docs.python.org/3/library/stdtypes.html#str.swapcase).
    """
    return self.swapcase()


@pxt.udf(is_method=True)
def title(self: str) -> str:
    """
    Return a titlecased version of string, i.e. words start with uppercase characters, all remaining cased characters
    are lowercase.

    Equivalent to [`str.title()`](https://docs.python.org/3/library/stdtypes.html#str.title).
    """
    return self.title()


@pxt.udf(is_method=True)
def upper(self: str) -> str:
    """
    Return a copy of string converted to uppercase.

    Equivalent to [`str.upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper).
    """
    return self.upper()


@upper.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.upper(self)


@pxt.udf(is_method=True)
def wrap(self: str, width: int, **kwargs: Any) -> list[str]:
    """
    Wraps the single paragraph in string so every line is at most `width` characters long.
    Returns a list of output lines, without final newlines.

    Equivalent to [`textwrap.fill()`](https://docs.python.org/3/library/textwrap.html#textwrap.fill).

    Args:
        width: Maximum line width.
        kwargs: Additional keyword arguments to pass to `textwrap.fill()`.
    """
    return textwrap.wrap(self, width, **kwargs)


@pxt.udf(is_method=True)
def zfill(self: str, width: int) -> str:
    """
    Pad a numeric string with ASCII `0` on the left to a total length of `width`.

    Equivalent to [`str.zfill()`](https://docs.python.org/3/library/stdtypes.html#str.zfill).

    Args:
        width: Minimum width of resulting string.
    """
    return self.zfill(width)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
