"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `StringType`.
It closely follows the Pandas `pandas.Series.str` API.

Example:
```python
import pixeltable as pxt
from pixeltable.functions import string as pxt_str

t = pxt.get_table(...)
t.select(pxt_str.capitalize(t.str_col)).collect()
```
"""

from typing import Any, Optional

import pixeltable.func as func
from pixeltable.utils.code import local_public_names


@func.udf(is_method=True)
def capitalize(self: str) -> str:
    """
    Return string with its first character capitalized and the rest lowercased.

    Equivalent to [`str.capitalize()`](https://docs.python.org/3/library/stdtypes.html#str.capitalize).
    """
    return self.capitalize()

@func.udf(is_method=True)
def casefold(self: str) -> str:
    """
    Return a casefolded copy of string.

    Equivalent to [`str.casefold()`](https://docs.python.org/3/library/stdtypes.html#str.casefold).
    """
    return self.casefold()

@func.udf(is_method=True)
def center(self: str, width: int, fillchar: str = ' ') -> str:
    """
    Return a centered string of length `width`.

    Equivalent to [`str.center()`](https://docs.python.org/3/library/stdtypes.html#str.center).

    Args:
        width: Total width of the resulting string.
        fillchar: Character used for padding.
    """
    return self.center(width, fillchar)

@func.udf(is_method=True)
def contains(self: str, pattern: str, case: bool = True, flags: int = 0, regex: bool = True) -> bool:
    """
    Test if string contains pattern or regex.

    Args:
        pattern: string literal or regular expression
        case: if False, ignore case
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
        regex: if True, treat pattern as a regular expression
    """
    if regex:
        import re
        if not case:
            flags |= re.IGNORECASE
        return bool(re.search(pattern, self, flags))
    else:
        if case:
            return pattern in self
        else:
            return pattern.lower() in self.lower()

@func.udf(is_method=True)
def count(self: str, pattern: str, flags: int = 0) -> int:
    """
    Count occurrences of pattern or regex.

    Args:
        pattern: string literal or regular expression
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    import re
    from builtins import len
    return len(re.findall(pattern, self, flags))

@func.udf(is_method=True)
def endswith(self: str, pattern: str) -> bool:
    """
    Return `True` if the string ends with the specified suffix, otherwise return `False`.

    Equivalent to [`str.endswith()`](https://docs.python.org/3/library/stdtypes.html#str.endswith).

    Args:
        pattern: string literal
    """
    return self.endswith(pattern)

@func.udf(is_method=True)
def fill(self: str, width: int, **kwargs: Any) -> str:
    """
    Wraps the single paragraph in string, and returns a single string containing the wrapped paragraph.

    Equivalent to [`textwrap.fill()`](https://docs.python.org/3/library/textwrap.html#textwrap.fill).

    Args:
        width: Maximum line width.
        kwargs: Additional keyword arguments to pass to `textwrap.fill()`.
    """
    import textwrap
    return textwrap.fill(self, width, **kwargs)

@func.udf(is_method=True)
def find(self: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the lowest index in string where `substr` is found within the slice `s[start:end]`.

    Equivalent to [`str.find()`](https://docs.python.org/3/library/stdtypes.html#str.find).

    Args:
        substr: substring to search for
        start: slice start
        end: slice end
    """
    return self.find(substr, start, end)

@func.udf(is_method=True)
def findall(self: str, pattern: str, flags: int = 0) -> list:
    """
    Find all occurrences of a regular expression pattern in string.

    Equivalent to [`re.findall()`](https://docs.python.org/3/library/re.html#re.findall).

    Args:
        pattern: regular expression pattern
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    import re
    return re.findall(pattern, self, flags)

@func.udf(is_method=True)
def format(self: str, *args: Any, **kwargs: Any) -> str:
    """
    Perform string formatting.

    Equivalent to [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format).
    """
    return self.format(*args, **kwargs)

@func.udf(is_method=True)
def fullmatch(self: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
    """
    Determine if string fully matches a regular expression.

    Equivalent to [`re.fullmatch()`](https://docs.python.org/3/library/re.html#re.fullmatch).

    Args:
        pattern: regular expression pattern
        case: if False, ignore case
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    import re
    if not case:
        flags |= re.IGNORECASE
    _ = bool(re.fullmatch(pattern, self, flags))
    return bool(re.fullmatch(pattern, self, flags))

@func.udf(is_method=True)
def index(self: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
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

@func.udf(is_method=True)
def isalnum(self: str) -> bool:
    """
    Return `True` if all characters in the string are alphanumeric and there is at least one character, `False`
    otherwise.

    Equivalent to [`str.isalnum()`](https://docs.python.org/3/library/stdtypes.html#str.isalnum
    """
    return self.isalnum()

@func.udf(is_method=True)
def isalpha(self: str) -> bool:
    """
    Return `True` if all characters in the string are alphabetic and there is at least one character, `False` otherwise.

    Equivalent to [`str.isalpha()`](https://docs.python.org/3/library/stdtypes.html#str.isalpha).
    """
    return self.isalpha()

@func.udf(is_method=True)
def isascii(self: str) -> bool:
    """
    Return `True` if the string is empty or all characters in the string are ASCII, `False` otherwise.

    Equivalent to [`str.isascii()`](https://docs.python.org/3/library/stdtypes.html#str.isascii).
    """
    return self.isascii()

@func.udf(is_method=True)
def isdecimal(self: str) -> bool:
    """
    Return `True` if all characters in the string are decimal characters and there is at least one character, `False`
    otherwise.

    Equivalent to [`str.isdecimal()`](https://docs.python.org/3/library/stdtypes.html#str.isdecimal).
    """
    return self.isdecimal()

@func.udf(is_method=True)
def isdigit(self: str) -> bool:
    """
    Return `True` if all characters in the string are digits and there is at least one character, `False` otherwise.

    Equivalent to [`str.isdigit()`](https://docs.python.org/3/library/stdtypes.html#str.isdigit).
    """
    return self.isdigit()

@func.udf(is_method=True)
def isidentifier(self: str) -> bool:
    """
    Return `True` if the string is a valid identifier according to the language definition, `False` otherwise.

    Equivalent to [`str.isidentifier()`](https://docs.python.org/3/library/stdtypes.html#str.isidentifier)
    """
    return self.isidentifier()


@func.udf(is_method=True)
def islower(self: str) -> bool:
    """
    Return `True` if all cased characters in the string are lowercase and there is at least one cased character, `False` otherwise.

    Equivalent to [`str.islower()`](https://docs.python.org/3/library/stdtypes.html#str.islower)
    """
    return self.islower()

@func.udf(is_method=True)
def isnumeric(self: str) -> bool:
    """
    Return `True` if all characters in the string are numeric characters, `False` otherwise.

    Equivalent to [`str.isnumeric()`](https://docs.python.org/3/library/stdtypes.html#str.isnumeric)
    """
    return self.isnumeric()

@func.udf(is_method=True)
def isupper(self: str) -> bool:
    """
    Return `True` if all cased characters in the string are uppercase and there is at least one cased character, `False` otherwise.

    Equivalent to [`str.isupper()`](https://docs.python.org/3/library/stdtypes.html#str.isupper)
    """
    return self.isupper()

@func.udf(is_method=True)
def istitle(self: str) -> bool:
    """
    Return `True` if the string is a titlecased string and there is at least one character, `False` otherwise.

    Equivalent to [`str.istitle()`](https://docs.python.org/3/library/stdtypes.html#str.istitle)
    """
    return self.istitle()

@func.udf(is_method=True)
def isspace(self: str) -> bool:
    """
    Return `True` if there are only whitespace characters in the string and there is at least one character, `False` otherwise.

    Equivalent to [`str.isspace()`](https://docs.python.org/3/library/stdtypes.html#str.isspace)
    """
    return self.isspace()

@func.udf(is_method=True)
def len(self: str) -> int:
    """
    Return the number of characters in the string.

    Equivalent to [`len(str)`](https://docs.python.org/3/library/functions.html#len)
    """
    return self.__len__()

@func.udf(is_method=True)
def ljust(self: str, width: int, fillchar: str = ' ') -> str:
    """
    Return the string left-justified in a string of length `width`.

    Equivalent to [`str.ljust()`](https://docs.python.org/3/library/stdtypes.html#str.ljust)

    Args:
        width: Minimum width of resulting string; additional characters will be filled with character defined in `fillchar`.
        fillchar: Additional character for filling.
    """
    return self.ljust(width, fillchar)

@func.udf(is_method=True)
def lower(self: str) -> str:
    """
    Return a copy of the string with all the cased characters converted to lowercase.

    Equivalent to [`str.lower()`](https://docs.python.org/3/library/stdtypes.html#str.lower)
    """
    return self.lower()

@func.udf(is_method=True)
def lstrip(self: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of the string with leading characters removed. The `chars` argument is a string specifying the set of
    characters to be removed. If omitted or `None`, whitespace characters are removed.

    Equivalent to [`str.lstrip()`](https://docs.python.org/3/library/stdtypes.html#str.lstrip)

    Args:
        chars: The set of characters to be removed.
    """
    return self.lstrip(chars)

@func.udf(is_method=True)
def match(self: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
    """
    Determine if string starts with a match of a regular expression

    Args:
        pattern: regular expression pattern
        case: if False, ignore case
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    import re
    if not case:
        flags |= re.IGNORECASE
    return bool(re.match(pattern, self, flags))

@func.udf(is_method=True)
def normalize(self: str, form: str) -> str:
    """
    Return the Unicode normal form.

    Equivalent to [`unicodedata.normalize()`](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)

    Args:
        form: Unicode normal form (`‘NFC’`, `‘NFKC’`, `‘NFD’`, `‘NFKD’`)
    """
    import unicodedata
    return unicodedata.normalize(form, self)

@func.udf(is_method=True)
def pad(self: str, width: int, side: str = 'left', fillchar: str = ' ') -> str:
    """
    Pad string up to width

    Args:
        width: Minimum width of resulting string; additional characters will be filled with character defined in `fillchar`.
        side: Side from which to fill resulting string (`‘left’`, `‘right’`, `‘both’`)
        fillchar: Additional character for filling
    """
    if side == 'left':
        return self.ljust(width, fillchar)
    elif side == 'right':
        return self.rjust(width, fillchar)
    elif side == 'both':
        return self.center(width, fillchar)
    else:
        raise ValueError(f"Invalid side: {side}")

@func.udf(is_method=True)
def partition(self: str, sep: str = ' ') -> list:
    """
    Splits string at the first occurrence of `sep`, and returns 3 elements containing the part before the
    separator, the separator itself, and the part after the separator. If the separator is not found, return 3 elements
    containing string itself, followed by two empty strings.
    """
    idx = self.find(sep)
    if idx == -1:
        return [self, '', '']
    from builtins import len
    return [self[:idx], sep, self[idx + len(sep):]]

@func.udf(is_method=True)
def removeprefix(self: str, prefix: str) -> str:
    """
    Remove prefix. If the prefix is not present, returns string.
    """
    if self.startswith(prefix):
        # we need to avoid referring to our symbol 'len'
        from builtins import len
        return self[len(prefix):]
    return self

@func.udf(is_method=True)
def removesuffix(self: str, suffix: str) -> str:
    """
    Remove suffix. If the suffix is not present, returns string.
    """
    if self.endswith(suffix):
        # we need to avoid referring to our symbol 'len'
        from builtins import len
        return self[:-len(suffix)]
    return self

@func.udf(is_method=True)
def repeat(self: str, n: int) -> str:
    """
    Repeat string `n` times.
    """
    return self * n

@func.udf(is_method=True)
def replace(
        self: str, pattern: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = False
) -> str:
    """
    Replace occurrences of `pattern` with `repl`.

    Equivalent to [`str.replace()`](https://docs.python.org/3/library/stdtypes.html#str.replace) or
    [`re.sub()`](https://docs.python.org/3/library/re.html#re.sub), depending on the value of regex.

    Args:
        pattern: string literal or regular expression
        repl: replacement string
        n: number of replacements to make (-1 for all)
        case: if False, ignore case
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
        regex: if True, treat pattern as a regular expression
    """
    if regex:
        import re
        if not case:
            flags |= re.IGNORECASE
        return re.sub(pattern, repl, self, 0 if n == -1 else n, flags)
    else:
        return self.replace(pattern, repl, n)

@func.udf(is_method=True)
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

@func.udf(is_method=True)
def rindex(self: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the highest index where `substr` is found, such that `substr` is contained within `[start:end]`.
    Raises ValueError if `substr` is not found.

    Equivalent to [`str.rindex()`](https://docs.python.org/3/library/stdtypes.html#str.rindex).
    """
    return self.rindex(substr, start, end)

@func.udf(is_method=True)
def rjust(self: str, width: int, fillchar: str = ' ') -> str:
    """
    Return the string right-justified in a string of length `width`.

    Equivalent to [`str.rjust()`](https://docs.python.org/3/library/stdtypes.html#str.rjust).

    Args:
        width: Minimum width of resulting string.
        fillchar: Additional character for filling.
    """
    return self.rjust(width, fillchar)

@func.udf(is_method=True)
def rpartition(self: str, sep: str = ' ') -> list:
    """
    This method splits string at the last occurrence of `sep`, and returns a list containing the part before the
    separator, the separator itself, and the part after the separator.
    """
    idx = self.rfind(sep)
    if idx == -1:
        return [self, '', '']
    from builtins import len
    return [self[:idx], sep, self[idx + len(sep):]]

@func.udf(is_method=True)
def rstrip(self: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of string with trailing characters removed.

    Equivalent to [`str.rstrip()`](https://docs.python.org/3/library/stdtypes.html#str.rstrip).

    Args:
        chars: The set of characters to be removed. If omitted or `None`, whitespace characters are removed.
    """
    return self.rstrip(chars)

@func.udf(is_method=True)
def slice(self: str, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> str:
    """
    Return a slice.

    Args:
        start: slice start
        stop: slice end
        step: slice step
    """
    return self[start:stop:step]

@func.udf(is_method=True)
def slice_replace(self: str, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> str:
    """
    Replace a positional slice with another value.

    Args:
        start: slice start
        stop: slice end
        repl: replacement value
    """
    return self[:start] + repl + self[stop:]

@func.udf(is_method=True)
def startswith(self: str, pattern: str) -> int:
    """
    Return `True` if string starts with `pattern`, otherwise return `False`.

    Equivalent to [`str.startswith()`](https://docs.python.org/3/library/stdtypes.html#str.startswith).

    Args:
        pattern: string literal
    """
    return self.startswith(pattern)

@func.udf(is_method=True)
def strip(self: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of string with leading and trailing characters removed.

    Equivalent to [`str.strip()`](https://docs.python.org/3/library/stdtypes.html#str.strip).

    Args:
        chars: The set of characters to be removed. If omitted or `None`, whitespace characters are removed.
    """
    return self.strip(chars)

@func.udf(is_method=True)
def swapcase(self: str) -> str:
    """
    Return a copy of string with uppercase characters converted to lowercase and vice versa.

    Equivalent to [`str.swapcase()`](https://docs.python.org/3/library/stdtypes.html#str.swapcase).
    """
    return self.swapcase()

@func.udf(is_method=True)
def title(self: str) -> str:
    """
    Return a titlecased version of string, i.e. words start with uppercase characters, all remaining cased characters
    are lowercase.

    Equivalent to [`str.title()`](https://docs.python.org/3/library/stdtypes.html#str.title).
    """
    return self.title()

@func.udf(is_method=True)
def upper(self: str) -> str:
    """
    Return a copy of string converted to uppercase.

    Equivalent to [`str.upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper).
    """
    return self.upper()

@func.udf(is_method=True)
def wrap(self: str, width: int, **kwargs: Any) -> dict:
    """
    Wraps the single paragraph in string so every line is at most `width` characters long.
    Returns a list of output lines, without final newlines.

    Equivalent to [`textwrap.fill()`](https://docs.python.org/3/library/textwrap.html#textwrap.fill).

    Args:
        width: Maximum line width.
        kwargs: Additional keyword arguments to pass to `textwrap.fill()`.
    """
    import textwrap
    return textwrap.wrap(self, width, **kwargs)

@func.udf(is_method=True)
def zfill(self: str, width: int) -> str:
    """
    Pad a numeric string with ASCII `0` on the left to a total length of `width`.

    Equivalent to [`str.zfill()`](https://docs.python.org/3/library/stdtypes.html#str.zfill).

    Args:
        width: Minimum width of resulting string.
    """
    return self.zfill(width)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
