"""
A collection of Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `StringType`.
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
from pixeltable.type_system import StringType
from pixeltable.utils.code import local_public_names


@func.udf
def capitalize(s: str) -> str:
    """
    Return `s` with its first character capitalized and the rest lowercased.

    Equivalent to [`str.capitalize()`](https://docs.python.org/3/library/stdtypes.html#str.capitalize).
    """
    return s.capitalize()

@func.udf
def casefold(s: str) -> str:
    """
    Return a casefolded copy of `s`.

    Equivalent to [`str.casefold()`](https://docs.python.org/3/library/stdtypes.html#str.casefold).
    """
    return s.casefold()

@func.udf
def center(s: str, width: int, fillchar: str = ' ') -> str:
    """
    Return a centered string of length `width`.

    Equivalent to [`str.center()`](https://docs.python.org/3/library/stdtypes.html#str.center).

    Args:
        width: Total width of the resulting string.
        fillchar: Character used for padding.
    """
    return s.center(width, fillchar)

@func.udf
def contains(s: str, pattern: str, case: bool = True, flags: int = 0, regex: bool = True) -> bool:
    """
    Test if pattern or regex is contained within a string.

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
        return bool(re.search(pattern, s, flags))
    else:
        if case:
            return pattern in s
        else:
            return pattern.lower() in s.lower()

@func.udf
def count(s: str, pattern: str, flags: int = 0) -> int:
    """
    Count occurrences of pattern or regex in `s`.

    Args:
        pattern: string literal or regular expression
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    import re
    return len(re.findall(pattern, s, flags))

@func.udf
def endswith(s: str, pattern: str) -> bool:
    """
    Return `True` if the string ends with the specified suffix, otherwise return `False`.

    Equivalent to [`str.endswith()`](https://docs.python.org/3/library/stdtypes.html#str.endswith).

    Args:
        pattern: string literal
    """
    return s.endswith(pattern)

@func.udf
def fill(s: str, width: int, **kwargs: Any) -> str:
    """
    Wraps the single paragraph in `s`, and returns a single string containing the wrapped paragraph.

    Equivalent to [`textwrap.fill()`](https://docs.python.org/3/library/textwrap.html#textwrap.fill).

    Args:
        width: Maximum line width.
        kwargs: Additional keyword arguments to pass to `textwrap.fill()`.
    """
    import textwrap
    return textwrap.fill(s, width, **kwargs)

@func.udf
def find(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the lowest index in `s` where `substr` is found within the slice `s[start:end]`.

    Equivalent to [`str.find()`](https://docs.python.org/3/library/stdtypes.html#str.find).

    Args:
        substr: substring to search for
        start: slice start
        end: slice end
    """
    return s.find(substr, start, end)

@func.udf
def findall(s: str, pattern: str, flags: int = 0) -> list:
    """
    Find all occurrences of a regular expression pattern in a string.

    Equivalent to [`re.findall()`](https://docs.python.org/3/library/re.html#re.findall).

    Args:
        pattern: regular expression pattern
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    import re
    return re.findall(pattern, s, flags)

@func.udf
def format(s: str, *args: Any, **kwargs: Any) -> str:
    """
    Perform string formatting.

    Equivalent to [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format).
    """
    return s.format(*args, **kwargs)

@func.udf
def fullmatch(s: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
    """
    Determine if `s` fully matches a regular expression.

    Equivalent to [`re.fullmatch()`](https://docs.python.org/3/library/re.html#re.fullmatch).

    Args:
        pattern: regular expression pattern
        case: if False, ignore case
        flags: [flags](https://docs.python.org/3/library/re.html#flags) for the `re` module
    """
    import re
    if not case:
        flags |= re.IGNORECASE
    _ = bool(re.fullmatch(pattern, s, flags))
    return bool(re.fullmatch(pattern, s, flags))

@func.udf
def index(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the lowest index in `s` where `substr` is found within the slice `s[start:end]`. Raises ValueError if `substr` is not found.

    Equivalent to [`str.index()`](https://docs.python.org/3/library/stdtypes.html#str.index).

    Args:
        substr: substring to search for
        start: slice start
        end: slice end
    """
    return s.index(substr, start, end)

@func.udf
def isalnum(s: str) -> bool:
    """
    Return `True` if all characters in the string are alphanumeric and there is at least one character, `False`
    otherwise.

    Equivalent to [`str.isalnum()`](https://docs.python.org/3/library/stdtypes.html#str.isalnum
    """
    return s.isalnum()

@func.udf
def isalpha(s: str) -> bool:
    """
    Return `True` if all characters in the string are alphabetic and there is at least one character, `False` otherwise.

    Equivalent to [`str.isalpha()`](https://docs.python.org/3/library/stdtypes.html#str.isalpha).
    """
    return s.isalpha()

@func.udf
def isascii(s: str) -> bool:
    """
    Return `True` if the string is empty or all characters in the string are ASCII, `False` otherwise.

    Equivalent to [`str.isascii()`](https://docs.python.org/3/library/stdtypes.html#str.isascii).
    """
    return s.isascii()

@func.udf
def isdecimal(s: str) -> bool:
    """
    Return `True` if all characters in the string are decimal characters and there is at least one character, `False`
    otherwise.

    Equivalent to [`str.isdecimal()`](https://docs.python.org/3/library/stdtypes.html#str.isdecimal).
    """
    return s.isdecimal()

@func.udf
def isdigit(s: str) -> bool:
    """
    Return `True` if all characters in the string are digits and there is at least one character, `False` otherwise.

    Equivalent to [`str.isdigit()`](https://docs.python.org/3/library/stdtypes.html#str.isdigit).
    """
    return s.isdigit()

@func.udf
def isidentifier(s: str) -> bool:
    """
    Return `True` if the string is a valid identifier according to the language definition, `False` otherwise.

    Equivalent to [`str.isidentifier()`](https://docs.python.org/3/library/stdtypes.html#str.isidentifier)
    """
    return s.isidentifier()


@func.udf
def islower(s: str) -> bool:
    """
    Return `True` if all cased characters in the string are lowercase and there is at least one cased character, `False` otherwise.

    Equivalent to [`str.islower()`](https://docs.python.org/3/library/stdtypes.html#str.islower)
    """
    return s.islower()

@func.udf
def isnumeric(s: str) -> bool:
    """
    Return `True` if all characters in the string are numeric characters, `False` otherwise.

    Equivalent to [`str.isnumeric()`](https://docs.python.org/3/library/stdtypes.html#str.isnumeric)
    """
    return s.isnumeric()

@func.udf
def isupper(s: str) -> bool:
    """
    Return `True` if all cased characters in the string are uppercase and there is at least one cased character, `False` otherwise.

    Equivalent to [`str.isupper()`](https://docs.python.org/3/library/stdtypes.html#str.isupper)
    """
    return s.isupper()

@func.udf
def istitle(s: str) -> bool:
    """
    Return `True` if the string is a titlecased string and there is at least one character, `False` otherwise.

    Equivalent to [`str.istitle()`](https://docs.python.org/3/library/stdtypes.html#str.istitle)
    """
    return s.istitle()

@func.udf
def isspace(s: str) -> bool:
    """
    Return `True` if there are only whitespace characters in the string and there is at least one character, `False` otherwise.

    Equivalent to [`str.isspace()`](https://docs.python.org/3/library/stdtypes.html#str.isspace)
    """
    return s.isspace()

@func.udf
def len(s: str) -> int:
    """
    Return the number of characters in the string.

    Equivalent to [`len(str)`](https://docs.python.org/3/library/functions.html#len)
    """
    return s.__len__()

@func.udf
def ljust(s: str, width: int, fillchar: str = ' ') -> str:
    """
    Return the string left-justified in a string of length `width`. Padding is done using the specified `fillchar` (default is a space).

    Equivalent to [`str.ljust()`](https://docs.python.org/3/library/stdtypes.html#str.ljust)

    Args:
        width: Minimum width of resulting string; additional characters will be filled with character defined in `fillchar`.
        fillchar: Additional character for filling.
    """
    return s.ljust(width, fillchar)

@func.udf
def lower(s: str) -> str:
    """
    Return a copy of the string with all the cased characters converted to lowercase.

    Equivalent to [`str.lower()`](https://docs.python.org/3/library/stdtypes.html#str.lower)
    """
    return s.lower()

@func.udf
def lstrip(s: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of the string with leading characters removed. The `chars` argument is a string specifying the set of
    characters to be removed. If omitted or `None`, whitespace characters are removed.

    Equivalent to [`str.lstrip()`](https://docs.python.org/3/library/stdtypes.html#str.lstrip)

    Args:
        chars: The set of characters to be removed.
    """
    return s.lstrip(chars)

@func.udf
def match(s: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
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
    return bool(re.match(pattern, s, flags))

@func.udf
def normalize(s: str, form: str) -> str:
    """
    Return the Unicode normal form for `s`.

    Equivalent to [`unicodedata.normalize()`](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)

    Args:
        form: Unicode normal form (`‘NFC’`, `‘NFKC’`, `‘NFD’`, `‘NFKD’`)
    """
    import unicodedata
    return unicodedata.normalize(form, s)

@func.udf
def pad(s: str, width: int, side: str = 'left', fillchar: str = ' ') -> str:
    """
    Pad string up to width

    Args:
        width: Minimum width of resulting string; additional characters will be filled with character defined in `fillchar`.
        side: Side from which to fill resulting string (`‘left’`, `‘right’`, `‘both’`)
        fillchar: Additional character for filling
    """
    if side == 'left':
        return s.ljust(width, fillchar)
    elif side == 'right':
        return s.rjust(width, fillchar)
    elif side == 'both':
        return s.center(width, fillchar)
    else:
        raise ValueError(f"Invalid side: {side}")

@func.udf
def partition(s: str, sep: str = ' ') -> list:
    """
    Splits `s` at the first occurrence of `sep`, and returns 3 elements containing the part before the
    separator, the separator itself, and the part after the separator. If the separator is not found, return 3 elements
    containing `s` itself, followed by two empty strings.
    """
    idx = s.find(sep)
    if idx == -1:
        return [s, '', '']
    from builtins import len
    return [s[:idx], sep, s[idx + len(sep):]]

@func.udf
def removeprefix(s: str, prefix: str) -> str:
    """
    Remove prefix from `s`. If the prefix is not present, returns `s`.
    """
    if s.startswith(prefix):
        # we need to avoid referring to our symbol 'len'
        from builtins import len
        return s[len(prefix):]
    return s

@func.udf
def removesuffix(s: str, suffix: str) -> str:
    """
    Remove suffix from `s`. If the suffix is not present, returns `s`.
    """
    if s.endswith(suffix):
        # we need to avoid referring to our symbol 'len'
        from builtins import len
        return s[:-len(suffix)]
    return s

@func.udf
def repeat(s: str, count: int) -> str:
    """
    Repeat `s` `count` times.
    """
    return s * count

@func.udf
def replace(
        s: str, pattern: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = False
) -> str:
    """
    Replace occurrences of `pattern` in `s` with `repl`.

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
        return re.sub(pattern, repl, s, 0 if n == -1 else n, flags)
    else:
        return s.replace(pattern, repl, n)

@func.udf
def rfind(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the highest index in `s` where `substr` is found, such that `substr` is contained within `s[start:end]`.

    Equivalent to [`str.rfind()`](https://docs.python.org/3/library/stdtypes.html#str.rfind).

    Args:
        substr: substring to search for
        start: slice start
        end: slice end
    """
    return s.rfind(substr, start, end)

@func.udf
def rindex(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """
    Return the highest index in `s` where `substr` is found, such that `substr` is contained within `s[start:end]`.
    Raises ValueError if `substr` is not found.

    Equivalent to [`str.rindex()`](https://docs.python.org/3/library/stdtypes.html#str.rindex).
    """
    return s.rindex(substr, start, end)

@func.udf
def rjust(s: str, width: int, fillchar: str = ' ') -> str:
    """
    Return `s` right-justified in a string of length `width`.

    Equivalent to [`str.rjust()`](https://docs.python.org/3/library/stdtypes.html#str.rjust).

    Args:
        width: Minimum width of resulting string.
        fillchar: Additional character for filling.
    """
    return s.rjust(width, fillchar)

@func.udf
def rpartition(s: str, sep: str = ' ') -> list:
    """
    This method splits `s` at the last occurrence of `sep`, and returns a list containing the part before the
    separator, the separator itself, and the part after the separator.
    """
    idx = s.rfind(sep)
    if idx == -1:
        return [s, '', '']
    from builtins import len
    return [s[:idx], sep, s[idx + len(sep):]]

@func.udf
def rstrip(s: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of `s` with trailing characters removed.

    Equivalent to [`str.rstrip()`](https://docs.python.org/3/library/stdtypes.html#str.rstrip).

    Args:
        chars: The set of characters to be removed. If omitted or `None`, whitespace characters are removed.
    """
    return s.rstrip(chars)

@func.udf
def slice(s: str, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> str:
    """
    Return a slice of `s`.

    Args:
        start: slice start
        stop: slice end
        step: slice step
    """
    return s[start:stop:step]

@func.udf
def slice_replace(s: str, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> str:
    """
    Replace a positional slice of a string with another value.

    Args:
        start: slice start
        stop: slice end
        repl: replacement value
    """
    return s[:start] + repl + s[stop:]

@func.udf
def startswith(s: str, pattern: str) -> int:
    """
    Return `True` if `s` starts with `pattern`, otherwise return `False`.

    Equivalent to [`str.startswith()`](https://docs.python.org/3/library/stdtypes.html#str.startswith).

    Args:
        pattern: string literal
    """
    return s.startswith(pattern)

@func.udf
def strip(s: str, chars: Optional[str] = None) -> str:
    """
    Return a copy of `s` with leading and trailing characters removed.

    Equivalent to [`str.strip()`](https://docs.python.org/3/library/stdtypes.html#str.strip).

    Args:
        chars: The set of characters to be removed. If omitted or `None`, whitespace characters are removed.
    """
    return s.strip(chars)

@func.udf
def swapcase(s: str) -> str:
    """
    Return a copy of `s` with uppercase characters converted to lowercase and vice versa.

    Equivalent to [`str.swapcase()`](https://docs.python.org/3/library/stdtypes.html#str.swapcase).
    """
    return s.swapcase()

@func.udf
def title(s: str) -> str:
    """
    Return a titlecased version of `s`, i.e. words start with uppercase characters, all remaining cased characters are
    lowercase.

    Equivalent to [`str.title()`](https://docs.python.org/3/library/stdtypes.html#str.title).
    """
    return s.title()

@func.udf
def upper(s: str) -> str:
    """
    Return a copy of `s` converted to uppercase.
    Equivalent to [`str.upper()`](https://docs.python.org/3/library/stdtypes.html#str.upper).
    """
    return s.upper()

@func.udf
def wrap(s: str, width: int, **kwargs: Any) -> dict:
    """
    Wraps the single paragraph in `s` so every line is at most `width` characters long.
    Returns a list of output lines, without final newlines.

    Equivalent to [`textwrap.fill()`](https://docs.python.org/3/library/textwrap.html#textwrap.fill).

    Args:
        width: Maximum line width.
        kwargs: Additional keyword arguments to pass to `textwrap.fill()`.
    """
    """Equivalent to textwrap.wrap()"""
    import textwrap
    return textwrap.wrap(s, width, **kwargs)

@func.udf
def zfill(s: str, width: int) -> str:
    """
    Pad a numeric string `s` with ASCII `0` on the left to a total length of `width`.

    Equivalent to [`str.zfill()`](https://docs.python.org/3/library/stdtypes.html#str.zfill).

    Args:
        width: Minimum width of resulting string.
    """
    return s.zfill(width)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
