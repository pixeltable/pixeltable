from typing import Any, Optional

import pixeltable.func as func
from pixeltable.type_system import StringType
from pixeltable.utils.code import local_public_names


@func.udf
def capitalize(s: str) -> str:
    """Equivalent to str.capitalize()"""
    return s.capitalize()

@func.udf
def casefold(s: str) -> str:
    """Equivalent to str.casefold()"""
    return s.casefold()

@func.udf
def center(s: str, width: int, fillchar: str = ' ') -> str:
    """Equivalent to str.center()"""
    return s.center(width, fillchar)

@func.udf
def contains(s: str, pattern: str, case: bool = True, flags: int = 0, regex: bool = True) -> bool:
    """
    Test if pattern or regex is contained within a string.

    Args:
        pattern: character sequence or regular expression
        case: if False, ignore case
        flags: `re` flags
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
def count(s: str, pattern: str) -> int:
    """Count occurrences of pattern in each string"""
    return s.count(pattern)

# TODO: what should we do with bytearray data?
# @func.udf
# def decode(s: str, encoding: str = 'utf-8', errors: str = 'strict') -> str:
#     """Count occurrences of pattern in each string"""
#     return s.decode(encoding, errors)

@func.udf
def endswith(s: str, pattern: str) -> int:
    """Equivalent to str.endswith()"""
    return s.endswith(pattern)

@func.udf
def fill(s: str, width: int, **kwargs: Any) -> str:
    """Equivalent to textwrap.fill()"""
    import textwrap
    return textwrap.fill(s, width, **kwargs)

@func.udf
def find(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """Equivalent to str.find()"""
    return s.find(substr, start, end)

@func.udf
def format(s: str, *args: Any, **kwargs: Any) -> str:
    """Equivalent to str.format()"""
    return s.format(*args, **kwargs)

@func.udf
def fullmatch(s: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
    """Determine if each string entirely matches a regular expression."""
    import re
    if not case:
        flags |= re.IGNORECASE
    _ = bool(re.fullmatch(pattern, s, flags))
    return bool(re.fullmatch(pattern, s, flags))

@func.udf
def index(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """Equivalent to str.index()"""
    return s.index(substr, start, end)

@func.udf
def isalnum(s: str) -> bool:
    """Equivalent to str.isalnum()"""
    return s.isalnum()

@func.udf
def isalpha(s: str) -> bool:
    """Equivalent to str.isalpha()"""
    return s.isalpha()

@func.udf
def isascii(s: str) -> bool:
    """Equivalent to str.isascii()"""
    return s.isascii()

@func.udf
def isdecimal(s: str) -> bool:
    """Equivalent to str.isdecimal()"""
    return s.isdecimal()

@func.udf
def isdigit(s: str) -> bool:
    """Equivalent to str.isdigit()"""
    return s.isdigit()

@func.udf
def isidentifier(s: str) -> bool:
    """Equivalent to str.isidentifier()"""
    return s.isidentifier()

@func.udf
def islower(s: str) -> bool:
    """Equivalent to str.islower()"""
    return s.islower()

@func.udf
def isnumeric(s: str) -> bool:
    """Equivalent to str.isnumeric()"""
    return s.isnumeric()

@func.udf
def isupper(s: str) -> bool:
    """Equivalent to str.isupper()"""
    return s.isupper()

@func.udf
def istitle(s: str) -> bool:
    """Equivalent to str.istitle()"""
    return s.istitle()

@func.udf
def isspace(s: str) -> bool:
    """Equivalent to str.isspace()"""
    return s.isspace()

@func.udf
def len(s: str) -> int:
    """Equivalent to len(str)"""
    return s.__len__()

@func.udf
def ljust(s: str, width: int, fillchar: str = ' ') -> int:
    """Equivalent to str.ljust()"""
    return s.ljust(width, fillchar)

@func.udf
def lower(s: str) -> str:
    """Equivalent to str.lower()"""
    return s.lower()

@func.udf
def lstrip(s: str, chars: Optional[str] = None) -> str:
    """Equivalent to str.lstrip()"""
    return s.lstrip(chars)

@func.udf
def match(s: str, pattern: str, case: bool = True, flags: int = 0) -> bool:
    """Determine if string starts with a match of a regular expression"""
    import re
    if not case:
        flags |= re.IGNORECASE
    return bool(re.match(pattern, s, flags))

@func.udf
def normalize(s: str, form: str) -> str:
    """Return the Unicode normal form for the string"""
    import unicodedata
    return unicodedata.normalize(form, s)

@func.udf
def pad(s: str, width: int, side: str = 'left', fillchar: str = ' ') -> str:
    """
    Pad string up to width
    Args:
        width: Minimum width of resulting string; additional characters will be filled with character defined in fillchar.
        side: {‘left’, ‘right’, ‘both’}, default ‘left’
            Side from which to fill resulting string.
        fillchar: str, default ‘ ‘
            Additional character for filling, default is whitespace.
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
    This method splits the string at the first occurrence of sep, and returns 3 elements containing the part before the
    separator, the separator itself, and the part after the separator. If the separator is not found, return 3 elements
    containing the string itself, followed by two empty strings.
    """
    idx = s.find(sep)
    if idx == -1:
        return [s, '', '']
    from builtins import len
    return [s[:idx], sep, s[idx + len(sep):]]

@func.udf
def removeprefix(s: str, prefix: str) -> str:
    """
    Remove prefix from a string. If the prefix is not present, the original string will be returned.
    """
    if s.startswith(prefix):
        # we need to avoid referring to our symbol 'len'
        from builtins import len
        return s[len(prefix):]
    return s

@func.udf
def removesuffix(s: str, suffix: str) -> str:
    """
    Remove suffix from a string. If the suffix is not present, the original string will be returned.
    """
    if s.endswith(suffix):
        # we need to avoid referring to our symbol 'len'
        from builtins import len
        return s[:-len(suffix)]
    return s

@func.udf
def repeat(s: str, count: int) -> str:
    """Repeat string count times"""
    return s * count

@func.udf
def replace(
        s: str, pattern: str, repl: str, n: int = -1, case: bool = True, flags: int = 0, regex: bool = False
) -> str:
    """Equivalent to str.replace() or re.sub(), depending on the value of regex"""
    if regex:
        import re
        if not case:
            flags |= re.IGNORECASE
        return re.sub(pattern, repl, s, 0 if n == -1 else n, flags)
    else:
        return s.replace(pattern, repl, n)

@func.udf
def rfind(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """Equivalent to str.rfind()"""
    return s.rfind(substr, start, end)

@func.udf
def rindex(s: str, substr: str, start: Optional[int] = 0, end: Optional[int] = None) -> int:
    """Equivalent to str.rindex()"""
    return s.rindex(substr, start, end)

@func.udf
def rjust(s: str, width: int, fillchar: str = ' ') -> int:
    """Equivalent to str.rjust()"""
    return s.rjust(width, fillchar)

@func.udf
def rpartition(s: str, sep: str = ' ') -> list:
    """
    This method splits the string at the last occurrence of sep, and returns a list containing the part before the
    separator, the separator itself, and the part after the separator.
    """
    idx = s.rfind(sep)
    if idx == -1:
        return [s, '', '']
    from builtins import len
    return [s[:idx], sep, s[idx + len(sep):]]

@func.udf
def rstrip(s: str, chars: Optional[str] = None) -> str:
    """Equivalent to str.rstrip()"""
    return s.rstrip(chars)

@func.udf
def slice(s: str, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> str:
    """Slice substring"""
    return s[start:stop:step]

@func.udf
def slice_replace(s: str, start: Optional[int] = None, stop: Optional[int] = None, repl: Optional[str] = None) -> str:
    """Replace a positional slice of a string with another value."""
    return s[:start] + repl + s[stop:]

@func.udf
def startswith(s: str, pattern: str) -> int:
    """Equivalent to str.startswith()"""
    return s.startswith(pattern)

@func.udf
def strip(s: str, chars: Optional[str] = None) -> str:
    """Equivalent to str.strip()"""
    return s.strip(chars)

@func.udf
def swapcase(s: str) -> str:
    """Equivalent to str.swapcase()"""
    return s.swapcase()

@func.udf
def title(s: str) -> str:
    """Equivalent to str.title()"""
    return s.title()

@func.udf
def upper(s: str) -> str:
    """Equivalent to str.upper()"""
    return s.upper()

@func.udf
def wrap(s: str, width: int, **kwargs: Any) -> dict:
    """Equivalent to textwrap.wrap()"""
    import textwrap
    return textwrap.wrap(s, width, **kwargs)

@func.udf
def zfill(s: str, width: int) -> str:
    """Pad s with '0' characters on the left"""
    return s.zfill(width)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__
