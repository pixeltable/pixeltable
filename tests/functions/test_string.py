import re
import textwrap
import unicodedata
from typing import Callable

import pytest

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.functions.string import (
    capitalize,
    casefold,
    center,
    contains_re,
    count,
    endswith,
    find,
    format,
    fullmatch,
    isalnum,
    isalpha,
    isascii,
    isdecimal,
    isdigit,
    isidentifier,
    islower,
    isnumeric,
    isspace,
    istitle,
    isupper,
    ljust,
    lower,
    lstrip,
    match,
    replace_re,
    reverse,
    rfind,
    rjust,
    rstrip,
    startswith,
    string_splitter,
    strip,
    swapcase,
    title,
    upper,
    zfill,
)

from ..utils import reload_catalog, skip_test_if_not_installed, validate_update_status


class TestString:
    TEST_STR = """
        The concept of relational database was defined by E. F. Codd at IBM in 1970. Codd introduced the term
        relational in his research paper "A Relational Model of Data for Large Shared Data Banks". In this paper
        and later papers, he defined what he meant by relation. One well-known definition of what constitutes a
        relational database system is composed of Codd's 12 rules. However, no commercial implementations of the
        relational model conform to all of Codd's rules, so the term has gradually come to describe a broader class
        of database systems, which at a minimum: Present the data to the user as relations (a presentation in tabular
        form, i.e. as a collection of tables with each table consisting of a set of rows and columns);
        Provide relational operators to manipulate the data in tabular form.
        In 1974, IBM began developing System R, a research project to develop a prototype RDBMS. The first system sold
        as an RDBMS was Multics Relational Data Store (June 1976). Oracle was released in 1979 by Relational Software,
        now Oracle Corporation. Ingres and IBM BS12 followed. Other examples of an RDBMS include IBM Db2, SAP Sybase
        ASE, and Informix. In 1984, the first RDBMS for Macintosh began being developed, code-named Silver Surfer,
        and was released in 1987 as 4th Dimension and known today as 4D.
        The first systems that were relatively faithful implementations of the relational model were from:
        University of Michigan - Micro DBMS (1969)
        Massachusetts Institute of Technology (1971)]
        IBM UK Scientific Centre at Peterlee - IS1 (1970-72), and its successor, PRTV (1973-79).
        """

    TEST_STRS = (
        *textwrap.dedent(TEST_STR.strip()).split('. '),
        '   \v\t\rWhite\n\nSpace\n\f \n\n',
        r'%%!!#__\\Symbols%%!!#\\@@__%',
        'a',
        ' ',
        '',
    )

    def test_all(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        test_params: list[tuple[pxt.Function, Callable, list, dict]] = [
            # (pxt_fn, str_fn, args, **kwargs)
            (capitalize, str.capitalize, [], {}),
            (casefold, str.casefold, [], {}),
            (center, str.center, [100], {}),
            (count, str.count, ['relation'], {}),
            (endswith, str.endswith, ['1970.'], {}),
            (endswith, str.endswith, ['%'], {}),
            (endswith, str.endswith, ['_'], {}),
            (endswith, str.endswith, [r's%%!!#\\@@__%'], {}),
            (find, str.find, ['relation', 10, -10], {}),
            (isalnum, str.isalnum, [], {}),
            (isalpha, str.isalpha, [], {}),
            (isascii, str.isascii, [], {}),
            (isdecimal, str.isdecimal, [], {}),
            (isdigit, str.isdigit, [], {}),
            (isidentifier, str.isidentifier, [], {}),
            (islower, str.islower, [], {}),
            (isnumeric, str.isnumeric, [], {}),
            (isupper, str.isupper, [], {}),
            (istitle, str.istitle, [], {}),
            (isspace, str.isspace, [], {}),
            (pxt.functions.string.len, str.__len__, [], {}),
            (ljust, str.ljust, [100], {}),
            (lower, str.lower, [], {}),
            (lstrip, str.lstrip, [], {}),
            (lstrip, str.lstrip, ['ST'], {}),
            (reverse, lambda s: s[::-1], [], {}),
            (rfind, str.rfind, ['relation', 10, -10], {}),
            (rjust, str.rjust, [100], {}),
            (rstrip, str.rstrip, [], {}),
            (rstrip, str.rstrip, ['ST'], {}),
            (startswith, str.startswith, ['Codd'], {}),
            (strip, str.strip, [], {}),
            (strip, str.strip, ['ST'], {}),
            (swapcase, str.swapcase, [], {}),
            (title, str.title, [], {}),
            (upper, str.upper, [], {}),
            (zfill, str.zfill, [100], {}),
        ]

        for pxt_fn, str_fn, args, kwargs in test_params:
            try:
                actual = t.select(out=pxt_fn(t.s, *args, **kwargs)).collect()['out']
                expected = [str_fn(s, *args, **kwargs) for s in self.TEST_STRS]
                assert actual == expected, pxt_fn
                # Run the same query, forcing the calculations to be done in Python (not SQL)
                # by interposing a non-SQLizable identity function
                actual_py = t.select(
                    out=pxt_fn(t.s.apply(lambda x: x, col_type=pxt.String), *args, **kwargs)
                ).collect()['out']
                assert actual_py == expected, pxt_fn
            except Exception as e:
                print(pxt_fn)
                raise e

        # Check that they can all be called with method syntax too
        for pxt_fn, _, _, _ in test_params:
            mref = t.s.__getattr__(pxt_fn.name)  # noqa: PLC2801
            assert isinstance(mref, exprs.MethodRef)
            assert mref.method_name == pxt_fn.name, pxt_fn

    def test_removeprefix(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=t.s.removeprefix('Codd')).collect()
        for row in res:
            if row['s'].startswith('Codd'):
                assert row['out'] == row['s'][4:]
            else:
                assert row['out'] == row['s']

    def test_removesuffix(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=t.s.removesuffix('1970')).collect()
        for row in res:
            if row['s'].endswith('1970'):
                assert row['out'] == row['s'][:-4]
            else:
                assert row['out'] == row['s']

    def test_replace(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        n = len(t.where(t.s.contains('Codd')).collect())
        t.add_computed_column(s2=t.s.replace('Codd', 'Mohan'))
        m = len(t.where(t.s2.contains('Mohan')).collect())
        assert n == m

        t.add_computed_column(s3=t.s.replace_re('C.dd', 'Mohan'))
        o = len(t.where(t.s3.contains('Mohan')).collect())
        assert n == o

    def test_slice_replace(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=t.s.slice_replace(50, 51, 'abc')).collect()
        for row in res:
            assert row['out'] == row['s'][:50] + 'abc' + row['s'][51:]

    def test_partition(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        status = t.add_computed_column(parts=t.s.partition('IBM'))
        assert status.num_excs == 0
        res = t.where(t.s.contains('IBM')).select(t.s, t.parts).collect()
        for row in res:
            assert len(row['parts']) == 3
            assert len(row['parts'][0]) == row['s'].find('IBM')
            assert row['parts'][1] == 'IBM'

    def test_rpartition(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        status = t.add_computed_column(parts=t.s.rpartition('IBM'))
        assert status.num_excs == 0
        res = t.where(t.s.contains('IBM')).select(t.s, t.parts).collect()
        for row in res:
            assert len(row['parts']) == 3
            assert len(row['parts'][0]) == row['s'].rfind('IBM')
            assert row['parts'][1] == 'IBM'

    def test_wrap(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        res = t.select(t.s, out=t.s.fill(5)).collect()
        for row in res:
            assert row['out'] == textwrap.fill(row['s'], 5)
        res = t.select(t.s, out=t.s.wrap(5)).collect()
        for row in res:
            assert row['out'] == textwrap.wrap(row['s'], 5)

    def test_slice(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))
        res = t.select(t.s, out=t.s.slice(0, 4)).collect()
        for row in res:
            assert row['out'] == row['s'][0:4]

    def test_match(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        assert len(t.where(t.s.match('Codd')).collect()) == 2
        assert len(t.where(t.s.match('codd', case=False)).collect()) == 2

    def test_fullmatch(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        assert len(t.where(t.s.fullmatch('F')).collect()) == 1
        assert len(t.where(t.s.fullmatch('f', case=False)).collect()) == 1

    def test_pad(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))
        res = t.select(t.s, out=t.s.pad(width=100, side='both')).collect()
        for row in res:
            assert row['out'] == row['s'].center(100)

    def test_normalize(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        res = t.select(t.s, out=t.s.normalize('NFC')).collect()
        for row in res:
            assert row['out'] == unicodedata.normalize('NFC', row['s'])

    def test_repeat(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String, 'n': pxt.Int})
        strs = ['a', 'b', 'c', 'd', 'e']
        validate_update_status(t.insert({'s': s, 'n': n} for n, s in enumerate(strs)), expected_rows=len(strs))
        res = t.select(t.s, t.n, out=t.s.repeat(t.n)).collect()
        for row in res:
            assert row['out'] == row['s'] * row['n']

    def test_contains(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        assert t.select(out=t.s.contains('IBM')).collect()['out'] == ['IBM' in s for s in self.TEST_STRS]
        assert t.select(out=t.s.contains('ibm', case=True)).collect()['out'] == ['ibm' in s for s in self.TEST_STRS]
        assert t.select(out=t.s.contains('ibm', case=False)).collect()['out'] == [
            'ibm' in s.lower() for s in self.TEST_STRS
        ]

        assert t.select(out=t.s.contains_re('ibm', flags=re.IGNORECASE)).collect()['out'] == [
            'ibm' in s.lower() for s in self.TEST_STRS
        ]
        assert t.select(out=t.s.contains_re('i.m', flags=re.IGNORECASE)).collect()['out'] >= [
            'ibm' in s.lower() for s in self.TEST_STRS
        ]

    def test_index(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.select(t.s.index('IBM')).collect()
        assert 'ValueError' in str(exc_info.value)

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.select(t.s.rindex('IBM')).collect()
        assert 'ValueError' in str(exc_info.value)

        res = t.where(t.s.contains('IBM')).select(t.s, idx=t.s.index('IBM')).collect()
        for s, idx in zip(res['s'], res['idx']):
            assert s[idx : idx + 3] == 'IBM'

        res = t.where(t.s.contains('IBM')).select(t.s, idx=t.s.rindex('IBM')).collect()
        for s, idx in zip(res['s'], res['idx']):
            assert s[idx : idx + 3] == 'IBM'

    def test_format(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'input': pxt.String})

        t.add_computed_column(s1=format('ABC {0}', t.input, t.input))
        t.add_computed_column(s2=format('DEF {this}', this=t.input))
        t.add_computed_column(s3=format('GHI {0} JKL {this}', t.input, this=t.input))
        status = t.insert(input='MNO')
        assert status.num_rows == 1
        assert status.num_excs == 0
        row = t.head()[0]
        assert row == {'input': 'MNO', 's1': 'ABC MNO', 's2': 'DEF MNO', 's3': 'GHI MNO JKL MNO'}

        reload_catalog()
        t = pxt.get_table('test_tbl')
        status = t.insert(input='PQR')
        assert status.num_rows == 1
        assert status.num_excs == 0
        row = t.head()[1]
        assert row == {'input': 'PQR', 's1': 'ABC PQR', 's2': 'DEF PQR', 's3': 'GHI PQR JKL PQR'}

    def test_join(self, uses_db: None) -> None:
        from pixeltable.functions.string import join

        t = pxt.create_table('test_tbl', {'elements': pxt.Json})
        test_data = [['a', 'b', 'c'], ['hello', 'world'], ['single'], []]
        validate_update_status(t.insert({'elements': e} for e in test_data), expected_rows=len(test_data))

        # Test with comma separator
        res = t.select(out=join(', ', t.elements)).collect()['out']
        expected = [', '.join(e) for e in test_data]
        assert res == expected

        # Test with empty separator
        res = t.select(out=join('', t.elements)).collect()['out']
        expected = [''.join(e) for e in test_data]
        assert res == expected

        # Force Python execution and compare
        res_py = t.select(out=join(', ', t.elements.apply(lambda x: x, col_type=pxt.Json))).collect()['out']
        expected = [', '.join(e) for e in test_data]
        assert res_py == expected

    def test_zfill_signed(self, uses_db: None) -> None:
        """Test zfill with signed numbers to ensure sign is preserved at front."""
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        test_strs = ['-42', '+42', '42', '-1', '+1', '1', '-123456', '+123456', '']
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))

        # Test zfill with width 6
        res = t.select(out=zfill(t.s, 6)).collect()['out']
        expected = [s.zfill(6) for s in test_strs]
        assert res == expected, f'SQL result {res} != expected {expected}'

        # Force Python execution and compare
        res_py = t.select(out=zfill(t.s.apply(lambda x: x, col_type=pxt.String), 6)).collect()['out']
        assert res_py == expected, f'Python result {res_py} != expected {expected}'

    def test_is_validation_functions(self, uses_db: None) -> None:
        """Test is* validation functions with edge cases to ensure SQL/Python equivalence.

        Note: SQL implementations use PostgreSQL POSIX character classes which are locale-dependent.
        These tests focus on ASCII strings for reliable cross-locale behavior.
        """
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        # Test strings covering various edge cases (ASCII only, no duplicates)
        test_strs = [
            # Empty and whitespace
            '',
            ' ',
            '   ',
            '\t',
            '\n',
            '\t\n',
            # Alphanumeric
            'abc',
            'ABC',
            '123',
            'abc123',
            'ABC123',
            # Alpha only
            'Hello',
            'HELLO',
            'hello',
            # Digit only
            '0',
            '12345',
            # Mixed with symbols
            'abc!',
            'a b',
            'hello world',
            'a1b2',
            '123.45',
            '-123',
            # Case variations
            'Hello World',
            'HELLO WORLD',
            # Identifiers
            'valid_name',
            '_private',
            '__dunder__',
            'CamelCase',
            # Non-identifiers
            '123invalid',
            '123Invalid',
            '!invalid',
            # Title case edge cases
            'Hello world',
            'hello WORLD',
            # Swapcase inputs
            'HeLLo',
            'hELLO wORLD',
        ]
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))

        test_params = [
            (isalnum, str.isalnum),
            (isalpha, str.isalpha),
            (isascii, str.isascii),
            (isdecimal, str.isdecimal),
            (isdigit, str.isdigit),
            (islower, str.islower),
            (isupper, str.isupper),
            (isspace, str.isspace),
            (istitle, str.istitle),
            (isidentifier, str.isidentifier),
            (swapcase, str.swapcase),
            (casefold, str.casefold),
        ]

        for pxt_fn, str_fn in test_params:
            # SQL execution
            res_sql = t.select(out=pxt_fn(t.s)).collect()['out']
            expected = [str_fn(s) for s in test_strs]
            assert res_sql == expected, f'{pxt_fn.name} SQL mismatch: {res_sql} != {expected}'

            # Force Python execution
            res_py = t.select(out=pxt_fn(t.s.apply(lambda x: x, col_type=pxt.String))).collect()['out']
            assert res_py == expected, f'{pxt_fn.name} Python mismatch: {res_py} != {expected}'

    def test_regex_sql_equivalence(self, uses_db: None) -> None:
        """Test regex-based string functions for SQL/Python equivalence, including edge cases.

        For each function the test runs two queries:
          1. Normal query — Pixeltable uses the SQL implementation when possible.
          2. Forced-Python query — the identity `.apply()` wrapper prevents SQL pushdown.
        Both must produce the same result as the equivalent Python expression.
        """
        test_strs = [
            # Plain words
            'cat',
            'dog',
            'catdog',
            'cat dog',
            'dog food',
            # Casing variants
            'Cat',
            'CAT',
            'Hello World',
            # Digits and mixed
            'abc123',
            '123abc',
            '123',
            # Regex-special characters in the *input*
            'hello.world',
            'price $9.99',
            '1+2=3',
            # Vowel-rich strings
            'hello',
            'aeiou',
            # Repeated characters (tests non-overlapping count behaviour)
            'aaa',
            'aaaa',
            # Empty / whitespace
            '',
            ' ',
        ]
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))

        def check_sql_and_py(pxt_fn: pxt.Function, *args, **kwargs) -> tuple[list, list]:
            """Return (sql_results, python_results) for pxt_fn applied to the test table."""
            res_sql = t.select(out=pxt_fn(t.s, *args, **kwargs)).collect()['out']
            res_py = t.select(out=pxt_fn(t.s.apply(lambda x: x, col_type=pxt.String), *args, **kwargs)).collect()['out']
            return res_sql, res_py

        # ── contains_re ───────────────────────────────────────────────────────
        # Basic literal pattern
        for pat in ['cat', 'dog', 'hello']:
            res_sql, res_py = check_sql_and_py(contains_re, pat)
            expected = [bool(re.search(pat, s)) for s in test_strs]
            assert res_sql == expected, f'contains_re SQL pat={pat!r}'
            assert res_py == expected, f'contains_re Py pat={pat!r}'

        # Regex metacharacters: dot, anchors, character class, alternation
        for pat in ['hel.o', '^cat', '[0-9]+', 'cat|dog']:
            res_sql, res_py = check_sql_and_py(contains_re, pat)
            expected = [bool(re.search(pat, s)) for s in test_strs]
            assert res_sql == expected, f'contains_re SQL pat={pat!r}'
            assert res_py == expected, f'contains_re Py pat={pat!r}'

        # '.' matches any non-newline character; only the empty string has no match
        assert t.where(~contains_re(t.s, '.')).select(t.s).collect()['s'] == ['']

        # flags parameter causes Python fallback — result must still be correct
        res_flags = t.select(out=contains_re(t.s, 'cat', flags=re.IGNORECASE)).collect()['out']
        assert res_flags == [bool(re.search('cat', s, re.IGNORECASE)) for s in test_strs]

        # ── count ─────────────────────────────────────────────────────────────
        for pat in ['[aeiou]', 'a', 'cat', 'zzz']:
            res_sql, res_py = check_sql_and_py(count, pat)
            expected = [len(re.findall(pat, s)) for s in test_strs]
            assert res_sql == expected, f'count SQL pat={pat!r}'
            assert res_py == expected, f'count Py pat={pat!r}'

        # Non-overlapping behaviour: 'aaa' with 'aa' → 1 (not 2)
        res_sql, res_py = check_sql_and_py(count, 'aa')
        expected = [len(re.findall('aa', s)) for s in test_strs]
        assert res_sql == expected, f'count non-overlap SQL: {res_sql} != {expected}'
        assert res_py == expected

        # flags causes Python fallback — result must still be correct
        res_flags = t.select(out=count(t.s, 'cat', flags=re.IGNORECASE)).collect()['out']
        assert res_flags == [len(re.findall('cat', s, re.IGNORECASE)) for s in test_strs]

        # ── match ─────────────────────────────────────────────────────────────
        for pat in ['cat', 'dog', 'hello', '[0-9]+', r'\w+']:
            res_sql, res_py = check_sql_and_py(match, pat)
            expected = [bool(re.match(pat, s)) for s in test_strs]
            assert res_sql == expected, f'match SQL pat={pat!r}'
            assert res_py == expected, f'match Py pat={pat!r}'

        # match only succeeds at the *start*; 'dog food' matches 'dog' but not 'food'
        matched_dog = t.where(match(t.s, 'dog')).select(t.s).collect()['s']
        assert 'dog food' in matched_dog
        assert 'cat dog' not in matched_dog

        # Alternation: (?:cat|dog) must be anchored at start only — 'cat dog' starts with 'cat'
        res_sql, res_py = check_sql_and_py(match, 'cat|dog')
        expected = [bool(re.match('cat|dog', s)) for s in test_strs]
        assert res_sql == expected, f'match alternation SQL ((?:...) anchoring)'
        assert res_py == expected

        # Case-insensitive match (dynamic `case` param)
        res_sql, res_py = check_sql_and_py(match, 'cat', case=False)
        expected = [bool(re.match('cat', s, re.IGNORECASE)) for s in test_strs]
        assert res_sql == expected, f'match case=False SQL'
        assert res_py == expected

        # ── fullmatch ─────────────────────────────────────────────────────────
        for pat in ['cat', 'dog', r'\w+', r'[a-z]+']:
            res_sql, res_py = check_sql_and_py(fullmatch, pat)
            expected = [bool(re.fullmatch(pat, s)) for s in test_strs]
            assert res_sql == expected, f'fullmatch SQL pat={pat!r}'
            assert res_py == expected, f'fullmatch Py pat={pat!r}'

        # fullmatch must match the *entire* string; 'catdog' must NOT fullmatch 'cat'
        assert 'catdog' not in t.where(fullmatch(t.s, 'cat')).select(t.s).collect()['s']
        assert 'cat dog' not in t.where(fullmatch(t.s, 'cat')).select(t.s).collect()['s']

        # Alternation: (?:cat|dog) with end-anchor means 'catdog' must NOT match 'cat|dog'
        # This verifies that '^(?:cat|dog)$' is used, not '^cat|dog$'.
        res_sql, res_py = check_sql_and_py(fullmatch, 'cat|dog')
        expected = [bool(re.fullmatch('cat|dog', s)) for s in test_strs]
        assert res_sql == expected, f'fullmatch alternation SQL ((?:...) anchoring)'
        assert res_py == expected
        fullmatch_alt = set(t.where(fullmatch(t.s, 'cat|dog')).select(t.s).collect()['s'])
        assert fullmatch_alt == {'cat', 'dog'}, f'fullmatch alternation set: {fullmatch_alt}'

        # Case-insensitive fullmatch
        res_sql, res_py = check_sql_and_py(fullmatch, 'cat', case=False)
        expected = [bool(re.fullmatch('cat', s, re.IGNORECASE)) for s in test_strs]
        assert res_sql == expected, f'fullmatch case=False SQL'
        assert res_py == expected

        # ── replace_re ────────────────────────────────────────────────────────
        for pat, repl in [
            ('[aeiou]', '*'),  # vowel replacement
            (r'(\w+)', r'[\1]'),  # backreference
            ('cat', 'feline'),  # literal pattern
            ('zzz', 'XXX'),  # no-match: string unchanged
            (r'\d+', '#'),  # digit replacement
        ]:
            res_sql, res_py = check_sql_and_py(replace_re, pat, repl)
            expected = [re.sub(pat, repl, s) for s in test_strs]
            assert res_sql == expected, f'replace_re SQL pat={pat!r} repl={repl!r}'
            assert res_py == expected, f'replace_re Py pat={pat!r} repl={repl!r}'

        # n=1 limits to first replacement only (falls back to Python); must still be correct
        res_n1 = t.select(out=replace_re(t.s, '[aeiou]', '*', n=1)).collect()['out']
        expected_n1 = [re.sub('[aeiou]', '*', s, count=1) for s in test_strs]
        assert res_n1 == expected_n1, f'replace_re n=1: {res_n1} != {expected_n1}'

        # flags causes Python fallback — result must still be correct
        res_flags = t.select(out=replace_re(t.s, 'cat', 'feline', flags=re.IGNORECASE)).collect()['out']
        assert res_flags == [re.sub('cat', 'feline', s, flags=re.IGNORECASE) for s in test_strs]

    def test_string_splitter(self, uses_db: None) -> None:
        skip_test_if_not_installed('spacy')
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert([{'s': self.TEST_STR}]), expected_rows=1)
        v = pxt.create_view('test_view', t, iterator=string_splitter(t.s, 'sentence'))
        results = v.select(v.text).collect()
        # Verify we got multiple sentences from the TEST_STR
        assert len(results) > 1
        # Verify each result has a 'text' field that is a non-empty string
        for row in results:
            assert isinstance(row['text'], str)
            assert len(row['text']) > 0
