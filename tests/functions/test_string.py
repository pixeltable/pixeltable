import re
import textwrap
import unicodedata
from typing import Any, Callable, ClassVar

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

from ..utils import pxt_raises, reload_catalog, skip_test_if_not_installed, validate_update_status


@pxt.udf
def noop_str(s: str) -> str:
    """Identity UDF with no SQL translation; forces Python-path execution."""
    return s


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

    PUSHDOWN_STRS: ClassVar[list[str]] = [
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
        # Padding / justification edge cases
        '',
        ' ',
        'ab',
        'abcdef',
        '  spaced  ',
        # Signed numbers (zfill)
        '-42',
        '+42',
        '42',
        '-',
        '+',
        '--5',
        # UTF-8: CJK, emoji, accented
        '你好',
        '😀',
        'café',
        'hello 🌍 world',
        '😀😀',
        '-😀',
        '+éè',
        # Strings with newlines (validates '.' exclusion from PG pushdown)
        'cat\ndog',
        'hello\nworld',
        'a\nb',
        # Unicode decimals / whitespace (isascii, isdecimal, isspace edge cases)
        '٠',  # noqa: RUF001  # Arabic-Indic zero
        '०',  # noqa: RUF001  # Devanagari zero
        '０',  # noqa: RUF001  # Fullwidth zero
        '\xa0',  # Non-breaking space
        '\u2002',  # En space
        '\u3000',  # Ideographic space
    ]

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

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.select(t.s.index('IBM')).collect()
        assert 'ValueError' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
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

    def _assert_pushdown(
        self,
        t: Any,
        s: Any,
        s_py: Any,
        pxt_fn: pxt.Function,
        *args: Any,
        expected: list[Any],
        label: str,
        **kwargs: Any,
    ) -> None:
        """Assert SQL pushdown and Python fallback paths produce identical expected results."""
        res_sql = t.select(out=pxt_fn(s, *args, **kwargs)).collect()['out']
        res_py = t.select(out=pxt_fn(s_py, *args, **kwargs)).collect()['out']
        assert res_sql == expected, f'{label} SQL'
        assert res_py == expected, f'{label} Python'

    def test_sql_pushdown(self, uses_db: None) -> None:
        """Test all to_sql pushdown implementations for SQL/Python equivalence.

        For each function the test runs two queries:
          1. Normal query -- Pixeltable uses the SQL implementation when possible.
          2. Forced-Python query -- the identity .apply() wrapper prevents SQL pushdown.
        Both must produce the same result as the equivalent Python expression.
        """
        strs = self.PUSHDOWN_STRS
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in strs), expected_rows=len(strs))

        s = t.s
        s_py = t.s.apply(lambda x: x, col_type=pxt.String)  # forces Python fallback

        def chk(fn: pxt.Function, *a: Any, expected: list[Any], label: str, **kw: Any) -> None:
            self._assert_pushdown(t, s, s_py, fn, *a, expected=expected, label=label, **kw)

        # ── isascii / isdecimal / isspace ─────────────────────────────────────
        # isascii has SQL pushdown; isdecimal and isspace do not (PG LC_CTYPE='C' is ASCII-only)
        for pxt_fn, str_fn in [(isascii, str.isascii), (isdecimal, str.isdecimal), (isspace, str.isspace)]:
            res = t.select(out=pxt_fn(s)).collect()['out']
            expected: list[Any] = [str_fn(x) for x in strs]
            assert res == expected, f'{pxt_fn.name} mismatch'

        # ── contains_re ───────────────────────────────────────────────────────
        for pat in ['cat', 'dog', 'hello', '^cat', '[0-9]+', 'cat|dog']:
            chk(contains_re, pat, expected=[bool(re.search(pat, x)) for x in strs], label=f'contains_re {pat!r}')

        # flags parameter causes Python fallback
        res_flags = t.select(out=contains_re(s, 'cat', flags=re.IGNORECASE)).collect()['out']
        assert res_flags == [bool(re.search(r'cat', x, re.IGNORECASE)) for x in strs]

        # ── count ─────────────────────────────────────────────────────────────
        for pat in ['[aeiou]', 'a', 'cat', 'zzz', 'aa']:
            chk(count, pat, expected=[len(re.findall(pat, x)) for x in strs], label=f'count {pat!r}')

        # ── match ─────────────────────────────────────────────────────────────
        for pat in ['cat', 'dog', 'hello', '[0-9]+']:
            chk(match, pat, expected=[bool(re.match(pat, x)) for x in strs], label=f'match {pat!r}')

        # match anchors at start only
        matched_dog = t.where(match(s, 'dog')).select(s).collect()['s']
        assert 'dog food' in matched_dog
        assert 'cat dog' not in matched_dog

        # Alternation anchoring: 'cat|dog' must be wrapped as '^(?:cat|dog)'
        chk(match, 'cat|dog', expected=[bool(re.match(r'cat|dog', x)) for x in strs], label='match alternation')

        # case=False (ASCII pattern)
        chk(
            match,
            'cat',
            case=False,
            expected=[bool(re.match(r'cat', x, re.IGNORECASE)) for x in strs],
            label='match case=False',
        )

        # ── fullmatch ─────────────────────────────────────────────────────────
        for pat in ['cat', 'dog', r'[a-z]+']:
            chk(fullmatch, pat, expected=[bool(re.fullmatch(pat, x)) for x in strs], label=f'fullmatch {pat!r}')

        # fullmatch must match the *entire* string
        assert 'catdog' not in t.where(fullmatch(s, 'cat')).select(s).collect()['s']

        # Alternation anchoring: '^(?:cat|dog)$' not '^cat|dog$'
        fullmatch_alt = set(t.where(fullmatch(s, 'cat|dog')).select(s).collect()['s'])
        assert fullmatch_alt == {'cat', 'dog'}, f'fullmatch alternation: {fullmatch_alt}'

        # case=False (ASCII pattern)
        chk(
            fullmatch,
            'cat',
            case=False,
            expected=[bool(re.fullmatch(r'cat', x, re.IGNORECASE)) for x in strs],
            label='fullmatch case=False',
        )

        # ── replace_re ────────────────────────────────────────────────────────
        for pat, repl in [('[aeiou]', '*'), ('cat', 'feline'), ('zzz', 'XXX')]:
            chk(replace_re, pat, repl, expected=[re.sub(pat, repl, x) for x in strs], label=f'replace_re {pat!r}')

        # n=1 falls back to Python
        res_n1 = t.select(out=replace_re(s, '[aeiou]', '*', n=1)).collect()['out']
        assert res_n1 == [re.sub(r'[aeiou]', '*', x, count=1) for x in strs]

        # ── Patterns that must fall back to Python ────────────────────────────
        # All of these use constructs that PG handles differently. The whitelist
        # rejects them, so both SQL and Python paths run through Python and
        # produce identical, correct results.
        fallback_patterns = [
            # Non-POSIX: \b \B \A \Z \0 (?P<name>...)
            r'\bcat\b',
            r'cat\B',
            r'(?P<name>cat)',
            r'\Acat',
            r'cat\Z',
            r'\0',
            # CATEGORY: \d \w \s (PG LC_CTYPE='C' is ASCII-only)
            r'\w+',
            r'\d+',
            r'\s+',
            # ANY: '.' (PG matches \n, Python does not)
            'hel.o',
            r'a.*b',
            # Inline flags: (?i) (?s) (?m) (PG interprets differently)
            '(?i)cat',
            '(?s)cat',
            '(?m)^cat',
        ]
        for pat in fallback_patterns:
            chk(contains_re, pat, expected=[bool(re.search(pat, x)) for x in strs], label=f'fallback {pat!r}')

        # Verify '.' patterns produce correct results with newline-containing strings
        assert t.select(out=contains_re(s, 'a.b')).collect()['out'] == [bool(re.search(r'a.b', x)) for x in strs], (
            'dot must not match newline (a.b vs a\\nb)'
        )

        # Non-ASCII pattern with case=False must fall back to Python
        chk(
            match,
            'é',
            case=False,
            expected=[bool(re.match(r'é', x, re.IGNORECASE)) for x in strs],
            label='match non-ASCII case=False',
        )

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

    UNICODE_STRS = ('café', 'ñoño', 'ÀÉÎÕÜ')

    def test_unicode_case(self, uses_db: None) -> None:
        """`upper`/`lower`/`capitalize` produce identical, correct results on SQL and Python paths
        for non-ASCII input.
        """
        t = pxt.create_table('test_tbl', {'idx': pxt.Int, 's': pxt.String})
        validate_update_status(
            t.insert({'idx': i, 's': s} for i, s in enumerate(self.UNICODE_STRS)), expected_rows=len(self.UNICODE_STRS)
        )

        test_params: list[tuple[pxt.Function, Callable]] = [
            (upper, str.upper),
            (lower, str.lower),
            (capitalize, str.capitalize),
        ]
        for pxt_fn, str_fn in test_params:
            sql_actual = t.order_by(t.idx).select(out=pxt_fn(t.s)).collect()['out']
            py_actual = t.order_by(t.idx).select(out=pxt_fn(noop_str(t.s))).collect()['out']
            py_expected = [str_fn(s) for s in self.UNICODE_STRS]

            assert py_actual == py_expected, pxt_fn
            assert sql_actual == py_expected, pxt_fn
            assert sql_actual == py_actual, pxt_fn

    def test_unicode_contains_and_filter(self, uses_db: None) -> None:
        """Case-insensitive `contains` and `where(upper(...) == ...)` produce identical, correct
        results on SQL and Python paths for non-ASCII input.
        """
        t = pxt.create_table('test_tbl', {'idx': pxt.Int, 's': pxt.String})
        validate_update_status(
            t.insert({'idx': i, 's': s} for i, s in enumerate(self.UNICODE_STRS)), expected_rows=len(self.UNICODE_STRS)
        )

        needle = 'CAFÉ'
        sql_actual = t.order_by(t.idx).select(out=t.s.contains(needle, case=False)).collect()['out']
        py_actual = t.order_by(t.idx).select(out=noop_str(t.s).contains(needle, case=False)).collect()['out']
        py_expected = [needle.lower() in s.lower() for s in self.UNICODE_STRS]

        assert py_actual == py_expected
        assert sql_actual == py_expected
        assert sql_actual == py_actual

        needle_upper = 'CAFÉ'
        sql_hits = sorted(t.where(upper(t.s) == needle_upper).select(t.s).collect()['s'])
        py_hits = sorted(t.where(upper(noop_str(t.s)) == needle_upper).select(t.s).collect()['s'])
        py_hits_expected = sorted(s for s in self.UNICODE_STRS if s.upper() == needle_upper)

        assert py_hits == py_hits_expected
        assert sql_hits == py_hits_expected
        assert sql_hits == py_hits

    def test_computed_column_vs_live_select_unicode(self, uses_db: None) -> None:
        """Insert-time computed-column materialization (Python) and live `select upper(t.s)` (SQL)
        produce identical results on non-ASCII input, so `where(upper(t.s) == t.upper_s)` is a tautology.
        """
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        t.add_computed_column(upper_s=upper(t.s))
        s = 'café'
        validate_update_status(t.insert([{'s': s}]), expected_rows=1)

        stored = t.select(t.upper_s).collect()['upper_s'][0]
        live = t.select(out=upper(t.s)).collect()['out'][0]

        assert stored == s.upper()
        assert live == s.upper()
        assert stored == live

        tautology_hits = list(t.where(upper(t.s) == t.upper_s).select(t.s).collect()['s'])
        assert tautology_hits == [s]
