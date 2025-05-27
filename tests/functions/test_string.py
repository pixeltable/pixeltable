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
    count,
    endswith,
    find,
    format,
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
    rfind,
    rjust,
    rstrip,
    startswith,
    strip,
    swapcase,
    title,
    upper,
    zfill,
)

from ..utils import reload_catalog, validate_update_status


class TestString:
    TEST_STR = """
        The concept of relational database was defined by E. F. Codd at IBM in 1970. Codd introduced the term relational in his
        research paper "A Relational Model of Data for Large Shared Data Banks". In this paper and later papers, he defined
        what he meant by relation. One well-known definition of what constitutes a relational database system is composed of
        Codd's 12 rules. However, no commercial implementations of the relational model conform to all of Codd's rules, so
        the term has gradually come to describe a broader class of database systems, which at a minimum:
        Present the data to the user as relations (a presentation in tabular form, i.e. as a collection of tables with each
        table consisting of a set of rows and columns);
        Provide relational operators to manipulate the data in tabular form.
        In 1974, IBM began developing System R, a research project to develop a prototype RDBMS. The first system sold as
        an RDBMS was Multics Relational Data Store (June 1976). Oracle was released in 1979 by Relational
        Software, now Oracle Corporation. Ingres and IBM BS12 followed. Other examples of an RDBMS include IBM Db2, SAP
        Sybase ASE, and Informix. In 1984, the first RDBMS for Macintosh began being developed, code-named Silver Surfer,
        and was released in 1987 as 4th Dimension and known today as 4D.
        The first systems that were relatively faithful implementations of the relational model were from:
        University of Michigan – Micro DBMS (1969)
        Massachusetts Institute of Technology (1971)]
        IBM UK Scientific Centre at Peterlee – IS1 (1970–72), and its successor, PRTV (1973–79).
        """  # noqa: RUF001

    TEST_STRS = textwrap.dedent(TEST_STR.strip()).split('. ')

    [
        'The concept of relational database was defined by E. F. Codd at IBM in 1970.',
        'Codd introduced the term relational in his research paper\n'
        '"A Relational Model of Data for Large Shared Data Banks".',
        'In this paper and later papers, he defined\nwhat he meant by relation.',
        '   White\n\nSpace\n\n\n',
        r'%%!!#__\\Symbols%%!!#\\@@__%',
    ]

    def test_all(self, reset_db: None) -> None:
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
            (rfind, str.rfind, ['relation', 10, -10], {}),
            (rjust, str.rjust, [100], {}),
            (rstrip, str.rstrip, [], {}),
            (startswith, str.startswith, ['Codd'], {}),
            (strip, str.strip, [], {}),
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

    def test_removeprefix(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=t.s.removeprefix('Codd')).collect()
        for row in res:
            if row['s'].startswith('Codd'):
                assert row['out'] == row['s'][4:]
            else:
                assert row['out'] == row['s']

    def test_removesuffix(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=t.s.removesuffix('1970')).collect()
        for row in res:
            if row['s'].endswith('1970'):
                assert row['out'] == row['s'][:-4]
            else:
                assert row['out'] == row['s']

    def test_replace(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        n = len(t.where(t.s.contains('Codd')).collect())
        t.add_computed_column(s2=t.s.replace('Codd', 'Mohan'))
        m = len(t.where(t.s2.contains('Mohan')).collect())
        assert n == m
        t.add_computed_column(s3=t.s.replace('C.dd', 'Mohan', regex=True))
        o = len(t.where(t.s3.contains('Mohan')).collect())
        assert n == o

    def test_slice_replace(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=t.s.slice_replace(50, 51, 'abc')).collect()
        for row in res:
            assert row['out'] == row['s'][:50] + 'abc' + row['s'][51:]

    def test_partition(self, reset_db: None) -> None:
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

    def test_rpartition(self, reset_db: None) -> None:
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

    def test_wrap(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        res = t.select(t.s, out=t.s.fill(5)).collect()
        for row in res:
            assert row['out'] == textwrap.fill(row['s'], 5)
        res = t.select(t.s, out=t.s.wrap(5)).collect()
        for row in res:
            assert row['out'] == textwrap.wrap(row['s'], 5)

    def test_slice(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))
        res = t.select(t.s, out=t.s.slice(0, 4)).collect()
        for row in res:
            assert row['out'] == row['s'][0:4]

    def test_match(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        assert len(t.where(t.s.match('Codd')).collect()) == 2
        assert len(t.where(t.s.match('codd', case=False)).collect()) == 2

    def test_fullmatch(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        # count() doesn't yet support non-SQL Where clauses
        assert len(t.where(t.s.fullmatch('F')).collect()) == 1
        assert len(t.where(t.s.fullmatch('f', case=False)).collect()) == 1

    def test_pad(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))
        res = t.select(t.s, out=t.s.pad(width=100, side='both')).collect()
        for row in res:
            assert row['out'] == row['s'].center(100)

    def test_normalize(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))

        res = t.select(t.s, out=t.s.normalize('NFC')).collect()
        for row in res:
            assert row['out'] == unicodedata.normalize('NFC', row['s'])

    def test_repeat(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String, 'n': pxt.Int})
        strs = ['a', 'b', 'c', 'd', 'e']
        validate_update_status(t.insert({'s': s, 'n': n} for n, s in enumerate(strs)), expected_rows=len(strs))
        res = t.select(t.s, t.n, out=t.s.repeat(t.n)).collect()
        for row in res:
            assert row['out'] == row['s'] * row['n']

    def testcontains(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.String})
        validate_update_status(t.insert({'s': s} for s in self.TEST_STRS), expected_rows=len(self.TEST_STRS))
        assert t.select(out=t.s.contains('IBM')).collect()['out'] == ['IBM' in s for s in self.TEST_STRS]
        assert t.select(out=t.s.contains('IBM', regex=False)).collect()['out'] == ['IBM' in s for s in self.TEST_STRS]
        assert t.select(out=t.s.contains('ibm', regex=False, case=True)).collect()['out'] == [
            'ibm' in s for s in self.TEST_STRS
        ]
        assert t.select(out=t.s.contains('ibm', regex=False, case=False)).collect()['out'] == [
            'ibm' in s.lower() for s in self.TEST_STRS
        ]
        assert t.select(out=t.s.contains('ibm', regex=True, flags=re.IGNORECASE)).collect()['out'] == [
            'ibm' in s.lower() for s in self.TEST_STRS
        ]
        assert t.select(out=t.s.contains('i.m', regex=True, flags=re.IGNORECASE)).collect()['out'] >= [
            'ibm' in s.lower() for s in self.TEST_STRS
        ]

    def test_index(self, reset_db: None) -> None:
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

    def test_format(self, reset_db: None) -> None:
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
