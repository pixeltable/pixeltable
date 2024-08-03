import re

import pytest

import pixeltable as pxt
from ..utils import validate_update_status


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
            """

    def test_all(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))

        from pixeltable.functions.string import \
            capitalize, casefold, center, count, endswith, find, isalnum, isalpha, isascii, isdecimal, isdigit, \
            isidentifier, islower,  isnumeric, isupper, isspace, istitle, ljust, lower, lstrip, rfind, \
            rjust, rstrip, startswith, strip, swapcase, title, upper, zfill
        test_params = [  # (pxt_fn, str_fn, args, kwargs)
            (capitalize, str.capitalize, [], {}),
            (casefold, str.casefold, [], {}),
            (center, str.center, [100], {}),
            (count, str.count, ['relation'], {}),
            (endswith, str.endswith, ['1970'], {}),
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

        _ = t.select(out=count(t.s, 'relation')).collect()

        assert t.select(out=count(t.s, 'relation')).collect()['out'] == \
               [str.count(s, 'relation') for s in test_strs]

        for pxt_fn, str_fn, args, kwargs in test_params:
            try:
                assert t.select(out=pxt_fn(t.s, *args, **kwargs)).collect()['out'] == \
                    [str_fn(s, *args, **kwargs) for s in test_strs], \
                    pxt_fn
            except Exception as e:
                print(pxt_fn)
                raise e

    def test_removeprefix(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import removeprefix
        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=removeprefix(t.s, 'Codd')).collect()
        for row in res:
            if row['s'].startswith('Codd'):
                assert row['out'] == row['s'][4:]
            else:
                assert row['out'] == row['s']

    def test_removesuffix(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import removesuffix
        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=removesuffix(t.s, '1970')).collect()
        for row in res:
            if row['s'].endswith('1970'):
                assert row['out'] == row['s'][:-4]
            else:
                assert row['out'] == row['s']

    def test_replace(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import replace, contains
        # count() doesn't yet support non-SQL Where clauses
        n = len(t.where(contains(t.s, 'Codd')).collect())
        t['s2'] = replace(t.s, 'Codd', 'Mohan')
        m = len(t.where(contains(t.s2, 'Mohan')).collect())
        assert n == m
        t['s3'] = replace(t.s, 'C.dd', 'Mohan', regex=True)
        o = len(t.where(contains(t.s3, 'Mohan')).collect())
        assert n == o

    def test_slice_replace(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import slice_replace, contains
        # count() doesn't yet support non-SQL Where clauses
        res = t.select(t.s, out=slice_replace(t.s, 50, 51, 'abc')).collect()
        for row in res:
            assert row['out'] == row['s'][:50] + 'abc' + row['s'][51:]

    def test_partition(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import partition, contains
        # count() doesn't yet support non-SQL Where clauses
        status = t.add_column(parts=partition(t.s, 'IBM'))
        assert status.num_excs == 0
        res = t.where(contains(t.s, 'IBM')).select(t.s, t.parts).collect()
        for row in res:
            assert len(row['parts']) == 3
            assert len(row['parts'][0]) == row['s'].find('IBM')
            assert row['parts'][1] == 'IBM'

    def test_rpartition(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import rpartition, contains
        # count() doesn't yet support non-SQL Where clauses
        status = t.add_column(parts=rpartition(t.s, 'IBM'))
        assert status.num_excs == 0
        res = t.where(contains(t.s, 'IBM')).select(t.s, t.parts).collect()
        for row in res:
            assert len(row['parts']) == 3
            assert len(row['parts'][0]) == row['s'].rfind('IBM')
            assert row['parts'][1] == 'IBM'

    def test_wrap(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import fill, wrap
        import textwrap
        res = t.select(t.s, out=fill(t.s, 5)).collect()
        for row in res:
            assert row['out'] == textwrap.fill(row['s'], 5)
        res = t.select(t.s, out=wrap(t.s, 5)).collect()
        for row in res:
            assert row['out'] == textwrap.wrap(row['s'], 5)

    def test_slice(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import slice
        res = t.select(t.s, out=slice(t.s, 0, 4)).collect()
        for row in res:
            assert row['out'] == row['s'][0:4]

    def test_match(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import match
        # count() doesn't yet support non-SQL Where clauses
        assert len(t.where(match(t.s, 'Codd')).collect()) == 2
        assert len(t.where(match(t.s, 'codd', case=False)).collect()) == 2

    def test_fullmatch(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import fullmatch
        # count() doesn't yet support non-SQL Where clauses
        assert len(t.where(fullmatch(t.s, 'F')).collect()) == 1
        assert len(t.where(fullmatch(t.s, 'f', case=False)).collect()) == 1

    def test_pad(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import pad
        res = t.select(t.s, out=pad(t.s, width=100, side='both')).collect()
        for row in res:
            assert row['out'] == row['s'].center(100)

    def test_normalize(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import normalize
        import unicodedata
        res = t.select(t.s, out=normalize(t.s, 'NFC')).collect()
        for row in res:
            assert row['out'] == unicodedata.normalize('NFC', row['s'])

    def test_repeat(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType(), 'n': pxt.IntType()})
        strs = ['a', 'b', 'c', 'd', 'e']
        validate_update_status(t.insert({'s': s, 'n': n} for n, s in enumerate(strs)), expected_rows=len(strs))
        from pixeltable.functions.string import repeat
        res = t.select(t.s, t.n, out=repeat(t.s, t.n)).collect()
        for row in res:
            assert row['out'] == row['s'] * row['n']

    def test_contains(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import contains

        assert t.select(out=contains(t.s, 'IBM')).collect()['out'] == ['IBM' in s for s in test_strs]
        assert t.select(out=contains(t.s, 'IBM', regex=False)).collect()['out'] == ['IBM' in s for s in test_strs]
        assert t.select(out=contains(t.s, 'ibm', regex=False, case=True)).collect()['out'] == \
               ['ibm' in s for s in test_strs]
        assert t.select(out=contains(t.s, 'ibm', regex=False, case=False)).collect()['out'] == \
               ['ibm' in s.lower() for s in test_strs]
        assert t.select(out=contains(t.s, 'ibm', regex=True, flags=re.IGNORECASE)).collect()['out'] == \
               ['ibm' in s.lower() for s in test_strs]
        assert t.select(out=contains(t.s, 'i.m', regex=True, flags=re.IGNORECASE)).collect()['out'] >= \
               ['ibm' in s.lower() for s in test_strs]

    def test_index(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'s': pxt.StringType()})
        test_strs = self.TEST_STR.split('. ')
        validate_update_status(t.insert({'s': s} for s in test_strs), expected_rows=len(test_strs))
        from pixeltable.functions.string import index, rindex, contains

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.select(index(t.s, 'IBM')).collect()
        assert 'ValueError' in str(exc_info.value)

        with pytest.raises(pxt.Error) as exc_info:
            _ = t.select(rindex(t.s, 'IBM')).collect()
        assert 'ValueError' in str(exc_info.value)

        res = t.where(contains(t.s, 'IBM')).select(t.s, idx=index(t.s, 'IBM')).collect()
        for s, idx in zip(res['s'], res['idx']):
            assert s[idx:idx + 3] == 'IBM'

        res = t.where(contains(t.s, 'IBM')).select(t.s, idx=rindex(t.s, 'IBM')).collect()
        for s, idx in zip(res['s'], res['idx']):
            assert s[idx:idx + 3] == 'IBM'

    def test_format(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'input': pxt.StringType()})
        from pixeltable.functions.string import format

        t.add_column(s1=format('ABC {0}', t.input))
        t.add_column(s2=format('DEF {this}', this=t.input))
        t.add_column(s3=format('GHI {0} JKL {this}', t.input, this=t.input))
        status = t.insert(input='MNO')
        assert status.num_rows == 1
        assert status.num_excs == 0
        row = t.head()[0]
        assert row == {'input': 'MNO', 's1': 'ABC MNO', 's2': 'DEF MNO', 's3': 'GHI MNO JKL MNO'}