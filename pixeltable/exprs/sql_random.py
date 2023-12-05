from __future__ import annotations
from typing import Optional, List, Dict, Any
import pixeltable.type_system as ts
import sqlalchemy as sql

from .expr import Expr
import pixeltable.catalog as catalog

class SqlRandom(Expr):
    def __init__(self, ):
        super().__init__(ts.FloatType())

    def __str__(self) -> str:
        return 'random()'

    def _equals(self, other: 'SqlRandom') -> bool:
        return True
        # TODD: for this to be used outside of order by, we need to
        # clarify semantics, as postgres random() has
        # some odd behavior (used 15.4)
        # NB. postgres random() behaves oddly depending on
        # whether it is called in order by or not.
        # example: comparing the following results:
        # 0. create table foo (row int);
        # insert into foo values (1), (2), (3);
        # 1. select random() as a, random() as b from foo # all a and b ar equal, and consistent with value order
        # 2. select random() as a, random() as b from foo order by random(); # a and b are different.
        # 3. with tmp as (select random() as a, random() as b from foo) # a and b are different, and order is different
        # select * from tmp order by random();

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return sql.sql.functions.random()

    def eval(self, data_row: catalog.DataRow, row_builder : Any) -> None:
        assert False, 'SqlRandom should be evaluated in the database'
        return None

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        return cls()
