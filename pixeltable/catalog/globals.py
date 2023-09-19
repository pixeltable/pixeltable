from typing import List
import dataclasses
import logging

import sqlalchemy.orm as orm
import sqlalchemy as sql

from pixeltable.metadata import schema
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')

@dataclasses.dataclass
class UpdateStatus:
    num_rows: int = 0
    # TODO: disambiguate what this means: # of slots computed or # of columns computed?
    num_computed_values: int = 0
    num_excs: int = 0
    updated_cols: List[str] = dataclasses.field(default_factory=list)
    cols_with_excs: List[str] = dataclasses.field(default_factory=list)

def init_catalog() -> None:
    """One-time initialization of the catalog. Idempotent."""
    with orm.Session(Env.get().engine, future=True) as session:
        if session.query(sql.func.count(schema.Dir.id)).scalar() > 0:
            return
        # create a top-level directory, so that every schema object has a directory
        dir_md = schema.DirMd(name='')
        dir_record = schema.Dir(parent_id=None, md=dataclasses.asdict(dir_md))
        session.add(dir_record)
        session.flush()
        session.commit()
        _logger.info(f'Initialized catalog')
