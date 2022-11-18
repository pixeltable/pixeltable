import pathlib
from typing import Optional, List, Set, Dict, Any, Type, Union, Callable
import re

import PIL
import numpy as np
from PIL import Image
from tqdm import tqdm

import pandas as pd
import sqlalchemy as sql
import sqlalchemy.orm as orm

from pixeltable import store, env
from pixeltable import exceptions as exc
from pixeltable.type_system import ColumnType
from pixeltable.utils import clip
from pixeltable.index import VectorIndex


_ID_RE = r'[a-zA-Z]\w*'
_PATH_RE = f'{_ID_RE}(\\.{_ID_RE})*'


class Column:
    def __init__(
            self, name: str, t: ColumnType, primary_key: bool = False, nullable: bool = True,
            col_id: Optional[int] = None):
        self.name = name
        self.col_type = t
        self.id = col_id
        self.primary_key = primary_key
        self.nullable = nullable
        if self.id is not None:
            self.sa_col = sql.Column(self.storage_name(), self.col_type.to_sa_type(), nullable=self.nullable)
        self.idx: Optional[VectorIndex] = None

    def to_sql(self) -> str:
        return f'{self.storage_name()} {self.col_type.to_sql()}'

    def set_id(self, col_id: int) -> None:
        self.id = col_id
        self.sa_col = sql.Column(self.storage_name(), self.col_type.to_sa_type(), nullable=self.nullable)

    def set_idx(self, idx: VectorIndex) -> None:
        self.idx = idx

    def storage_name(self) -> str:
        assert self.id is not None
        return f'col_{self.id}'

    def __str__(self) -> str:
        return f'{self.name}: {self.col_type.name}'


# base class of all addressable objects within a Db
class SchemaObject:
    def __init__(self, obj_id: int):
        self.id = obj_id

    @classmethod
    def display_name(cls) -> str:
        """
        Return name displayed in error messages.
        """
        assert False


class DirBase(SchemaObject):
    def __init__(self, dir_id: int):
        super().__init__(dir_id)

    @classmethod
    def display_name(cls) -> str:
        return 'directory'


# contains only MutableTables
class Dir(DirBase):
    def __init__(self, dir_id: int):
        super().__init__(dir_id)


# contains only TableSnapshots
class SnapshotDir(DirBase):
    def __init__(self, dir_id: int):
        super().__init__(dir_id)


class Table(SchemaObject):
    #def __init__(self, tbl_record: store.Table, schema: List[Column]):
    def __init__(self, db_id: int, tbl_id: int, dir_id: int, name: str, version: int, cols: List[Column]):
        super().__init__(tbl_id)
        self.db_id = db_id
        self.dir_id = dir_id
        # TODO: this will be out-of-date after a rename()
        self.name = name
        for pos, col in enumerate(cols):
            if re.fullmatch(_ID_RE, col.name) is None:
                raise exc.BadFormatError(f"Invalid column name: '{col.name}'")
            assert col.id is not None
        self.cols = cols
        self.cols_by_name = {col.name: col for col in cols}
        self.version = version

        # we can't call _load_valid_rowids() here because the storage table may not exist yet
        self.valid_rowids: Set[int] = set()

        # sqlalchemy-related metadata; used to insert and query the storage table
        self.sa_md = sql.MetaData()
        self._create_sa_tbl()
        self.is_dropped = False

    def _load_valid_rowids(self) -> None:
        if not any(col.col_type == ColumnType.IMAGE for col in self.cols):
            return
        with env.get_engine().connect() as conn:
            with conn.begin():
                stmt = sql.select(self.rowid_col) \
                    .where(self.v_min_col <= self.version) \
                    .where(self.v_max_col > self.version)
                rows = conn.execute(stmt)
                for row in rows:
                    rowid = row[0]
                    self.valid_rowids.add(rowid)

    def __getattr__(self, col_name: str) -> 'pixeltable.exprs.ColumnRef':
        if col_name not in self.cols_by_name:
            raise AttributeError(f'Column {col_name} unknown')
        col = self.cols_by_name[col_name]
        from pixeltable.exprs import ColumnRef
        return ColumnRef(col)

    def __getitem__(self, index: object) -> Union['pixeltable.exprs.ColumnRef', 'pixeltable.dataframe.DataFrame']:
        if isinstance(index, str):
            # basically <tbl>.<colname>
            return self.__getattr__(index)
        from pixeltable.dataframe import DataFrame
        return DataFrame(self).__getitem__(index)

    def df(self) -> 'pixeltable.dataframe.DataFrame':
        # local import: avoid circular imports
        from pixeltable.dataframe import DataFrame
        return DataFrame(self)

    def show(self, *args, **kwargs) -> 'pixeltable.dataframe.DataFrameResultSet':  # type: ignore[name-defined, no-untyped-def]
        from pixeltable.dataframe import DataFrame
        return self.df().show(*args, **kwargs)

    def count(self) -> int:
        from pixeltable.dataframe import DataFrame
        return self.df().count()

    def columns(self) -> List[Column]:
        return self.cols

    def storage_name(self) -> str:
        return f'tbl_{self.id}'

    def _check_is_dropped(self) -> None:
        if self.is_dropped:
            raise exc.OperationalError('Table has been dropped')

    def _create_sa_tbl(self) -> None:
        self.rowid_col = sql.Column('rowid', sql.BigInteger, nullable=False)
        self.v_min_col = sql.Column('v_min', sql.BigInteger, nullable=False)
        self.v_max_col = \
            sql.Column('v_max', sql.BigInteger, nullable=False, server_default=str(store.Table.MAX_VERSION))
        sa_cols = [self.rowid_col, self.v_min_col, self.v_max_col]
        sa_cols.extend([col.sa_col for col in self.cols])
        self.sa_tbl = sql.Table(self.storage_name(), self.sa_md, *sa_cols)

    @classmethod
    def _vector_idx_name(cls, tbl_id: int, col: Column) -> str:
        return f'{tbl_id}_{col.id}'

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def load_cols(cls, tbl_id: int, schema_version: int, session: orm.Session) -> List[Column]:
        col_records = session.query(store.SchemaColumn) \
            .where(store.SchemaColumn.tbl_id == tbl_id) \
            .where(store.SchemaColumn.schema_version == schema_version) \
            .order_by(store.SchemaColumn.pos.asc()).all()
        cols = [
            Column(r.name, r.col_type, primary_key=r.is_pk, nullable=r.is_nullable, col_id=r.col_id)
            for r in col_records
        ]
        for col in cols:
            if col.col_type == ColumnType.IMAGE:
                col.set_idx(VectorIndex.load(cls._vector_idx_name(tbl_id, col), dim=512))
        return cols


class TableSnapshot(Table):
    def __init__(self, snapshot_record: store.TableSnapshot, cols: List[Column]):
        assert snapshot_record.db_id is not None
        assert snapshot_record.id is not None
        assert snapshot_record.dir_id is not None
        assert snapshot_record.name is not None
        assert snapshot_record.tbl_version is not None
        super().__init__(
            snapshot_record.db_id, snapshot_record.id, snapshot_record.dir_id, snapshot_record.name,
            snapshot_record.tbl_version, cols)
        # it's safe to call _load_valid_rowids() here because the storage table already exists
        self._load_valid_rowids()

    def __repr__(self) -> str:
        return f'TableSnapshot(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table snapshot'

class MutableTable(Table):
    def __init__(self, tbl_record: store.Table, schema_version: int, cols: List[Column]):
        assert tbl_record.db_id is not None
        assert tbl_record.id is not None
        assert tbl_record.dir_id is not None
        assert tbl_record.name is not None
        assert tbl_record.current_version is not None
        super().__init__(
            tbl_record.db_id, tbl_record.id, tbl_record.dir_id, tbl_record.name, tbl_record.current_version, cols)
        assert tbl_record.next_col_id is not None
        self.next_col_id = tbl_record.next_col_id
        assert tbl_record.next_row_id is not None
        self.next_row_id = tbl_record.next_row_id
        self.schema_version = schema_version

    def __repr__(self) -> str:
        return f'MutableTable(name={self.name})'

    @classmethod
    def display_name(cls) -> str:
        return 'table'

    def add_column(self, c: Column) -> None:
        self._check_is_dropped()
        if re.fullmatch(_ID_RE, c.name) is None:
            raise exc.BadFormatError(f"Invalid column name: '{c.name}'")
        if c.name in self.cols_by_name:
            raise exc.DuplicateNameError(f'Column {c.name} already exists')
        assert self.next_col_id is not None
        c.id = self.next_col_id
        self.next_col_id += 1
        self.cols.append(c)
        self.cols_by_name[c.name] = c

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with env.get_engine().connect() as conn:
            with conn.begin():
                conn.execute(
                    sql.update(store.Table.__table__)
                        .values({
                            store.Table.current_version: self.version,
                            store.Table.current_schema_version: self.schema_version,
                            store.Table.next_col_id: self.next_col_id
                        })
                        .where(store.Table.id == self.id))
                conn.execute(
                    sql.insert(store.TableSchemaVersion.__table__)
                        .values(
                            tbl_id=self.id, schema_version=self.schema_version,
                            preceding_schema_version=preceding_schema_version))
                conn.execute(
                    sql.insert(store.StorageColumn.__table__)
                        .values(tbl_id=self.id, col_id=c.id, schema_version_add=self.schema_version))
                self._create_col_md(conn)
                stmt = f'ALTER TABLE {self.storage_name()} ADD COLUMN {c.to_sql()}'
                conn.execute(sql.text(stmt))

    def drop_column(self, name: str) -> None:
        self._check_is_dropped()
        if name not in self.cols_by_name:
            raise exc.UnknownEntityError
        col = self.cols_by_name[name]
        self.cols.remove(col)
        del self.cols_by_name[name]

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with env.get_engine().connect() as conn:
            with conn.begin():
                conn.execute(
                    sql.update(store.Table.__table__)
                        .values({
                            store.Table.current_version: self.version,
                            store.Table.current_schema_version: self.schema_version
                        })
                        .where(store.Table.id == self.id))
                conn.execute(
                    sql.insert(store.TableSchemaVersion.__table__)
                        .values(
                            tbl_id=self.id, schema_version=self.schema_version,
                            preceding_schema_version=preceding_schema_version))
                conn.execute(
                    sql.update(store.StorageColumn.__table__)
                        .values({store.StorageColumn.schema_version_drop: self.schema_version})
                        .where(store.StorageColumn.tbl_id == self.id)
                        .where(store.StorageColumn.col_id == col.id))
                self._create_col_md(conn)

    def rename_column(self, old_name: str, new_name: str) -> None:
        self._check_is_dropped()
        if old_name not in self.cols_by_name:
            raise exc.UnknownEntityError(f'Unknown column: {old_name}')
        if re.fullmatch(_ID_RE, new_name) is None:
            raise exc.BadFormatError(f"Invalid column name: '{new_name}'")
        if new_name in self.cols_by_name:
            raise exc.DuplicateNameError(f'Column {new_name} already exists')
        col = self.cols_by_name[old_name]
        del self.cols_by_name[old_name]
        col.name = new_name
        self.cols_by_name[new_name] = col

        # we're creating a new schema version
        self.version += 1
        preceding_schema_version = self.schema_version
        self.schema_version = self.version

        with env.get_engine().connect() as conn:
            with conn.begin():
                conn.execute(
                    sql.update(store.Table.__table__)
                        .values({
                            store.Table.current_version: self.version,
                            store.Table.current_schema_version: self.schema_version
                        })
                        .where(store.Table.id == self.id))
                conn.execute(
                    sql.insert(store.TableSchemaVersion.__table__)
                        .values(tbl_id=self.id, schema_version=self.schema_version,
                                preceding_schema_version=preceding_schema_version))
                self._create_col_md(conn)

    def _create_col_md(self, conn: sql.engine.base.Connection) -> None:
        for pos, c in enumerate(self.cols):
            conn.execute(
                sql.insert(store.SchemaColumn.__table__)
                .values(tbl_id=self.id, schema_version=self.version, col_id=c.id,
                        pos=pos, name=c.name, col_type=c.col_type, is_nullable=c.nullable, is_pk=c.primary_key))

    def insert_pandas(self, data: pd.DataFrame) -> None:
        self._check_is_dropped()
        reqd_col_names = set([col.name for col in self.cols if not col.nullable])
        given_col_names = set(data.columns)
        if not(reqd_col_names <= given_col_names):
            raise exc.InsertError(f'Missing columns: {", ".join(reqd_col_names - given_col_names)}')

        # check types
        all_col_names = set([col.name for col in self.cols])
        inserted_cols = [self.cols_by_name[name] for name in all_col_names & given_col_names]
        for col in inserted_cols:
            if col.col_type == ColumnType.STRING and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires string data')
            if col.col_type == ColumnType.INT and not pd.api.types.is_integer_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires integer data')
            if col.col_type == ColumnType.FLOAT and not pd.api.types.is_numeric_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires numerical data')
            if col.col_type == ColumnType.BOOL and not pd.api.types.is_bool_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires boolean data')
            if col.col_type == ColumnType.TIMESTAMP \
                    and not pd.api.types.is_datetime64_any_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires datetime data')
            if col.col_type == ColumnType.IMAGE and not pd.api.types.is_string_dtype(data.dtypes[col.name]):
                raise exc.InsertError(f'Column {col.name} requires local file paths')

        # check image data and build index
        image_cols = [col for col in inserted_cols if col.col_type == ColumnType.IMAGE]
        print('creating index')
        rowids = range(self.next_row_id, self.next_row_id + len(data))
        for col in image_cols:
            embeddings = np.zeros((len(data), 512))
            for i, (_, path_str) in tqdm(enumerate(data[col.name].items())):
                try:
                    img = Image.open(path_str)
                    embeddings[i] = clip.encode_image(img)
                except FileNotFoundError:
                    raise exc.OperationalError(f'Column {col.name}: file does not exist: {path_str}')
                except PIL.UnidentifiedImageError:
                    raise exc.OperationalError(f'Column {col.name}: not a valid image file: {path_str}')
            assert col.idx is not None
            col.idx.insert(embeddings, np.array(rowids))

        # we're creating a new version
        self.version += 1
        # construct new df with the storage column names, in order to iterate over it more easily
        stored_data = {col.storage_name(): data[col.name] for col in inserted_cols}
        stored_data_df = pd.DataFrame(data=stored_data)
        insert_values: List[Dict[str, Any]] = []
        print('preparing data ')
        for i, row in tqdm(enumerate(stored_data_df.itertuples(index=False))):
            insert_values.append({'rowid': rowids[i], 'v_min': self.version, **row._asdict()})

        with env.get_engine().connect() as conn:
            with conn.begin():
                conn.execute(sql.insert(self.sa_tbl), insert_values)
                self.next_row_id += len(data)
                conn.execute(
                    sql.update(store.Table.__table__)
                        .values({store.Table.current_version: self.version, store.Table.next_row_id: self.next_row_id})
                        .where(store.Table.id == self.id))

        self.valid_rowids.update(rowids)

    def insert_csv(self, file_path: str) -> None:
        pass

    # TODO: update() signature?
    #def update(self, data: pd.DataFrame) -> None:

    # TODO: delete() signature?
    #def delete(self, data: DataFrame) -> None:

    def revert(self) -> None:
        self._check_is_dropped()
        if self.version == 0:
            raise exc.OperationalError('Cannot revert version 0')
        # check if the current version is referenced by a snapshot
        with orm.Session(env.get_engine()) as session:
            # make sure we don't have a snapshot referencing this version
            num_references = session.query(sql.func.count(store.TableSnapshot.id)) \
                .where(store.TableSnapshot.db_id == self.db_id) \
                .where(store.TableSnapshot.tbl_id == self.id) \
                .where(store.TableSnapshot.tbl_version == self.version) \
                .scalar()
            if num_references > 0:
                raise exc.OperationalError(
                    f'Current version is needed for {num_references} snapshot{"s" if num_references > 1 else ""}')

            conn = session.connection()
            # delete newly-added data
            conn.execute(sql.delete(self.sa_tbl).where(self.sa_tbl.c.v_min == self.version))
            # revert new deletions
            conn.execute(
                sql.update(self.sa_tbl).values({self.sa_tbl.c.v_max: store.Table.MAX_VERSION})
                    .where(self.sa_tbl.c.v_max == self.version))

            if self.version == self.schema_version:
                # the current version involved a schema change:
                # we need to determine the preceding schema version and reload the schema
                preceding_schema_version = session.query(store.TableSchemaVersion.preceding_schema_version) \
                    .where(store.TableSchemaVersion.tbl_id == self.id) \
                    .where(store.TableSchemaVersion.schema_version == self.schema_version) \
                    .scalar()
                self.cols = self.load_cols(self.id, preceding_schema_version, session)
                conn.execute(
                    sql.delete(store.TableSchemaVersion.__table__)
                        .where(store.TableSchemaVersion.tbl_id == self.id)
                        .where(store.TableSchemaVersion.schema_version == self.schema_version))
                self.schema_version = preceding_schema_version

            conn.execute(
                sql.update(store.Table.__table__)
                    .values({
                        store.Table.current_version: self.version,
                        store.Table.current_schema_version: self.schema_version
                    })
                    .where(store.Table.id == self.id))

            session.commit()
        self.version -= 1

    # MODULE-LOCAL, NOT PUBLIC
    def rename(self, new_name: str) -> None:
        self._check_is_dropped()
        with env.get_engine().connect() as conn:
            with conn.begin():
                conn.execute(
                    sql.update(store.Table.__table__).values({store.Table.name: new_name})
                        .where(store.Table.id == self.id))

    # MODULE-LOCAL, NOT PUBLIC
    def drop(self) -> None:
        self._check_is_dropped()
        with env.get_engine().connect() as conn:
            with conn.begin():
                conn.execute(
                    sql.update(store.Table.__table__).values({store.Table.is_mutable: False})
                        .where(store.Table.id == self.id))

    # MODULE-LOCAL, NOT PUBLIC
    @classmethod
    def create(
        cls, db_id: int, dir_id: int, name: str, num_retained_versions: int, cols: List[Column]) -> 'MutableTable':
        # make sure col names are unique ids (within the table)j
        col_names: Set[str] = set()
        for col_name in [c.name for c in cols]:
            if re.fullmatch(_ID_RE, col_name) is None:
                raise exc.BadFormatError(f"Invalid column name: '{col_name}'")
            if col_name in col_names:
                raise exc.DuplicateNameError(f'Duplicate column: {col_name}')
            col_names.add(col_name)

        with orm.Session(env.get_engine()) as session:
            tbl_record = store.Table(
                db_id=db_id, dir_id=dir_id, name=name, num_retained_versions=num_retained_versions, current_version=0,
                current_schema_version=0, is_mutable=True, next_col_id=len(cols), next_row_id=0)
            session.add(tbl_record)
            session.flush()  # sets tbl_record.id

            tbl_version_record = store.TableSchemaVersion(
                tbl_id=tbl_record.id, schema_version=0, preceding_schema_version=0)
            session.add(tbl_version_record)

            for pos, col in enumerate(cols):
                col.set_id(pos)
                session.add(store.StorageColumn(tbl_id=tbl_record.id, col_id=col.id, schema_version_add=0))
                session.add(
                    store.SchemaColumn(
                        tbl_id=tbl_record.id, schema_version=0, col_id=col.id, pos=pos, name=col.name,
                        col_type=col.col_type, is_nullable=col.nullable, is_pk=col.primary_key)
                )

                # for image cols, add VectorIndex for kNN search
                if col.col_type == ColumnType.IMAGE:
                    col.set_idx(VectorIndex.create(Table._vector_idx_name(tbl_record.id, col), 512))
            session.flush()

            assert tbl_record.id is not None
            tbl = MutableTable(tbl_record, 0, cols)
            tbl.sa_md.create_all(bind=session.connection())
            session.commit()
            return tbl


class Path:
    def __init__(self, path: str, empty_is_valid: bool=False):
        if path == '' and not empty_is_valid or path != '' and re.fullmatch(_PATH_RE, path) is None:
            raise exc.BadFormatError(f"Invalid path format: '{path}'")
        self.components = path.split('.')

    @property
    def len(self) -> int:
        return 0 if self.is_root else len(self.components)

    @property
    def name(self) -> str:
        assert len(self.components) > 0
        return self.components[-1]

    @property
    def is_root(self) -> bool:
        return self.components[0] == ''

    @property
    def parent(self) -> 'Path':
        if len(self.components) == 1:
            if self.is_root:
                return self
            else:
                return Path('', empty_is_valid=True)
        else:
            return Path('.'.join(self.components[:-1]))

    def append(self, name: str) -> 'Path':
        if self.is_root:
            return Path(name)
        else:
            return Path(f'{str(self)}.{name}')

    def is_ancestor(self, other: 'Path', is_parent: bool = False) -> bool:
        """
        True if self as an ancestor path of other.
        """
        if self.len >= other.len or other.is_root:
            return False
        if self.is_root and (other.len == 1 or not is_parent):
            return True
        is_prefix = self.components == other.components[:self.len]
        return is_prefix and (self.len == (other.len - 1) or not is_parent)

    def __str__(self) -> str:
        return '.'.join(self.components)


class PathDict:
    def __init__(self) -> None:
        # *not* Dict[Path, SchemaObject]
        self.paths: Dict[str, SchemaObject] = {}  # all paths

    def __getitem__(self, path: Path) -> SchemaObject:
        return self.paths[str(path)]

    def __setitem__(self, path: Path, val: SchemaObject) -> None:
        self.paths[str(path)] = val

    def __delitem__(self, path: Path) -> None:
        del self.paths[str(path)]

    def update(self, paths: Dict[str, SchemaObject]) -> None:
        self.paths.update(paths)

    # checks that the parent of path exists and is a Dir
    # and that the object of path has 'expected' type
    def check_is_valid(
            self, path: Path, expected: Optional[Type[SchemaObject]],
            expected_parent_type: Type[DirBase] = DirBase) -> None:
        path_str = str(path)
        # check for existence
        if expected is not None:
            if path_str not in self.paths:
                raise exc.UnknownEntityError(path_str)
            obj = self.paths[path_str]
            if not isinstance(obj, expected):
                raise exc.UnknownEntityError(f'{path_str} needs to be a {expected.display_name()}')
        if expected is None and path_str in self.paths:
            raise exc.DuplicateNameError(f'{path_str} already exists')
        # check for containing directory
        parent_path = path.parent
        if str(parent_path) not in self.paths:
            raise exc.UnknownEntityError(f'Directory {str(parent_path)}')
        parent = self.paths[str(parent_path)]
        if not isinstance(parent, expected_parent_type):
            raise exc.UnknownEntityError(f'{str(parent_path)} needs to be a {expected_parent_type.display_name()}')

    def get_children(self, parent: Path, child_type: Optional[Type[SchemaObject]], recursive: bool) -> List[Path]:
        candidates = [
            Path(path, empty_is_valid=True)
            for path, obj in self.paths.items() if child_type is None or isinstance(obj, child_type)
        ]
        result = [path for path in candidates if parent.is_ancestor(path, is_parent=(not recursive))]
        return result


class Db:
    def __init__(self, db_id: int, name: str):
        self.id = db_id
        self.name = name
        self.paths = PathDict()
        self.paths.update(self._load_dirs())
        self.paths.update(self._load_tables())

    def create_table(self, path_str: str, schema: List[Column], num_retained_versions: int = 10) -> MutableTable:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None, expected_parent_type=Dir)
        dir = self.paths[path.parent]

        tbl = MutableTable.create(self.id, dir.id, path.name, num_retained_versions, schema)
        self.paths[path] = tbl
        return tbl

    def get_table(self, path_str: str) -> Table:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=Table)
        obj = self.paths[path]
        assert isinstance(obj, Table)
        return obj

    # TODO: move into path_utils.py
    @classmethod
    def _get_parent_path(cls, path: str) -> str:
        if re.fullmatch(_PATH_RE, path) is None:
            raise exc.BadFormatError(f"Invalid path: '{path}'")
        path_components = path.split('.')
        return '' if len(path_components) == 1 else '.'.join(path_components[:-1])

    @classmethod
    def _create_path(cls, dir_path: str, name: str) -> str:
        if dir_path != '' and re.fullmatch(_PATH_RE, dir_path) is None:
            raise exc.BadFormatError(f"Invalid path: '{dir_path}'")
        return name if dir_path == '' else f'{dir_path}.{name}'

    def rename_table(self, path_str: str, new_name: str) -> None:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=MutableTable)
        if re.fullmatch(_ID_RE, new_name) is None:
            raise exc.BadFormatError(f"Invalid table name: '{new_name}'")
        new_path = path.parent.append(new_name)
        self.paths.check_is_valid(new_path, expected=None, expected_parent_type=Dir)

        tbl = self.paths[path]
        assert isinstance(tbl, MutableTable)
        del self.paths[path]
        self.paths[new_path] = tbl
        tbl.rename(new_name)

    def move_table(self, tbl_path: str, dir_path: str) -> None:
        pass

    def list_tables(self, dir_path: str = '', recursive: bool = True) -> List[str]:
        assert dir_path is not None
        path = Path(dir_path, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=DirBase)
        return [str(p) for p in self.paths.get_children(path, child_type=Table, recursive=recursive)]

    def drop_table(self, path_str: str, force: bool = False, ignore_errors: bool = False) -> None:
        path = Path(path_str)
        try:
            self.paths.check_is_valid(path, expected=MutableTable)
        except Exception as e:
            if ignore_errors:
                return
            else:
                raise e
        tbl = self.paths[path]
        assert isinstance(tbl, MutableTable)
        tbl.drop()
        del self.paths[path]

    def create_snapshot(self, path_str: str, tbl_paths: List[str]) -> None:
        snapshot_dir_path = Path(path_str)
        self.paths.check_is_valid(snapshot_dir_path, expected=None, expected_parent_type=Dir)
        tbls: List[MutableTable] = []
        for tbl_path_str in tbl_paths:
            tbl_path = Path(tbl_path_str)
            self.paths.check_is_valid(tbl_path, expected=MutableTable)
            tbl = self.paths[tbl_path]
            assert isinstance(tbl, MutableTable)
            tbls.append(tbl)

        with orm.Session(env.get_engine()) as session:
            dir_record = store.Dir(db_id=self.id, path=path_str, is_snapshot=True)
            session.add(dir_record)
            session.flush()
            assert dir_record.id is not None
            self.paths[snapshot_dir_path] = Dir(dir_record.id)

            for tbl in tbls:
                snapshot_record = store.TableSnapshot(
                    db_id=self.id, dir_id=dir_record.id, name=tbl.name, tbl_id=tbl.id, tbl_version=tbl.version,
                    tbl_schema_version=tbl.schema_version)
                session.add(snapshot_record)
                session.flush()
                assert snapshot_record.id is not None
                cols = Table.load_cols(tbl.id, tbl.schema_version, session)
                snapshot_path = snapshot_dir_path.append(tbl.name)
                self.paths[snapshot_path] = TableSnapshot(snapshot_record, cols)

            session.commit()

    def create_dir(self, path_str: str) -> None:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=None, expected_parent_type=Dir)
        with orm.Session(env.get_engine()) as session:
            dir_record = store.Dir(db_id=self.id, path=path_str, is_snapshot=False)
            session.add(dir_record)
            session.flush()
            assert dir_record.id is not None
            self.paths[path] = Dir(dir_record.id)
            session.commit()

    def rm_dir(self, path_str: str, force: bool = False) -> None:
        path = Path(path_str)
        self.paths.check_is_valid(path, expected=Dir)

        if not force:
            # make sure it's empty
            if len(self.paths.get_children(path, child_type=None, recursive=True)) > 0:
                raise exc.DirectoryNotEmptyError(f'Directory {path_str}')
        else:
            # delete tables
            for tbl_path in self.paths.get_children(path, child_type=Table, recursive=True):
                self.drop_table(str(tbl_path), force=True)
            # rm subdirs
            for dir_path in self.paths.get_children(path, child_type=DirBase, recursive=False):
                self.rm_dir(str(dir_path), force=True)

        with env.get_engine().connect() as conn:
            with conn.begin():
                dir = self.paths[path]
                conn.execute(sql.delete(store.Dir.__table__).where(store.Dir.id == dir.id))
        del self.paths[path]

    def list_dirs(self, path_str: str = '', recursive: bool = True) -> List[str]:
        path = Path(path_str, empty_is_valid=True)
        self.paths.check_is_valid(path, expected=DirBase)
        return [str(p) for p in self.paths.get_children(path, child_type=DirBase, recursive=recursive)]

    def _load_dirs(self) -> Dict[str, SchemaObject]:
        result: Dict[str, SchemaObject] = {}
        with orm.Session(env.get_engine()) as session:
            for dir_record in session.query(store.Dir).where(store.Dir.db_id == self.id).all():
                result[dir_record.path] = SnapshotDir(dir_record.id) if dir_record.is_snapshot else Dir(dir_record.id)
        return result

    def _load_tables(self) -> Dict[str, SchemaObject]:
        result: Dict[str, SchemaObject] = {}
        with orm.Session(env.get_engine()) as session:
            # load all reachable (= mutable) tables
            q = session.query(store.Table, store.Dir.path) \
                .join(store.Dir)\
                .where(store.Table.db_id == self.id) \
                .where(store.Table.is_mutable == True)
            for tbl_record, dir_path in q.all():
                cols = Table.load_cols(tbl_record.id, tbl_record.current_schema_version, session)
                tbl = MutableTable(tbl_record, tbl_record.current_schema_version, cols)
                tbl._load_valid_rowids()  # TODO: move this someplace more appropriate
                path = Path(dir_path, empty_is_valid=True).append(tbl_record.name)
                result[str(path)] = tbl

            # load all table snapshots
            q = session.query(store.TableSnapshot, store.Dir.path) \
                .join(store.Dir) \
                .where(store.TableSnapshot.db_id == self.id)
            for snapshot_record, dir_path in q.all():
                cols = Table.load_cols(snapshot_record.tbl_id, snapshot_record.tbl_schema_version, session)
                snapshot = TableSnapshot(snapshot_record, cols)
                path = Path(dir_path, empty_is_valid=True).append(snapshot_record.name)
                result[str(path)] = snapshot

        return result

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'Db(name={self.name})'

    @classmethod
    def create(cls, name: str) -> 'Db':
        db_id: int = -1
        with orm.Session(env.get_engine()) as session:
            # check for duplicate name
            is_duplicate = session.query(sql.func.count(store.Db.id)).where(store.Db.name == name).scalar() > 0
            if is_duplicate:
                raise exc.DuplicateNameError(f"Db '{name}' already exists")

            db_record = store.Db(name=name)
            session.add(db_record)
            session.flush()
            assert db_record.id is not None
            db_id = db_record.id
            # also create a top-level directory, so that every schema object has a directory
            dir_record = store.Dir(db_id=db_id, path='', is_snapshot=False)
            session.add(dir_record)
            session.flush()
            session.commit()
        assert db_id is not None
        return Db(db_id, name)

    @classmethod
    def load(cls, name: str) -> 'Db':
        if re.fullmatch(_ID_RE, name) is None:
            raise exc.BadFormatError(f"Invalid db name: '{name}'")
        with orm.Session(env.get_engine()) as session:
            try:
                db_record = session.query(store.Db).where(store.Db.name == name).one()
                return Db(db_record.id, db_record.name)
            except sql.exc.NoResultFound:
                raise exc.UnknownEntityError(f'Db {name}')
