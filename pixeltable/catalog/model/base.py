"""model_base(): the registry of declared models and the operations that reconcile them with the catalog."""

from __future__ import annotations

from typing import Literal

from pixeltable import catalog, exceptions as excs, type_system as ts
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec

from .declaration import BtreeIndex, EmbeddingIndex, IndexDeclaration, TableModelMeta
from .diff import (
    _PY_MISMATCH_HINT,
    PY_DESTRUCTIVE_HINT,
    TableDiff,
    base_query_columns,
    format_diff,
    user_columns,
    validate_models,
)
from .resolution import TableSchemaChangeSet


def model_base(cls_name: str = 'TableModel') -> type[TableModelMeta]:
    # mypy fundamentally does not understand metaclasses.
    cls = TableModelMeta(cls_name, (), {}, name='')
    registered_models: dict[str, TableModelMeta] = {}
    cls.__registered_models__ = registered_models  # type: ignore[attr-defined]

    def _bind_all(catalog_dir: str = '') -> None:
        for model in registered_models.values():
            model._bind(catalog_dir)

    def _create_models(catalog_dir: str, expect_created: set[str]) -> None:
        """Create every model that doesn't exist yet and bind all of them.

        Raises ConcurrencyError if a model named in expect_created already exists.
        """
        for name, model in registered_models.items():
            tbl, was_created = model._create(catalog_dir)
            if name in expect_created and not was_created:
                raise excs.ConcurrencyError(
                    excs.ErrorCode.CONCURRENT_MODIFICATION,
                    f'Table {str(tbl._path())!r} was created concurrently; re-run the operation.',
                )

    def _create_all(catalog_dir: str = '') -> dict[str, TableDiff]:
        """Returns the diff that was applied, per model: 'create' for the tables created now, 'up_to_date'
        for those that already matched. Raises rather than returning a partially applied diff."""
        # create_all() only creates tables; it never mutates an existing one. If any existing table differs from
        # its model, refuse.
        diffs = validate_models(registered_models, catalog_dir)
        changed = [(name, d) for name, d in diffs.items() if d['exists'] and d['resolution'] != 'up_to_date']
        if len(changed) > 0:
            detail = '\n'.join(line for name, d in changed for line in format_diff(name, d))
            raise excs.RequestError(
                excs.ErrorCode.SCHEMA_MISMATCH,
                f'One or more existing tables differ from their models.\n{detail}\n{_PY_MISMATCH_HINT}',
            )

        _create_models(catalog_dir, {name for name, d in diffs.items() if not d['exists']})
        return diffs

    def _get_model_diff(catalog_dir: str = '') -> dict[str, TableDiff]:
        return validate_models(registered_models, catalog_dir)

    def _diff_all(catalog_dir: str = '') -> None:
        diffs = _get_model_diff(catalog_dir)
        lines: list[str] = []
        for name, d in diffs.items():
            lines.extend(format_diff(name, d))
        Env.get().console_logger.info('\n'.join(lines) if len(lines) > 0 else 'Catalog is up to date.')

    def _update_all(catalog_dir: str = '', *, allow_destructive: bool = False) -> dict[str, TableDiff]:
        """Reconcile every registered model with the catalog.

        Returns the diff that was applied, per model. The compare-and-swap in update_from_model() and the
        concurrency check in _create_models() both abort if the catalog moved, so the returned diff is what
        reached the store.

        Not atomic: migrations and creations run in separate transactions, so a failure raises with part of the
        diff applied. Re-running reconciles whatever is left.

        Destructive changes without allow_destructive raise DESTRUCTIVE_SCHEMA_CHANGE.
        """
        diffs = validate_models(registered_models, catalog_dir)

        if len(diffs) == 0:
            # No updates *or* create statements.
            Env.get().console_logger.info('Catalog is up to date.')
            return diffs

        fatal = [(name, d) for name, d in diffs.items() if d['resolution'] == 'unsupported']
        if len(fatal) > 0:
            detail = '\n'.join(line for name, d in fatal for line in format_diff(name, d))
            raise excs.RequestError(
                excs.ErrorCode.SCHEMA_MISMATCH,
                'One or more tables cannot be updated, because their models are inconsistent '
                'with the existing table(s) in the catalog.\n'
                f'{detail}\n'
                'Adjust the existing table(s) manually, or adjust the models to be consistent with the catalog.',
            )

        destructive = [(name, d) for name, d in diffs.items() if d['resolution'] == 'update_destructive']
        if len(destructive) > 0 and not allow_destructive:
            detail = '\n'.join(line for name, d in destructive for line in format_diff(name, d))
            raise excs.RequestError(
                excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
                f'The following updates would result in destructive catalog changes.\n{detail}\n{PY_DESTRUCTIVE_HINT}',
            )

        # Apply column/index changes to existing tables. New tables are handled by _create_all() below.
        update_diffs = [
            (name, d) for name, d in diffs.items() if d['resolution'] in ('update_additive', 'update_destructive')
        ]

        if len(update_diffs) > 0:
            catalog_dir = catalog.Path.dir_prefix(catalog_dir)
            change_sets: list[TableSchemaChangeSet] = []
            for name, d in update_diffs:
                model = registered_models[name]
                new_col_names = {c['name'] for c in d['ops'] if c['target'] == 'column' and c['op'] == 'add'}
                dropped_col_names = [c['name'] for c in d['ops'] if c['target'] == 'column' and c['op'] == 'drop']
                new_idx_refs = [
                    c['details']['index_ref'] for c in d['ops'] if c['target'] == 'index' and c['op'] == 'add'
                ]
                dropped_idx_names = [c['name'] for c in d['ops'] if c['target'] == 'index' and c['op'] == 'drop']
                # Resolve type annotations to ColumnTypes, mirroring _create(), and tag each column's origin.
                # Iterate in declaration order (not the diff's sorted order), so a new column may depend on an
                # earlier new column, as it can at create time.
                user_cols = user_columns(model)
                base_query_cols = base_query_columns(model)
                new_columns: dict[str, tuple[ColumnSpec, Literal['base_query', 'model_body']]] = {}
                for col_name, col_spec in user_cols.items():
                    if col_name not in new_col_names:
                        continue
                    spec = col_spec.copy()
                    if 'type' in spec:
                        spec['type'] = ts.ColumnType.normalize_type(  # type: ignore[typeddict-item]
                            spec['type'], allow_builtin_types=False
                        )
                    origin: Literal['base_query', 'model_body'] = (
                        'base_query' if col_name in base_query_cols else 'model_body'
                    )
                    new_columns[col_name] = (spec, origin)

                # resolve idx_refs to IndexDeclarations. (We can't simply go by index name, since there may be unnamed
                # indexes.) Instead we compare the (index_type, name, columns) tuple; if there are two unnamed indexes
                # with the same type, then they *must* have different columns, so the tuple uniquely identifies the
                # index.
                new_idxs: list[IndexDeclaration] = []
                for idx_ref in new_idx_refs:
                    matching_idxs = [
                        idx
                        for idx in model.__indexes__
                        if (idx_ref['index_type'] == 'btree') == isinstance(idx, BtreeIndex)
                        and idx_ref['name'] == (idx.name if isinstance(idx, EmbeddingIndex) else None)
                        and [idx.column.name] == idx_ref['columns']
                    ]
                    assert len(matching_idxs) == 1
                    new_idxs.append(matching_idxs[0])

                # only an existing table is updated, so the diff recorded what it was computed against
                assert d['tbl_id'] is not None and d['schema_versions'] is not None
                change_sets.append(
                    TableSchemaChangeSet(
                        path=catalog.Path.parse(f'{catalog_dir}{name}'),
                        new_columns=new_columns,
                        dropped_columns=dropped_col_names,
                        new_idxs=new_idxs,
                        dropped_idxs=dropped_idx_names,
                        tbl_id=d['tbl_id'],
                        schema_versions=d['schema_versions'],
                    )
                )

            # All models share catalog_dir, hence a single catalog; apply every table's changes in one transaction.
            cat = get_runtime().get_catalog(change_sets[0]['path'])
            cat.update_from_model(change_sets)

        # Now create any new tables, and bind every model to its table. The diff computed above is the one being
        # applied, so the models it found up-to-date are not re-examined against the catalog.
        try:
            _create_models(catalog_dir, {name for name, d in diffs.items() if d['resolution'] == 'create'})
        except excs.Error as e:
            # the migrations above are already committed; name them, so that a failure here doesn't read as
            # though the catalog were untouched. Augmenting in place keeps the exception's type and fields.
            # e.message excludes e.detail, which is diagnostic text that must not become part of the message
            if len(update_diffs) > 0:
                migrated = ', '.join(repr(d['path']) for _, d in update_diffs)
                e.args = (
                    f'{e.message}\n\nThe following table(s) were already migrated: {migrated}. '
                    'Re-run update_all() to finish reconciling.',
                )
            raise
        return diffs

    cls.bind_all = _bind_all  # type: ignore[attr-defined]
    cls.create_all = _create_all  # type: ignore[attr-defined]
    cls.get_model_diff = _get_model_diff  # type: ignore[attr-defined]
    cls.diff_all = _diff_all  # type: ignore[attr-defined]
    cls.update_all = _update_all  # type: ignore[attr-defined]

    return cls  # type: ignore[return-value]
