//! Functions behind `Table.update()`, `batch_update()`, `recompute_columns()`, `delete()`, `revert()` and
//! `get_versions()`: argument conversion; the logic is in `exec::update` and `catalog::versions`.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use uuid::Uuid;

use super::catalog::parse_uuid;
use super::expr::{as_expr, from_object};
use super::table::{returned_rows_to_py, status_dict};
use crate::catalog::Catalog;
use crate::catalog::path::{fold_identifier, is_valid_identifier};
use crate::catalog::runtime::with_catalog;
use crate::catalog::table::{ColId, ResolvedColumn};
use crate::error::{Error, Result, py_repr};
use crate::exec::compute::ComputedCol;
use crate::exec::insert::{self, InputRow, InsertPlan, UpdateStatus};
use crate::exec::scan::PathInput;
use crate::exec::update::{self, QCol, RowSelection, UpdatePlan, dependent_columns, view_update_tree};
use crate::expr::{Expr, ExprRef};
use crate::store::rows::RowKey;
use crate::value::Value;
use crate::value::py as vpy;

fn obj_repr(obj: &Bound<'_, PyAny>) -> String {
    obj.repr().map(|s| s.to_string()).unwrap_or_default()
}

/// Upstream `_validate_update_value_spec()` (before any catalog access).
fn prevalidate_spec(spec: &Bound<'_, PyDict>) -> Result<()> {
    for (k, v) in spec.iter() {
        let Ok(name) = k.cast::<PyString>() else {
            crate::bail!(InvalidArgument, "Update specification: dict key must be column name; got {}", obj_repr(&k));
        };
        let name = name.to_string();
        if !is_valid_identifier(&name, false) {
            crate::bail!(InvalidArgument, "Update specification: invalid column name {}", py_repr(&name));
        }
        if !matches!(from_object(&v), Ok(Some(_))) {
            crate::bail!(
                UnsupportedOperation,
                "Column {}: value is not a recognized literal or expression: {}",
                py_repr(&name),
                obj_repr(&v)
            );
        }
    }
    Ok(())
}

/// The `where` argument as an expression.
fn where_arg(where_: Option<&Bound<'_, PyAny>>) -> Result<Option<ExprRef>> {
    let Some(w) = where_.filter(|w| !w.is_none()) else { return Ok(None) };
    match as_expr(w) {
        Some(e) => Ok(Some(e)),
        None => crate::bail!(
            InvalidExpression,
            "`where` argument must be a valid Pixeltable expression; got `{}`",
            w.get_type().str()?
        ),
    }
}

/// Folds the keys of a user-supplied column mapping (upstream `fold_mapping_keys()`).
fn fold_keys<'py>(spec: &Bound<'py, PyDict>) -> Result<Vec<(String, Bound<'py, PyAny>)>> {
    let mut out: Vec<(String, Bound<'py, PyAny>)> = Vec::new();
    for (k, v) in spec.iter() {
        let name: String = k.str()?.to_string();
        let folded = fold_identifier(&name);
        if out.iter().any(|(n, _)| *n == folded) {
            let spellings: Vec<String> = spec
                .keys()
                .iter()
                .filter_map(|k| k.str().ok().map(|s| s.to_string()))
                .filter(|n| fold_identifier(n) == folded)
                .map(|n| py_repr(&n))
                .collect();
            crate::bail!(InvalidSchema, "Column names are case-insensitive, but {} were specified", spellings.join(", "));
        }
        out.push((folded, v));
    }
    Ok(out)
}

/// What a value of an update spec may be.
#[derive(Clone, Copy)]
struct SpecRules {
    allow_pk: bool,
    allow_exprs: bool,
    allow_media: bool,
}

/// A validated update target.
enum Target {
    Literal(ColId, Value),
    Expr(ColId, ExprRef),
}

/// Upstream `_validate_update_spec()`: resolves the columns of a (folded) spec and converts its values. `_rowid`
/// entries are returned separately.
fn validate_spec(
    py: Python<'_>,
    cols: &[ResolvedColumn],
    tbl_id: Uuid,
    spec: &[(String, Bound<'_, PyAny>)],
    rules: SpecRules,
) -> Result<(Vec<(ResolvedColumn, Target)>, Option<Bound<'static, PyAny>>)> {
    let _ = py;
    let _: Option<&Bound<'_, PyAny>> = None;
    let mut out = Vec::new();
    for (name, val) in spec {
        if name == "_rowid" {
            continue;
        }
        let Some(col) = cols.iter().find(|c| c.name == *name) else {
            crate::bail!(ColumnNotFound, "Unknown column: {name}");
        };
        if col.tbl_id != tbl_id {
            crate::bail!(UnsupportedOperation, "Column {} is a base table column and cannot be updated", py_repr(&col.name));
        }
        let is_match_col = col.md.is_pk && rules.allow_pk;
        if col.md.is_computed() && !is_match_col {
            crate::bail!(UnsupportedOperation, "Column {} is computed and cannot be updated", py_repr(name));
        }
        if col.md.is_pk && !rules.allow_pk {
            crate::bail!(UnsupportedOperation, "Column {} is a primary key column and cannot be updated", py_repr(name));
        }
        if col.md.col_type.is_media() && !rules.allow_media {
            crate::bail!(UnsupportedOperation, "Column {} is a media column and cannot be updated", py_repr(name));
        }
        let col_type = &col.md.col_type;
        let target = match as_expr(val).is_none().then(|| vpy::coerce(val, col_type)) {
            Some(Ok(v)) => Target::Literal(col.md.id, v),
            Some(Err(e)) if !matches!(e, Error::Type(_) | Error::Value(_) | Error::Pxt { .. }) => return Err(e),
            _ => {
                if !rules.allow_exprs {
                    crate::bail!(
                        UnsupportedOperation,
                        "Column {}: value is not a valid literal for this column (expected `{}`): {}",
                        py_repr(name),
                        col_type.repr(),
                        obj_repr(val)
                    );
                }
                let Some(e) = from_object(val).ok().flatten() else {
                    crate::bail!(
                        UnsupportedOperation,
                        "Column {}: value is not a recognized literal or expression: {}",
                        py_repr(name),
                        obj_repr(val)
                    );
                };
                if !col_type.is_supertype_of(&e.col_type, true, true) {
                    crate::bail!(
                        TypeMismatch,
                        "The literal value {} is not compatible with the type `{}` of column {}",
                        obj_repr(val),
                        col_type.repr(),
                        py_repr(name)
                    );
                }
                match e.as_literal() {
                    Some(v) if as_expr(val).is_none() => Target::Literal(col.md.id, v.clone()),
                    _ => Target::Expr(col.md.id, e),
                }
            }
        };
        out.push((col.clone(), target));
    }
    Ok((out, None))
}

/// Builds the rest of an update plan once the changed columns are known (under the lock).
fn build_plan(
    cat: &mut Catalog,
    tbl_id: Uuid,
    selection: RowSelection,
    mut updated_cols: Vec<String>,
    mut changed: HashSet<QCol>,
    explicit_recompute: Vec<ComputedCol>,
    cascade: bool,
) -> Result<UpdatePlan> {
    let md = cat.table(tbl_id)?.clone();
    let mut recompute = explicit_recompute;
    if cascade {
        for c in dependent_columns(&md, &mut changed)? {
            if !recompute.iter().any(|r| r.col_id == c.col_id) {
                recompute.push(c);
            }
        }
    }
    for c in &recompute {
        let q = format!("{}.{}", md.name, c.name);
        if !updated_cols.contains(&q) {
            updated_cols.push(q);
        }
    }
    // the indexes on updated or recomputed columns are always recomputed (not reported as updated columns)
    let modified: HashSet<QCol> = changed.iter().copied().chain(recompute.iter().map(|c| (tbl_id, c.col_id))).collect();
    recompute.extend(crate::exec::compute::index_value_columns(cat, tbl_id, Some(&modified))?);
    let tbl_display = crate::functions::udf::tbl_version_display(&md);
    for c in &recompute {
        crate::functions::udf::check_valid_column(cat, &tbl_display, &c.name, &c.expr, "updated in")?;
    }
    let views = if cascade { view_update_tree(cat, tbl_id, &changed, false, &mut updated_cols)? } else { Vec::new() };
    let path = cat.tbl_path(tbl_id, None)?;
    Ok(UpdatePlan {
        tbl_id,
        tbl_name: md.name.clone(),
        schema_version: md.current_schema_version,
        is_data_versioned: md.is_data_versioned,
        path: PathInput::snapshot(cat, &path)?,
        selection,
        recompute,
        updated_cols,
        cascade,
        views,
        errors_only: None,
        return_cols: None,
        schema_change: None,
        writes_rows: true,
        media: crate::exec::compute::MediaTarget {
            tbl_id,
            dir: cat.store().media_dir(tbl_id),
            version: md.current_version + 1,
        },
    })
}


/// Fails unless `pred` references only the table's path.
fn check_where_bound(cat: &Catalog, tbl_id: Uuid, pred: &Expr) -> Result<()> {
    let chain = cat.base_chain(tbl_id);
    if pred.column_refs().iter().all(|r| chain.contains(&r.tbl_id)) {
        return Ok(());
    }
    crate::bail!(
        UnsupportedOperation,
        "`where` predicate ({}) is not bound by {}",
        crate::expr::display::display(pred),
        cat.display_str(tbl_id)?
    );
}

/// Computes and commits an update; returns its status dict (with the updated rows if the plan asks for them).
fn run_update<'py>(py: Python<'py>, plan: UpdatePlan, fail_on_exception: bool) -> Result<Bound<'py, PyDict>> {
    let out = py.detach(|| plan.compute(fail_on_exception))?;
    let rows = if plan.return_cols.is_some() { out.own.new_rows.clone() } else { Vec::new() };
    let status = with_catalog(py, |cat| plan.commit(cat, out))?;
    let d = status_dict(py, &status)?;
    if let Some(cols) = &plan.return_cols {
        d.set_item("rows", returned_rows_to_py(py, cols, &rows.iter().collect::<Vec<_>>())?)?;
    }
    Ok(d)
}

/// `Table.update()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version, value_spec, where_, cascade, return_rows))]
pub fn tbl_update<'py>(
    py: Python<'py>,
    tbl_id: &str,
    version: Option<i64>,
    value_spec: &Bound<'py, PyAny>,
    where_: Option<&Bound<'py, PyAny>>,
    cascade: bool,
    return_rows: bool,
) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    let spec = value_spec
        .cast::<PyDict>()
        .map_err(|_| crate::pxt_err!(InvalidArgument, "Update specification must be a dictionary"))?;
    prevalidate_spec(spec)?;
    let pred = where_arg(where_)?;
    let folded = fold_keys(spec)?;
    let cols = with_catalog(py, |cat| {
        cat.check_mutable(id, version, "update")?;
        cat.resolve_columns(id, None)
    })?;
    let rules = SpecRules { allow_pk: false, allow_exprs: true, allow_media: true };
    let (targets, _) = validate_spec(py, &cols, id, &folded, rules)?;
    let plan = with_catalog(py, |cat| {
        if let Some(p) = &pred {
            check_where_bound(cat, id, p)?;
            update::validate_where(p)?;
        }
        let tbl_name = cat.table(id)?.name.clone();
        let mut changed: HashSet<QCol> = HashSet::new();
        let mut updated_cols = Vec::new();
        let mut exprs = Vec::new();
        for (col, t) in &targets {
            changed.insert((id, col.md.id));
            updated_cols.push(format!("{tbl_name}.{}", col.name));
            exprs.push(match t {
                Target::Literal(c, v) => (*c, Expr::literal(v.clone(), col.md.col_type.clone())),
                Target::Expr(c, e) => {
                    let chain = cat.base_chain(id);
                    if !e.column_refs().iter().all(|r| chain.contains(&r.tbl_id)) {
                        crate::bail!(
                            UnsupportedOperation,
                            "The value expression for column {} ({}) is not bound by {}",
                            py_repr(&col.name),
                            crate::expr::display::display(e),
                            cat.display_str(id)?
                        );
                    }
                    (*c, e.clone())
                }
            });
        }
        let mut plan =
            build_plan(cat, id, RowSelection::Where { pred: pred.clone(), targets: exprs }, updated_cols, changed, Vec::new(), cascade)?;
        if return_rows {
            plan.return_cols = Some(insert::return_columns(cat, id)?);
        }
        Ok(plan)
    })?;
    run_update(py, plan, false)
}

/// How a `batch_update()` row identifies the row it updates.
enum RowIdent {
    Rowid(RowKey),
    Pk(Vec<Value>),
}

/// `Table.batch_update()`; returns the status dict.
#[pyfunction]
#[pyo3(signature = (tbl_id, version, rows, cascade, if_not_exists, return_rows))]
pub fn tbl_batch_update<'py>(
    py: Python<'py>,
    tbl_id: &str,
    version: Option<i64>,
    rows: &Bound<'py, PyAny>,
    cascade: bool,
    if_not_exists: &str,
    return_rows: bool,
) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    let if_not_exists = if_not_exists.to_ascii_lowercase();
    if !["error", "ignore", "insert"].contains(&if_not_exists.as_str()) {
        crate::bail!(InvalidArgument, "if_not_exists must be one of: ['error', 'ignore', 'insert']");
    }
    let rows: Vec<Bound<'py, PyAny>> = rows.try_iter()?.collect::<PyResult<_>>()?;
    let (cols, is_data_versioned, num_pos) = with_catalog(py, |cat| {
        cat.check_mutable(id, version, "update")?;
        let md = cat.table(id)?;
        Ok((cat.resolve_columns(id, None)?, md.is_data_versioned, cat.num_pos(id)))
    })?;
    if rows.is_empty() {
        let d = status_dict(py, &UpdateStatus::default())?;
        if return_rows {
            d.set_item("rows", PyList::empty(py))?;
        }
        return Ok(d);
    }
    let dicts: Vec<Bound<'py, PyDict>> = rows
        .iter()
        .map(|r| r.cast::<PyDict>().cloned().map_err(|_| crate::pxt_err!(UnsupportedOperation, "row {} is not a dictionary", obj_repr(r))))
        .collect::<Result<_>>()?;
    let pk_cols: Vec<&ResolvedColumn> = cols.iter().filter(|c| c.md.is_pk && c.tbl_id == id).collect();
    let has_rowid = dicts[0].contains("_rowid")?;
    if !is_data_versioned && dicts.iter().any(|d| d.contains("_rowid").unwrap_or(false)) {
        crate::bail!(UnsupportedOperation, "batch_update(): _rowid is not supported for operational tables");
    }
    if pk_cols.is_empty() && !has_rowid {
        crate::bail!(InvalidArgument, "Table must have primary key for batch update");
    }
    let rules = SpecRules { allow_pk: !has_rowid, allow_exprs: false, allow_media: false };
    // per row: the values to set, and the rowid or primary key values that identify the row
    // per row: the values to set, how the row is identified, and its primary key values
    let mut updates: Vec<(Vec<(ColId, Value)>, RowIdent, Vec<(ColId, Value)>)> = Vec::new();
    for d in &dicts {
        for k in d.keys() {
            if !k.is_instance_of::<PyString>() {
                crate::bail!(InvalidArgument, "Update specification: dict key must be column name; got {}", obj_repr(&k));
            }
        }
        let folded = fold_keys(d)?;
        let (targets, _) = validate_spec(py, &cols, id, &folded, rules)?;
        let mut values = Vec::new();
        let mut pk_vals: HashMap<ColId, Value> = HashMap::new();
        for (col, t) in targets {
            let Target::Literal(c, v) = t else { return Err(Error::internal("batch_update() value is not a literal")) };
            if col.md.is_pk {
                pk_vals.insert(c, v);
            } else {
                values.push((c, v));
            }
        }
        let ident = if has_rowid {
            let Some(rowid) = d.get_item("_rowid")? else {
                crate::bail!(InternalError, "Malformed batch update: row is missing _rowid");
            };
            let parts: Vec<i64> = match rowid.cast::<PyTuple>() {
                Ok(t) => t.iter().map(|x| x.extract::<i64>()).collect::<PyResult<_>>().map_err(|_| {
                    crate::pxt_err!(InternalError, "Malformed _rowid: all components must be int, got {}", obj_repr(&rowid))
                })?,
                Err(_) => rowid.extract::<Vec<i64>>().map_err(|_| {
                    crate::pxt_err!(InternalError, "Malformed _rowid: all components must be int, got {}", obj_repr(&rowid))
                })?,
            };
            if parts.len() != 1 + num_pos {
                crate::bail!(
                    InternalError,
                    "Malformed _rowid: expected {} components, got {}",
                    1 + num_pos,
                    parts.len()
                );
            }
            RowIdent::Rowid((parts[0], parts[1..].to_vec()))
        } else {
            let missing: Vec<String> = pk_cols.iter().filter(|c| !pk_vals.contains_key(&c.md.id)).map(|c| py_repr(&c.name)).collect();
            if !missing.is_empty() {
                crate::bail!(UnsupportedOperation, "Primary key column(s) {} missing in {}", missing.join(", "), obj_repr(d));
            }
            RowIdent::Pk(pk_cols.iter().map(|c| pk_vals[&c.md.id].clone()).collect())
        };
        let pk_list: Vec<(ColId, Value)> = pk_cols.iter().filter_map(|c| pk_vals.get(&c.md.id).map(|v| (c.md.id, v.clone()))).collect();
        updates.push((values, ident, pk_list));
    }
    // resolve the rows (primary key lookups need the table's rows) and plan
    let (plan, unmatched_idx) = with_catalog(py, |cat| {
        // the live rows with the given primary keys (by the table's primary-key index)
        let pk_keys: Vec<Vec<Value>> = updates
            .iter()
            .filter_map(|(_, ident, _)| match ident {
                RowIdent::Pk(pk) => Some(pk.clone()),
                RowIdent::Rowid(_) => None,
            })
            .collect();
        let mut by_pk: Vec<Option<RowKey>> = Vec::new();
        if !pk_keys.is_empty() {
            let data = cat.data(id)?;
            let pk_ids: Vec<ColId> = pk_cols.iter().map(|c| c.md.id).collect();
            let truncate = cat.table(id)?.is_data_versioned;
            by_pk = data
                .find_live_by_pks(&pk_ids, truncate, &pk_keys, true)
                .into_iter()
                .map(|r| r.map(|r| (r.rowid, r.pos.clone())))
                .collect();
        }
        let mut by_pk = by_pk.into_iter();
        let mut sel: HashMap<RowKey, Vec<(ColId, Value)>> = HashMap::new();
        let mut keys_in_order: Vec<Option<RowKey>> = Vec::new();
        let mut changed: HashSet<QCol> = HashSet::new();
        let mut updated_cols: Vec<String> = Vec::new();
        let tbl_name = cat.table(id)?.name.clone();
        for (values, ident, _) in &updates {
            for (c, _) in values {
                changed.insert((id, *c));
                if let Some(col) = cols.iter().find(|x| x.tbl_id == id && x.md.id == *c) {
                    let q = format!("{tbl_name}.{}", col.name);
                    if !updated_cols.contains(&q) {
                        updated_cols.push(q);
                    }
                }
            }
            let key = match ident {
                RowIdent::Rowid(k) => Some(k.clone()),
                RowIdent::Pk(_) => by_pk.next().flatten(),
            };
            if let Some(k) = &key {
                sel.entry(k.clone()).or_default().extend(values.iter().cloned());
            }
            keys_in_order.push(key);
        }
        let mut plan = build_plan(cat, id, RowSelection::Keys(sel), updated_cols, changed, Vec::new(), cascade)?;
        if return_rows {
            plan.return_cols = Some(insert::return_columns(cat, id)?);
        }
        let unmatched: Vec<usize> = keys_in_order.iter().enumerate().filter(|(_, k)| k.is_none()).map(|(i, _)| i).collect();
        Ok((plan, unmatched))
    })?;
    let out = py.detach(|| plan.compute(false))?;
    let mut returned = if plan.return_cols.is_some() { out.own.new_rows.clone() } else { Vec::new() };
    let mut unmatched_idx = unmatched_idx;
    // rowids that match no live row
    for (i, (_, ident, _)) in updates.iter().enumerate() {
        if let RowIdent::Rowid(k) = ident {
            if out.unmatched.contains(k) {
                unmatched_idx.push(i);
            }
        }
    }
    unmatched_idx.sort();
    if !unmatched_idx.is_empty() && if_not_exists == "error" {
        crate::bail!(RowNotFound, "batch_update(): {} row(s) not found", unmatched_idx.len());
    }
    let mut status = with_catalog(py, |cat| plan.commit(cat, out))?;
    if if_not_exists == "insert" && !unmatched_idx.is_empty() {
        // upstream inserts the unmatched rows as given (primary key and values), without checking required columns
        let rows: Vec<InputRow> = unmatched_idx
            .iter()
            .map(|i| {
                let (values, _, pk) = &updates[*i];
                pk.iter().chain(values.iter()).cloned().collect()
            })
            .collect();
        let ins_plan = with_catalog(py, |cat| InsertPlan::new(cat, id, None))?;
        let built = py.detach(|| insert::build_rows(&ins_plan, rows, false))?;
        if plan.return_cols.is_some() {
            returned.extend(built.rows.iter().cloned());
        }
        let ins_status = with_catalog(py, |cat| insert::commit(cat, &ins_plan, built))?;
        status = status + ins_status.to_cascade();
    }
    let d = status_dict(py, &status)?;
    if let Some(cols) = &plan.return_cols {
        d.set_item("rows", returned_rows_to_py(py, cols, &returned.iter().collect::<Vec<_>>())?)?;
    }
    Ok(d)
}

/// `Table.recompute_columns()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version, columns, where_, errors_only, cascade))]
pub fn tbl_recompute_columns<'py>(
    py: Python<'py>,
    tbl_id: &str,
    version: Option<i64>,
    columns: &Bound<'py, PyList>,
    where_: Option<&Bound<'py, PyAny>>,
    errors_only: bool,
    cascade: bool,
) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    let pred = where_arg(where_)?;
    let plan = with_catalog(py, |cat| {
        cat.check_mutable(id, version, "recompute columns of")?;
        if columns.is_empty() {
            crate::bail!(MissingRequired, "At least one column must be specified to recompute");
        }
        if errors_only && columns.len() > 1 {
            crate::bail!(UnsupportedOperation, "Cannot use errors_only=True with multiple columns");
        }
        let md = cat.table(id)?.clone();
        if !md.is_data_versioned {
            crate::bail!(UnsupportedOperation, "recompute_columns() is supported on data-versioned tables only");
        }
        let path = cat.tbl_path(id, None)?;
        let all_cols = cat.path_columns(&path)?;
        let visible = cat.resolve_columns(id, None)?;
        let mut targets: Vec<ResolvedColumn> = Vec::new();
        for c in columns.iter() {
            let col = if let Some(e) = as_expr(&c) {
                let Some(r) = e.as_column_ref() else {
                    crate::bail!(TypeMismatch, "Invalid column parameter type: {}", c.get_type().str()?);
                };
                match all_cols.iter().find(|x| x.tbl_id == r.tbl_id && x.md.id == r.col_id) {
                    Some(x) => x.clone(),
                    None => crate::bail!(ColumnNotFound, "Unknown column: {}", r.name),
                }
            } else {
                let name: String = c.extract()?;
                match visible.iter().find(|x| x.name == fold_identifier(&name)) {
                    Some(x) => x.clone(),
                    None => crate::bail!(ColumnNotFound, "Unknown column: {name}"),
                }
            };
            if !col.md.is_computed() {
                crate::bail!(UnsupportedOperation, "Column {} is not a computed column", py_repr(&col.name));
            }
            if col.tbl_id != id {
                crate::bail!(UnsupportedOperation, "Cannot recompute column of a base: {}", col.name);
            }
            targets.push(col);
        }
        if let Some(p) = &pred {
            check_where_bound(cat, id, p)?;
            update::validate_where(p)?;
        }
        let changed: HashSet<QCol> = targets.iter().map(|c| (id, c.md.id)).collect();
        // the explicit targets (in schema order), then their dependents
        let target_ids: HashSet<ColId> = targets.iter().map(|c| c.md.id).collect();
        let explicit: Vec<ComputedCol> = {
            let mut ordered = Vec::new();
            for c in md.columns_at(None)? {
                if target_ids.contains(&c.md.id) && c.md.stored {
                    ordered.push(ComputedCol {
                        col_id: c.md.id,
                        name: c.name.clone(),
                        is_media: c.md.col_type.is_media(),
                        expr: crate::expr::serde::from_json(c.md.value_expr.as_ref().ok_or_else(|| Error::internal("no value"))?)?,
                    });
                }
            }
            ordered
        };
        let selection = RowSelection::Where { pred: pred.clone(), targets: Vec::new() };
        let mut plan = build_plan(cat, id, selection, Vec::new(), changed, explicit, cascade)?;
        if errors_only {
            plan.errors_only = Some(targets[0].md.id);
        }
        Ok(plan)
    })?;
    run_update(py, plan, false)
}

/// Resolves a column parameter (a name or a column reference) of `tbl_id`.
fn column_param(cat: &Catalog, tbl_id: Uuid, column: &Bound<'_, PyAny>) -> Result<ResolvedColumn> {
    if let Some(e) = as_expr(column) {
        let Some(r) = e.as_column_ref() else {
            crate::bail!(TypeMismatch, "Invalid column parameter type: {}", column.get_type().str()?);
        };
        let path = cat.tbl_path(tbl_id, None)?;
        return match cat.path_columns(&path)?.into_iter().find(|c| c.tbl_id == r.tbl_id && c.md.id == r.col_id) {
            Some(c) => Ok(c),
            None => {
                let tbl_name = cat.table(r.tbl_id).map(|t| t.name.clone()).unwrap_or_default();
                crate::bail!(ColumnNotFound, "Unknown column: {tbl_name}.{}", r.name)
            }
        };
    }
    let name: String = column
        .extract()
        .map_err(|_| crate::pxt_err!(TypeMismatch, "Invalid column parameter type: {}", column.get_type().str().map(|s| s.to_string()).unwrap_or_default()))?;
    match cat.resolve_column_by_name(tbl_id, None, &name)? {
        Some(c) => Ok(c),
        None => crate::bail!(ColumnNotFound, "Unknown column: {name}"),
    }
}

/// `Table.alter_column()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version, column, type_))]
pub fn tbl_alter_column(py: Python<'_>, tbl_id: &str, version: Option<i64>, column: &Bound<'_, PyAny>, type_: &Bound<'_, PyAny>) -> Result<()> {
    let id = parse_uuid(tbl_id)?;
    let new_type = super::types::normalize_type(type_, false)?;
    with_catalog(py, |cat| {
        cat.check_mutable(id, version, "alter columns of")?;
        let col = column_param(cat, id, column)?;
        cat.alter_column(id, &col, &new_type)
    })
}

/// `Table.alter_computed_column()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version, kwargs, recompute, cascade))]
pub fn tbl_alter_computed_column<'py>(
    py: Python<'py>,
    tbl_id: &str,
    version: Option<i64>,
    kwargs: &Bound<'py, PyDict>,
    recompute: bool,
    cascade: bool,
) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    if kwargs.len() != 1 {
        let names: Vec<String> = kwargs.keys().iter().map(|k| k.str().map(|s| s.to_string()).unwrap_or_default()).collect();
        crate::bail!(
            UnsupportedOperation,
            "alter_computed_column() requires exactly one keyword argument of the form `col_name=expression`; got {} arguments instead ({})",
            kwargs.len(),
            names.join(", ")
        );
    }
    let (name, spec) = kwargs.iter().next().ok_or_else(|| Error::internal("missing column"))?;
    let col_name: String = name.extract()?;
    let plan = with_catalog(py, |cat| {
        cat.check_mutable(id, version, "alter columns of")?;
        let Some(col) = cat.resolve_column_by_name(id, None, &col_name)? else {
            crate::bail!(ColumnNotFound, "Unknown column: {col_name}");
        };
        if col.tbl_id != id {
            crate::bail!(UnsupportedOperation, "Cannot alter base table column {}", py_repr(&col.name));
        }
        if !col.md.is_computed() {
            crate::bail!(UnsupportedOperation, "Column {} is not a computed column", py_repr(&col.name));
        }
        let Some(new_expr) = from_object(&spec).ok().flatten() else {
            crate::bail!(
                InvalidExpression,
                "Column {}: the new value needs to be a Pixeltable expression, but it is a `{}`",
                py_repr(&col.name),
                spec.get_type().str()?
            );
        };
        crate::catalog::tables::check_no_cellmd_refs(&new_expr)?;
        let path = cat.tbl_path(id, None)?;
        if !cat.is_bound_by(&new_expr, &path)? {
            crate::bail!(
                UnsupportedOperation,
                "The value expression of column {} ({}) is not bound by {}",
                py_repr(&col.name),
                crate::expr::display::display(&new_expr),
                cat.display_str(id)?
            );
        }
        let md = cat.table(id)?.clone();
        let old_json = col.md.value_expr.clone();
        let new_json = crate::expr::serde::to_json(&new_expr)?;
        if old_json.as_ref() == Some(&new_json) {
            return Ok(None);
        }
        if new_expr.col_type != col.md.col_type {
            crate::bail!(
                UnsupportedOperation,
                "Column {}: the new value expression has type `{}`, but the column has type `{}`. Changing the type of a computed column is not supported.",
                py_repr(&col.name),
                new_expr.col_type.repr(),
                col.md.col_type.repr()
            );
        }
        check_no_cycle(&md, col.md.id, &new_expr)?;
        if recompute && !md.is_data_versioned {
            crate::bail!(UnsupportedOperation, "alter_computed_column(recompute=True) is supported on data-versioned tables only");
        }
        let changed: HashSet<QCol> = HashSet::from([(id, col.md.id)]);
        let explicit = if col.md.stored {
            vec![ComputedCol { col_id: col.md.id, name: col.name.clone(), expr: new_expr.clone(), is_media: col.md.col_type.is_media() }]
        } else {
            Vec::new()
        };
        let selection = RowSelection::Where { pred: None, targets: Vec::new() };
        let mut plan = build_plan(cat, id, selection, vec![format!("{}.{}", md.name, col.name)], changed, explicit, recompute && cascade)?;
        if !col.md.stored {
            // the dependents read the new expression of the unstored column
            for e in plan.path.elems.iter_mut().filter(|e| e.info.tbl_id == id) {
                e.info.unstored.insert(col.md.id, new_expr.clone());
            }
        }
        plan.schema_change = Some((col.md.id, new_expr));
        plan.writes_rows = recompute;
        Ok(Some(plan))
    })?;
    let Some(plan) = plan else {
        return Ok(status_dict(py, &UpdateStatus::default())?);
    };
    run_update(py, plan, false)
}

/// Fails if giving column `col_id` of `md` the value expression `new_expr` creates a dependency cycle among the
/// table's columns.
fn check_no_cycle(md: &crate::catalog::table::TableMd, col_id: ColId, new_expr: &Expr) -> Result<()> {
    let mut deps: HashMap<ColId, Vec<ColId>> = HashMap::new();
    let mut names: HashMap<ColId, String> = HashMap::new();
    for c in md.columns_at(None)? {
        names.insert(c.md.id, c.name.clone());
        let refs: Vec<ColId> = if c.md.id == col_id {
            new_expr.column_refs().iter().filter(|r| r.tbl_id == md.id).map(|r| r.col_id).collect()
        } else {
            match &c.md.value_expr {
                Some(e) => crate::expr::serde::from_json(e)?.column_refs().iter().filter(|r| r.tbl_id == md.id).map(|r| r.col_id).collect(),
                None => Vec::new(),
            }
        };
        deps.insert(c.md.id, refs);
    }
    // a path from col_id back to itself
    let mut stack = vec![(col_id, vec![col_id])];
    let mut seen: HashSet<ColId> = HashSet::new();
    while let Some((cur, path)) = stack.pop() {
        for d in deps.get(&cur).cloned().unwrap_or_default() {
            if d == col_id {
                let cycle: Vec<String> = path.iter().map(|c| py_repr(names.get(c).map(String::as_str).unwrap_or(""))).collect();
                crate::bail!(
                    UnsupportedOperation,
                    "The new value expressions create a circular dependency between columns {}.",
                    cycle.join(", ")
                );
            }
            if seen.insert(d) {
                let mut p = path.clone();
                p.push(d);
                stack.push((d, p));
            }
        }
    }
    Ok(())
}

/// `Table.delete()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version, where_))]
pub fn tbl_delete<'py>(py: Python<'py>, tbl_id: &str, version: Option<i64>, where_: Option<&Bound<'py, PyAny>>) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| {
        let md = cat.table(id)?;
        if md.is_view() || version.is_some() {
            let kind = if version.is_some() { "snapshot" } else { md.kind.display_name() };
            crate::bail!(UnsupportedOperation, "{}: Cannot delete from a {kind}.", cat.display_str(id)?);
        }
        Ok(())
    })?;
    let pred = where_arg(where_)?;
    let status = with_catalog(py, |cat| {
        if let Some(p) = &pred {
            check_where_bound(cat, id, p)?;
            update::validate_where(p)?;
        }
        update::delete(cat, id, pred.as_ref())
    })?;
    Ok(status_dict(py, &status)?)
}

/// `Table.revert()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version))]
pub fn tbl_revert(py: Python<'_>, tbl_id: &str, version: Option<i64>) -> Result<()> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| cat.revert(id, version))
}

/// Versions of a table, newest first (for `get_versions()`): tuples of
/// (version, created_at, user, change_type, inserts, updates, deletes, errors, schema_change).
#[pyfunction]
#[pyo3(signature = (tbl_id, n=None))]
pub fn tbl_versions<'py>(py: Python<'py>, tbl_id: &str, n: Option<&Bound<'py, PyAny>>) -> Result<Vec<Bound<'py, PyTuple>>> {
    let id = parse_uuid(tbl_id)?;
    let n = match n.filter(|n| !n.is_none()) {
        None => None,
        Some(obj) => {
            let v = if obj.is_instance_of::<pyo3::types::PyBool>() { None } else { obj.extract::<i64>().ok() };
            match v {
                Some(v) if v >= 1 => Some(v as usize),
                _ => crate::bail!(InvalidArgument, "Invalid value for `n`: {}", obj.str()?),
            }
        }
    };
    let versions = with_catalog(py, |cat| cat.version_history(id, n))?;
    let mut out = Vec::new();
    for v in versions {
        out.push(PyTuple::new(
            py,
            [
                v.version.into_pyobject(py)?.into_any(),
                v.created_at.into_pyobject(py)?.into_any(),
                v.user.into_pyobject(py)?.into_any(),
                v.change_type.into_pyobject(py)?.into_any(),
                v.inserts.into_pyobject(py)?.into_any(),
                v.updates.into_pyobject(py)?.into_any(),
                v.deletes.into_pyobject(py)?.into_any(),
                v.errors.into_pyobject(py)?.into_any(),
                v.schema_change.into_pyobject(py)?.into_any(),
            ],
        )?);
    }
    Ok(out)
}
