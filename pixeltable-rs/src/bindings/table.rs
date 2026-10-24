//! Functions behind table handles (`pixeltable.Table` holds a table id and an optional pinned version).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use super::catalog::{parse_schema, parse_uuid};
use super::expr::{PyExpr, as_expr};
use super::sources::Conduit;
use super::types::wrap_type;
use crate::catalog::path::fold_identifier;
use crate::catalog::runtime::with_catalog;
use crate::catalog::table::RowCountStats;
use crate::catalog::{IfExists, IfNotExists};
use crate::error::{Error, Result, py_repr};
use crate::exec::compute::AddComputedPlan;
use crate::exec::compute_rows::ComputeRowsPlan;
use crate::exec::insert::{self, InputRow, InsertPlan, UpdateStatus};
use crate::types::ColumnType;
use crate::value::Value;
use crate::exec::column_ref_expr;
use crate::value::py as vpy;

fn stats_dict<'py>(py: Python<'py>, s: &RowCountStats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ins_rows", s.ins_rows)?;
    d.set_item("del_rows", s.del_rows)?;
    d.set_item("upd_rows", s.upd_rows)?;
    d.set_item("num_excs", s.num_excs)?;
    d.set_item("computed_values", s.computed_values)?;
    Ok(d)
}

/// An `UpdateStatus` as the dict `pixeltable.UpdateStatus._from_core()` takes.
pub fn status_dict<'py>(py: Python<'py>, s: &UpdateStatus) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("updated_cols", s.updated_cols.clone())?;
    d.set_item("cols_with_excs", s.cols_with_excs.clone())?;
    d.set_item("row_count_stats", stats_dict(py, &s.row_count_stats)?)?;
    d.set_item("cascade_row_count_stats", stats_dict(py, &s.cascade_row_count_stats)?)?;
    Ok(d)
}

#[pyfunction]
pub fn tbl_kind(py: Python<'_>, tbl_id: &str) -> Result<String> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| Ok(cat.table(id)?.kind.display_name().to_string()))
}

#[pyfunction]
pub fn tbl_name(py: Python<'_>, tbl_id: &str) -> Result<String> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| Ok(cat.table(id)?.name.clone()))
}

#[pyfunction]
pub fn tbl_path(py: Python<'_>, tbl_id: &str) -> Result<String> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| Ok(cat.table_path(id)?.to_string()))
}

#[pyfunction]
pub fn tbl_version(py: Python<'_>, tbl_id: &str) -> Result<i64> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| Ok(cat.table(id)?.current_version))
}

#[pyfunction]
#[pyo3(signature = (tbl_id, version=None))]
pub fn tbl_columns(py: Python<'_>, tbl_id: &str, version: Option<i64>) -> Result<Vec<String>> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| Ok(cat.resolve_columns(id, version)?.into_iter().map(|c| c.name).collect()))
}

/// name -> ColumnType of the visible columns
#[pyfunction]
#[pyo3(signature = (tbl_id, version=None))]
pub fn tbl_schema<'py>(py: Python<'py>, tbl_id: &str, version: Option<i64>) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    let cols = with_catalog(py, |cat| cat.resolve_columns(id, version))?;
    let d = PyDict::new(py);
    for c in cols {
        d.set_item(&c.name, wrap_type(py, &c.md.col_type)?)?;
    }
    Ok(d)
}

/// `t.<name>` / `t['<name>']`
#[pyfunction]
#[pyo3(signature = (tbl_id, version, name))]
pub fn tbl_column_ref(py: Python<'_>, tbl_id: &str, version: Option<i64>, name: &str) -> Result<PyExpr> {
    let id = parse_uuid(tbl_id)?;
    let col = with_catalog(py, |cat| cat.resolve_column_by_name(id, version, name))?;
    match col {
        Some(c) => Ok(PyExpr::new(column_ref_expr(&c, version))),
        None => Err(Error::Attribute(format!("Unknown column: {name}"))),
    }
}

#[pyfunction]
#[pyo3(signature = (tbl_id, recursive))]
pub fn tbl_list_views(py: Python<'_>, tbl_id: &str, recursive: bool) -> Result<Vec<String>> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| cat.list_views(id, recursive))
}

fn parse_on_error(on_error: &str) -> Result<bool> {
    match on_error.to_ascii_lowercase().as_str() {
        "abort" => Ok(true),
        "ignore" => Ok(false),
        _ => Err(Error::Value(format!("Invalid value for on_error: {on_error}"))),
    }
}

/// A result value of a row-producing operation (`compute()`, `return_rows`): a media cell is reported by its file
/// URL (upstream `DataRow.get_stored_val()`), an in-memory media object as itself.
pub fn row_value_to_py<'py>(py: Python<'py>, v: &Value, t: &ColumnType) -> Result<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Media(s) => PyString::new(py, &crate::media::file_url(s)).into_any(),
        other => vpy::to_py(py, other, t)?,
    })
}

/// Upstream `QueryTableDataConduit.prepare_for_insert_into_table()` + the insert's query execution: the query's
/// result rows as input rows of `plan`.
pub fn query_input_rows(py: Python<'_>, q: &crate::exec::QuerySpec, plan: &InsertPlan) -> Result<Vec<InputRow>> {
    let input = with_catalog(py, |cat| crate::exec::QueryInput::snapshot(cat, q))?;
    let result = crate::exec::execute(&input, q, false)?;
    let mut targets = Vec::with_capacity(result.names.len());
    for name in &result.names {
        let col = plan.column(name).ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column name {name}"))?;
        if col.is_computed() {
            crate::bail!(UnsupportedOperation, "Value for computed column {name}");
        }
        targets.push(col);
    }
    let missing: Vec<&str> = plan
        .columns
        .iter()
        .filter(|c| c.is_required() && !result.names.contains(&c.name))
        .map(|c| c.name.as_str())
        .collect();
    if !missing.is_empty() {
        crate::bail!(MissingRequired, "Missing required column(s) ({})", missing.join(", "));
    }
    let mut rows = Vec::with_capacity(result.rows.len());
    for r in result.rows {
        let mut row = Vec::with_capacity(r.len());
        for (v, col) in r.into_iter().zip(targets.iter()) {
            let v = match v {
                // in-memory media (unstored computed columns of the source) are written by the insert
                v @ Value::PyObj(_) if col.col_type.is_media() => v,
                v => crate::value::cast_value(v, &col.col_type).map_err(|e| match e {
                    Error::Type(msg) => crate::pxt_err!(
                        UnsupportedOperation,
                        "Error in column {}: {}",
                        col.name,
                        super::sources::lower_first(&msg)
                    ),
                    other => other,
                })?,
            };
            row.push((col.col_id, v));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Converts any insert source (or a single row given as keyword arguments) into input rows of `plan`.
pub fn source_input_rows<'py>(
    py: Python<'py>,
    source: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
    plan: &InsertPlan,
) -> Result<Vec<InputRow>> {
    match source {
        None => {
            let row = kwargs.cloned().unwrap_or_else(|| PyDict::new(py));
            Ok(vec![super::sources::convert_row(row.as_any(), plan, true)?])
        }
        Some(s) => match super::sources::classify(py, s)? {
            Conduit::Query(q) => query_input_rows(py, &q, plan),
            c => super::sources::convert_rows(py, &c, plan, true),
        },
    }
}

/// Inserts input rows into a table: evaluates computed columns, commits, and returns the status dict (with the
/// stored rows if `return_rows`).
pub fn insert_rows<'py>(
    py: Python<'py>,
    plan: &InsertPlan,
    rows: Vec<InputRow>,
    fail_on_exception: bool,
    return_rows: bool,
) -> Result<Bound<'py, PyDict>> {
    let built = insert::build_rows(plan, rows, fail_on_exception)?;
    let returned = if return_rows {
        let list = PyList::empty(py);
        for r in &built.rows {
            let d = PyDict::new(py);
            for c in plan.columns.iter().filter(|c| c.stored) {
                d.set_item(&c.name, row_value_to_py(py, r.cell(c.col_id), &c.col_type)?)?;
            }
            list.append(d)?;
        }
        Some(list)
    } else {
        None
    };
    let status = with_catalog(py, |cat| insert::commit(cat, plan, built))?;
    let d = status_dict(py, &status)?;
    if let Some(rows) = returned {
        d.set_item("rows", rows)?;
    }
    Ok(d)
}

/// `Table.insert()`; returns the status dict.
#[pyfunction]
#[pyo3(signature = (tbl_id, version, source, kwargs, on_error, return_rows=false, source_format=None, schema_overrides=None))]
#[allow(clippy::too_many_arguments)]
pub fn tbl_insert<'py>(
    py: Python<'py>,
    tbl_id: &str,
    version: Option<i64>,
    source: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
    on_error: &str,
    return_rows: bool,
    source_format: Option<&str>,
    schema_overrides: Option<&Bound<'py, PyAny>>,
) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    if source_format.is_some() {
        return Err(Error::not_implemented("insert() with `source_format`"));
    }
    if schema_overrides.is_some_and(|o| !o.is_none()) {
        return Err(Error::not_implemented("insert() with `schema_overrides`"));
    }
    let source = source.filter(|s| !s.is_none());
    if let Some(s) = source {
        let seq = py.import("collections.abc")?.getattr("Sequence")?;
        if s.is_instance(&seq)? && !s.is_instance_of::<PyString>() && s.len()? == 0 {
            crate::bail!(UnsupportedOperation, "Cannot insert an empty sequence.");
        }
    }
    let fail_on_exception = parse_on_error(on_error)?;
    let plan = with_catalog(py, |cat| InsertPlan::new(cat, id, version))?;
    // conversion and computed columns run Python code (pydantic, generators, PIL, UDFs): no catalog lock here
    let rows = source_input_rows(py, source, kwargs, &plan)?;
    insert_rows(py, &plan, rows, fail_on_exception, return_rows)
}

/// `Table.compute()`: returns (rows as tuples, {name: ColumnType}, per-row error dicts).
#[pyfunction]
#[pyo3(signature = (tbl_id, version, source, outputs, on_error))]
pub fn tbl_compute<'py>(
    py: Python<'py>,
    tbl_id: &str,
    version: Option<i64>,
    source: &Bound<'py, PyAny>,
    outputs: Option<&Bound<'py, PyAny>>,
    on_error: &str,
) -> Result<(Bound<'py, PyList>, Bound<'py, PyDict>, Bound<'py, PyList>)> {
    let id = parse_uuid(tbl_id)?;
    let seq = py.import("collections.abc")?.getattr("Sequence")?;
    let is_str = source.is_instance_of::<PyString>() || source.is_instance_of::<pyo3::types::PyBytes>();
    if is_str || !source.is_instance(&seq)? || source.len()? == 0 {
        crate::bail!(
            UnsupportedOperation,
            "compute() requires a non-empty sequence of dicts or pydantic models; got {}",
            source.get_type().name()?
        );
    }
    let fail_on_exception = parse_on_error(on_error)?;
    let output_names: Option<Vec<String>> = match outputs.filter(|o| !o.is_none()) {
        None => None,
        Some(o) => Some(
            o.try_iter()?
                .map(|item| {
                    let item = item?;
                    if let Some(e) = as_expr(&item) {
                        match e.as_column_ref() {
                            Some(c) => Ok(c.name.clone()),
                            None => crate::bail!(InvalidArgument, "compute() outputs must be column names or references"),
                        }
                    } else {
                        Ok(item.extract::<String>()?)
                    }
                })
                .collect::<Result<_>>()?,
        ),
    };
    let list = PyList::new(py, source.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
    let conduit = super::sources::classify(py, list.as_any())?;
    if matches!(conduit, Conduit::Query(_)) {
        crate::bail!(
            UnsupportedOperation,
            "compute() requires a sequence of dicts or pydantic models; got {}",
            source.get_type().name()?
        );
    }
    let plan = with_catalog(py, |cat| ComputeRowsPlan::new(cat, id, version, output_names))?;
    let inputs = super::sources::convert_rows(py, &conduit, &plan.insert, false)?;
    let required = plan.required_input_names();
    if !required.is_empty() {
        let present = |row: &InputRow, name: &str| {
            plan.insert.column(name).is_some_and(|c| row.iter().any(|(id, _)| *id == c.col_id))
        };
        for row in &inputs {
            let missing: Vec<&str> = required.iter().filter(|n| !present(row, n)).map(|n| n.as_str()).collect();
            if !missing.is_empty() {
                let shown: Vec<String> = row
                    .iter()
                    .filter_map(|(id, v)| {
                        plan.insert.columns.iter().find(|c| c.col_id == *id).map(|c| {
                            let pv = vpy::to_py(py, v, &c.col_type).and_then(|o| o.repr().map(|r| r.to_string()));
                            format!("{}: {}", py_repr(&c.name), pv.unwrap_or_default())
                        })
                    })
                    .collect();
                crate::bail!(
                    MissingRequired,
                    "Missing required column(s) ({}) in row {{{}}}",
                    missing.join(", "),
                    shown.join(", ")
                );
            }
        }
    }
    let computed = plan.run(inputs, fail_on_exception)?;
    let col_types = PyDict::new(py);
    for (name, _, t) in &plan.outputs {
        col_types.set_item(name, wrap_type(py, t)?)?;
    }
    let data = PyList::empty(py);
    let errors = PyList::empty(py);
    for r in computed {
        let vals = r
            .values
            .iter()
            .zip(plan.outputs.iter())
            .map(|(v, (_, _, t))| row_value_to_py(py, v, t))
            .collect::<Result<Vec<_>>>()?;
        data.append(PyTuple::new(py, vals)?)?;
        let errs = PyDict::new(py);
        for (name, e) in r.errors {
            let ed = PyDict::new(py);
            ed.set_item("errortype", e.errortype)?;
            ed.set_item("errormsg", e.errormsg)?;
            errs.set_item(name, ed)?;
        }
        errors.append(errs)?;
    }
    Ok((data, col_types, errors))
}

/// Upstream `Column.calls_custom_udf`: the expression calls a function that is not part of Pixeltable.
fn calls_custom_udf(e: &crate::expr::Expr) -> bool {
    let mut custom = false;
    e.visit(&mut |n| {
        if let crate::expr::ExprKind::FunctionCall(fc) = &n.kind {
            if let crate::functions::FuncRef::Udf(call) = &fc.func {
                if !call.def.self_path.as_deref().is_some_and(|p| p.starts_with("pixeltable.")) {
                    custom = true;
                }
            }
        }
    });
    custom
}

/// `Table.get_metadata()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version=None))]
pub fn tbl_metadata<'py>(py: Python<'py>, tbl_id: &str, version: Option<i64>) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| {
        let md = cat.table(id)?.clone();
        let path = cat.table_path(id)?;
        let tbl_path = cat.tbl_path(id, version)?;
        let cols = cat.resolve_columns(id, version)?;
        let tbl_media_validation = md.current_schema()?.media_validation;
        let columns = PyDict::new(py);
        for c in &cols {
            let owner = cat.table(c.tbl_id)?;
            let cm = PyDict::new(py);
            cm.set_item("name", &c.name)?;
            cm.set_item("type_", c.md.col_type.repr())?;
            cm.set_item("version_added", c.md.schema_version_add)?;
            cm.set_item("is_stored", c.md.stored)?;
            cm.set_item("is_primary_key", c.md.is_pk)?;
            let mv = if c.md.col_type.is_media() {
                let owner_default = owner.current_schema().map(|s| s.media_validation).unwrap_or(tbl_media_validation);
                Some(c.md.media_validation.unwrap_or(owner_default).name())
            } else {
                None
            };
            cm.set_item("media_validation", mv)?;
            cm.set_item("is_computed", c.md.is_computed())?;
            let computed_with = match &c.md.value_expr {
                Some(e) => Some(crate::expr::display::display(&cat.with_column_names(&crate::expr::serde::from_json(e)?, &tbl_path))),
                None => None,
            };
            cm.set_item("computed_with", computed_with)?;
            let is_builtin = match &c.md.value_expr {
                Some(e) => Some(!calls_custom_udf(&*crate::expr::serde::from_json(e)?)),
                None => None,
            };
            cm.set_item("is_builtin", is_builtin)?;
            let depends_on: Option<Vec<(String, String)>> = match &c.md.value_expr {
                Some(e) => {
                    let e = cat.with_column_names(&crate::expr::serde::from_json(e)?, &tbl_path);
                    let mut deps: Vec<(String, String)> = e
                        .column_refs()
                        .iter()
                        .map(|r| (cat.table(r.tbl_id).map(|t| t.name.clone()).unwrap_or_default(), r.name.clone()))
                        .collect();
                    deps.sort();
                    deps.dedup();
                    Some(deps)
                }
                None => None,
            };
            cm.set_item("depends_on", depends_on)?;
            cm.set_item("defined_in", &owner.name)?;
            cm.set_item("comment", &c.md.comment)?;
            cm.set_item("custom_metadata", vpy::json_to_py(py, &c.md.custom_metadata)?)?;
            cm.set_item("is_iterator_col", false)?;
            cm.set_item("destination", &c.md.destination)?;
            columns.set_item(&c.name, cm)?;
        }
        let pk: Vec<String> = cols.iter().filter(|c| c.md.is_pk).map(|c| c.name.clone()).collect();
        let sv = md.schema_version_md(md.schema_version_at(version)?)?;
        let tv = version.unwrap_or(md.current_version);
        let created = md.version_md(tv).map(|v| v.created_at).unwrap_or(md.created_at);
        let d = PyDict::new(py);
        let uuid_cls = py.import("uuid")?.getattr("UUID")?;
        d.set_item("id", uuid_cls.call1((id.to_string(),))?)?;
        // a version-pinned handle is an anonymous snapshot
        let (name, path_str, kind) = match version {
            Some(v) => (format!("{}:{v}", md.name), format!("{path}:{v}"), "snapshot"),
            None => (md.name.clone(), path.to_string(), md.kind.display_name()),
        };
        d.set_item("name", name)?;
        d.set_item("path", path_str)?;
        d.set_item("kind", kind)?;
        d.set_item("columns", columns)?;
        d.set_item("indexes", super::indexes::indexes_metadata(py, cat, id)?)?;
        d.set_item("is_data_versioned", md.is_data_versioned)?;
        d.set_item("has_default_idxs", md.has_default_idxs)?;
        d.set_item("is_view", md.is_view() || version.is_some())?;
        d.set_item("is_snapshot", md.is_snapshot() || version.is_some())?;
        d.set_item("version", tv)?;
        let dt = py.import("datetime")?;
        let utc = dt.getattr("timezone")?.getattr("utc")?;
        d.set_item("version_created", dt.getattr("datetime")?.call_method1("fromtimestamp", (created, utc))?)?;
        d.set_item("schema_version", sv.schema_version)?;
        d.set_item("comment", &sv.comment)?;
        d.set_item("custom_metadata", vpy::json_to_py(py, &sv.custom_metadata)?)?;
        d.set_item("media_validation", sv.media_validation.name())?;
        d.set_item("primary_key", if pk.is_empty() { None } else { Some(pk) })?;
        d.set_item(
            "base",
            match version {
                Some(v) => cat.pinned_base_display(id, v)?,
                None => cat.base_display(id)?,
            },
        )?;
        let view_filter = match md.view.as_ref().and_then(|v| v.predicate.as_ref()) {
            Some(p) => Some(crate::expr::display::display(&cat.with_column_names(&crate::expr::serde::from_json(p)?, &[]))),
            None => None,
        };
        d.set_item("view_filter", view_filter)?;
        d.set_item("view_sample", py.None())?;
        d.set_item("iterator_call", py.None())?;
        Ok(d)
    })
}

fn check_single_column_kwarg(method: &str, value_form: &str, kwargs: &Bound<'_, PyDict>) -> Result<()> {
    if kwargs.len() != 1 {
        let names: Vec<String> = kwargs.keys().iter().map(|k| k.str().map(|s| s.to_string()).unwrap_or_default()).collect();
        crate::bail!(
            UnsupportedOperation,
            "{method}() requires exactly one keyword argument of the form {value_form}; got {} arguments instead ({})",
            kwargs.len(),
            names.join(", ")
        );
    }
    Ok(())
}

/// `Table.add_column(name=type)`
#[pyfunction]
#[pyo3(signature = (tbl_id, kwargs, if_exists))]
pub fn tbl_add_column(py: Python<'_>, tbl_id: &str, kwargs: &Bound<'_, PyDict>, if_exists: &str) -> Result<Vec<String>> {
    check_single_column_kwarg("add_column", "`col_name=col_type`", kwargs)?;
    if let Some((_, v)) = kwargs.iter().next() {
        let ok = v.is_instance_of::<PyDict>() || super::types::is_type_form(&v)?;
        if !ok {
            crate::bail!(
                InvalidArgument,
                "The argument to add_column() must be a type; did you intend to use add_computed_column() instead?"
            );
        }
    }
    tbl_add_columns(py, tbl_id, kwargs, if_exists)
}

/// `Table.add_computed_column(name=expr, ...)`; returns the status dict.
#[pyfunction]
#[pyo3(signature = (tbl_id, kwargs, stored, destination, custom_metadata, comment, on_error, if_exists))]
#[allow(clippy::too_many_arguments)]
pub fn tbl_add_computed_column<'py>(
    py: Python<'py>,
    tbl_id: &str,
    kwargs: &Bound<'py, PyDict>,
    stored: Option<bool>,
    destination: Option<&Bound<'py, PyAny>>,
    custom_metadata: Option<&Bound<'py, PyAny>>,
    comment: Option<&Bound<'py, PyAny>>,
    on_error: &str,
    if_exists: &str,
) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| cat.check_mutable(id, None, "add columns to"))?;
    check_single_column_kwarg("add_computed_column", "`col_name=col_type` or `col_name=expression`", kwargs)?;
    let (name, value) = kwargs.iter().next().ok_or_else(|| Error::internal("missing column"))?;
    let name: String = name.extract()?;
    if !crate::catalog::path::is_valid_identifier(&name, false) {
        crate::bail!(InvalidColumnName, "Invalid column name: {name}");
    }
    let folded = fold_identifier(&name);
    let fail_on_exception = match on_error.to_ascii_lowercase().as_str() {
        "abort" => true,
        "ignore" => false,
        _ => return Err(Error::Value(format!("Invalid value for on_error: {on_error}"))),
    };
    let ie = IfExists::parse(if_exists, "if_exists")?;
    let spec_dict = PyDict::new(py);
    spec_dict.set_item("value", &value)?;
    if let Some(s) = stored {
        spec_dict.set_item("stored", s)?;
    }
    if let Some(d) = destination.filter(|d| !d.is_none()) {
        spec_dict.set_item("destination", d)?;
    }
    if let Some(cm) = custom_metadata.filter(|c| !c.is_none()) {
        spec_dict.set_item("custom_metadata", cm)?;
    }
    if let Some(c) = comment.filter(|c| !c.is_none() && c.str().map(|s| !s.to_string().is_empty()).unwrap_or(true)) {
        spec_dict.set_item("comment", c)?;
    }
    let schema = PyDict::new(py);
    schema.set_item(&folded, spec_dict)?;
    let mut specs = parse_schema(&schema)?;
    let spec = specs.pop().ok_or_else(|| Error::internal("missing column spec"))?;
    let Some(plan) = with_catalog(py, |cat| AddComputedPlan::new(cat, id, spec, ie))? else {
        return Ok(status_dict(py, &UpdateStatus::default())?);
    };
    let mut stats = RowCountStats::default();
    let mut cols_with_excs = Vec::new();
    let mut saved = Vec::new();
    let values = match plan.compute(fail_on_exception, &mut stats, &mut cols_with_excs, &mut saved) {
        Ok(v) => v,
        Err(e) => {
            crate::exec::compute::discard_media(&saved);
            return Err(e);
        }
    };
    let status = with_catalog(py, |cat| {
        cat.adopt_media(saved);
        plan.commit(cat, values, stats, cols_with_excs)
    })?;
    Ok(status_dict(py, &status)?)
}

/// `Table.add_columns()` for plain (non-computed) columns.
#[pyfunction]
#[pyo3(signature = (tbl_id, schema, if_exists))]
pub fn tbl_add_columns(py: Python<'_>, tbl_id: &str, schema: &Bound<'_, PyDict>, if_exists: &str) -> Result<Vec<String>> {
    let id = parse_uuid(tbl_id)?;
    let ie = IfExists::parse(if_exists, "if_exists")?;
    let specs = parse_schema(schema)?;
    with_catalog(py, |cat| cat.add_columns(id, specs, ie))
}

/// `Table.drop_column()`; `column` is a name or a column reference.
#[pyfunction]
#[pyo3(signature = (tbl_id, column, if_not_exists))]
pub fn tbl_drop_column(py: Python<'_>, tbl_id: &str, column: &Bound<'_, PyAny>, if_not_exists: &str) -> Result<()> {
    let id = parse_uuid(tbl_id)?;
    let ine = IfNotExists::parse(if_not_exists, "if_not_exists")?;
    let name: String = if let Some(e) = as_expr(column) {
        match e.as_column_ref() {
            Some(c) => {
                let c = c.clone();
                let name = with_catalog(py, |cat| {
                    let visible = cat.resolve_columns(id, None)?;
                    match visible.iter().find(|v| v.tbl_id == c.tbl_id && v.md.id == c.col_id) {
                        Some(v) => Ok(v.name.clone()),
                        None => {
                            let tbl_name = cat.table(c.tbl_id).map(|t| t.name.clone()).unwrap_or_default();
                            Err(crate::pxt_err!(ColumnNotFound, "Unknown column: {tbl_name}.{}", c.name))
                        }
                    }
                });
                match name {
                    Ok(n) => n,
                    Err(Error::Pxt { code: crate::error::ErrorCode::ColumnNotFound, .. }) if ine == IfNotExists::Ignore => {
                        return Ok(());
                    }
                    Err(e) => return Err(e),
                }
            }
            None => crate::bail!(TypeMismatch, "Invalid column parameter type: {}", column.get_type().str()?),
        }
    } else if column.is_instance_of::<PyString>() {
        column.extract()?
    } else {
        crate::bail!(TypeMismatch, "Invalid column parameter type: {}", column.get_type().str()?);
    };
    with_catalog(py, |cat| cat.drop_column(id, &name, ine))
}

#[pyfunction]
pub fn tbl_rename_column(py: Python<'_>, tbl_id: &str, old_name: &str, new_name: &str) -> Result<()> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| cat.rename_column(id, old_name, new_name))
}
