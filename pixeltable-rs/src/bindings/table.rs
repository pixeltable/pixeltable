//! Functions behind table handles (`pixeltable.Table` holds a table id and an optional pinned version).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use uuid::Uuid;

use super::catalog::{parse_schema, parse_uuid};
use super::expr::{PyExpr, as_expr};
use super::types::wrap_type;
use crate::catalog::path::fold_identifier;
use crate::catalog::runtime::with_catalog;
use crate::catalog::table::RowCountStats;
use crate::catalog::{IfExists, IfNotExists};
use crate::error::{Error, Result, py_repr};
use crate::exec::compute::AddComputedPlan;
use crate::exec::insert::{self, InputRow, InsertPlan, UpdateStatus};
use crate::exec::column_ref_expr;
use crate::value::Value;
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

/// Upstream `TableDataConduit.create()` for the sources supported so far: a list of dicts, an iterator of dicts,
/// or a list of pydantic models. Returns the rows as Python objects (each checked to be a dict later).
fn source_rows<'py>(source: &Bound<'py, PyAny>) -> Result<Vec<Bound<'py, PyAny>>> {
    let py = source.py();
    let base_model = py.import("pydantic")?.getattr("BaseModel")?;
    if let Ok(list) = source.cast::<PyList>() {
        if let Some(first) = list.iter().next() {
            if first.is_instance(&base_model)? {
                return list.iter().map(|m| Ok(m.call_method0("model_dump")?)).collect();
            }
        }
        if list.iter().all(|r| r.is_instance_of::<PyDict>()) {
            return Ok(list.iter().collect());
        }
    }
    let pd_df = py.import("pandas")?.getattr("DataFrame")?;
    if source.is_instance(&pd_df)? {
        return Err(Error::not_implemented("inserting from a pandas DataFrame"));
    }
    if source.is_instance_of::<PyString>() || source.hasattr("_tbl_ref")? || source.hasattr("_spec")? {
        return Err(Error::not_implemented(format!(
            "inserting from a source of type {}",
            source.get_type().str()?
        )));
    }
    let iterator_cls = py.import("collections.abc")?.getattr("Iterator")?;
    if source.is_instance(&iterator_cls)? {
        return source.try_iter()?.map(|r| Ok(r?)).collect();
    }
    crate::bail!(UnsupportedOperation, "Unsupported data source type: {}", source.get_type().str()?)
}

fn lower_first(msg: &str) -> String {
    let mut chars = msg.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Saves in-memory media (PIL images, bytes) to the table's media directory; returns the file path.
fn save_media(obj: &Bound<'_, PyAny>, media_dir: &std::path::Path) -> Result<String> {
    std::fs::create_dir_all(media_dir)?;
    let py = obj.py();
    if obj.is_instance(vpy::pil_image_cls(py)?)? {
        let path = media_dir.join(format!("{}.png", Uuid::new_v4().simple()));
        obj.call_method1("save", (path.to_string_lossy().to_string(),))?;
        return Ok(path.to_string_lossy().to_string());
    }
    if let Ok(b) = obj.extract::<Vec<u8>>() {
        let path = media_dir.join(Uuid::new_v4().simple().to_string());
        std::fs::write(&path, b)?;
        return Ok(path.to_string_lossy().to_string());
    }
    Err(Error::internal("unsupported in-memory media value"))
}

/// Upstream `RowDataTableDataConduit._translate_row()`.
fn convert_row(row: &Bound<'_, PyAny>, plan: &InsertPlan, media_dir: &std::path::Path) -> Result<InputRow> {
    let row_str = row.str().map(|s| s.to_string()).unwrap_or_default();
    let dict = row
        .cast::<PyDict>()
        .map_err(|_| crate::pxt_err!(UnsupportedOperation, "row {row_str} is not a dictionary"))?;
    let mut folded: Vec<(String, Bound<'_, PyAny>)> = Vec::new();
    for (k, v) in dict.iter() {
        let name: String = k.str()?.to_string();
        let f = fold_identifier(&name);
        if folded.iter().any(|(n, _)| *n == f) {
            let spellings: Vec<String> = dict
                .iter()
                .filter_map(|(k, _)| k.str().ok().map(|s| s.to_string()))
                .filter(|n| fold_identifier(n) == f)
                .map(|n| py_repr(&n))
                .collect();
            crate::bail!(InvalidSchema, "Column names are case-insensitive, but {} were specified", spellings.join(", "));
        }
        folded.push((f, v));
    }
    let missing: Vec<&str> = plan
        .columns
        .iter()
        .filter(|c| c.is_required() && !folded.iter().any(|(n, _)| *n == c.name))
        .map(|c| c.name.as_str())
        .collect();
    if !missing.is_empty() {
        crate::bail!(MissingRequired, "Missing required column(s) ({}) in row {row_str}", missing.join(", "));
    }
    let mut out = Vec::with_capacity(folded.len());
    for (name, v) in folded {
        let col = plan
            .column(&name)
            .ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column name {name} in row {row_str}"))?;
        if col.is_computed() {
            crate::bail!(UnsupportedOperation, "Value for computed column {name} in row {row_str}");
        }
        let mut value = match vpy::coerce(&v, &col.col_type) {
            Ok(value) => value,
            Err(Error::Type(msg)) => crate::bail!(
                UnsupportedOperation,
                "Error in column {name}: {}\nRow: {row_str}",
                lower_first(&msg)
            ),
            Err(e) => return Err(e),
        };
        if let Value::PyObj(o) = &value {
            value = Value::Media(save_media(o.bind(row.py()), media_dir)?);
        }
        out.push((col.col_id, value));
    }
    Ok(out)
}

/// `Table.insert()`; returns the status dict.
#[pyfunction]
#[pyo3(signature = (tbl_id, version, source, kwargs, on_error))]
pub fn tbl_insert<'py>(
    py: Python<'py>,
    tbl_id: &str,
    version: Option<i64>,
    source: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
    on_error: &str,
) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    let source = source.filter(|s| !s.is_none());
    if let Some(s) = source {
        let seq = py.import("collections.abc")?.getattr("Sequence")?;
        if s.is_instance(&seq)? && s.len()? == 0 {
            crate::bail!(UnsupportedOperation, "Cannot insert an empty sequence.");
        }
    }
    let fail_on_exception = match on_error.to_ascii_lowercase().as_str() {
        "abort" => true,
        "ignore" => false,
        _ => return Err(Error::Value(format!("Invalid value for on_error: {on_error}"))),
    };
    let rows: Vec<Bound<'py, PyAny>> = match source {
        None => vec![kwargs.cloned().unwrap_or_else(|| PyDict::new(py)).into_any()],
        Some(s) => source_rows(s)?,
    };
    let (plan, media_dir) = with_catalog(py, |cat| Ok((InsertPlan::new(cat, id, version)?, cat.store().media_dir(id))))?;
    // conversion and computed columns run Python code (pydantic, generators, PIL, UDFs): no catalog lock here
    let converted: Vec<InputRow> = rows.iter().map(|r| convert_row(r, &plan, &media_dir)).collect::<Result<_>>()?;
    let built = insert::build_rows(&plan, converted, fail_on_exception)?;
    let status = with_catalog(py, |cat| insert::commit(cat, &plan, built))?;
    Ok(status_dict(py, &status)?)
}

/// `Table.get_metadata()`
#[pyfunction]
#[pyo3(signature = (tbl_id, version=None))]
pub fn tbl_metadata<'py>(py: Python<'py>, tbl_id: &str, version: Option<i64>) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| {
        let md = cat.table(id)?.clone();
        let path = cat.table_path(id)?;
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
                Some(c.md.media_validation.unwrap_or(tbl_media_validation).name())
            } else {
                None
            };
            cm.set_item("media_validation", mv)?;
            cm.set_item("is_computed", c.md.is_computed())?;
            let computed_with = match &c.md.value_expr {
                Some(e) => Some(crate::expr::display::display(&*crate::expr::serde::from_json(e)?)),
                None => None,
            };
            cm.set_item("computed_with", computed_with)?;
            cm.set_item("is_builtin", if c.md.is_computed() { Some(true) } else { None })?;
            let depends_on: Option<Vec<(String, String)>> = match &c.md.value_expr {
                Some(e) => {
                    let e = crate::expr::serde::from_json(e)?;
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
        d.set_item("name", &md.name)?;
        d.set_item("path", path.to_string())?;
        d.set_item("kind", md.kind.display_name())?;
        d.set_item("columns", columns)?;
        d.set_item("indexes", PyDict::new(py))?;
        d.set_item("is_data_versioned", md.is_data_versioned)?;
        d.set_item("has_default_idxs", md.has_default_idxs)?;
        d.set_item("is_view", md.is_view())?;
        d.set_item("is_snapshot", md.is_snapshot())?;
        d.set_item("version", tv)?;
        let dt = py.import("datetime")?;
        let utc = dt.getattr("timezone")?.getattr("utc")?;
        d.set_item("version_created", dt.getattr("datetime")?.call_method1("fromtimestamp", (created, utc))?)?;
        d.set_item("schema_version", sv.schema_version)?;
        d.set_item("comment", &sv.comment)?;
        d.set_item("custom_metadata", vpy::json_to_py(py, &sv.custom_metadata)?)?;
        d.set_item("media_validation", sv.media_validation.name())?;
        d.set_item("primary_key", if pk.is_empty() { None } else { Some(pk) })?;
        d.set_item("base", cat.base_display(id)?)?;
        let view_filter = match md.view.as_ref().and_then(|v| v.predicate.as_ref()) {
            Some(p) => Some(crate::expr::display::display(&*crate::expr::serde::from_json(p)?)),
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
    let skip = with_catalog(py, |cat| cat.handle_existing_column(id, &folded, ie))?;
    if skip {
        return Ok(status_dict(py, &UpdateStatus::default())?);
    }
    crate::catalog::tables::validate_column_spec(&spec)?;
    let plan = with_catalog(py, |cat| AddComputedPlan::new(cat, id, spec))?;
    let mut stats = RowCountStats::default();
    let mut cols_with_excs = Vec::new();
    let values = plan.compute(fail_on_exception, &mut stats, &mut cols_with_excs)?;
    let status = with_catalog(py, |cat| plan.commit(cat, values, stats, cols_with_excs))?;
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
                with_catalog(py, |cat| {
                    let visible = cat.resolve_columns(id, None)?;
                    match visible.iter().find(|v| v.tbl_id == c.tbl_id && v.md.id == c.col_id) {
                        Some(v) => Ok(v.name.clone()),
                        None => {
                            let tbl_name = cat.table(c.tbl_id).map(|t| t.name.clone()).unwrap_or_default();
                            Err(crate::pxt_err!(ColumnNotFound, "Unknown column: {tbl_name}.{}", c.name))
                        }
                    }
                })?
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

/// Versions of a table, newest first (for `get_versions()`): tuples of
/// (version, created_at, change_type, inserts, updates, deletes, errors, schema_change).
#[pyfunction]
#[pyo3(signature = (tbl_id, n=None))]
pub fn tbl_versions<'py>(py: Python<'py>, tbl_id: &str, n: Option<usize>) -> Result<Vec<Bound<'py, PyTuple>>> {
    let id = parse_uuid(tbl_id)?;
    let md = with_catalog(py, |cat| Ok(cat.table(id)?.clone()))?;
    let mut out = Vec::new();
    for v in md.versions.iter().rev().take(n.unwrap_or(usize::MAX)) {
        out.push(PyTuple::new(
            py,
            [
                v.version.into_pyobject(py)?.into_any(),
                v.created_at.into_pyobject(py)?.into_any(),
                v.change_type.clone().into_pyobject(py)?.into_any(),
                v.row_count_stats.ins_rows.into_pyobject(py)?.into_any(),
                v.row_count_stats.upd_rows.into_pyobject(py)?.into_any(),
                v.row_count_stats.del_rows.into_pyobject(py)?.into_any(),
                v.row_count_stats.num_excs.into_pyobject(py)?.into_any(),
                v.schema_change.clone().into_pyobject(py)?.into_any(),
            ],
        )?);
    }
    Ok(out)
}
