//! Data sources of `insert()`, `create_table(source=...)` and `compute()` (upstream `TableDataConduit`): lists and
//! iterators of dicts, sequences of pydantic models, and tables/queries. Python objects are converted to values
//! here, at the boundary; everything after that (computed columns, storage) is in `exec`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use super::query::PyQuerySpec;
use crate::catalog::path::fold_identifier;
use crate::error::{Error, Result, py_repr};
use crate::exec::QuerySpec;
use crate::exec::insert::{InputRow, InsertPlan};
use crate::types::ColumnType;
use crate::value::py as vpy;

/// A classified data source.
pub enum Conduit<'py> {
    /// dicts (a list, or the materialized rows of an iterator)
    Rows(Vec<Bound<'py, PyAny>>),
    /// pydantic model instances of one class, dumped to dicts with folded keys
    Pydantic { model: Bound<'py, PyAny>, model_name: String, rows: Vec<Bound<'py, PyDict>> },
    /// a table (all its columns) or a query
    Query(QuerySpec),
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
}

fn type_str(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().str().map(|s| s.to_string()).unwrap_or_default()
}

/// The query behind a `Table` or `Query` object, if it is one.
pub fn as_query(py: Python<'_>, source: &Bound<'_, PyAny>) -> Result<Option<QuerySpec>> {
    if let Ok(spec) = source.getattr("_spec") {
        if let Ok(spec) = spec.cast::<PyQuerySpec>() {
            return Ok(Some(spec.get().q.clone()));
        }
    }
    if source.hasattr("_tbl_ref")? {
        let (id, version): (String, Option<i64>) = source.getattr("_tbl_ref")?.extract()?;
        let _ = py;
        return Ok(Some(QuerySpec::new(super::catalog::parse_uuid(&id)?, version)));
    }
    Ok(None)
}

fn dump_pydantic<'py>(py: Python<'py>, items: &[Bound<'py, PyAny>]) -> Result<Conduit<'py>> {
    let model = items[0].get_type().into_any();
    let model_name = type_name(&items[0]);
    let mut rows = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        if !item.get_type().is(&model) {
            crate::bail!(
                TypeMismatch,
                "Expected an instance of `{model_name}`; got `{}` (in row {i})",
                type_name(item)
            );
        }
        // model serializers are user code: dumping happens here, before any catalog access
        let dumped = item.call_method0("model_dump").map_err(|e| {
            let is_ser_err = py
                .import("pydantic_core")
                .and_then(|m| m.getattr("PydanticSerializationError"))
                .map(|c| e.get_type(py).is_subclass(&c).unwrap_or(false))
                .unwrap_or(false);
            if is_ser_err {
                crate::pxt_err!(
                    UnsupportedOperation,
                    "Row {i}: error serializing pydantic model:\n{}",
                    e.value(py).str().map(|s| s.to_string()).unwrap_or_default()
                )
            } else {
                Error::Python(e)
            }
        })?;
        let folded = PyDict::new(py);
        for (k, v) in dumped.cast::<PyDict>().map_err(|_| Error::internal("model_dump() did not return a dict"))?.iter() {
            folded.set_item(fold_identifier(&k.str()?.to_string()), v)?;
        }
        rows.push(folded);
    }
    Ok(Conduit::Pydantic { model, model_name, rows })
}

/// Upstream `TableDataConduit.create()` for the supported source types.
pub fn classify<'py>(py: Python<'py>, source: &Bound<'py, PyAny>) -> Result<Conduit<'py>> {
    let base_model = py.import("pydantic")?.getattr("BaseModel")?;
    let sequence = py.import("collections.abc")?.getattr("Sequence")?;
    let is_str = source.is_instance_of::<PyString>() || source.is_instance_of::<pyo3::types::PyBytes>();
    if !is_str && source.is_instance(&sequence)? && source.len()? > 0 {
        let items: Vec<Bound<'py, PyAny>> = source.try_iter()?.collect::<PyResult<_>>()?;
        if items[0].is_instance(&base_model)? {
            return dump_pydantic(py, &items);
        }
    }
    if let Some(q) = as_query(py, source)? {
        return Ok(Conduit::Query(q));
    }
    if source.is_instance(&py.import("pandas")?.getattr("DataFrame")?)? {
        return Err(Error::not_implemented("a pandas DataFrame as a data source"));
    }
    if source.is_instance_of::<PyString>() || source.is_instance(&py.import("pathlib")?.getattr("PurePath")?)? {
        return Err(Error::not_implemented("a file or URL as a data source"));
    }
    if let Ok(list) = source.cast::<PyList>() {
        if list.len() > 0 && list.iter().all(|r| r.is_instance_of::<PyDict>()) {
            return Ok(Conduit::Rows(list.iter().collect()));
        }
    }
    let iterator_cls = py.import("collections.abc")?.getattr("Iterator")?;
    if source.is_instance(&iterator_cls)? {
        return Ok(Conduit::Rows(source.try_iter()?.collect::<PyResult<_>>()?));
    }
    crate::bail!(UnsupportedOperation, "Unsupported data source type: {}", type_str(source))
}

pub fn lower_first(msg: &str) -> String {
    let mut chars = msg.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// The folded keys of a row dict, with upstream's case-collision check.
fn folded_items<'py>(dict: &Bound<'py, PyDict>) -> Result<Vec<(String, Bound<'py, PyAny>)>> {
    let mut folded: Vec<(String, Bound<'py, PyAny>)> = Vec::new();
    for (k, v) in dict.iter() {
        let f = fold_identifier(&k.str()?.to_string());
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
    Ok(folded)
}

/// Upstream `RowDataTableDataConduit._translate_row()`. `check_required`: false for `compute()`, which checks the
/// required columns of its outputs itself.
pub fn convert_row(row: &Bound<'_, PyAny>, plan: &InsertPlan, check_required: bool) -> Result<InputRow> {
    let row_str = row.str().map(|s| s.to_string()).unwrap_or_default();
    let dict = row
        .cast::<PyDict>()
        .map_err(|_| crate::pxt_err!(UnsupportedOperation, "row {row_str} is not a dictionary"))?;
    let folded = folded_items(dict)?;
    if check_required {
        let missing: Vec<&str> = plan
            .columns
            .iter()
            .filter(|c| c.is_required() && !folded.iter().any(|(n, _)| *n == c.name))
            .map(|c| c.name.as_str())
            .collect();
        if !missing.is_empty() {
            crate::bail!(MissingRequired, "Missing required column(s) ({}) in row {row_str}", missing.join(", "));
        }
    }
    let mut out = Vec::with_capacity(folded.len());
    for (name, v) in folded {
        let col = plan
            .column(&name)
            .ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column name {name} in row {row_str}"))?;
        if col.is_computed() {
            crate::bail!(UnsupportedOperation, "Value for computed column {name} in row {row_str}");
        }
        let value = match vpy::coerce(&v, &col.col_type) {
            Ok(value) => value,
            Err(Error::Type(msg)) => crate::bail!(
                UnsupportedOperation,
                "Error in column {name}: {}\nRow: {row_str}",
                lower_first(&msg)
            ),
            Err(e) => return Err(e),
        };
        out.push((col.col_id, value));
    }
    Ok(out)
}

/// Upstream `PydanticTableDataConduit._validate_pydantic_model()`.
fn validate_pydantic_model(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    model_name: &str,
    plan: &InsertPlan,
    check_required: bool,
) -> Result<()> {
    let fields = model.getattr("model_fields")?;
    let fields = fields.cast::<PyDict>().map_err(|_| Error::internal("model_fields is not a dict"))?;
    let model_fields = folded_items(fields)?;
    let has_field = |name: &str| model_fields.iter().any(|(n, _)| n == name);
    let mut missing: Vec<&str> =
        plan.columns.iter().filter(|c| c.is_required() && !has_field(&c.name)).map(|c| c.name.as_str()).collect();
    missing.sort();
    if check_required && !missing.is_empty() {
        crate::bail!(MissingRequired, "Pydantic model `{model_name}` is missing required columns: {}", missing.join(", "));
    }
    let mut computed: Vec<&str> =
        plan.columns.iter().filter(|c| c.is_computed() && has_field(&c.name)).map(|c| c.name.as_str()).collect();
    computed.sort();
    if !computed.is_empty() {
        crate::bail!(
            UnsupportedOperation,
            "Pydantic model `{model_name}` has fields for computed columns: {}",
            computed.join(", ")
        );
    }
    let mut common: Vec<&(String, Bound<'_, PyAny>)> =
        model_fields.iter().filter(|(n, _)| plan.column(n).is_some()).collect();
    if common.is_empty() {
        crate::bail!(
            UnsupportedOperation,
            "Pydantic model `{model_name}` has no fields that map to columns in table {}",
            py_repr(&plan.tbl_name)
        );
    }
    common.sort_by(|a, b| a.0.cmp(&b.0));
    let ts = py.import("pixeltable.type_system")?;
    for (name, info) in common {
        let Some(col) = plan.column(name) else { continue };
        let annotation = info.getattr("annotation")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("infer_pydantic_json", true)?;
        let inferred = ts.getattr("ColumnType")?.call_method("from_python_type", (&annotation,), Some(&kwargs))?;
        let model_type_name = annotation
            .getattr("__name__")
            .and_then(|n| n.extract::<String>())
            .unwrap_or_else(|_| annotation.str().map(|s| s.to_string()).unwrap_or_default());
        let Some(inferred) = super::types::core_type(&inferred) else {
            crate::bail!(InvalidType, "Pydantic model `{model_name}`: cannot infer Pixeltable type for column {}", py_repr(name));
        };
        if col.col_type.is_media() {
            if !inferred.is_string() {
                crate::bail!(
                    UnsupportedOperation,
                    "Column {} requires a `str` or `Path` field in `{model_name}`, but it is `{model_type_name}`",
                    py_repr(name)
                );
            }
            continue;
        }
        if !col.col_type.is_supertype_of(&inferred, true, true) {
            crate::bail!(
                TypeMismatch,
                "Pydantic model `{model_name}` has incompatible type `{model_type_name}` for column {} (of Pixeltable type `{}`)",
                py_repr(name),
                col.col_type.repr()
            );
        }
        let base_model = py.import("pydantic")?.getattr("BaseModel")?;
        let is_model = annotation.is_instance_of::<pyo3::types::PyType>()
            && annotation.cast::<pyo3::types::PyType>().map(|t| t.is_subclass(&base_model).unwrap_or(false)).unwrap_or(false);
        if is_model {
            let convertible: bool = py.import("pixeltable._typehints")?.call_method1("is_json_convertible", (&annotation,))?.extract()?;
            if !convertible {
                crate::bail!(
                    UnsupportedOperation,
                    "Pydantic model `{model_name}` has field {} with nested model `{model_type_name}`, which is not JSON-convertible",
                    py_repr(name)
                );
            }
        }
    }
    Ok(())
}

/// Upstream `PydanticTableDataConduit.prepare_for_insert_into_table()`. `check_required`: see `convert_row()`.
pub fn convert_pydantic_rows(
    py: Python<'_>,
    model: &Bound<'_, PyAny>,
    model_name: &str,
    rows: &[Bound<'_, PyDict>],
    plan: &InsertPlan,
    check_required: bool,
) -> Result<Vec<InputRow>> {
    validate_pydantic_model(py, model, model_name, plan, check_required)?;
    let mut required: Vec<&str> = plan.columns.iter().filter(|c| c.is_required()).map(|c| c.name.as_str()).collect();
    required.sort();
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        if check_required {
            for name in &required {
                if row.get_item(name)?.is_none_or(|v| v.is_none()) {
                    crate::bail!(MissingRequired, "Missing required column {} in row {i}", py_repr(name));
                }
            }
        }
        let mut converted = Vec::new();
        for (k, v) in row.iter() {
            let name: String = k.extract()?;
            // fields that are not table columns are ignored
            let Some(col) = plan.column(&name) else { continue };
            let value = match vpy::coerce(&v, &col.col_type) {
                Ok(value) => value,
                Err(Error::Type(msg)) => crate::bail!(
                    UnsupportedOperation,
                    "Error in column {} (row {i}): {}",
                    py_repr(&name),
                    lower_first(&msg)
                ),
                Err(e) => return Err(e),
            };
            converted.push((col.col_id, value));
        }
        out.push(converted);
    }
    Ok(out)
}

/// Converts the Python rows of a (non-query) conduit into input rows for `plan`.
pub fn convert_rows(py: Python<'_>, conduit: &Conduit<'_>, plan: &InsertPlan, check_required: bool) -> Result<Vec<InputRow>> {
    match conduit {
        Conduit::Rows(rows) => rows.iter().map(|r| convert_row(r, plan, check_required)).collect(),
        Conduit::Pydantic { model, model_name, rows } => convert_pydantic_rows(py, model, model_name, rows, plan, check_required),
        Conduit::Query(_) => Err(Error::internal("query sources are not converted from Python")),
    }
}

/// Upstream `_infer_schema_from_rows()`: column types (in first-seen order) of a list of dicts.
pub fn infer_schema_from_rows(
    rows: &[Bound<'_, PyAny>],
    overrides: &[(String, ColumnType)],
    primary_key: &[String],
) -> Result<Vec<(String, ColumnType)>> {
    let mut schema: Vec<(String, ColumnType)> = Vec::new();
    let mut cols_with_nones: Vec<String> = Vec::new();
    for (n, row) in rows.iter().enumerate() {
        let row_str = row.str().map(|s| s.to_string()).unwrap_or_default();
        let dict = row.cast::<PyDict>().map_err(|_| crate::pxt_err!(UnsupportedOperation, "row {row_str} is not a dictionary"))?;
        for (k, v) in dict.iter() {
            let col_name: String = k.str()?.to_string();
            if let Some((_, t)) = overrides.iter().find(|(o, _)| *o == col_name) {
                if !schema.iter().any(|(s, _)| *s == col_name) {
                    schema.push((col_name, t.clone()));
                }
                continue;
            }
            if v.is_none() {
                if !cols_with_nones.contains(&col_name) {
                    cols_with_nones.push(col_name);
                }
                continue;
            }
            let Some((t, _)) = vpy::infer_literal(&v)? else {
                crate::bail!(
                    InvalidType,
                    "Could not infer type for column `{col_name}`; the value in row {n} has an unsupported type: {}",
                    type_str(&v)
                );
            };
            // JSON type schemas are not inferred when importing data
            let t = if t.is_json() { ColumnType::json(false) } else { t };
            let t = t.with_nullable(!primary_key.contains(&col_name));
            match schema.iter_mut().find(|(s, _)| *s == col_name) {
                None => schema.push((col_name, t)),
                Some((_, existing)) => match existing.supertype(&t, true) {
                    Some(s) => *existing = s,
                    None => crate::bail!(
                        InvalidType,
                        "Could not infer type of column `{col_name}`; the value in row {n} does not match preceding type {}: {}\nConsider specifying the type explicitly in `schema_overrides`.",
                        existing.repr(),
                        v.repr()?
                    ),
                },
            }
        }
    }
    let entirely_none: Vec<&String> = cols_with_nones.iter().filter(|c| !schema.iter().any(|(s, _)| s == *c)).collect();
    if !entirely_none.is_empty() {
        crate::bail!(
            UnsupportedOperation,
            "The following columns have no non-null values: {}\nConsider specifying the type(s) explicitly in `schema_overrides`.",
            entirely_none.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        );
    }
    Ok(schema)
}

/// `schema_overrides` as normalized column types (upstream raises INVALID_TYPE with the column name).
pub fn parse_schema_overrides(overrides: Option<&Bound<'_, PyAny>>) -> Result<Vec<(String, ColumnType)>> {
    let Some(o) = overrides.filter(|o| !o.is_none()) else { return Ok(Vec::new()) };
    let d = o.cast::<PyDict>().map_err(|_| crate::pxt_err!(InvalidArgument, "`schema_overrides` must be a dictionary"))?;
    let mut out = Vec::new();
    for (k, v) in d.iter() {
        let name: String = k.str()?.to_string();
        let t = super::types::normalize_type(&v, false).map_err(|e| match e {
            Error::Pxt { message, .. } => {
                crate::pxt_err!(InvalidType, "Invalid type for schema_overrides[{}]: {message}", py_repr(&name))
            }
            other => other,
        })?;
        out.push((name, t));
    }
    Ok(out)
}

/// Upstream `normalize_schema_names()` for row sources (names must be valid identifiers).
pub fn check_source_names(
    schema: &[(String, ColumnType)],
    overrides: &[(String, ColumnType)],
    primary_key: &[String],
) -> Result<()> {
    let extraneous: Vec<&str> =
        overrides.iter().filter(|(o, _)| !schema.iter().any(|(s, _)| s == o)).map(|(o, _)| o.as_str()).collect();
    if !extraneous.is_empty() {
        crate::bail!(
            UnsupportedOperation,
            "Some column(s) specified in `schema_overrides` are not present in the source: {}",
            extraneous.join(", ")
        );
    }
    let invalid: Vec<&str> = schema
        .iter()
        .filter(|(n, _)| !crate::catalog::path::is_valid_identifier(n, false))
        .map(|(n, _)| n.as_str())
        .collect();
    if !invalid.is_empty() {
        crate::bail!(
            InvalidArgument,
            "Column names must be valid pixeltable identifiers. Invalid names: {}",
            invalid.join(", ")
        );
    }
    let missing_pk: Vec<&str> =
        primary_key.iter().filter(|pk| !schema.iter().any(|(s, _)| s == *pk)).map(|s| s.as_str()).collect();
    if !missing_pk.is_empty() {
        crate::bail!(ColumnNotFound, "Primary key column(s) are not found in the source: {}", missing_pk.join(", "));
    }
    Ok(())
}

/// `py_tuple` helper for building result rows.
pub fn tuple<'py>(py: Python<'py>, items: Vec<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, items)
}
