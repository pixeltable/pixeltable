//! Module-level catalog functions behind `pixeltable.globals`-style API (`pxt.create_dir()`, `pxt.create_table()`,
//! ...) and the test hooks.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyString};
use uuid::Uuid;

use super::expr::{as_expr, from_object};
use super::query::PyQuerySpec;
use super::types::{is_type_form, normalize_type};
use crate::catalog::dirs::{ContentNode, assemble_contents, extract_paths};
use crate::catalog::path::{Path, fold_identifier};
use crate::catalog::runtime::{self, with_catalog};
use crate::catalog::table::MediaValidation;
use crate::catalog::tables::ColumnSpec;
use crate::catalog::{Catalog, IfExists, IfNotExists};
use crate::error::{Error, Result, py_repr};
use crate::value::Json;
use crate::value::py as vpy;

pub fn parse_uuid(s: &str) -> Result<Uuid> {
    Uuid::parse_str(s).map_err(|_| Error::internal(format!("invalid table id: {s}")))
}

fn py_str(obj: &Bound<'_, PyAny>) -> String {
    obj.str().map(|s| s.to_string()).unwrap_or_default()
}

fn py_repr_obj(obj: &Bound<'_, PyAny>) -> String {
    obj.repr().map(|s| s.to_string()).unwrap_or_default()
}

/// `json.dumps(obj)`-serializability check, returning the value as JSON.
fn to_json_checked(obj: &Bound<'_, PyAny>) -> std::result::Result<Json, String> {
    let json = obj.py().import("json").map_err(|e| e.to_string())?;
    match json.call_method1("dumps", (obj,)) {
        Ok(s) => {
            let s: String = s.extract().map_err(|e: PyErr| e.to_string())?;
            serde_json::from_str(&s).map_err(|e| e.to_string())
        }
        Err(e) => Err(e.value(obj.py()).str().map(|s| s.to_string()).unwrap_or_else(|_| e.to_string())),
    }
}

pub fn custom_metadata_json(obj: Option<&Bound<'_, PyAny>>) -> Result<Json> {
    match obj {
        None => Ok(Json::Null),
        Some(o) => to_json_checked(o).map_err(|_| crate::pxt_err!(InvalidArgument, "`custom_metadata` must be JSON-serializable")),
    }
}

pub fn comment_arg(obj: Option<&Bound<'_, PyAny>>) -> Result<Option<String>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            if !o.is_instance_of::<PyString>() {
                crate::bail!(InvalidArgument, "`comment` must be a string or None");
            }
            let s: String = o.extract()?;
            Ok(if s.is_empty() { None } else { Some(s) })
        }
    }
}

const VALID_SPEC_KEYS: [&str; 8] =
    ["type", "value", "primary_key", "stored", "media_validation", "destination", "custom_metadata", "comment"];

/// One schema entry (`name: type | ColumnSpec dict | Expr`) as a `ColumnSpec` (upstream `normalize_schema()` and
/// `Column._validate_column_spec()`).
fn parse_column_spec(name: &str, spec: &Bound<'_, PyAny>) -> Result<ColumnSpec> {
    let qname = py_repr(name);
    let mut cs = ColumnSpec {
        name: name.to_string(),
        col_type: crate::types::ColumnType::invalid(true),
        value_expr: None,
        is_pk: false,
        stored: None,
        media_validation: None,
        comment: None,
        custom_metadata: Json::Null,
        destination: None,
    };
    if let Some(e) = as_expr(spec) {
        cs.col_type = e.col_type.clone();
        cs.value_expr = Some(e);
        return Ok(cs);
    }
    if let Ok(d) = spec.cast::<PyDict>() {
        for (k, _) in d.iter() {
            let k_str = py_str(&k);
            if !VALID_SPEC_KEYS.contains(&k_str.as_str()) {
                crate::bail!(InvalidArgument, "Column {qname}: invalid key {}", py_repr_obj(&k));
            }
        }
        let type_obj = d.get_item("type")?;
        let value_obj = d.get_item("value")?;
        if type_obj.is_none() && value_obj.is_none() {
            crate::bail!(MissingRequired, "Column {qname}: 'type' or 'value' must be specified");
        }
        if let Some(t) = &type_obj {
            if !is_type_form(t)? {
                crate::bail!(InvalidArgument, "Column {qname}: 'type' must be a type; got {}", py_str(t));
            }
        }
        if let Some(v) = &value_obj {
            let e = from_object(v)?
                .ok_or_else(|| crate::pxt_err!(InvalidExpression, "Column {qname}: 'value' must be a Pixeltable expression."))?;
            if type_obj.is_some() {
                crate::bail!(InvalidArgument, "Column {qname}: 'type' is redundant if 'value' is specified");
            }
            cs.col_type = e.col_type.clone();
            cs.value_expr = Some(e);
        }
        if let Some(mv) = d.get_item("media_validation")? {
            let s: String = mv.extract().map_err(|_| {
                crate::pxt_err!(InvalidArgument, "Column {qname}: media_validation must be one of: ['on_read', 'on_write']")
            })?;
            cs.media_validation = Some(MediaValidation::parse(&s, &format!("Column {qname}: media_validation"))?);
        }
        if let Some(st) = d.get_item("stored")? {
            if !st.is_instance_of::<PyBool>() {
                crate::bail!(TypeMismatch, "Column {qname}: 'stored' must be a bool; got {}", py_str(&st));
            }
            cs.stored = Some(st.extract()?);
        }
        if let Some(c) = d.get_item("comment")? {
            if !c.is_instance_of::<PyString>() {
                crate::bail!(TypeMismatch, "Column {qname}: 'comment' must be a string; got {}", py_str(&c));
            }
            cs.comment = Some(c.extract()?);
        }
        if let Some(dest) = d.get_item("destination")? {
            if !dest.is_none() {
                let py = spec.py();
                let is_path = dest.is_instance(&py.import("pathlib")?.getattr("PurePath")?)?;
                if !(dest.is_instance_of::<PyString>() || is_path) {
                    crate::bail!(
                        TypeMismatch,
                        "Column {qname}: `destination` must be a string, a path, or a ConfigVar; got {}",
                        py_repr_obj(&dest)
                    );
                }
                cs.destination = Some(py_str(&dest));
            }
        }
        if let Some(cm) = d.get_item("custom_metadata")? {
            cs.custom_metadata = to_json_checked(&cm).map_err(|err| {
                crate::pxt_err!(InvalidArgument, "Column {qname}: `custom_metadata` must be JSON-serializable; got Error: {err}")
            })?;
        }
        if let Some(pk) = d.get_item("primary_key")? {
            cs.is_pk = pk.is_truthy()?;
        }
        if let Some(t) = &type_obj {
            cs.col_type = normalize_type(t, false)?;
        }
        return Ok(cs);
    }
    if is_type_form(spec)? {
        cs.col_type = normalize_type(spec, false)?;
        return Ok(cs);
    }
    crate::bail!(TypeMismatch, "Invalid spec for column {qname}: {}", py_repr_obj(spec))
}

/// A schema mapping as column specs, with case-insensitive names folded (upstream `normalize_schema()`).
pub fn parse_schema(schema: &Bound<'_, PyDict>) -> Result<Vec<ColumnSpec>> {
    let mut seen: Vec<(String, String)> = Vec::new();
    for (k, _) in schema.iter() {
        let name: String = k.extract().map_err(|_| crate::pxt_err!(InvalidColumnName, "Invalid column name: {}", py_str(&k)))?;
        let folded = fold_identifier(&name);
        if seen.iter().any(|(f, _)| *f == folded) {
            let spellings: Vec<String> = schema
                .iter()
                .filter_map(|(k, _)| k.extract::<String>().ok())
                .filter(|n| fold_identifier(n) == folded)
                .map(|n| py_repr(&n))
                .collect();
            crate::bail!(InvalidSchema, "Column names are case-insensitive, but {} were specified", spellings.join(", "));
        }
        seen.push((folded, name));
    }
    let mut specs = Vec::new();
    for (folded, name) in seen {
        let spec = schema.get_item(&name)?.ok_or_else(|| Error::internal("schema entry disappeared"))?;
        let mut cs = parse_column_spec(&folded, &spec)?;
        cs.name = folded;
        specs.push(cs);
    }
    Ok(specs)
}

fn normalize_primary_key(pk: Option<&Bound<'_, PyAny>>) -> Result<Vec<String>> {
    let err = || crate::pxt_err!(InvalidArgument, "primary_key must be a single column name or a list of column names");
    match pk {
        None => Ok(Vec::new()),
        Some(p) if p.is_none() => Ok(Vec::new()),
        Some(p) if p.is_instance_of::<PyString>() => Ok(vec![p.extract()?]),
        Some(p) => {
            let list = p.cast::<PyList>().map_err(|_| err())?;
            list.iter().map(|i| if i.is_instance_of::<PyString>() { Ok(i.extract::<String>()?) } else { Err(err()) }).collect()
        }
    }
}

// ------------------------------------------------------------------------------------------------ functions

#[pyfunction]
pub fn init(py: Python<'_>) -> Result<()> {
    with_catalog(py, |_| Ok(()))
}

#[pyfunction]
pub fn home(py: Python<'_>) -> Result<String> {
    with_catalog(py, |cat| Ok(cat.store().home().to_string_lossy().to_string()))
}

#[pyfunction]
pub fn reload_catalog(py: Python<'_>) {
    runtime::reload(py);
}

#[pyfunction]
pub fn reset_db(py: Python<'_>) -> Result<()> {
    runtime::reset(py)
}

#[pyfunction]
pub fn register_reserved_names(names: Vec<String>) {
    runtime::set_reserved_names(names);
}

#[pyfunction]
#[pyo3(signature = (path, if_exists, parents))]
pub fn create_dir(py: Python<'_>, path: &str, if_exists: &str, parents: bool) -> Result<String> {
    let p = Path::parse(path, false, false)?;
    let ie = IfExists::parse(if_exists, "if_exists")?;
    with_catalog(py, |cat| Ok(cat.create_dir(&p, ie, parents)?.to_string()))
}

#[pyfunction]
#[pyo3(signature = (path, force, if_not_exists))]
pub fn drop_dir(py: Python<'_>, path: &str, force: bool, if_not_exists: &str) -> Result<()> {
    let p = Path::parse(path, false, false)?;
    let ine = IfNotExists::parse(if_not_exists, "if_not_exists")?;
    with_catalog(py, |cat| cat.drop_dir(&p, ine, force))
}

#[pyfunction]
#[pyo3(name = "move", signature = (path, new_path, if_exists, if_not_exists))]
pub fn move_(py: Python<'_>, path: &str, new_path: &str, if_exists: &str, if_not_exists: &str) -> Result<()> {
    let ie = IfExists::parse(if_exists, "if_exists")?;
    if !matches!(ie, IfExists::Error | IfExists::Ignore) {
        crate::bail!(InvalidArgument, "`if_exists` must be one of 'error' or 'ignore'");
    }
    let ine = IfNotExists::parse(if_not_exists, "if_not_exists")?;
    let p = Path::parse(path, false, false)?;
    let np = Path::parse(new_path, false, false)?;
    if p == np {
        crate::bail!(UnsupportedOperation, "move(): source and destination cannot be identical");
    }
    if p.is_ancestor(&np) {
        crate::bail!(UnsupportedOperation, "move(): cannot move {} into its own subdirectory", py_repr(path));
    }
    with_catalog(py, |cat| cat.move_entry(&p, &np, ie, ine))
}

#[pyfunction]
pub fn get_dir_contents<'py>(py: Python<'py>, dir_path: &str, recursive: bool) -> Result<Bound<'py, PyDict>> {
    let p = Path::parse(dir_path, true, false)?;
    let nodes = with_catalog(py, |cat| cat.dir_contents(&p, recursive))?;
    let (mut dirs, mut tables) = (Vec::new(), Vec::new());
    assemble_contents(dir_path, &nodes, &mut dirs, &mut tables);
    dirs.sort();
    tables.sort();
    let d = PyDict::new(py);
    d.set_item("dirs", dirs)?;
    d.set_item("tables", tables)?;
    Ok(d)
}

#[pyfunction]
pub fn list_tables(py: Python<'_>, dir_path: &str, recursive: bool) -> Result<Vec<String>> {
    let p = Path::parse(dir_path, true, false)?;
    let nodes = with_catalog(py, |cat| cat.dir_contents(&p, recursive))?;
    Ok(extract_paths(&nodes, &p, false))
}

#[pyfunction]
pub fn list_dirs(py: Python<'_>, path: &str, recursive: bool) -> Result<Vec<String>> {
    let p = Path::parse(path, true, false)?;
    let nodes = with_catalog(py, |cat| cat.dir_contents(&p, recursive))?;
    Ok(extract_paths(&nodes, &p, true))
}

fn collect_ids(prefix: &str, nodes: &[ContentNode], out: &mut std::collections::HashMap<Uuid, String>) {
    for n in nodes {
        match n {
            ContentNode::Dir { name, entries } => {
                let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
                collect_ids(&path, entries, out);
            }
            ContentNode::Table { name, id } => {
                let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
                out.insert(*id, path);
            }
        }
    }
}

fn tree_nodes<'py>(
    py: Python<'py>,
    cat: &Catalog,
    prefix: &str,
    nodes: &[ContentNode],
    path_by_id: &std::collections::HashMap<Uuid, String>,
) -> Result<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    let mut sorted: Vec<&ContentNode> = nodes.iter().collect();
    sorted.sort_by_key(|n| match n {
        ContentNode::Dir { name, .. } | ContentNode::Table { name, .. } => name.clone(),
    });
    for n in sorted {
        let d = PyDict::new(py);
        match n {
            ContentNode::Dir { name, entries } => {
                if name.starts_with('_') {
                    continue;
                }
                let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
                d.set_item("name", name)?;
                d.set_item("path", &path)?;
                d.set_item("kind", "directory")?;
                d.set_item("entries", tree_nodes(py, cat, &path, entries, path_by_id)?)?;
            }
            ContentNode::Table { name, id } => {
                if name.starts_with('_') {
                    continue;
                }
                let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
                let md = cat.table(*id)?;
                let version: Option<i64> = if md.is_snapshot() { None } else { Some(md.current_version) };
                let base = md.view.as_ref().and_then(|v| {
                    path_by_id.get(&v.base_id).map(|bp| match v.base_version {
                        Some(ver) => format!("{bp}:{ver}"),
                        None => bp.clone(),
                    })
                });
                d.set_item("name", name)?;
                d.set_item("path", &path)?;
                d.set_item("kind", md.kind.display_name())?;
                d.set_item("version", version)?;
                d.set_item("error_count", md.total_error_count())?;
                d.set_item("base", base)?;
            }
        }
        list.append(d)?;
    }
    Ok(list)
}

#[pyfunction]
pub fn get_dir_tree<'py>(py: Python<'py>, path: &str) -> Result<Bound<'py, PyList>> {
    let p = Path::parse(path, true, false)?;
    with_catalog(py, |cat| {
        let nodes = cat.dir_contents(&p, true)?;
        let mut path_by_id = std::collections::HashMap::new();
        collect_ids("", &nodes, &mut path_by_id);
        tree_nodes(py, cat, "", &nodes, &path_by_id)
    })
}

/// Rows of `pxt.ls()`: [name, kind, version, base], sorted by name.
#[pyfunction]
pub fn ls_rows(py: Python<'_>, path: &str) -> Result<Vec<Vec<String>>> {
    let p = Path::parse(path, true, false)?;
    with_catalog(py, |cat| {
        let nodes = cat.dir_contents(&p, false)?;
        let mut rows = Vec::new();
        for n in nodes {
            match n {
                ContentNode::Dir { name, .. } => rows.push(vec![name, "dir".into(), String::new(), String::new()]),
                ContentNode::Table { name, id } => {
                    let md = cat.table(id)?;
                    let kind = md.kind.display_name().to_string();
                    let version = if md.is_snapshot() { String::new() } else { md.current_version.to_string() };
                    let base = cat.base_display(id)?.unwrap_or_default();
                    rows.push(vec![name, kind, version, base]);
                }
            }
        }
        rows.sort_by(|a, b| a[0].cmp(&b[0]));
        Ok(rows)
    })
}

/// Returns (table id, kind, created).
#[pyfunction]
#[pyo3(signature = (path, schema, source, primary_key, comment, custom_metadata, media_validation, if_exists, is_data_versioned))]
#[allow(clippy::too_many_arguments)]
pub fn create_table(
    py: Python<'_>,
    path: &str,
    schema: Option<&Bound<'_, PyAny>>,
    source: Option<&Bound<'_, PyAny>>,
    primary_key: Option<&Bound<'_, PyAny>>,
    comment: Option<&Bound<'_, PyAny>>,
    custom_metadata: Option<&Bound<'_, PyAny>>,
    media_validation: &str,
    if_exists: &str,
    is_data_versioned: bool,
) -> Result<(String, String, bool)> {
    let schema = schema.filter(|s| !s.is_none());
    let source = source.filter(|s| !s.is_none());
    if schema.is_none() == source.is_none() {
        crate::bail!(MissingRequired, "Either a `schema` or a `source` must be provided (but not both)");
    }
    if source.is_some() {
        return Err(Error::not_implemented("create_table() from a `source`"));
    }
    let schema = schema.ok_or_else(|| Error::internal("missing schema"))?;
    let schema_dict = match schema.cast::<PyDict>() {
        Ok(d) if !d.is_empty() => d.clone(),
        _ => crate::bail!(InvalidArgument, "`schema` must be a non-empty dictionary"),
    };
    let p = Path::parse(path, false, false)?;
    let ie = IfExists::parse(if_exists, "if_exists")?;
    let mv = MediaValidation::parse(media_validation, "media_validation")?;
    let pk = normalize_primary_key(primary_key)?;
    let comment = comment_arg(comment)?;
    let custom_metadata = custom_metadata_json(custom_metadata)?;
    let mut specs = parse_schema(&schema_dict)?;
    for pk_col in &pk {
        let folded = fold_identifier(pk_col);
        match specs.iter_mut().find(|s| s.name == folded) {
            Some(s) => s.is_pk = true,
            None => crate::bail!(ColumnNotFound, "Primary key column {} not found in table schema.", py_repr(pk_col)),
        }
    }
    let (id, created) =
        with_catalog(py, |cat| cat.create_table(&p, specs, ie, comment, custom_metadata, mv, is_data_versioned))?;
    Ok((id.to_string(), "table".to_string(), created))
}

/// Returns (view id, kind, created).
#[pyfunction]
#[pyo3(signature = (path, base, additional_columns, is_snapshot, iterator, comment, custom_metadata, media_validation, if_exists))]
#[allow(clippy::too_many_arguments)]
pub fn create_view(
    py: Python<'_>,
    path: &str,
    base: &Bound<'_, PyAny>,
    additional_columns: Option<&Bound<'_, PyAny>>,
    is_snapshot: bool,
    iterator: Option<&Bound<'_, PyAny>>,
    comment: Option<&Bound<'_, PyAny>>,
    custom_metadata: Option<&Bound<'_, PyAny>>,
    media_validation: &str,
    if_exists: &str,
) -> Result<(String, String, bool)> {
    let (base_id, base_version, predicate) = if let Ok(spec) = base.getattr("_spec") {
        let spec = spec.cast::<PyQuerySpec>().map_err(|_| Error::internal("Query._spec is not a QuerySpec"))?;
        let q = spec.get().q.clone();
        if q.select.is_some() {
            return Err(Error::not_implemented("create_view() over a query with a select list"));
        }
        if !q.order_by.is_empty() || q.limit.is_some() {
            crate::bail!(UnsupportedOperation, "Cannot use `order_by` or `limit` in a view definition");
        }
        (q.tbl_id, q.version, q.where_)
    } else if let Ok(tref) = base.getattr("_tbl_ref") {
        let (id, version): (String, Option<i64>) = tref.extract()?;
        (parse_uuid(&id)?, version, None)
    } else {
        crate::bail!(TypeMismatch, "`base` must be an instance of `Table` or `Query`");
    };
    let p = Path::parse(path, false, false)?;
    let ie = IfExists::parse(if_exists, "if_exists")?;
    let mv = MediaValidation::parse(media_validation, "media_validation")?;
    if additional_columns.is_some_and(|a| !a.is_none() && a.len().unwrap_or(0) > 0) {
        return Err(Error::not_implemented("create_view() with `additional_columns`"));
    }
    if iterator.is_some_and(|i| !i.is_none()) {
        return Err(Error::not_implemented("create_view() with an `iterator`"));
    }
    let comment = comment_arg(comment)?;
    let custom_metadata = custom_metadata_json(custom_metadata)?;
    let (id, created) = with_catalog(py, |cat| {
        cat.create_view(&p, base_id, base_version, predicate, is_snapshot, comment, custom_metadata, mv, ie)
    })?;
    let kind = with_catalog(py, |cat| Ok(cat.table(id)?.kind.display_name().to_string()))?;
    Ok((id.to_string(), kind, created))
}

/// Returns (table id, kind, version) or None.
#[pyfunction]
#[pyo3(signature = (path, if_not_exists))]
pub fn get_table(py: Python<'_>, path: &str, if_not_exists: &str) -> Result<Option<(String, String, Option<i64>)>> {
    let ine = IfNotExists::parse(if_not_exists, "if_not_exists")?;
    let p = Path::parse(path, false, true)?;
    let (result, warnings) = with_catalog(py, |cat| {
        Ok(match cat.get_table(&p, ine)? {
            Some(id) => (
                Some((id.to_string(), cat.table(id)?.kind.display_name().to_string(), p.version)),
                crate::functions::udf::load_warnings(cat, id)?,
            ),
            None => (None, Vec::new()),
        })
    })?;
    crate::functions::udf::emit_warnings(py, &warnings)?;
    Ok(result)
}

/// `table` is a path string or a table id (from a handle).
#[pyfunction]
#[pyo3(signature = (path, tbl_id, force, if_not_exists))]
pub fn drop_table(py: Python<'_>, path: Option<&str>, tbl_id: Option<&str>, force: bool, if_not_exists: &str) -> Result<()> {
    let p = match (path, tbl_id) {
        (Some(path), _) => Path::parse(path, false, false)?,
        (None, Some(id)) => {
            let id = parse_uuid(id)?;
            with_catalog(py, |cat| cat.table_path(id))?
        }
        (None, None) => return Err(Error::internal("drop_table() needs a path or an id")),
    };
    let ine = IfNotExists::parse(if_not_exists, "if_not_exists")?;
    with_catalog(py, |cat| cat.drop_table(&p, force, ine))
}

/// The current default time zone (for timestamps without one).
#[pyfunction]
pub fn default_time_zone(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    vpy::default_tz(py).cloned()
}
