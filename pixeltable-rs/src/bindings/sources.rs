//! Data sources of `insert()`, `create_table(source=...)` and `compute()` (upstream `TableDataConduit`): lists and
//! iterators of dicts, sequences of pydantic models, tables/queries, pandas DataFrames, and CSV, Excel, JSON and
//! Parquet files (read with pandas, `json` and pyarrow). Python objects are converted to values here, at the boundary; everything
//! after that (computed columns, storage) is in `exec`.

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
    /// a pandas DataFrame (a CSV or Excel file is read into one); `overrides`: `schema_overrides` for its columns
    Pandas { df: Bound<'py, PyAny>, overrides: Vec<(String, ColumnType)> },
    /// a Parquet file, as a `pyarrow.parquet.ParquetDataset`
    Parquet { ds: Bound<'py, PyAny>, overrides: Vec<(String, ColumnType)> },
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
    classify_with(py, source, None, None)
}

const SOURCE_FORMATS: [&str; 4] = ["csv", "excel", "parquet", "json"];

/// The local path of a file source (upstream `fetch_url(allow_local_file=True)`): a local path or `file://` URL as
/// is, a remote URL downloaded to the file cache.
fn source_file_path(py: Python<'_>, source: &Bound<'_, PyAny>) -> Result<String> {
    let s: String = py.import("os")?.call_method1("fspath", (source,))?.extract()?;
    if crate::media::is_local(&s) {
        let expanded: String = py.import("os.path")?.call_method1("expanduser", (crate::media::normalize_input(&s),))?.extract()?;
        return Ok(expanded);
    }
    Ok(crate::media::local_path(py, &s)?.to_string_lossy().into_owned())
}

/// Upstream `_parse_jsonl_from_path()` / `JsonTableDataConduit`: the rows of a JSON or JSON Lines file.
fn json_rows<'py>(py: Python<'py>, path: &str, extra: Option<&Bound<'py, PyDict>>) -> Result<Vec<Bound<'py, PyAny>>> {
    let json = py.import("json")?;
    let read_kwargs = PyDict::new(py);
    read_kwargs.set_item("encoding", "utf-8")?;
    let text: String = py.import("pathlib")?.getattr("Path")?.call1((path,))?.call_method("read_text", (), Some(&read_kwargs))?.extract()?;
    if path.to_ascii_lowercase().trim_end_matches('/').ends_with(".jsonl") {
        let mut rows = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let obj = json.call_method("loads", (line,), extra).map_err(|e| {
                if e.is_instance(py, &json.getattr("JSONDecodeError").expect("json.JSONDecodeError")) {
                    crate::pxt_err!(InvalidDataFormat, "Invalid JSONL line {}: {}", i + 1, py_repr(line))
                } else {
                    Error::Python(e)
                }
            })?;
            if !obj.is_instance_of::<PyDict>() {
                crate::bail!(InvalidArgument, "JSONL line {} must be a JSON object, got {}", i + 1, type_name(&obj));
            }
            rows.push(obj);
        }
        return Ok(rows);
    }
    let obj = json.call_method("loads", (text,), extra)?;
    match obj.cast::<PyList>() {
        Ok(list) => Ok(list.iter().collect()),
        Err(_) => crate::bail!(UnsupportedOperation, "Unsupported data source type: {}", type_str(&obj)),
    }
}

/// `classify()` with `source_format` and the extra arguments of a file reader (`extra_args`, or `insert()`'s keyword
/// arguments): upstream passes them to `pandas.read_csv()`, `pandas.read_excel()` and `json.loads()`.
pub fn classify_with<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    source_format: Option<&str>,
    extra: Option<&Bound<'py, PyDict>>,
) -> Result<Conduit<'py>> {
    if let Some(f) = source_format {
        if !SOURCE_FORMATS.contains(&f) {
            crate::bail!(InvalidArgument, "Invalid source_format {}; must be one of {SOURCE_FORMATS:?}", py_repr(f));
        }
    }
    let file_format = source_format.or_else(|| {
        let s = source.cast::<PyString>().ok()?.to_str().ok()?.to_ascii_lowercase();
        if s.contains(".csv") {
            Some("csv")
        } else if s.contains(".xls") {
            Some("excel")
        } else if s.contains(".json") {
            Some("json")
        } else if [".parquet", ".pq", ".parq"].iter().any(|x| s.contains(x)) {
            Some("parquet")
        } else {
            None
        }
    });
    let is_frame_or_query = as_query(py, source)?.is_some() || source.is_instance(&py.import("pandas")?.getattr("DataFrame")?)?;
    if let (Some(f), false) = (file_format, is_frame_or_query) {
        let pd = py.import("pandas")?;
        return match f {
            "csv" | "excel" => {
                let path = source_file_path(py, source)?;
                let reader = if f == "csv" { "read_csv" } else { "read_excel" };
                Ok(Conduit::Pandas { df: pd.call_method(reader, (path,), extra)?, overrides: Vec::new() })
            }
            "json" => Ok(Conduit::Rows(json_rows(py, &source_file_path(py, source)?, extra)?)),
            _ => {
                let path = source_file_path(py, source)?;
                let ds = py.import("pyarrow.parquet")?.getattr("ParquetDataset")?.call1((path,))?;
                Ok(Conduit::Parquet { ds, overrides: Vec::new() })
            }
        };
    }
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
        return Ok(Conduit::Pandas { df: source.clone(), overrides: Vec::new() });
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
        Conduit::Pandas { df, overrides } => {
            let cols = pandas_schema(py, df, overrides, &[])?;
            pandas_rows(py, df, &cols, plan)
        }
        Conduit::Parquet { ds, overrides } => {
            let cols = parquet_schema(py, ds, overrides, &[])?;
            parquet_rows(py, ds, &cols, plan)
        }
    }
}

/// A column of a DataFrame source: its name in the DataFrame, the Pixeltable column name it maps to and the type
/// inferred for it (upstream `df_infer_schema()` + `normalize_schema_names()`).
pub struct PdColumn {
    pub src: String,
    pub name: String,
    pub col_type: ColumnType,
}

/// Upstream `__pd_coltype_to_pxt_type()`: the type of a DataFrame column, from its dtype (an `object` column from its
/// values).
fn pandas_column_type(py: Python<'_>, df: &Bound<'_, PyAny>, src: &Bound<'_, PyAny>, dtype: &Bound<'_, PyAny>, nullable: bool) -> Result<ColumnType> {
    let pd = py.import("pandas")?;
    let api = pd.getattr("api")?.getattr("types")?;
    if api.call_method1("is_datetime64_any_dtype", (dtype,))?.is_truthy()? {
        return Ok(ColumnType::new(crate::types::TypeKind::Timestamp, nullable));
    }
    if dtype.is_instance(&pd.getattr("StringDtype")?)? {
        return Ok(ColumnType::string(nullable));
    }
    let kind: String = if api.call_method1("is_extension_array_dtype", (dtype,))?.is_truthy()? {
        String::new()
    } else {
        dtype.getattr("kind")?.extract()?
    };
    match kind.as_str() {
        "i" | "u" => return Ok(ColumnType::int(nullable)),
        "f" => return Ok(ColumnType::float(nullable)),
        "b" => return Ok(ColumnType::bool(nullable)),
        "U" | "S" => return Ok(ColumnType::string(nullable)),
        "O" => {
            // the type of the non-null values (all null: float, as pandas has it)
            let values = df.get_item(src)?.call_method0("dropna")?;
            if values.len()? == 0 {
                return Ok(ColumnType::float(nullable));
            }
            let mut inferred: Option<ColumnType> = None;
            for v in values.try_iter()? {
                let t = vpy::infer_literal(&v?)?.map(|(t, _)| t);
                inferred = match (inferred, t) {
                    (_, None) => None,
                    (None, Some(t)) => Some(t),
                    (Some(a), Some(t)) => a.supertype(&t, true),
                };
                if inferred.is_none() {
                    break;
                }
            }
            return Ok(inferred.unwrap_or_else(|| ColumnType::string(false)).with_nullable(nullable));
        }
        _ => {}
    }
    crate::bail!(
        InvalidType,
        "Could not infer Pixeltable type of column: {} (dtype: {})",
        src.str()?,
        dtype.str()?
    )
}

/// Upstream `normalize_pxt_col_name()`: an arbitrary column name as a Pixeltable identifier.
fn normalize_pxt_col_name(name: &str) -> String {
    let id: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let id = match id.chars().next() {
        Some(c) if c.is_ascii_digit() => format!("c_{id}"),
        Some('_') => format!("c{id}"),
        None => "c_".to_string(),
        _ => id,
    };
    fold_identifier(&id)
}

/// Upstream `normalize_schema_names()`: the source columns (name, type, None if none could be inferred) with
/// Pixeltable names (normalized to identifiers, made unique with a numeric suffix).
fn normalize_source_names(
    typed: Vec<(String, Option<ColumnType>)>,
    overrides: &[(String, ColumnType)],
    primary_key: &[String],
) -> Result<Vec<PdColumn>> {
    let untyped: Vec<&str> = typed.iter().filter(|(_, t)| t.is_none()).map(|(n, _)| n.as_str()).collect();
    if !untyped.is_empty() {
        crate::bail!(InvalidType, "Could not infer pixeltable type for column(s): {}", untyped.join(", "));
    }
    let extraneous: Vec<&str> =
        overrides.iter().filter(|(o, _)| !typed.iter().any(|(s, _)| s == o)).map(|(o, _)| o.as_str()).collect();
    if !extraneous.is_empty() {
        crate::bail!(
            UnsupportedOperation,
            "Some column(s) specified in `schema_overrides` are not present in the source: {}",
            extraneous.join(", ")
        );
    }
    let mut cols: Vec<PdColumn> = Vec::new();
    for (src, col_type) in typed {
        let base = normalize_pxt_col_name(&src);
        let mut name = base.clone();
        let mut n = 1;
        while crate::catalog::runtime::is_reserved_name(&name) || cols.iter().any(|c| c.name == name) {
            name = format!("{base}_{n}");
            n += 1;
        }
        cols.push(PdColumn { src, name, col_type: col_type.unwrap_or_else(|| ColumnType::invalid(true)) });
    }
    let missing_pk: Vec<&str> =
        primary_key.iter().filter(|pk| !cols.iter().any(|c| c.src == **pk)).map(|s| s.as_str()).collect();
    if !missing_pk.is_empty() {
        crate::bail!(ColumnNotFound, "Primary key column(s) are not found in the source: {}", missing_pk.join(", "));
    }
    Ok(cols)
}

/// Upstream `check_source_columns_are_insertable()`: every source column maps to a non-computed column of the table
/// of `plan`, and every required column has a source column.
fn check_insertable(cols: &[PdColumn], plan: &InsertPlan) -> Result<()> {
    let mut seen: Vec<(&str, &str)> = Vec::new();
    for c in cols {
        if let Some((_, prev)) = seen.iter().find(|(n, _)| *n == c.name) {
            crate::bail!(
                InvalidSchema,
                "Pixeltable column names are case-insensitive; columns {} and {} are the same",
                py_repr(prev),
                py_repr(&c.src)
            );
        }
        seen.push((&c.name, &c.src));
        let col = plan.column(&c.name).ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column name {}", c.name))?;
        if col.is_computed() {
            crate::bail!(UnsupportedOperation, "Value for computed column {}", c.name);
        }
    }
    let missing: Vec<&str> = plan
        .columns
        .iter()
        .filter(|c| c.is_required() && !cols.iter().any(|x| x.name == c.name))
        .map(|c| c.name.as_str())
        .collect();
    if !missing.is_empty() {
        crate::bail!(MissingRequired, "Missing required column(s) ({})", missing.join(", "));
    }
    Ok(())
}

/// Upstream `df_infer_schema()` and `normalize_schema_names()` for a DataFrame: its columns with their types
/// (`overrides` first) and Pixeltable names (normalized, made unique with a numeric suffix).
pub fn pandas_schema(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
    overrides: &[(String, ColumnType)],
    primary_key: &[String],
) -> Result<Vec<PdColumn>> {
    let mut typed: Vec<(String, Option<ColumnType>)> = Vec::new();
    let dtypes = df.getattr("dtypes")?;
    for (src_obj, dtype) in df.getattr("columns")?.try_iter()?.zip(dtypes.try_iter()?) {
        let (src_obj, dtype) = (src_obj?, dtype?);
        let src: String = src_obj.str()?.to_string();
        let col_type = match overrides.iter().find(|(o, _)| *o == src) {
            Some((_, t)) => t.clone(),
            None => pandas_column_type(py, df, &src_obj, &dtype, !primary_key.contains(&src))?,
        };
        typed.push((src, Some(col_type)));
    }
    let cols = normalize_source_names(typed, overrides, primary_key)?;
    for pk in primary_key {
        let series = df.get_item(pk)?;
        if series.call_method0("count")?.extract::<usize>()? < df.len()? {
            crate::bail!(UnsupportedOperation, "Primary key column `{pk}` cannot contain null values.");
        }
    }
    Ok(cols)
}

/// Upstream `PandasTableDataConduit.prepare_insert()`: checks that the DataFrame's columns can be inserted into the
/// table of `plan` and converts its rows (`_df_row_to_pxt_row()`: values by the inferred type of their column, NaN
/// and NaT as None).
fn pandas_rows(py: Python<'_>, df: &Bound<'_, PyAny>, cols: &[PdColumn], plan: &InsertPlan) -> Result<Vec<InputRow>> {
    check_insertable(cols, plan)?;
    let builtins = py.import("builtins")?;
    let pd = py.import("pandas")?;
    let uuid_cls = py.import("uuid")?.getattr("UUID")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("index", false)?;
    kwargs.set_item("name", py.None())?;
    let mut rows = Vec::new();
    for tuple in df.call_method("itertuples", (), Some(&kwargs))?.try_iter()? {
        let tuple = tuple?;
        let row = PyDict::new(py);
        for (c, val) in cols.iter().zip(tuple.try_iter()?) {
            let val = val?;
            let t = &c.col_type;
            let is_nan = val.is_instance_of::<pyo3::types::PyFloat>() && val.extract::<f64>()?.is_nan();
            let is_null = || -> Result<bool> { Ok(pd.call_method1("isnull", (&val,))?.is_truthy()?) };
            let nval = if t.is_float() {
                builtins.getattr("float")?.call1((&val,))?
            } else if is_nan {
                py.None().into_bound(py)
            } else if t.is_int() {
                builtins.getattr("int")?.call1((&val,))?
            } else if t.is_bool() {
                builtins.getattr("bool")?.call1((&val,))?
            } else if t.is_string() {
                builtins.getattr("str")?.call1((&val,))?
            } else if t.is_date() || t.is_timestamp() {
                if is_null()? {
                    py.None().into_bound(py)
                } else {
                    let ts = pd.getattr("Timestamp")?.call1((&val,))?;
                    if t.is_date() { ts.call_method0("date")? } else { ts.call_method0("to_pydatetime")? }
                }
            } else if t.is_uuid() {
                if is_null()? {
                    py.None().into_bound(py)
                } else if val.is_instance(&uuid_cls)? {
                    val
                } else {
                    uuid_cls.call1((&val,))?
                }
            } else {
                val
            };
            row.set_item(&c.name, nval)?;
        }
        rows.push(convert_row(row.as_any(), plan, false)?);
    }
    Ok(rows)
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

/// Upstream `utils.arrow.to_pxt_type()`: the type of an Arrow column (None if it has none).
fn arrow_column_type(py: Python<'_>, t: &Bound<'_, PyAny>, nullable: bool) -> Result<Option<ColumnType>> {
    use crate::types::TypeKind;
    let pa = py.import("pyarrow")?;
    if t.is_instance(&pa.getattr("TimestampType")?)? {
        return Ok(Some(ColumnType::new(TypeKind::Timestamp, nullable)));
    }
    if t.is_instance(&pa.getattr("StructType")?)? {
        return Ok(Some(ColumnType::json(nullable)));
    }
    if t.is_instance(&pa.getattr("FixedShapeTensorType")?)? {
        let Some(dtype) = arrow_column_type(py, &t.getattr("value_type")?, nullable)? else { return Ok(None) };
        let Some(dtype) = crate::types::DType::for_scalar(&dtype.kind) else { return Ok(None) };
        let shape: Vec<Option<u64>> = t.getattr("shape")?.extract::<Vec<u64>>()?.into_iter().map(Some).collect();
        return Ok(Some(ColumnType::new(TypeKind::Array { shape: Some(shape), dtype: Some(dtype) }, nullable)));
    }
    let kinds: [(&str, TypeKind); 18] = [
        ("string", TypeKind::String),
        ("large_string", TypeKind::String),
        ("bool_", TypeKind::Bool),
        ("int8", TypeKind::Int),
        ("int16", TypeKind::Int),
        ("int32", TypeKind::Int),
        ("int64", TypeKind::Int),
        ("uint8", TypeKind::Int),
        ("uint16", TypeKind::Int),
        ("uint32", TypeKind::Int),
        ("uint64", TypeKind::Int),
        ("float32", TypeKind::Float),
        ("float64", TypeKind::Float),
        ("date32", TypeKind::Date),
        ("date64", TypeKind::Date),
        ("uuid", TypeKind::Uuid),
        ("binary", TypeKind::Binary),
        ("large_binary", TypeKind::Invalid),
    ];
    for (ctor, kind) in kinds {
        if matches!(kind, TypeKind::Invalid) || !pa.hasattr(ctor)? {
            continue;
        }
        if t.call_method1("equals", (pa.call_method0(ctor)?,))?.is_truthy()? {
            return Ok(Some(ColumnType::new(kind, nullable)));
        }
    }
    Ok(None)
}

/// Upstream `ParquetTableDataConduit`: the columns of a Parquet dataset (`pyarrow.parquet.ParquetDataset`), typed
/// by their Arrow types (`overrides` first).
pub fn parquet_schema(
    py: Python<'_>,
    ds: &Bound<'_, PyAny>,
    overrides: &[(String, ColumnType)],
    primary_key: &[String],
) -> Result<Vec<PdColumn>> {
    let mut typed: Vec<(String, Option<ColumnType>)> = Vec::new();
    for field in ds.getattr("schema")?.try_iter()? {
        let field = field?;
        let src: String = field.getattr("name")?.extract()?;
        let col_type = match overrides.iter().find(|(o, _)| *o == src) {
            Some((_, t)) => Some(t.clone()),
            None => arrow_column_type(py, &field.getattr("type")?, !primary_key.contains(&src))?,
        };
        typed.push((src, col_type));
    }
    normalize_source_names(typed, overrides, primary_key)
}

/// Upstream `ParquetTableDataConduit.valid_row_batch()`: the rows of a Parquet dataset, each value converted by the
/// type of the table column it goes to (`utils.arrow._ar_val_to_pxt_val()`).
fn parquet_rows(py: Python<'_>, ds: &Bound<'_, PyAny>, cols: &[PdColumn], plan: &InsertPlan) -> Result<Vec<InputRow>> {
    check_insertable(cols, plan)?;
    let pa = py.import("pyarrow")?;
    let builtins = py.import("builtins")?;
    let uuid_cls = py.import("uuid")?.getattr("UUID")?;
    let tensor_type = pa.getattr("FixedShapeTensorType")?;
    let mut rows = Vec::new();
    for fragment in ds.getattr("fragments")?.try_iter()? {
        for batch in fragment?.call_method0("to_batches")?.try_iter()? {
            let batch = batch?;
            // column by column, as Python values (fixed-shape tensors as arrays)
            let mut columns: Vec<Vec<Bound<'_, PyAny>>> = Vec::with_capacity(cols.len());
            for k in 0..cols.len() {
                let mut col = batch.call_method1("column", (k,))?;
                let values = if col.getattr("type")?.is_instance(&tensor_type)? {
                    if col.hasattr("combine_chunks")? {
                        col = col.call_method0("combine_chunks")?;
                    }
                    col.call_method0("to_numpy_ndarray")?
                } else {
                    col.call_method0("to_pylist")?
                };
                columns.push(values.try_iter()?.collect::<PyResult<_>>()?);
            }
            let n = columns.first().map_or(0, Vec::len);
            for i in 0..n {
                let row = PyDict::new(py);
                for (c, values) in cols.iter().zip(&columns) {
                    let val = &values[i];
                    let t = plan.column(&c.name).map(|x| x.col_type.clone()).unwrap_or_else(|| c.col_type.clone());
                    let nval = if val.is_none() {
                        val.clone()
                    } else if t.is_float() {
                        builtins.getattr("float")?.call1((val,))?
                    } else if t.is_int() {
                        builtins.getattr("int")?.call1((val,))?
                    } else if t.is_bool() {
                        builtins.getattr("bool")?.call1((val,))?
                    } else if t.is_string() {
                        builtins.getattr("str")?.call1((val,))?
                    } else if t.is_uuid() && !val.is_instance(&uuid_cls)? {
                        if val.is_instance_of::<pyo3::types::PyBytes>() {
                            let kwargs = PyDict::new(py);
                            kwargs.set_item("bytes", val)?;
                            uuid_cls.call((), Some(&kwargs))?
                        } else {
                            uuid_cls.call1((val,))?
                        }
                    } else if (t.is_date() || t.is_timestamp()) && val.is_instance_of::<PyString>() {
                        let cls = py.import("datetime")?.getattr(if t.is_date() { "date" } else { "datetime" })?;
                        cls.call_method1("fromisoformat", (val,))?
                    } else {
                        val.clone()
                    };
                    row.set_item(&c.name, nval)?;
                }
                rows.push(convert_row(row.as_any(), plan, false)?);
            }
        }
    }
    Ok(rows)
}
