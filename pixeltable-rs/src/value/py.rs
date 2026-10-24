//! Python <-> `Value` conversion.
//!
//! - `coerce()` is upstream's `ColumnType.create_literal()`: it applies the documented conversions (int -> float,
//!   ISO strings -> timestamps, tuples -> lists, pydantic models -> dicts, ...) and then validates; validation
//!   failures are `Error::Type` with upstream's `TypeError` message.
//! - `validate()` is `ColumnType.validate_literal()` (no conversions).
//! - `infer_literal()` is `ColumnType.infer_literal_type()` plus the conversion, for expression literals.
//! - `to_py()` produces what a query returns for a value of a given type.

use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::types::{ColumnType, DType, JsonSchema, JsonSpec, TypeKind};
use crate::value::{ArrayData, Json, MAX_JSON_DEPTH, NdArray, Value, json_too_deep_msg};

static DATETIME: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static DATE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static TIMEDELTA: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static UUID_CLS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static NDARRAY: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static PIL_IMAGE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static BASE_MODEL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static ENUM_CLS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static PATH_CLS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static EPOCH: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ONE_US: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static DEFAULT_TZ: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

const DATE_EPOCH_ORDINAL: i64 = 719_163; // date(1970, 1, 1).toordinal()

fn datetime_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    DATETIME.import(py, "datetime", "datetime")
}
fn date_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    DATE.import(py, "datetime", "date")
}
fn uuid_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    UUID_CLS.import(py, "uuid", "UUID")
}
fn ndarray_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    NDARRAY.import(py, "numpy", "ndarray")
}
pub fn pil_image_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    PIL_IMAGE.import(py, "PIL.Image", "Image")
}
fn base_model_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    BASE_MODEL.import(py, "pydantic", "BaseModel")
}
fn enum_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    ENUM_CLS.import(py, "enum", "Enum")
}
fn path_cls(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    PATH_CLS.import(py, "pathlib", "PurePath")
}

fn epoch(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    EPOCH
        .get_or_try_init(py, || {
            let utc = py.import("datetime")?.getattr("timezone")?.getattr("utc")?;
            Ok::<_, PyErr>(datetime_cls(py)?.call1((1970, 1, 1, 0, 0, 0, 0, utc))?.unbind())
        })
        .map(|o| o.bind(py))
}

fn one_us(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    ONE_US
        .get_or_try_init(py, || {
            let kwargs = PyDict::new(py);
            kwargs.set_item("microseconds", 1)?;
            Ok::<_, PyErr>(TIMEDELTA.import(py, "datetime", "timedelta")?.call((), Some(&kwargs))?.unbind())
        })
        .map(|o| o.bind(py))
}

/// The name of the default time zone: `PIXELTABLE_TIME_ZONE`, then `TZ`, then the system zone, else UTC.
fn default_tz_name() -> String {
    for var in ["PIXELTABLE_TIME_ZONE", "TZ"] {
        if let Ok(v) = std::env::var(var) {
            let v = v.trim_start_matches(':').to_string();
            if !v.is_empty() {
                return v;
            }
        }
    }
    if let Ok(target) = std::fs::read_link("/etc/localtime") {
        let s = target.to_string_lossy().to_string();
        if let Some(idx) = s.find("zoneinfo/") {
            return s[idx + "zoneinfo/".len()..].to_string();
        }
    }
    "UTC".to_string()
}

/// The time zone in which naive timestamps are interpreted and in which query results are returned.
pub fn default_tz(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    DEFAULT_TZ
        .get_or_try_init(py, || {
            let zoneinfo = py.import("zoneinfo")?.getattr("ZoneInfo")?;
            let tz = match zoneinfo.call1((default_tz_name(),)) {
                Ok(tz) => tz,
                Err(_) => zoneinfo.call1(("UTC",))?,
            };
            Ok::<_, PyErr>(tz.unbind())
        })
        .map(|o| o.bind(py))
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "object".to_string())
}

fn py_str(obj: &Bound<'_, PyAny>) -> String {
    obj.str().map(|s| s.to_string()).unwrap_or_else(|_| "<unprintable>".to_string())
}

fn py_repr(obj: &Bound<'_, PyAny>) -> String {
    obj.repr().map(|s| s.to_string()).unwrap_or_else(|_| "<unprintable>".to_string())
}

fn type_err(msg: impl Into<String>) -> Error {
    Error::Type(msg.into())
}

/// A Python str as a Rust string. A str that UTF-8 cannot encode (a lone surrogate) is a `TypeError` with Python's
/// message, so that insert reports it as an invalid value of its column.
fn str_value(obj: &Bound<'_, PyAny>) -> Result<String> {
    let py = obj.py();
    obj.extract::<String>().map_err(|e| {
        if e.is_instance_of::<pyo3::exceptions::PyUnicodeEncodeError>(py) {
            type_err(e.value(py).to_string())
        } else {
            e.into()
        }
    })
}

/// Fails once JSON containers are nested more than `MAX_JSON_DEPTH` deep (`depth` is the level of the container
/// being entered, starting at 1).
fn check_json_depth(depth: usize) -> Result<()> {
    if depth > MAX_JSON_DEPTH { Err(type_err(json_too_deep_msg())) } else { Ok(()) }
}

// ------------------------------------------------------------------------------------------------ scalars

pub fn timestamp_to_micros(dt: &Bound<'_, PyAny>) -> Result<i64> {
    let py = dt.py();
    let mut dt = dt.clone();
    if dt.getattr("tzinfo")?.is_none() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("tzinfo", default_tz(py)?)?;
        dt = dt.call_method("replace", (), Some(&kwargs))?;
    }
    Ok(dt.sub(epoch(py)?)?.floor_div(one_us(py)?)?.extract::<i64>()?)
}

pub fn micros_to_timestamp<'py>(py: Python<'py>, micros: i64) -> PyResult<Bound<'py, PyAny>> {
    let delta = one_us(py)?.mul(micros)?;
    epoch(py)?.add(delta)?.call_method1("astimezone", (default_tz(py)?,))
}

fn date_to_days(d: &Bound<'_, PyAny>) -> Result<i32> {
    let ordinal: i64 = d.call_method0("toordinal")?.extract()?;
    i32::try_from(ordinal - DATE_EPOCH_ORDINAL).map_err(|_| Error::internal("date out of range"))
}

fn days_to_date<'py>(py: Python<'py>, days: i32) -> PyResult<Bound<'py, PyAny>> {
    date_cls(py)?.call_method1("fromordinal", (days as i64 + DATE_EPOCH_ORDINAL,))
}

fn uuid_from_py(obj: &Bound<'_, PyAny>) -> Result<uuid::Uuid> {
    let bytes: Vec<u8> = obj.getattr("bytes")?.extract()?;
    uuid::Uuid::from_slice(&bytes).map_err(|e| Error::internal(format!("invalid UUID bytes: {e}")))
}

// ------------------------------------------------------------------------------------------------ arrays

fn dtype_of(arr: &Bound<'_, PyAny>) -> Result<Option<DType>> {
    let dt = arr.getattr("dtype")?;
    let kind: String = dt.getattr("kind")?.extract()?;
    if kind == "U" {
        return Ok(Some(DType::Str));
    }
    let name: String = dt.getattr("name")?.extract()?;
    Ok(DType::from_name(&name))
}

fn dtype_str(arr: &Bound<'_, PyAny>) -> String {
    arr.getattr("dtype").map(|d| py_str(&d)).unwrap_or_default()
}

fn shape_of(arr: &Bound<'_, PyAny>) -> Result<Vec<usize>> {
    Ok(arr.getattr("shape")?.extract()?)
}

fn typed_vec<T: numpy::Element + Clone>(arr: &Bound<'_, PyAny>) -> Result<Vec<T>> {
    let a = arr.cast::<PyArrayDyn<T>>().map_err(|e| Error::internal(format!("array conversion: {e}")))?;
    Ok(a.to_owned_array().iter().cloned().collect())
}

/// Converts a numpy array of a supported dtype into an `NdArray`.
pub fn ndarray_from_py(arr: &Bound<'_, PyAny>) -> Result<NdArray> {
    let py = arr.py();
    let dtype = dtype_of(arr)?.ok_or_else(|| type_err(format!("Unsupported dtype {}", dtype_str(arr))))?;
    let shape = shape_of(arr)?;
    let data = match dtype {
        DType::Bool => ArrayData::Bool(typed_vec::<bool>(arr)?),
        DType::U8 => ArrayData::U8(typed_vec::<u8>(arr)?),
        DType::U16 => ArrayData::U16(typed_vec::<u16>(arr)?),
        DType::U32 => ArrayData::U32(typed_vec::<u32>(arr)?),
        DType::U64 => ArrayData::U64(typed_vec::<u64>(arr)?),
        DType::I8 => ArrayData::I8(typed_vec::<i8>(arr)?),
        DType::I16 => ArrayData::I16(typed_vec::<i16>(arr)?),
        DType::I32 => ArrayData::I32(typed_vec::<i32>(arr)?),
        DType::I64 => ArrayData::I64(typed_vec::<i64>(arr)?),
        DType::F16 => {
            let np = py.import("numpy")?;
            let bits = arr.call_method1("view", (np.getattr("uint16")?,))?;
            ArrayData::F16(typed_vec::<u16>(&bits)?)
        }
        DType::F32 => ArrayData::F32(typed_vec::<f32>(arr)?),
        DType::F64 => ArrayData::F64(typed_vec::<f64>(arr)?),
        DType::Str => {
            let flat = arr.call_method0("ravel")?.call_method0("tolist")?;
            ArrayData::Str(flat.extract()?)
        }
    };
    Ok(NdArray { shape, data })
}

pub fn ndarray_to_py<'py>(py: Python<'py>, a: &NdArray) -> PyResult<Bound<'py, PyAny>> {
    fn mk<'py, T: numpy::Element + Clone>(py: Python<'py>, v: &[T], shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
        let flat = numpy::PyArray1::from_vec(py, v.to_vec());
        Ok(flat.reshape(shape.to_vec())?.into_any())
    }
    let shape = &a.shape;
    match &a.data {
        ArrayData::Bool(v) => mk(py, v, shape),
        ArrayData::U8(v) => mk(py, v, shape),
        ArrayData::U16(v) => mk(py, v, shape),
        ArrayData::U32(v) => mk(py, v, shape),
        ArrayData::U64(v) => mk(py, v, shape),
        ArrayData::I8(v) => mk(py, v, shape),
        ArrayData::I16(v) => mk(py, v, shape),
        ArrayData::I32(v) => mk(py, v, shape),
        ArrayData::I64(v) => mk(py, v, shape),
        ArrayData::F16(v) => {
            let np = py.import("numpy")?;
            mk(py, v, shape)?.call_method1("view", (np.getattr("float16")?,))
        }
        ArrayData::F32(v) => mk(py, v, shape),
        ArrayData::F64(v) => mk(py, v, shape),
        ArrayData::Str(v) => {
            let np = py.import("numpy")?;
            let list = PyList::new(py, v)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", np.getattr("str_")?)?;
            np.call_method("array", (list,), Some(&kwargs))?.call_method1("reshape", (PyTuple::new(py, shape)?,))
        }
    }
}

fn shape_repr_py(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
    if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) }
}

fn validate_array(obj: &Bound<'_, PyAny>, shape: &Option<crate::types::Shape>, dtype: &Option<DType>) -> Result<()> {
    let py = obj.py();
    if !obj.is_instance(ndarray_cls(py)?)? {
        return Err(type_err(format!("Expected numpy.ndarray, got {}", type_name(obj))));
    }
    let val_dtype = dtype_of(obj)?;
    let val_dtype_str = dtype_str(obj);
    let col_dtype_str = match dtype {
        Some(DType::Str) => "<U0".to_string(),
        Some(d) => d.name().to_string(),
        None => "None".to_string(),
    };
    if let Some(d) = dtype {
        let ok = match d {
            DType::Str => val_dtype == Some(DType::Str),
            d => val_dtype == Some(*d),
        };
        if !ok {
            return Err(type_err(format!(
                "Expected numpy.ndarray of dtype {col_dtype_str}, got numpy.ndarray of dtype {val_dtype_str}"
            )));
        }
    }
    if val_dtype.is_none() {
        return Err(type_err(format!("Unsupported dtype {val_dtype_str}")));
    }
    if let Some(expected) = shape {
        let val_shape = shape_of(obj)?;
        let mismatch = val_shape.len() != expected.len()
            || val_shape.iter().zip(expected.iter()).any(|(n1, n2)| matches!(n2, Some(n2) if *n2 as usize != *n1));
        if mismatch {
            return Err(type_err(format!(
                "Expected numpy.ndarray({}, dtype={}), got numpy.ndarray({}, dtype={})",
                crate::types::shape_repr(expected),
                col_dtype_str,
                shape_repr_py(&val_shape),
                val_dtype_str
            )));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ json

fn is_valid_json_literal(obj: &Bound<'_, PyAny>, depth: usize) -> Result<bool> {
    let py = obj.py();
    if obj.is_none()
        || obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyInt>()
        || obj.is_instance_of::<PyFloat>()
        || obj.is_instance_of::<PyBool>()
        || obj.is_instance_of::<PyBytes>()
        || obj.is_instance(pil_image_cls(py)?)?
    {
        return Ok(true);
    }
    if obj.is_instance(ndarray_cls(py)?)? {
        return Ok(dtype_of(obj)?.is_some());
    }
    if let Ok(list) = obj.cast::<PyList>() {
        check_json_depth(depth)?;
        for item in list.iter() {
            if !is_valid_json_literal(&item, depth + 1)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        check_json_depth(depth)?;
        for item in tuple.iter() {
            if !is_valid_json_literal(&item, depth + 1)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        check_json_depth(depth)?;
        for (k, v) in dict.iter() {
            if !k.is_instance_of::<PyString>() || !is_valid_json_literal(&v, depth + 1)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    Ok(false)
}

/// Converts a (validated) Python JSON literal into a JSON value.
pub fn json_from_py(obj: &Bound<'_, PyAny>) -> Result<Json> {
    json_from_py_at(obj, 1)
}

fn json_from_py_at(obj: &Bound<'_, PyAny>, depth: usize) -> Result<Json> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(Json::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Json::Bool(obj.extract()?));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Json::from(i));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(Json::from(u));
        }
        let f: f64 = obj.extract()?;
        return Ok(serde_json::Number::from_f64(f).map_or(Json::Null, Json::Number));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f: f64 = obj.extract()?;
        return serde_json::Number::from_f64(f).map(Json::Number).ok_or_else(|| {
            Error::not_implemented(format!("storing the non-finite float {} inside a JSON value", py_repr(obj)))
        });
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Json::String(str_value(obj)?));
    }
    if obj.is_instance(base_model_cls(py)?)? {
        return json_from_py_at(&obj.call_method0("model_dump")?, depth);
    }
    if let Ok(list) = obj.cast::<PyList>() {
        check_json_depth(depth)?;
        return Ok(Json::Array(list.iter().map(|i| json_from_py_at(&i, depth + 1)).collect::<Result<_>>()?));
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        check_json_depth(depth)?;
        return Ok(Json::Array(tuple.iter().map(|i| json_from_py_at(&i, depth + 1)).collect::<Result<_>>()?));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        check_json_depth(depth)?;
        let mut m = serde_json::Map::new();
        for (k, v) in dict.iter() {
            m.insert(str_value(&k)?, json_from_py_at(&v, depth + 1)?);
        }
        return Ok(Json::Object(m));
    }
    Err(Error::not_implemented(format!(
        "storing a value of type `{}` inside a JSON value",
        type_name(obj)
    )))
}

pub fn json_to_py<'py>(py: Python<'py>, j: &Json) -> PyResult<Bound<'py, PyAny>> {
    Ok(match j {
        Json::Null => py.None().into_bound(py),
        Json::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Json::String(s) => PyString::new(py, s).into_any(),
        Json::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Json::Object(m) => {
            let dict = PyDict::new(py);
            for (k, v) in m {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.into_any()
        }
    })
}

fn validate_json_schema(obj: &Bound<'_, PyAny>, schema: &JsonSchema) -> Result<()> {
    match &schema.spec {
        JsonSpec::List(fixed) => {
            let list = obj.cast::<PyList>().map_err(|_| {
                type_err(format!("Expected a list; got `{}`: {}", type_name(obj), py_repr(obj)))
            })?;
            let n = list.len();
            if n < fixed.len() {
                let qualifier = if schema.variadic.is_none() { "exactly" } else { "at least" };
                return Err(type_err(format!(
                    "Too few items in list: expected {qualifier} {}; got {n}",
                    fixed.len()
                )));
            }
            if n > fixed.len() && schema.variadic.is_none() {
                return Err(type_err(format!("Too many items in list: expected exactly {}; got {n}", fixed.len())));
            }
            for (i, item) in list.iter().enumerate() {
                let expected = fixed.get(i).or(schema.variadic.as_ref());
                if let Some(t) = expected {
                    validate(&item, t)?;
                }
            }
            Ok(())
        }
        JsonSpec::Dict(fields) => {
            let dict = obj.cast::<PyDict>().map_err(|_| {
                type_err(format!("Expected a dict; got `{}`: {}", type_name(obj), py_repr(obj)))
            })?;
            for (key, t) in fields {
                match dict.get_item(key)? {
                    Some(v) => validate(&v, t)?,
                    None if schema.optional_keys.contains(key) => {}
                    None => return Err(type_err(format!("Missing required key: {}", crate::error::py_repr(key)))),
                }
            }
            for (k, _) in dict.iter() {
                let k: String = k.extract()?;
                if !fields.contains_key(&k) {
                    return Err(type_err(format!("Unexpected key: {k}")));
                }
            }
            Ok(())
        }
    }
}

// ------------------------------------------------------------------------------------------------ media

fn validate_file_path(obj: &Bound<'_, PyAny>) -> Result<()> {
    let py = obj.py();
    if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        let parsed = py.import("urllib.parse")?.call_method1("urlparse", (&s,))?;
        let scheme: String = parsed.getattr("scheme")?.extract()?;
        let local_path = if scheme.len() <= 1 {
            Some(std::path::absolute(&s).unwrap_or_else(|_| std::path::PathBuf::from(&s)))
        } else if scheme == "file" {
            let path: String = parsed.getattr("path")?.extract()?;
            let path = py.import("urllib.request")?.call_method1("url2pathname", (path,))?;
            let path: String = py.import("urllib.parse")?.call_method1("unquote", (path,))?.extract()?;
            Some(std::path::PathBuf::from(path))
        } else {
            None
        };
        if let Some(p) = local_path {
            if !p.is_file() {
                return Err(type_err(format!("File not found: {}", p.display())));
            }
        }
        Ok(())
    } else if obj.is_instance_of::<PyBytes>() {
        Ok(())
    } else {
        Err(type_err(format!("expected file path or bytes, got {}", py_str(&obj.get_type().into_any()))))
    }
}

// ------------------------------------------------------------------------------------------------ validate / coerce

/// `ColumnType.validate_literal()`: no conversions; raises `Error::Type`.
pub fn validate(obj: &Bound<'_, PyAny>, t: &ColumnType) -> Result<()> {
    let py = obj.py();
    if obj.is_none() {
        if !t.nullable {
            return Err(type_err("Expected non-None value"));
        }
        return Ok(());
    }
    match &t.kind {
        TypeKind::Invalid => Err(Error::internal("cannot validate a literal of an invalid type")),
        TypeKind::String => {
            if obj.is_instance_of::<PyString>() {
                Ok(())
            } else {
                Err(type_err(format!("Expected string, got {}", type_name(obj))))
            }
        }
        TypeKind::Int => {
            if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
                Ok(())
            } else {
                Err(type_err(format!("Expected int, got {}", type_name(obj))))
            }
        }
        TypeKind::Float => {
            if obj.is_instance_of::<PyFloat>() {
                Ok(())
            } else {
                Err(type_err(format!("Expected float, got {}", type_name(obj))))
            }
        }
        TypeKind::Bool => {
            if obj.is_instance_of::<PyBool>() {
                Ok(())
            } else {
                Err(type_err(format!("Expected bool, got {}", type_name(obj))))
            }
        }
        TypeKind::Timestamp => {
            if obj.is_instance(datetime_cls(py)?)? {
                Ok(())
            } else {
                Err(type_err(format!("Expected datetime.datetime, got {}", type_name(obj))))
            }
        }
        TypeKind::Date => {
            if obj.is_instance(date_cls(py)?)? {
                Ok(())
            } else {
                Err(type_err(format!("Expected datetime.date, got {}", type_name(obj))))
            }
        }
        TypeKind::Uuid => {
            if obj.is_instance(uuid_cls(py)?)? {
                Ok(())
            } else {
                Err(type_err(format!("Expected uuid.UUID, got {}", type_name(obj))))
            }
        }
        TypeKind::Binary => {
            if obj.is_instance_of::<PyBytes>() {
                Ok(())
            } else {
                Err(type_err(format!("Expected `bytes`, got `{}`", type_name(obj))))
            }
        }
        TypeKind::Json(schema) => {
            let mut v = obj.clone();
            if let Ok(t) = v.cast::<PyTuple>() {
                v = t.to_list().into_any();
            }
            if v.is_instance(base_model_cls(py)?)? {
                v = v.call_method0("model_dump")?;
            }
            if !is_valid_json_literal(&v, 1)? {
                return Err(type_err(format!("That literal is not a valid Pixeltable JSON object: {}", py_str(&v))));
            }
            if let Some(s) = schema {
                validate_json_schema(&v, s)?;
            }
            Ok(())
        }
        TypeKind::Array { shape, dtype } => validate_array(obj, shape, dtype),
        TypeKind::Image { .. } => {
            if obj.is_instance(pil_image_cls(py)?)? {
                Ok(())
            } else {
                validate_file_path(obj)
            }
        }
        TypeKind::Video | TypeKind::Audio | TypeKind::Document { .. } => validate_file_path(obj),
    }
}

/// Upstream `_create_literal()` conversions (applied to non-None values before validation).
fn pre_convert<'py>(obj: &Bound<'py, PyAny>, t: &ColumnType) -> Result<Bound<'py, PyAny>> {
    let py = obj.py();
    let mut v = obj.clone();
    if v.is_instance(enum_cls(py)?)? {
        v = v.getattr("value")?;
    }
    if v.is_instance(path_cls(py)?)? {
        v = v.str()?.into_any();
    }
    Ok(match &t.kind {
        TypeKind::Float if v.is_instance_of::<PyInt>() && !v.is_instance_of::<PyBool>() => {
            v.extract::<f64>()?.into_pyobject(py)?.into_any()
        }
        TypeKind::Bool if v.is_instance_of::<PyInt>() && !v.is_instance_of::<PyBool>() => {
            PyBool::new(py, v.is_truthy()?).to_owned().into_any()
        }
        TypeKind::Timestamp if v.is_instance_of::<PyString>() => datetime_cls(py)?.call_method1("fromisoformat", (v,))?,
        TypeKind::Date if v.is_instance_of::<PyString>() => {
            datetime_cls(py)?.call_method1("fromisoformat", (v,))?.call_method0("date")?
        }
        TypeKind::Uuid if v.is_instance_of::<PyString>() => uuid_cls(py)?.call1((v,))?,
        TypeKind::Binary if v.is_instance_of::<PyByteArray>() => v,
        TypeKind::Json(_) => {
            if let Ok(t) = v.cast::<PyTuple>() {
                t.to_list().into_any()
            } else if v.is_instance(base_model_cls(py)?)? {
                v.call_method0("model_dump")?
            } else {
                v
            }
        }
        TypeKind::Array { dtype, .. } if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() => {
            let np = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            match dtype {
                Some(DType::Str) => kwargs.set_item("dtype", np.getattr("str_")?)?,
                Some(d) => kwargs.set_item("dtype", d.name())?,
                None => kwargs.set_item("dtype", py.None())?,
            }
            np.call_method("array", (v,), Some(&kwargs))?
        }
        TypeKind::Image { .. } if v.is_instance_of::<PyString>() => {
            let s: String = v.extract()?;
            if s.starts_with("data:") {
                decode_data_url_image(py, &s)?
            } else {
                v
            }
        }
        _ => v,
    })
}

fn decode_data_url_image<'py>(py: Python<'py>, url: &str) -> Result<Bound<'py, PyAny>> {
    let attempt = || -> PyResult<Bound<'py, PyAny>> {
        let response = py.import("urllib.request")?.call_method1("urlopen", (url,))?;
        let data = response.call_method0("read")?;
        let bio = py.import("io")?.call_method1("BytesIO", (data,))?;
        let img = py.import("PIL.Image")?.call_method1("open", (bio,))?;
        img.call_method0("load")?;
        Ok(img)
    };
    attempt().map_err(|_| {
        // upstream truncates to 50 characters (not bytes: slicing bytes could split a character)
        let shown = if url.chars().count() < 50 {
            url.to_string()
        } else {
            format!("{}...", url.chars().take(50).collect::<String>())
        };
        crate::pxt_err!(InvalidDataFormat, "data URL could not be decoded into a valid image: {shown}")
    })
}

/// Converts an already-validated Python value of type `t` into a `Value`.
fn convert_validated(obj: &Bound<'_, PyAny>, t: &ColumnType) -> Result<Value> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(Value::Null);
    }
    Ok(match &t.kind {
        TypeKind::Invalid => Value::Null,
        TypeKind::String => {
            let s = str_value(obj)?;
            // upstream replaces NUL characters (which Postgres cannot store) with spaces
            Value::String(if s.contains('\0') { s.replace('\0', " ") } else { s })
        }
        TypeKind::Int => Value::Int(obj.extract::<i64>().map_err(|_| {
            crate::pxt_err!(UnsupportedOperation, "Integer value {} is out of range for type Int", py_str(obj))
        })?),
        TypeKind::Float => Value::Float(obj.extract()?),
        TypeKind::Bool => Value::Bool(obj.extract()?),
        TypeKind::Timestamp => Value::Timestamp(timestamp_to_micros(obj)?),
        TypeKind::Date => {
            let d = if obj.is_instance(datetime_cls(py)?)? { obj.call_method0("date")? } else { obj.clone() };
            Value::Date(date_to_days(&d)?)
        }
        TypeKind::Uuid => Value::Uuid(uuid_from_py(obj)?),
        TypeKind::Binary => Value::Binary(obj.extract()?),
        TypeKind::Json(_) => Value::Json(json_from_py(obj)?),
        TypeKind::Array { .. } => Value::Array(Arc::new(ndarray_from_py(obj)?)),
        TypeKind::Image { .. } | TypeKind::Video | TypeKind::Audio | TypeKind::Document { .. } => {
            if obj.is_instance_of::<PyString>() {
                Value::Media(obj.extract()?)
            } else {
                // in-memory media (a PIL image or bytes); the store writes it to the media directory
                Value::PyObj(Arc::new(obj.clone().unbind()))
            }
        }
    })
}

/// `ColumnType.create_literal()`: convert, validate, and produce a `Value`.
pub fn coerce(obj: &Bound<'_, PyAny>, t: &ColumnType) -> Result<Value> {
    let v = if obj.is_none() { obj.clone() } else { pre_convert(obj, t)? };
    validate(&v, t)?;
    convert_validated(&v, t)
}

/// `ColumnType.infer_literal_type()` plus conversion: the type and value of a Python literal, or None. A value that
/// cannot be stored (a str with a lone surrogate, JSON nested too deeply) is an INVALID_ARGUMENT error.
pub fn infer_literal(obj: &Bound<'_, PyAny>) -> Result<Option<(ColumnType, Value)>> {
    infer_literal_at(obj, 1).map_err(|e| match e {
        Error::Type(msg) => crate::pxt_err!(InvalidArgument, "{msg}"),
        e => e,
    })
}

fn infer_literal_at(obj: &Bound<'_, PyAny>, depth: usize) -> Result<Option<(ColumnType, Value)>> {
    let py = obj.py();
    let mut v = obj.clone();
    if v.is_instance(enum_cls(py)?)? {
        v = v.getattr("value")?;
    }
    if v.is_none() {
        return Ok(Some((ColumnType::invalid(true), Value::Null)));
    }
    let t = if v.is_instance_of::<PyString>() {
        ColumnType::string(false)
    } else if v.is_instance_of::<PyBool>() {
        ColumnType::bool(false)
    } else if v.is_instance_of::<PyInt>() {
        ColumnType::int(false)
    } else if v.is_instance_of::<PyFloat>() {
        ColumnType::float(false)
    } else if v.is_instance(datetime_cls(py)?)? {
        ColumnType::new(TypeKind::Timestamp, false)
    } else if v.is_instance(date_cls(py)?)? {
        ColumnType::new(TypeKind::Date, false)
    } else if v.is_instance(uuid_cls(py)?)? {
        ColumnType::new(TypeKind::Uuid, false)
    } else if v.is_instance_of::<PyBytes>() {
        ColumnType::new(TypeKind::Binary, false)
    } else if v.is_instance(pil_image_cls(py)?)? {
        let (w, h): (u64, u64) = v.getattr("size")?.extract()?;
        let mode: String = v.getattr("mode")?.extract()?;
        ColumnType::new(TypeKind::Image { width: Some(w), height: Some(h), mode: Some(mode) }, false)
    } else if v.is_instance(ndarray_cls(py)?)? {
        match dtype_of(&v)? {
            Some(d) => {
                let shape = shape_of(&v)?.into_iter().map(|n| Some(n as u64)).collect();
                ColumnType::new(TypeKind::Array { shape: Some(shape), dtype: Some(d) }, false)
            }
            None => return Ok(None),
        }
    } else if v.is_instance_of::<PyList>()
        || v.is_instance_of::<PyTuple>()
        || v.is_instance_of::<PyDict>()
        || v.is_instance(base_model_cls(py)?)?
    {
        let converted = pre_convert(&v, &ColumnType::json(false))?;
        if !is_valid_json_literal(&converted, depth)? {
            return Ok(None);
        }
        return Ok(Some((json_literal_type(&converted, depth)?, Value::Json(json_from_py_at(&converted, depth)?))));
    } else {
        return Ok(None);
    };
    let value = convert_validated(&v, &t)?;
    Ok(Some((t, value)))
}

/// The structural type of a JSON literal (`JsonType.from_literal`); `obj` has been validated at `depth`.
fn json_literal_type(obj: &Bound<'_, PyAny>, depth: usize) -> Result<ColumnType> {
    let item_type = |item: &Bound<'_, PyAny>| -> Result<ColumnType> {
        Ok(match infer_literal_at(item, depth + 1)? {
            Some((t, _)) => t,
            None => ColumnType::json(false),
        })
    };
    let spec = if let Ok(list) = obj.cast::<PyList>() {
        JsonSpec::List(list.iter().map(|i| item_type(&i)).collect::<Result<_>>()?)
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        let mut fields = indexmap::IndexMap::new();
        for (k, v) in dict.iter() {
            fields.insert(str_value(&k)?, item_type(&v)?);
        }
        JsonSpec::Dict(fields)
    } else {
        return Ok(ColumnType::json(false));
    };
    Ok(ColumnType::new(
        TypeKind::Json(Some(Box::new(JsonSchema { spec, variadic: None, optional_keys: Default::default() }))),
        false,
    ))
}

// ------------------------------------------------------------------------------------------------ to Python

/// The Python object a query returns for `v` (of column type `t`).
pub fn to_py<'py>(py: Python<'py>, v: &Value, t: &ColumnType) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => {
            if t.is_float() {
                (*i as f64).into_pyobject(py)?.into_any()
            } else {
                i.into_pyobject(py)?.into_any()
            }
        }
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Timestamp(us) => micros_to_timestamp(py, *us)?,
        Value::Date(d) => days_to_date(py, *d)?,
        Value::Uuid(u) => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("bytes", PyBytes::new(py, u.as_bytes()))?;
            uuid_cls(py)?.call((), Some(&kwargs))?
        }
        Value::Binary(b) => PyBytes::new(py, b).into_any(),
        Value::Json(j) => json_to_py(py, j)?,
        Value::Array(a) => ndarray_to_py(py, a)?,
        Value::Media(path) => {
            if t.is_image() {
                open_image(py, path)?
            } else {
                PyString::new(py, path).into_any()
            }
        }
        Value::PyObj(o) => o.bind(py).clone(),
    })
}

/// Opens a stored image (a local path or a file:// URL) as a PIL image.
fn open_image<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let local = path.strip_prefix("file://").unwrap_or(path);
    let img = py.import("PIL.Image")?.call_method1("open", (local,))?;
    Ok(img)
}

/// `ColumnType.create_literal()` returning the converted Python object (upstream returns Python values, not
/// stored values: media stay paths, timestamps stay datetimes).
pub fn create_literal_py<'py>(obj: &Bound<'py, PyAny>, t: &ColumnType) -> Result<Bound<'py, PyAny>> {
    let py = obj.py();
    if obj.is_none() {
        validate(obj, t)?;
        return Ok(obj.clone());
    }
    let mut v = pre_convert(obj, t)?;
    validate(&v, t)?;
    if t.is_string() {
        let s: String = v.extract()?;
        if s.contains('\0') {
            v = PyString::new(py, &s.replace('\0', " ")).into_any();
        }
    }
    if t.is_timestamp() && v.getattr("tzinfo")?.is_none() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("tzinfo", default_tz(py)?)?;
        v = v.call_method("replace", (), Some(&kwargs))?;
    }
    Ok(v)
}
