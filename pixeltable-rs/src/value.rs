//! Cell values.
//!
//! `Value` is what rows store and what expressions evaluate to. Conversion to and from Python objects (with
//! type-directed coercion and upstream's validation messages) is in `value::py`.

pub mod py;

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::Arc;
use uuid::Uuid;

use crate::types::DType;

pub type Json = serde_json::Value;

/// An n-dimensional array (C order).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NdArray {
    pub shape: Vec<usize>,
    pub data: ArrayData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArrayData {
    Bool(Vec<bool>),
    U8(#[serde(with = "serde_bytes")] Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    /// float16, as raw bits
    F16(Vec<u16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Str(Vec<String>),
}

impl ArrayData {
    pub fn dtype(&self) -> DType {
        match self {
            ArrayData::Bool(_) => DType::Bool,
            ArrayData::U8(_) => DType::U8,
            ArrayData::U16(_) => DType::U16,
            ArrayData::U32(_) => DType::U32,
            ArrayData::U64(_) => DType::U64,
            ArrayData::I8(_) => DType::I8,
            ArrayData::I16(_) => DType::I16,
            ArrayData::I32(_) => DType::I32,
            ArrayData::I64(_) => DType::I64,
            ArrayData::F16(_) => DType::F16,
            ArrayData::F32(_) => DType::F32,
            ArrayData::F64(_) => DType::F64,
            ArrayData::Str(_) => DType::Str,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ArrayData::Bool(v) => v.len(),
            ArrayData::U8(v) => v.len(),
            ArrayData::U16(v) => v.len(),
            ArrayData::U32(v) => v.len(),
            ArrayData::U64(v) => v.len(),
            ArrayData::I8(v) => v.len(),
            ArrayData::I16(v) => v.len(),
            ArrayData::I32(v) => v.len(),
            ArrayData::I64(v) => v.len(),
            ArrayData::F16(v) => v.len(),
            ArrayData::F32(v) => v.len(),
            ArrayData::F64(v) => v.len(),
            ArrayData::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl NdArray {
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }
}

/// A cell value. Media values are file paths or URLs; in-memory media (for example a PIL image produced by a UDF)
/// is written to the media store before it is stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    /// microseconds since the Unix epoch, UTC
    Timestamp(i64),
    /// days since 1970-01-01
    Date(i32),
    Uuid(Uuid),
    Binary(#[serde(with = "serde_bytes")] Vec<u8>),
    Json(#[serde(with = "json_as_string")] Json),
    Array(Arc<NdArray>),
    /// a local file path or a URL
    Media(String),
    /// An opaque Python object, only while a row is in flight (for example a PIL image returned by a UDF before it
    /// is written to the media store). Never persisted.
    #[serde(skip)]
    PyObj(Arc<pyo3::Py<pyo3::PyAny>>),
}

/// Maximum nesting depth of a JSON value (a scalar has depth 0, `[]` and `{}` 1, `[[]]` 2). Values coming from
/// Python are checked against it (`value::py`), and the store refuses to write a deeper value and reads with the same
/// limit, so every stored value can be read back, and recursive conversion, comparison and drop cannot exhaust the
/// stack. Upstream has no explicit limit; its Python recursion fails at about 490 levels.
pub const MAX_JSON_DEPTH: usize = 500;

/// The message for a value nested more deeply than `MAX_JSON_DEPTH` (starts with a word that `lower_first()` keeps
/// readable, since insert prefixes it with the column).
pub fn json_too_deep_msg() -> String {
    format!("Value is nested too deeply to be stored as JSON (more than {MAX_JSON_DEPTH} levels)")
}

/// The nesting depth of serialized JSON (number of `[`/`{` on the deepest path), by a linear scan that does not
/// recurse; `bytes` is assumed to be well-formed JSON.
pub fn json_nesting_depth(bytes: &[u8]) -> usize {
    let (mut depth, mut max, mut in_str, mut escaped) = (0usize, 0usize, false, false);
    for &b in bytes {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' | b'{' => {
                depth += 1;
                max = max.max(depth);
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    max
}

/// Parses JSON nested at most `max_depth` levels deep. serde_json's own recursion limit (128) is replaced by this
/// one, which every writer enforces too (see `MAX_JSON_DEPTH`, `store::MAX_CATALOG_DEPTH`).
pub fn parse_json_bounded<T>(bytes: &[u8], max_depth: usize) -> std::result::Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    let depth = json_nesting_depth(bytes);
    if depth > max_depth {
        return Err(format!("JSON nested {depth} levels deep exceeds the limit of {max_depth}"));
    }
    let mut de = serde_json::Deserializer::from_slice(bytes);
    de.disable_recursion_limit();
    let v = T::deserialize(&mut de).map_err(|e| e.to_string())?;
    de.end().map_err(|e| e.to_string())?;
    Ok(v)
}

mod json_as_string {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &serde_json::Value, s: S) -> Result<S::Ok, S::Error> {
        let text = v.to_string();
        if super::json_nesting_depth(text.as_bytes()) > super::MAX_JSON_DEPTH {
            return Err(serde::ser::Error::custom(super::json_too_deep_msg()));
        }
        s.serialize_str(&text)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<serde_json::Value, D::Error> {
        let s = String::deserialize(d)?;
        super::parse_json_bounded(s.as_bytes(), super::MAX_JSON_DEPTH).map_err(serde::de::Error::custom)
    }
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Json(Json::Number(n)) => n.as_f64(),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(*b as i64),
            Value::Json(Json::Number(n)) => n.as_i64(),
            _ => None,
        }
    }

    /// Name of the Python class this value converts to, for error messages.
    pub fn py_type_name(&self) -> &'static str {
        match self {
            Value::Null => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) | Value::Media(_) => "str",
            Value::Timestamp(_) => "datetime",
            Value::Date(_) => "date",
            Value::Uuid(_) => "UUID",
            Value::Binary(_) => "bytes",
            Value::Json(j) => match j {
                Json::Null => "NoneType",
                Json::Bool(_) => "bool",
                Json::Number(n) if n.is_f64() => "float",
                Json::Number(_) => "int",
                Json::String(_) => "str",
                Json::Array(_) => "list",
                Json::Object(_) => "dict",
            },
            Value::Array(_) => "ndarray",
            Value::PyObj(_) => "object",
        }
    }

    /// Unwrap scalar JSON values into the corresponding scalar values (for arithmetic and comparisons on json
    /// operands, whose runtime type is only known at evaluation time).
    pub fn json_scalar(&self) -> Value {
        match self {
            Value::Json(Json::Null) => Value::Null,
            Value::Json(Json::Bool(b)) => Value::Bool(*b),
            Value::Json(Json::Number(n)) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else {
                    Value::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            Value::Json(Json::String(s)) => Value::String(s.clone()),
            other => other.clone(),
        }
    }
}

/// Total order used for sorting non-null values (Postgres semantics: NaN sorts after every other float).
pub fn cmp_values(a: &Value, b: &Value) -> Ordering {
    use Value::*;
    match (a, b) {
        (Null, Null) => Ordering::Equal,
        (Null, _) => Ordering::Greater,
        (_, Null) => Ordering::Less,
        (Bool(x), Bool(y)) => x.cmp(y),
        (Int(x), Int(y)) => x.cmp(y),
        (Float(x), Float(y)) => cmp_f64(*x, *y),
        (Int(x), Float(y)) => cmp_f64(*x as f64, *y),
        (Float(x), Int(y)) => cmp_f64(*x, *y as f64),
        (String(x), String(y)) | (Media(x), Media(y)) => x.cmp(y),
        (Timestamp(x), Timestamp(y)) => x.cmp(y),
        (Date(x), Date(y)) => x.cmp(y),
        (Timestamp(x), Date(y)) => x.cmp(&(*y as i64 * 86_400_000_000)),
        (Date(x), Timestamp(y)) => (*x as i64 * 86_400_000_000).cmp(y),
        (Uuid(x), Uuid(y)) => x.cmp(y),
        (Binary(x), Binary(y)) => x.cmp(y),
        (Json(x), Json(y)) => cmp_json(x, y),
        _ => type_rank(a).cmp(&type_rank(b)),
    }
}

fn type_rank(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Int(_) | Value::Float(_) => 2,
        Value::String(_) => 3,
        Value::Timestamp(_) | Value::Date(_) => 4,
        Value::Uuid(_) => 5,
        Value::Binary(_) => 6,
        Value::Json(_) => 7,
        Value::Array(_) => 8,
        Value::Media(_) => 9,
        Value::PyObj(_) => 10,
    }
}

pub fn cmp_f64(x: f64, y: f64) -> Ordering {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
    }
}

/// Order of jsonb values in Postgres: Object > Array > Boolean > Number > String > Null.
fn cmp_json(a: &Json, b: &Json) -> Ordering {
    fn rank(j: &Json) -> u8 {
        match j {
            Json::Null => 0,
            Json::String(_) => 1,
            Json::Number(_) => 2,
            Json::Bool(_) => 3,
            Json::Array(_) => 4,
            Json::Object(_) => 5,
        }
    }
    match (a, b) {
        (Json::String(x), Json::String(y)) => x.cmp(y),
        (Json::Number(x), Json::Number(y)) => cmp_f64(x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN)),
        (Json::Bool(x), Json::Bool(y)) => x.cmp(y),
        (Json::Array(x), Json::Array(y)) => {
            x.len().cmp(&y.len()).then_with(|| x.iter().zip(y.iter()).map(|(p, q)| cmp_json(p, q)).find(|o| o.is_ne()).unwrap_or(Ordering::Equal))
        }
        (Json::Object(x), Json::Object(y)) => x.len().cmp(&y.len()).then_with(|| a.to_string().cmp(&b.to_string())),
        _ => rank(a).cmp(&rank(b)),
    }
}

/// SQL equality of two non-null values (numeric types compare by value).
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => x == y,
        (Value::Json(x), Value::Json(y)) => x == y,
        _ => cmp_values(a, b) == Ordering::Equal && type_rank(a) == type_rank(b),
    }
}

/// Python's `repr(float)`.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // shortest round-trip digits in scientific form, e.g. "1.2345e3"
    let sci = format!("{f:e}");
    let (mantissa, exp) = match sci.split_once('e') {
        Some((m, e)) => (m.to_string(), e.parse::<i32>().unwrap_or(0)),
        None => return sci,
    };
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{first}{rest}e{exp_sign}{:02}", exp.abs());
    }
    if exp < 0 {
        let zeros = "0".repeat((-exp - 1) as usize);
        return format!("{sign}0.{zeros}{digits}");
    }
    let int_len = exp as usize + 1;
    if digits.len() <= int_len {
        let zeros = "0".repeat(int_len - digits.len());
        format!("{sign}{digits}{zeros}.0")
    } else {
        let (i, frac) = digits.split_at(int_len);
        format!("{sign}{i}.{frac}")
    }
}

/// Microseconds since the epoch for a UTC timestamp given as calendar fields.
pub fn timestamp_from_utc_parts(date: chrono::NaiveDate, h: u32, m: u32, s: u32, us: u32) -> Option<i64> {
    let t = chrono::NaiveTime::from_hms_micro_opt(h, m, s, us)?;
    Some(chrono::NaiveDateTime::new(date, t).and_utc().timestamp_micros())
}

pub fn date_from_days(days: i32) -> Option<chrono::NaiveDate> {
    chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?.checked_add_signed(chrono::Duration::days(days as i64))
}

pub fn days_from_date(d: chrono::NaiveDate) -> Option<i32> {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?;
    i32::try_from(d.signed_duration_since(epoch).num_days()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr() {
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(1234.5), "1234.5");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(123456789012345.0), "123456789012345.0");
        assert_eq!(py_float_repr(-2.5e-7), "-2.5e-07");
        assert_eq!(py_float_repr(0.0001), "0.0001");
    }
}

/// Converts a value to type `t` the way `ColumnType.create_literal()` converts the corresponding Python value
/// (used by `astype()`); failures are `Error::Type` with upstream's `validate_literal()` message.
pub fn cast_value(v: Value, t: &crate::types::ColumnType) -> crate::error::Result<Value> {
    use crate::error::Error;
    use crate::types::TypeKind;
    let v = if t.is_json() { v } else { v.json_scalar() };
    if v.is_null() {
        if t.nullable {
            return Ok(Value::Null);
        }
        return Err(Error::Type("Expected non-None value".into()));
    }
    let mismatch = |expected: &str, v: &Value| Error::Type(format!("Expected {expected}, got {}", v.py_type_name()));
    Ok(match (&t.kind, v) {
        (TypeKind::String, v @ Value::String(_)) => v,
        (TypeKind::String, v) => return Err(mismatch("string", &v)),
        (TypeKind::Int, v @ Value::Int(_)) => v,
        (TypeKind::Int, v) => return Err(mismatch("int", &v)),
        (TypeKind::Float, v @ Value::Float(_)) => v,
        (TypeKind::Float, Value::Int(i)) => Value::Float(i as f64),
        (TypeKind::Float, v) => return Err(mismatch("float", &v)),
        (TypeKind::Bool, v @ Value::Bool(_)) => v,
        (TypeKind::Bool, Value::Int(i)) => Value::Bool(i != 0),
        (TypeKind::Bool, v) => return Err(mismatch("bool", &v)),
        (TypeKind::Timestamp, v @ Value::Timestamp(_)) => v,
        (TypeKind::Timestamp, v) => return Err(mismatch("datetime.datetime", &v)),
        (TypeKind::Date, v @ Value::Date(_)) => v,
        (TypeKind::Date, Value::Timestamp(us)) => Value::Date(us.div_euclid(86_400_000_000) as i32),
        (TypeKind::Date, v) => return Err(mismatch("datetime.date", &v)),
        (TypeKind::Uuid, v @ Value::Uuid(_)) => v,
        (TypeKind::Uuid, Value::String(s)) => {
            Value::Uuid(uuid::Uuid::parse_str(&s).map_err(|_| Error::Value("badly formed hexadecimal UUID string".into()))?)
        }
        (TypeKind::Uuid, v) => return Err(mismatch("uuid.UUID", &v)),
        (TypeKind::Binary, v @ Value::Binary(_)) => v,
        (TypeKind::Binary, v) => return Err(Error::Type(format!("Expected `bytes`, got `{}`", v.py_type_name()))),
        (TypeKind::Json(_), v @ Value::Json(_)) => v,
        (TypeKind::Json(_), Value::Bool(b)) => Value::Json(Json::Bool(b)),
        (TypeKind::Json(_), Value::Int(i)) => Value::Json(Json::from(i)),
        (TypeKind::Json(_), Value::Float(f)) => Value::Json(serde_json::Number::from_f64(f).map_or(Json::Null, Json::Number)),
        (TypeKind::Json(_), Value::String(s)) => Value::Json(Json::String(s)),
        (TypeKind::Json(_), v) => {
            return Err(Error::Type(format!("That literal is not a valid Pixeltable JSON object: {}", v.py_type_name())));
        }
        (TypeKind::Array { dtype, .. }, Value::Array(a)) => {
            if let Some(d) = dtype {
                if a.dtype() != *d {
                    return Err(Error::Type(format!(
                        "Expected numpy.ndarray of dtype {}, got numpy.ndarray of dtype {}",
                        d.name(),
                        a.dtype().name()
                    )));
                }
            }
            Value::Array(a)
        }
        (TypeKind::Array { .. }, v) => return Err(mismatch("numpy.ndarray", &v)),
        (TypeKind::Image { .. }, Value::String(s)) if s.starts_with("data:") => {
            pyo3::Python::attach(|py| -> crate::error::Result<Value> {
                let img = crate::value::py::decode_data_url_image(py, &s)?;
                Ok(Value::PyObj(Arc::new(img.unbind())))
            })?
        }
        (TypeKind::Image { .. } | TypeKind::Video | TypeKind::Audio | TypeKind::Document { .. }, Value::String(s)) => {
            Value::Media(crate::media::normalize_input(&s))
        }
        (TypeKind::Image { .. } | TypeKind::Video | TypeKind::Audio | TypeKind::Document { .. }, v @ Value::Media(_)) => v,
        (TypeKind::Image { .. } | TypeKind::Video | TypeKind::Audio | TypeKind::Document { .. }, v) => {
            return Err(Error::Type(format!("expected file path or bytes, got <class '{}'>", v.py_type_name())));
        }
        (TypeKind::Invalid, v) => v,
    })
}

/// A value as JSON, for building JSON values from inline lists and dicts.
pub fn value_to_json(v: &Value) -> crate::error::Result<Json> {
    use crate::error::Error;
    Ok(match v {
        Value::Null => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Int(i) => Json::from(*i),
        Value::Float(f) => serde_json::Number::from_f64(*f).map_or(Json::Null, Json::Number),
        Value::String(s) | Value::Media(s) => Json::String(s.clone()),
        Value::Json(j) => j.clone(),
        other => return Err(Error::not_implemented(format!("storing a value of type `{}` inside a JSON value", other.py_type_name()))),
    })
}

#[cfg(test)]
mod json_depth_tests {
    use super::*;

    fn nested(depth: usize) -> Json {
        let mut j = Json::Array(Vec::new());
        for _ in 1..depth {
            j = Json::Array(vec![j]);
        }
        j
    }

    #[test]
    fn nesting_depth_ignores_strings() {
        assert_eq!(json_nesting_depth(b"1"), 0);
        assert_eq!(json_nesting_depth(b"[[], {\"a\": [1]}]"), 3);
        assert_eq!(json_nesting_depth(br#"{"k": "[[[{\"]]]"}"#), 1);
    }

    #[test]
    fn stored_json_round_trips_up_to_the_limit() {
        // serde_json alone stops at 128 levels; every value that can be written must read back
        for depth in [127, 128, 129, MAX_JSON_DEPTH] {
            let v = Value::Json(nested(depth));
            let bytes = rmp_serde::to_vec(&v).unwrap_or_else(|e| panic!("depth {depth}: {e}"));
            match rmp_serde::from_slice::<Value>(&bytes) {
                Ok(Value::Json(j)) => assert_eq!(json_nesting_depth(j.to_string().as_bytes()), depth),
                other => panic!("depth {depth}: {other:?}"),
            }
        }
    }

    #[test]
    fn deeper_json_is_not_written() {
        let err = rmp_serde::to_vec(&Value::Json(nested(MAX_JSON_DEPTH + 1))).err().map(|e| e.to_string());
        assert!(err.is_some_and(|e| e.contains("nested too deeply")));
    }

    #[test]
    fn bounded_parse() {
        let text = nested(300).to_string();
        assert!(parse_json_bounded::<Json>(text.as_bytes(), 300).is_ok());
        assert!(parse_json_bounded::<Json>(text.as_bytes(), 299).is_err());
    }
}
