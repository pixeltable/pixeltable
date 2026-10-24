//! Column types: the model behind `pixeltable.type_system`.
//!
//! The Python `ColumnType` classes are thin wrappers around `ColumnType` here; nullability, matching, the supertype
//! algebra, the `repr()` and the serialized form (`as_dict()`, identical to upstream's `_classname` dicts so that
//! stored metadata reads like upstream's) are all computed in Rust. Coercion and validation of Python values against
//! a type lives in `value::py`.

use crate::error::{Error, Result};
use indexmap::IndexMap;
use serde_json::{Map, Value as Json, json};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Mirrors `ColumnType.Type` (the enum values are part of the public API: `ColumnType.Type(i)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeEnum {
    String = 0,
    Int = 1,
    Float = 2,
    Bool = 3,
    Timestamp = 4,
    Json = 5,
    Array = 6,
    Image = 7,
    Video = 8,
    Audio = 9,
    Document = 10,
    Date = 11,
    Uuid = 12,
    Binary = 13,
    Invalid = 255,
}

impl TypeEnum {
    pub fn from_i64(v: i64) -> Option<TypeEnum> {
        use TypeEnum::*;
        Some(match v {
            0 => String,
            1 => Int,
            2 => Float,
            3 => Bool,
            4 => Timestamp,
            5 => Json,
            6 => Array,
            7 => Image,
            8 => Video,
            9 => Audio,
            10 => Document,
            11 => Date,
            12 => Uuid,
            13 => Binary,
            255 => Invalid,
            _ => return None,
        })
    }
}

/// Numpy dtypes supported in arrays (upstream's ARRAY_SUPPORTED_NUMPY_DTYPES).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    Str,
}

impl DType {
    /// The numpy name (`np.dtype(x).name`, and `str(np.dtype(x))` except for strings).
    pub fn name(self) -> &'static str {
        use DType::*;
        match self {
            Bool => "bool",
            U8 => "uint8",
            U16 => "uint16",
            U32 => "uint32",
            U64 => "uint64",
            I8 => "int8",
            I16 => "int16",
            I32 => "int32",
            I64 => "int64",
            F16 => "float16",
            F32 => "float32",
            F64 => "float64",
            Str => "str",
        }
    }

    pub fn from_name(name: &str) -> Option<DType> {
        use DType::*;
        Some(match name {
            "bool" | "bool_" => Bool,
            "uint8" => U8,
            "uint16" => U16,
            "uint32" => U32,
            "uint64" => U64,
            "int8" => I8,
            "int16" => I16,
            "int32" => I32,
            "int64" => I64,
            "float16" => F16,
            "float32" => F32,
            "float64" => F64,
            "str" | "str_" => Str,
            _ => return None,
        })
    }

    /// The scalar column type of an element (`ColumnType.from_np_dtype`).
    pub fn scalar_type(self, nullable: bool) -> ColumnType {
        use DType::*;
        let kind = match self {
            Bool => TypeKind::Bool,
            U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64 => TypeKind::Int,
            F16 | F32 | F64 => TypeKind::Float,
            Str => TypeKind::String,
        };
        ColumnType::new(kind, nullable)
    }

    /// The dtype an array of this pixeltable scalar type gets (`ArrayType.pxt_dtype_to_numpy_dtype`).
    pub fn for_scalar(kind: &TypeKind) -> Option<DType> {
        match kind {
            TypeKind::Int => Some(DType::I64),
            TypeKind::Float => Some(DType::F32),
            TypeKind::Bool => Some(DType::Bool),
            TypeKind::String => Some(DType::Str),
            _ => None,
        }
    }
}

pub type Shape = Vec<Option<u64>>;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonSpec {
    /// A tuple of fixed types (possibly followed by a variadic tail).
    List(Vec<ColumnType>),
    /// A typed dictionary.
    Dict(IndexMap<String, ColumnType>),
}

/// The type schema of a `Json[...]` type (upstream `JsonType.TypeSchema`).
#[derive(Debug, Clone, PartialEq)]
pub struct JsonSchema {
    pub spec: JsonSpec,
    pub variadic: Option<ColumnType>,
    pub optional_keys: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Invalid,
    String,
    Int,
    Float,
    Bool,
    Timestamp,
    Date,
    Uuid,
    Binary,
    Json(Option<Box<JsonSchema>>),
    Array { shape: Option<Shape>, dtype: Option<DType> },
    Image { width: Option<u64>, height: Option<u64>, mode: Option<String> },
    Video,
    Audio,
    Document { formats: Option<String> },
}

/// A column type: a kind plus nullability. `PartialEq` is upstream's `__eq__` (matches and same nullability).
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnType {
    pub kind: TypeKind,
    pub nullable: bool,
}

const DOC_FORMATS: [&str; 8] = ["HTML", "MD", "PDF", "XML", "TXT", "PPTX", "DOCX", "XLSX"];

impl ColumnType {
    pub fn new(kind: TypeKind, nullable: bool) -> Self {
        ColumnType { kind, nullable }
    }
    pub fn invalid(nullable: bool) -> Self {
        Self::new(TypeKind::Invalid, nullable)
    }
    pub fn string(nullable: bool) -> Self {
        Self::new(TypeKind::String, nullable)
    }
    pub fn int(nullable: bool) -> Self {
        Self::new(TypeKind::Int, nullable)
    }
    pub fn float(nullable: bool) -> Self {
        Self::new(TypeKind::Float, nullable)
    }
    pub fn bool(nullable: bool) -> Self {
        Self::new(TypeKind::Bool, nullable)
    }
    pub fn json(nullable: bool) -> Self {
        Self::new(TypeKind::Json(None), nullable)
    }

    pub fn document(formats: Option<String>, nullable: bool) -> Result<Self> {
        if let Some(f) = &formats {
            for part in f.split(',') {
                if !DOC_FORMATS.contains(&part) {
                    return Err(Error::Value(format!("Invalid document type: {part}")));
                }
            }
        }
        Ok(Self::new(TypeKind::Document { formats }, nullable))
    }

    pub fn type_enum(&self) -> TypeEnum {
        match &self.kind {
            TypeKind::Invalid => TypeEnum::Invalid,
            TypeKind::String => TypeEnum::String,
            TypeKind::Int => TypeEnum::Int,
            TypeKind::Float => TypeEnum::Float,
            TypeKind::Bool => TypeEnum::Bool,
            TypeKind::Timestamp => TypeEnum::Timestamp,
            TypeKind::Date => TypeEnum::Date,
            TypeKind::Uuid => TypeEnum::Uuid,
            TypeKind::Binary => TypeEnum::Binary,
            TypeKind::Json(_) => TypeEnum::Json,
            TypeKind::Array { .. } => TypeEnum::Array,
            TypeKind::Image { .. } => TypeEnum::Image,
            TypeKind::Video => TypeEnum::Video,
            TypeKind::Audio => TypeEnum::Audio,
            TypeKind::Document { .. } => TypeEnum::Document,
        }
    }

    /// The default instance of a type enum (`ColumnType.make_type`).
    pub fn make(t: TypeEnum, nullable: bool) -> ColumnType {
        let kind = match t {
            TypeEnum::String => TypeKind::String,
            TypeEnum::Int => TypeKind::Int,
            TypeEnum::Float => TypeKind::Float,
            TypeEnum::Bool => TypeKind::Bool,
            TypeEnum::Timestamp => TypeKind::Timestamp,
            TypeEnum::Json => TypeKind::Json(None),
            TypeEnum::Array => TypeKind::Array { shape: None, dtype: None },
            TypeEnum::Image => TypeKind::Image { width: None, height: None, mode: None },
            TypeEnum::Video => TypeKind::Video,
            TypeEnum::Audio => TypeKind::Audio,
            TypeEnum::Document => TypeKind::Document { formats: None },
            TypeEnum::Date => TypeKind::Date,
            TypeEnum::Uuid => TypeKind::Uuid,
            TypeEnum::Binary => TypeKind::Binary,
            TypeEnum::Invalid => TypeKind::Invalid,
        };
        ColumnType::new(kind, nullable)
    }

    pub fn with_nullable(&self, nullable: bool) -> ColumnType {
        ColumnType { kind: self.kind.clone(), nullable }
    }

    // ---- predicates

    pub fn is_invalid(&self) -> bool {
        matches!(self.kind, TypeKind::Invalid)
    }
    pub fn is_string(&self) -> bool {
        matches!(self.kind, TypeKind::String)
    }
    pub fn is_int(&self) -> bool {
        matches!(self.kind, TypeKind::Int)
    }
    pub fn is_float(&self) -> bool {
        matches!(self.kind, TypeKind::Float)
    }
    pub fn is_bool(&self) -> bool {
        matches!(self.kind, TypeKind::Bool)
    }
    pub fn is_timestamp(&self) -> bool {
        matches!(self.kind, TypeKind::Timestamp)
    }
    pub fn is_date(&self) -> bool {
        matches!(self.kind, TypeKind::Date)
    }
    pub fn is_uuid(&self) -> bool {
        matches!(self.kind, TypeKind::Uuid)
    }
    pub fn is_binary(&self) -> bool {
        matches!(self.kind, TypeKind::Binary)
    }
    pub fn is_json(&self) -> bool {
        matches!(self.kind, TypeKind::Json(_))
    }
    pub fn is_array(&self) -> bool {
        matches!(self.kind, TypeKind::Array { .. })
    }
    pub fn is_image(&self) -> bool {
        matches!(self.kind, TypeKind::Image { .. })
    }
    pub fn is_video(&self) -> bool {
        matches!(self.kind, TypeKind::Video)
    }
    pub fn is_audio(&self) -> bool {
        matches!(self.kind, TypeKind::Audio)
    }
    pub fn is_document(&self) -> bool {
        matches!(self.kind, TypeKind::Document { .. })
    }
    pub fn is_media(&self) -> bool {
        self.is_image() || self.is_video() || self.is_audio() || self.is_document()
    }
    pub fn is_numeric(&self) -> bool {
        self.is_int() || self.is_float()
    }
    pub fn is_scalar_json(&self) -> bool {
        self.is_string() || self.is_int() || self.is_float() || self.is_bool()
    }
    pub fn is_scalar(&self) -> bool {
        self.is_scalar_json() || self.is_timestamp() || self.is_date() || self.is_uuid()
    }

    /// Upstream `has_supertype()`: false for types that absorb nothing in literal inference.
    pub fn has_supertype(&self) -> bool {
        match self.kind {
            TypeKind::String | TypeKind::Timestamp | TypeKind::Date | TypeKind::Uuid => !self.nullable,
            _ => true,
        }
    }

    // ---- algebra

    /// Equal aside from nullability.
    pub fn matches(&self, other: &ColumnType) -> bool {
        self.kind == other.kind
    }

    fn scalar_supertype(a: &TypeKind, b: &TypeKind) -> Option<TypeKind> {
        use TypeKind::*;
        match (a, b) {
            (Bool, Int) | (Int, Bool) => Some(Int),
            (Bool, Float) | (Float, Bool) => Some(Float),
            (Int, Float) | (Float, Int) => Some(Float),
            _ => None,
        }
    }

    /// The most specific supertype of `self` and `other`, or None.
    pub fn supertype(&self, other: &ColumnType, for_inference: bool) -> Option<ColumnType> {
        let nullable = self.nullable || other.nullable;
        if self.matches(other) {
            return Some(self.with_nullable(nullable));
        }
        if self.is_invalid() {
            return Some(other.with_nullable(nullable));
        }
        if other.is_invalid() {
            return Some(self.with_nullable(nullable));
        }
        if self.is_scalar() && other.is_scalar() {
            if let Some(k) = Self::scalar_supertype(&self.kind, &other.kind) {
                return Some(ColumnType::new(k, nullable));
            }
        }
        if !for_inference
            && ((self.is_json() && other.is_scalar_json())
                || (self.is_scalar_json() && other.is_json())
                || (self.is_scalar_json() && other.is_scalar_json()))
        {
            return Some(ColumnType::json(nullable));
        }
        match (&self.kind, &other.kind) {
            (TypeKind::Json(s1), TypeKind::Json(s2)) => {
                let schema = match (s1, s2) {
                    (Some(a), Some(b)) => a.superschema(b).map(Box::new),
                    _ => None,
                };
                Some(ColumnType::new(TypeKind::Json(schema), nullable))
            }
            (TypeKind::Array { shape: sh1, dtype: d1 }, TypeKind::Array { shape: sh2, dtype: d2 }) => {
                if d1 != d2 {
                    return Some(ColumnType::new(TypeKind::Array { shape: None, dtype: None }, nullable));
                }
                let shape = match (sh1, sh2) {
                    (Some(a), Some(b)) if a.len() == b.len() => {
                        Some(a.iter().zip(b.iter()).map(|(x, y)| if x == y { *x } else { None }).collect())
                    }
                    _ => None,
                };
                Some(ColumnType::new(TypeKind::Array { shape, dtype: *d1 }, nullable))
            }
            (
                TypeKind::Image { width: w1, height: h1, mode: m1 },
                TypeKind::Image { width: w2, height: h2, mode: m2 },
            ) => Some(ColumnType::new(
                TypeKind::Image {
                    width: if w1 == w2 { *w1 } else { None },
                    height: if h1 == h2 { *h1 } else { None },
                    mode: if m1 == m2 { m1.clone() } else { None },
                },
                nullable,
            )),
            _ => None,
        }
    }

    pub fn common_supertype<'a>(types: impl IntoIterator<Item = &'a ColumnType>, for_inference: bool) -> Option<ColumnType> {
        let mut result: Option<ColumnType> = None;
        for t in types {
            result = match result {
                None => Some(t.clone()),
                Some(s) => Some(s.supertype(t, for_inference)?),
            };
        }
        result
    }

    pub fn is_supertype_of(&self, other: &ColumnType, ignore_nullable: bool, strict_json: bool) -> bool {
        let self_ = if !strict_json && self.is_json() { ColumnType::json(self.nullable) } else { self.clone() };
        match self_.supertype(other, false) {
            None => false,
            Some(s) if ignore_nullable => s.matches(&self_),
            Some(s) => s == self_,
        }
    }

    /// The type of `self[index]` for an array (numpy basic indexing); `index` items are Ok(int) or Err(slice).
    pub fn array_subscript_type(&self, index: &[ArrayIndex]) -> Result<ColumnType> {
        let TypeKind::Array { shape, dtype } = &self.kind else {
            return Err(Error::internal("array_subscript_type() on a non-array type"));
        };
        let Some(shape) = shape else { return Ok(self.clone()) };
        let dtype = dtype.ok_or_else(|| Error::internal("array type with a shape but no dtype"))?;
        if index.len() > shape.len() {
            crate::bail!(
                InvalidArgument,
                "Too many indices for `{}`: got {}; expected at most {}",
                self.repr(),
                index.len(),
                shape.len()
            );
        }
        let mut result_shape: Shape = Vec::new();
        for (n, el) in shape.iter().zip(index.iter()) {
            match el {
                ArrayIndex::Int(i) => {
                    if let Some(n) = n {
                        let n = *n as i64;
                        if !(-n <= *i && *i < n) {
                            crate::bail!(InvalidArgument, "Index {} is out of bounds for `{}`", i, self.repr());
                        }
                    }
                }
                ArrayIndex::Slice(start, stop, step) => {
                    result_shape.push(n.map(|n| slice_len(n as i64, *start, *stop, *step) as u64));
                }
            }
        }
        result_shape.extend(shape[index.len()..].iter().copied());
        if result_shape.is_empty() {
            return Ok(dtype.scalar_type(self.nullable));
        }
        Ok(ColumnType::new(TypeKind::Array { shape: Some(result_shape), dtype: Some(dtype) }, self.nullable))
    }

    // ---- display

    fn base_str(&self) -> String {
        match &self.kind {
            TypeKind::Invalid => "Invalid".into(),
            TypeKind::String => "String".into(),
            TypeKind::Int => "Int".into(),
            TypeKind::Float => "Float".into(),
            TypeKind::Bool => "Bool".into(),
            TypeKind::Timestamp => "Timestamp".into(),
            TypeKind::Date => "Date".into(),
            TypeKind::Uuid => "UUID".into(),
            TypeKind::Binary => "Binary".into(),
            TypeKind::Json(None) => "Json".into(),
            TypeKind::Json(Some(s)) => format!("Json[{}]", s.repr()),
            TypeKind::Array { shape, dtype } => match (shape, dtype) {
                (None, None) => "Array".into(),
                (None, Some(d)) => format!("Array[{}]", d.name()),
                (Some(s), Some(d)) => format!("Array[{}, {}]", shape_repr(s), d.name()),
                (Some(s), None) => format!("Array[{}]", shape_repr(s)),
            },
            TypeKind::Image { width, height, mode } => {
                let mut params = Vec::new();
                if width.is_some() || height.is_some() {
                    params.push(format!("({}, {})", opt_repr(width), opt_repr(height)));
                }
                if let Some(m) = mode {
                    params.push(crate::error::py_repr(m));
                }
                if params.is_empty() { "Image".into() } else { format!("Image[{}]", params.join(", ")) }
            }
            TypeKind::Video => "Video".into(),
            TypeKind::Audio => "Audio".into(),
            TypeKind::Document { .. } => "Document".into(),
        }
    }

    /// `repr(col_type)`: e.g. `Int`, `String | None`, `Array[(None,), float32]`.
    pub fn repr(&self) -> String {
        let base = self.base_str();
        if self.nullable { format!("{base} | None") } else { base }
    }

    /// The upstream class name (`_classname` in serialized form, and the Python wrapper class).
    pub fn class_name(&self) -> &'static str {
        match self.kind {
            TypeKind::Invalid => "InvalidType",
            TypeKind::String => "StringType",
            TypeKind::Int => "IntType",
            TypeKind::Float => "FloatType",
            TypeKind::Bool => "BoolType",
            TypeKind::Timestamp => "TimestampType",
            TypeKind::Date => "DateType",
            TypeKind::Uuid => "UUIDType",
            TypeKind::Binary => "BinaryType",
            TypeKind::Json(_) => "JsonType",
            TypeKind::Array { .. } => "ArrayType",
            TypeKind::Image { .. } => "ImageType",
            TypeKind::Video => "VideoType",
            TypeKind::Audio => "AudioType",
            TypeKind::Document { .. } => "DocumentType",
        }
    }

    // ---- serialization (upstream-compatible `as_dict()` form)

    pub fn to_dict(&self) -> Json {
        let mut m = Map::new();
        m.insert("_classname".into(), json!(self.class_name()));
        m.insert("nullable".into(), json!(self.nullable));
        match &self.kind {
            TypeKind::Json(Some(s)) => {
                m.insert("type_schema".into(), s.to_dict());
            }
            TypeKind::Array { shape, dtype } => {
                m.insert(
                    "shape".into(),
                    match shape {
                        None => Json::Null,
                        Some(s) => Json::Array(s.iter().map(|d| d.map_or(Json::Null, |n| json!(n))).collect()),
                    },
                );
                m.insert("numpy_dtype".into(), dtype.map_or(Json::Null, |d| json!(d.name())));
            }
            TypeKind::Image { width, height, mode } => {
                m.insert("width".into(), width.map_or(Json::Null, |w| json!(w)));
                m.insert("height".into(), height.map_or(Json::Null, |h| json!(h)));
                m.insert("mode".into(), mode.as_ref().map_or(Json::Null, |s| json!(s)));
            }
            TypeKind::Document { formats } => {
                if let Some(f) = formats {
                    m.insert("doc_formats".into(), json!(f));
                }
            }
            _ => {}
        }
        Json::Object(m)
    }

    pub fn from_dict(d: &Json) -> Result<ColumnType> {
        let bad = || Error::internal(format!("malformed column type: {d}"));
        let obj = d.as_object().ok_or_else(bad)?;
        let classname = obj.get("_classname").and_then(Json::as_str).ok_or_else(bad)?;
        let nullable = obj.get("nullable").and_then(Json::as_bool).unwrap_or(false);
        let get_u64 = |k: &str| obj.get(k).and_then(Json::as_u64);
        let kind = match classname {
            "InvalidType" => TypeKind::Invalid,
            "StringType" => TypeKind::String,
            "IntType" => TypeKind::Int,
            "FloatType" => TypeKind::Float,
            "BoolType" => TypeKind::Bool,
            "TimestampType" => TypeKind::Timestamp,
            "DateType" => TypeKind::Date,
            "UUIDType" => TypeKind::Uuid,
            "BinaryType" => TypeKind::Binary,
            "JsonType" => match obj.get("type_schema") {
                None | Some(Json::Null) => TypeKind::Json(None),
                Some(s) => TypeKind::Json(Some(Box::new(JsonSchema::from_dict(s)?))),
            },
            "ArrayType" => {
                let shape = match obj.get("shape") {
                    None | Some(Json::Null) => None,
                    Some(Json::Array(a)) => Some(a.iter().map(Json::as_u64).collect()),
                    Some(_) => return Err(bad()),
                };
                let dtype = match obj.get("numpy_dtype") {
                    None | Some(Json::Null) => None,
                    Some(Json::String(s)) => Some(DType::from_name(s).ok_or_else(bad)?),
                    Some(_) => return Err(bad()),
                };
                TypeKind::Array { shape, dtype }
            }
            "ImageType" => TypeKind::Image {
                width: get_u64("width"),
                height: get_u64("height"),
                mode: obj.get("mode").and_then(Json::as_str).map(str::to_owned),
            },
            "VideoType" => TypeKind::Video,
            "AudioType" => TypeKind::Audio,
            "DocumentType" => {
                TypeKind::Document { formats: obj.get("doc_formats").and_then(Json::as_str).map(str::to_owned) }
            }
            _ => return Err(bad()),
        };
        Ok(ColumnType::new(kind, nullable))
    }

    /// A hash consistent with `PartialEq` (upstream hashes only the type enum and nullability).
    pub fn hash_key(&self) -> (u8, bool) {
        (self.type_enum() as i64 as u8, self.nullable)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArrayIndex {
    Int(i64),
    Slice(Option<i64>, Option<i64>, Option<i64>),
}

/// `len(range(*slice(start, stop, step).indices(n)))`
pub fn slice_len(n: i64, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> i64 {
    let step = step.unwrap_or(1);
    if step == 0 {
        return 0;
    }
    let clamp = |v: Option<i64>, default: i64, lo: i64, hi: i64| -> i64 {
        match v {
            None => default,
            Some(v) if v < 0 => (v + n).max(lo),
            Some(v) => v.min(hi),
        }
    };
    let (start, stop) = if step > 0 {
        (clamp(start, 0, 0, n), clamp(stop, n, 0, n))
    } else {
        (clamp(start, n - 1, -1, n - 1), clamp(stop, -1, -1, n - 1))
    };
    if step > 0 {
        if stop > start { (stop - start + step - 1) / step } else { 0 }
    } else if start > stop {
        (start - stop + (-step) - 1) / (-step)
    } else {
        0
    }
}

fn opt_repr(v: &Option<u64>) -> String {
    v.map_or_else(|| "None".to_string(), |n| n.to_string())
}

/// Python tuple repr of a shape: `(3,)`, `(None, 2)`.
pub fn shape_repr(shape: &[Option<u64>]) -> String {
    let parts: Vec<String> = shape.iter().map(opt_repr).collect();
    if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) }
}

impl JsonSchema {
    pub fn repr(&self) -> String {
        match &self.spec {
            JsonSpec::List(items) => {
                let mut reprs: Vec<String> = items.iter().map(ColumnType::repr).collect();
                if let Some(v) = &self.variadic {
                    reprs.push(v.repr());
                    reprs.push("...".into());
                }
                let mut s = reprs.join(", ");
                if reprs.len() == 1 {
                    s.push(',');
                }
                format!("({s})")
            }
            JsonSpec::Dict(fields) => {
                let mut s = String::from("{");
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    let _ = write!(s, "{}: {}", crate::error::py_repr(k), v.repr());
                }
                s.push('}');
                if !self.optional_keys.is_empty() {
                    let keys: Vec<String> = self.optional_keys.iter().map(|k| crate::error::py_repr(k)).collect();
                    let _ = write!(s, ", optional_keys=[{}]", keys.join(", "));
                }
                s
            }
        }
    }

    pub fn superschema(&self, other: &JsonSchema) -> Option<JsonSchema> {
        if self == other {
            return Some(self.clone());
        }
        match (&self.spec, &other.spec) {
            (JsonSpec::List(a0), JsonSpec::List(b0)) => {
                let mut variadic = match (&self.variadic, &other.variadic) {
                    (Some(v1), Some(v2)) => Some(v1.supertype(v2, false)?),
                    (Some(v), None) | (None, Some(v)) => Some(v.clone()),
                    (None, None) => None,
                };
                let (a, b) = if a0.len() > b0.len() { (b0, a0) } else { (a0, b0) };
                let mut joined = Vec::new();
                for (i, item_b) in b.iter().enumerate() {
                    match a.get(i) {
                        None => {
                            variadic = Some(match variadic {
                                None => item_b.clone(),
                                Some(v) => v.supertype(item_b, false)?,
                            });
                        }
                        Some(item_a) => joined.push(item_a.supertype(item_b, false)?),
                    }
                }
                Some(JsonSchema { spec: JsonSpec::List(joined), variadic, optional_keys: BTreeSet::new() })
            }
            (JsonSpec::Dict(d1), JsonSpec::Dict(d2)) => {
                let mut joined = d1.clone();
                for (k, t) in d2 {
                    match joined.get(k) {
                        Some(existing) => {
                            let s = t.supertype(existing, false)?;
                            joined.insert(k.clone(), s);
                        }
                        None => {
                            joined.insert(k.clone(), t.clone());
                        }
                    }
                }
                let mut optional: BTreeSet<String> = self.optional_keys.union(&other.optional_keys).cloned().collect();
                for k in d1.keys() {
                    if !d2.contains_key(k) {
                        optional.insert(k.clone());
                    }
                }
                for k in d2.keys() {
                    if !d1.contains_key(k) {
                        optional.insert(k.clone());
                    }
                }
                Some(JsonSchema { spec: JsonSpec::Dict(joined), variadic: None, optional_keys: optional })
            }
            _ => None,
        }
    }

    pub fn to_dict(&self) -> Json {
        let (type_spec, key_order) = match &self.spec {
            JsonSpec::List(items) => (Json::Array(items.iter().map(ColumnType::to_dict).collect()), Json::Null),
            JsonSpec::Dict(fields) => {
                let mut m = Map::new();
                for (k, v) in fields {
                    m.insert(k.clone(), v.to_dict());
                }
                (Json::Object(m), Json::Array(fields.keys().map(|k| json!(k)).collect()))
            }
        };
        json!({
            "type_spec": type_spec,
            "type_spec_key_order": key_order,
            "variadic_type": self.variadic.as_ref().map_or(Json::Null, ColumnType::to_dict),
            "optional_keys": self.optional_keys.iter().collect::<Vec<_>>(),
        })
    }

    pub fn from_dict(d: &Json) -> Result<JsonSchema> {
        let bad = || Error::internal(format!("malformed json type schema: {d}"));
        let spec = match d.get("type_spec").ok_or_else(bad)? {
            Json::Array(items) => JsonSpec::List(items.iter().map(ColumnType::from_dict).collect::<Result<_>>()?),
            Json::Object(m) => {
                let order: Vec<String> = match d.get("type_spec_key_order") {
                    Some(Json::Array(keys)) => keys.iter().filter_map(|k| k.as_str().map(str::to_owned)).collect(),
                    _ => m.keys().cloned().collect(),
                };
                let mut fields = IndexMap::new();
                for k in order {
                    let t = m.get(&k).ok_or_else(bad)?;
                    fields.insert(k, ColumnType::from_dict(t)?);
                }
                JsonSpec::Dict(fields)
            }
            _ => return Err(bad()),
        };
        let variadic = match d.get("variadic_type") {
            None | Some(Json::Null) => None,
            Some(v) => Some(ColumnType::from_dict(v)?),
        };
        let optional_keys = match d.get("optional_keys") {
            Some(Json::Array(keys)) => keys.iter().filter_map(|k| k.as_str().map(str::to_owned)).collect(),
            _ => BTreeSet::new(),
        };
        Ok(JsonSchema { spec, variadic, optional_keys })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supertype_numeric() {
        let t = ColumnType::int(false).supertype(&ColumnType::float(true), false);
        assert_eq!(t, Some(ColumnType::float(true)));
        assert_eq!(ColumnType::string(false).supertype(&ColumnType::int(false), true), None);
        assert_eq!(ColumnType::string(false).supertype(&ColumnType::int(false), false), Some(ColumnType::json(false)));
    }

    #[test]
    fn reprs() {
        let a = ColumnType::new(TypeKind::Array { shape: Some(vec![None]), dtype: Some(DType::F32) }, true);
        assert_eq!(a.repr(), "Array[(None,), float32] | None");
        assert_eq!(ColumnType::make(TypeEnum::Uuid, false).repr(), "UUID");
        let img = ColumnType::new(TypeKind::Image { width: Some(3), height: None, mode: Some("RGB".into()) }, false);
        assert_eq!(img.repr(), "Image[(3, None), 'RGB']");
    }

    #[test]
    fn dict_roundtrip() {
        let a = ColumnType::new(TypeKind::Array { shape: Some(vec![Some(2), None]), dtype: Some(DType::I64) }, false);
        assert_eq!(ColumnType::from_dict(&a.to_dict()).ok(), Some(a));
    }

    #[test]
    fn slices() {
        assert_eq!(slice_len(10, None, None, None), 10);
        assert_eq!(slice_len(10, Some(2), Some(5), None), 3);
        assert_eq!(slice_len(10, None, None, Some(-1)), 10);
        assert_eq!(slice_len(10, Some(-3), None, None), 3);
    }
}
