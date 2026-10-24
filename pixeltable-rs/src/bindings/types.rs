//! `pixeltable._core.ColumnType`: the Rust half of every `pixeltable.type_system.ColumnType`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::error::{Error, Result};
use crate::types::{ArrayIndex, ColumnType, DType, JsonSchema, JsonSpec, TypeEnum, TypeKind};
use crate::value::py as vpy;

#[pyclass(name = "ColumnType", module = "pixeltable._core", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyColumnType {
    pub t: ColumnType,
}

impl PyColumnType {
    pub fn new(t: ColumnType) -> Self {
        PyColumnType { t }
    }
}

/// Extracts a core type from a `_core.ColumnType` or a Python `ColumnType` wrapper (its `_core` attribute).
pub fn core_type(obj: &Bound<'_, PyAny>) -> Option<ColumnType> {
    if let Ok(c) = obj.cast::<PyColumnType>() {
        return Some(c.get().t.clone());
    }
    let inner = obj.getattr("_core").ok()?;
    inner.cast::<PyColumnType>().ok().map(|c| c.get().t.clone())
}

/// Wraps a core type in the matching Python `ColumnType` subclass.
pub fn wrap_type<'py>(py: Python<'py>, t: &ColumnType) -> PyResult<Bound<'py, PyAny>> {
    let ts = py.import("pixeltable.type_system")?;
    ts.getattr("ColumnType")?.call_method1("_from_core", (PyColumnType::new(t.clone()),))
}

/// Converts any type form (a ColumnType, `pxt.Int`, `pxt.String | None`, `list[int]`, ...) to a core type, using
/// `pixeltable.type_system` to interpret Python type hints.
pub fn normalize_type(obj: &Bound<'_, PyAny>, allow_builtin_types: bool) -> Result<ColumnType> {
    if let Some(t) = core_type(obj) {
        return Ok(t);
    }
    let ts = obj.py().import("pixeltable.type_system")?;
    let kwargs = PyDict::new(obj.py());
    kwargs.set_item("allow_builtin_types", allow_builtin_types)?;
    let wrapped = ts.getattr("ColumnType")?.call_method("normalize_type", (obj,), Some(&kwargs))?;
    core_type(&wrapped).ok_or_else(|| Error::internal("normalize_type() did not return a ColumnType"))
}

pub fn is_type_form(obj: &Bound<'_, PyAny>) -> Result<bool> {
    if core_type(obj).is_some() {
        return Ok(true);
    }
    let ts = obj.py().import("pixeltable.type_system")?;
    Ok(ts.getattr("is_type_form")?.call1((obj,))?.is_truthy()?)
}

#[pymethods]
impl PyColumnType {
    /// The default type of a `ColumnType.Type` value.
    #[staticmethod]
    fn make(type_enum: i64, nullable: bool) -> Result<Self> {
        let te = TypeEnum::from_i64(type_enum).ok_or_else(|| Error::Value(format!("invalid type enum: {type_enum}")))?;
        Ok(PyColumnType::new(ColumnType::make(te, nullable)))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype, nullable))]
    fn array(shape: Option<Vec<Option<u64>>>, dtype: Option<String>, nullable: bool) -> Result<Self> {
        let dtype = match dtype {
            None => None,
            Some(d) => Some(DType::from_name(&d).ok_or_else(|| Error::Value(format!("Unsupported dtype: {d}")))?),
        };
        if shape.is_some() && dtype.is_none() {
            return Err(Error::internal("an array type with a shape needs a dtype"));
        }
        Ok(PyColumnType::new(ColumnType::new(TypeKind::Array { shape, dtype }, nullable)))
    }

    #[staticmethod]
    #[pyo3(signature = (width, height, mode, nullable))]
    fn image(width: Option<u64>, height: Option<u64>, mode: Option<String>, nullable: bool) -> Self {
        PyColumnType::new(ColumnType::new(TypeKind::Image { width, height, mode }, nullable))
    }

    #[staticmethod]
    #[pyo3(signature = (doc_formats, nullable))]
    fn document(doc_formats: Option<String>, nullable: bool) -> Result<Self> {
        Ok(PyColumnType::new(ColumnType::document(doc_formats, nullable)?))
    }

    /// `Json[(T1, T2, ...)]`, optionally with a variadic tail type.
    #[staticmethod]
    #[pyo3(signature = (items, variadic, nullable))]
    fn json_list(items: Vec<PyColumnType>, variadic: Option<PyColumnType>, nullable: bool) -> Self {
        let schema = JsonSchema {
            spec: JsonSpec::List(items.into_iter().map(|t| t.t).collect()),
            variadic: variadic.map(|t| t.t),
            optional_keys: Default::default(),
        };
        PyColumnType::new(ColumnType::new(TypeKind::Json(Some(Box::new(schema))), nullable))
    }

    /// `Json[{'k': T, ...}]`
    #[staticmethod]
    #[pyo3(signature = (fields, optional_keys, nullable))]
    fn json_dict(fields: Vec<(String, PyColumnType)>, optional_keys: Vec<String>, nullable: bool) -> Self {
        let schema = JsonSchema {
            spec: JsonSpec::Dict(fields.into_iter().map(|(k, t)| (k, t.t)).collect()),
            variadic: None,
            optional_keys: optional_keys.into_iter().collect(),
        };
        PyColumnType::new(ColumnType::new(TypeKind::Json(Some(Box::new(schema))), nullable))
    }

    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyAny>) -> Result<Self> {
        let s: String = d.py().import("json")?.call_method1("dumps", (d,))?.extract()?;
        let j: serde_json::Value = serde_json::from_str(&s)?;
        Ok(PyColumnType::new(ColumnType::from_dict(&j)?))
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vpy::json_to_py(py, &self.t.to_dict())
    }

    #[getter]
    fn nullable(&self) -> bool {
        self.t.nullable
    }

    #[getter]
    fn type_enum(&self) -> i64 {
        self.t.type_enum() as i64
    }

    #[getter]
    fn class_name(&self) -> &'static str {
        self.t.class_name()
    }

    fn copy(&self, nullable: bool) -> Self {
        PyColumnType::new(self.t.with_nullable(nullable))
    }

    fn __repr__(&self) -> String {
        self.t.repr()
    }

    fn matches(&self, other: &Bound<'_, PyAny>) -> bool {
        core_type(other).is_some_and(|o| self.t.matches(&o))
    }

    fn equals(&self, other: &Bound<'_, PyAny>) -> bool {
        core_type(other).is_some_and(|o| self.t == o)
    }

    fn hash_value(&self) -> (i64, bool) {
        let (t, n) = self.t.hash_key();
        (t as i64, n)
    }

    #[pyo3(signature = (other, for_inference=false))]
    fn supertype(&self, other: &Bound<'_, PyAny>, for_inference: bool) -> Result<Option<Self>> {
        let o = core_type(other).ok_or_else(|| Error::Type("expected a ColumnType".into()))?;
        Ok(self.t.supertype(&o, for_inference).map(PyColumnType::new))
    }

    #[pyo3(signature = (other, ignore_nullable=false, strict_json=true))]
    fn is_supertype_of(&self, other: &Bound<'_, PyAny>, ignore_nullable: bool, strict_json: bool) -> Result<bool> {
        let o = core_type(other).ok_or_else(|| Error::Type("expected a ColumnType".into()))?;
        Ok(self.t.is_supertype_of(&o, ignore_nullable, strict_json))
    }

    fn has_supertype(&self) -> bool {
        self.t.has_supertype()
    }

    /// Raises TypeError if `val` is not a valid literal of this type.
    fn validate_literal(&self, val: &Bound<'_, PyAny>) -> Result<()> {
        vpy::validate(val, &self.t)
    }

    /// Converts `val` to this type (raising TypeError if that is not possible).
    fn create_literal<'py>(&self, val: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
        vpy::create_literal_py(val, &self.t)
    }

    /// The type of a Python literal, or None.
    #[staticmethod]
    fn infer_literal_type(val: &Bound<'_, PyAny>) -> Result<Option<Self>> {
        Ok(vpy::infer_literal(val)?.map(|(t, _)| PyColumnType::new(t)))
    }

    #[staticmethod]
    #[pyo3(signature = (types, for_inference=false))]
    fn common_supertype(types: Vec<PyColumnType>, for_inference: bool) -> Option<Self> {
        let ts: Vec<ColumnType> = types.into_iter().map(|t| t.t).collect();
        ColumnType::common_supertype(ts.iter(), for_inference).map(PyColumnType::new)
    }

    // ---- kind-specific accessors

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match &self.t.kind {
            TypeKind::Array { shape: Some(s), .. } => Ok(Some(PyTuple::new(py, s)?)),
            _ => Ok(None),
        }
    }

    #[getter]
    fn dtype_name(&self) -> Option<&'static str> {
        match &self.t.kind {
            TypeKind::Array { dtype: Some(d), .. } => Some(d.name()),
            _ => None,
        }
    }

    #[getter]
    fn width(&self) -> Option<u64> {
        match &self.t.kind {
            TypeKind::Image { width, .. } => *width,
            _ => None,
        }
    }

    #[getter]
    fn height(&self) -> Option<u64> {
        match &self.t.kind {
            TypeKind::Image { height, .. } => *height,
            _ => None,
        }
    }

    #[getter]
    fn mode(&self) -> Option<String> {
        match &self.t.kind {
            TypeKind::Image { mode, .. } => mode.clone(),
            _ => None,
        }
    }

    #[getter]
    fn doc_formats(&self) -> Option<String> {
        match &self.t.kind {
            TypeKind::Document { formats } => formats.clone(),
            _ => None,
        }
    }

    /// For a typed Json: ('list', [types], variadic, []) or ('dict', [(key, type)], None, optional_keys); else None.
    fn json_schema<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let TypeKind::Json(Some(s)) = &self.t.kind else { return Ok(None) };
        let variadic = s.variadic.clone().map(PyColumnType::new);
        let optional: Vec<String> = s.optional_keys.iter().cloned().collect();
        let t = match &s.spec {
            JsonSpec::List(items) => {
                let items: Vec<PyColumnType> = items.iter().cloned().map(PyColumnType::new).collect();
                PyTuple::new(
                    py,
                    [
                        "list".into_pyobject(py)?.into_any(),
                        PyList::new(py, items)?.into_any(),
                        variadic.into_pyobject(py)?.into_any(),
                        PyList::new(py, optional)?.into_any(),
                    ],
                )?
            }
            JsonSpec::Dict(fields) => {
                let fields: Vec<(String, PyColumnType)> =
                    fields.iter().map(|(k, t)| (k.clone(), PyColumnType::new(t.clone()))).collect();
                PyTuple::new(
                    py,
                    [
                        "dict".into_pyobject(py)?.into_any(),
                        PyList::new(py, fields)?.into_any(),
                        py.None().into_bound(py),
                        PyList::new(py, optional)?.into_any(),
                    ],
                )?
            }
        };
        Ok(Some(t))
    }

    /// The type of `self[index]` for an array type; `index` items are ints or (start, stop, step) tuples.
    fn subscript_type(&self, index: Vec<Bound<'_, PyAny>>) -> Result<Self> {
        let idx = index
            .iter()
            .map(|i| {
                if let Ok(n) = i.extract::<i64>() {
                    Ok(ArrayIndex::Int(n))
                } else {
                    let (a, b, c): (Option<i64>, Option<i64>, Option<i64>) = i.extract()?;
                    Ok(ArrayIndex::Slice(a, b, c))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PyColumnType::new(self.t.array_subscript_type(&idx)?))
    }
}
