//! `pixeltable._core.Expr`: the Python expression object (operators build typed Rust nodes).

use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySlice, PyString, PyTuple};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::types::{PyColumnType, normalize_type, wrap_type};
use crate::error::{Error, Result};
use crate::expr::{ArithOp, CmpOp, ColumnProp, Expr, ExprKind, ExprRef, JsonPathElem, LogicalOp, MapperOp, StringOp, build, display};
use crate::types::ArrayIndex;
use crate::value::Value;
use crate::value::py as vpy;

#[pyclass(name = "Expr", module = "pixeltable", frozen)]
pub struct PyExpr {
    pub e: ExprRef,
    /// for a column reference obtained from a table: that table (upstream `ColumnVersionMd.tbl_id`), which is a view
    /// of the column's table for `view.base_column`
    pub path_tbl_id: Option<uuid::Uuid>,
}

impl PyExpr {
    pub fn new(e: ExprRef) -> Self {
        PyExpr { e, path_tbl_id: None }
    }
}

fn type_repr(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().str().map(|s| s.to_string()).unwrap_or_default()
}

/// The expression behind a Python object, if it is one.
pub fn as_expr(obj: &Bound<'_, PyAny>) -> Option<ExprRef> {
    obj.cast::<PyExpr>().ok().map(|e| e.get().e.clone())
}

/// True for the Python types upstream accepts as operands (`LiteralPythonTypes`).
fn is_literal_python_type(obj: &Bound<'_, PyAny>) -> Result<bool> {
    let py = obj.py();
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyInt>() || obj.is_instance_of::<PyFloat>() {
        return Ok(true);
    }
    let date = py.import("datetime")?.getattr("date")?;
    let uuid = py.import("uuid")?.getattr("UUID")?;
    Ok(obj.is_instance(&date)? || obj.is_instance(&uuid)?)
}

/// A literal node for a Python value (upstream `Literal(val)`).
pub fn literal(obj: &Bound<'_, PyAny>) -> Result<ExprRef> {
    match vpy::infer_literal(obj)? {
        Some((t, v)) => Ok(Expr::literal(v, t)),
        None => Err(Error::Type(format!("Not a valid literal: {}", obj.str()?))),
    }
}

/// An operand of an operator: an expression, or a literal of a `LiteralPythonTypes` type.
fn operand(obj: &Bound<'_, PyAny>) -> Result<ExprRef> {
    if let Some(e) = as_expr(obj) {
        return Ok(e);
    }
    if is_literal_python_type(obj)? {
        return literal(obj);
    }
    Err(Error::Type(format!("Other must be Expr or literal: {}", type_repr(obj))))
}

/// Upstream `Expr.from_object()`: an expression, an inline list/dict, or a literal; None if not convertible.
pub fn from_object(obj: &Bound<'_, PyAny>) -> Result<Option<ExprRef>> {
    let e = from_object_at(obj, 1)?;
    if let Some(e) = &e {
        e.check_depth()?;
    }
    Ok(e)
}

/// The nesting of inline lists and dicts becomes JSON nesting: fail beyond `MAX_JSON_DEPTH` (`depth` is the level of
/// the list or dict being entered).
fn check_inline_depth(depth: usize) -> Result<()> {
    if depth > crate::value::MAX_JSON_DEPTH {
        return Err(Error::new(crate::error::ErrorCode::InvalidArgument, crate::value::json_too_deep_msg()));
    }
    Ok(())
}

fn from_object_at(obj: &Bound<'_, PyAny>, depth: usize) -> Result<Option<ExprRef>> {
    if let Some(e) = as_expr(obj) {
        return Ok(Some(e));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        check_inline_depth(depth)?;
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            let item = item?;
            match from_object_at(&item, depth + 1)? {
                Some(e) => items.push(e),
                None => return Err(Error::Type(format!("Not a valid literal: {}", item.str()?))),
            }
        }
        return Ok(Some(build::inline_list(items)));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        check_inline_depth(depth)?;
        let mut items = Vec::new();
        for (k, v) in d.iter() {
            if !k.is_instance_of::<PyString>() {
                crate::bail!(
                    UnsupportedOperation,
                    "Dictionary requires string keys; {} has type {}",
                    k.str()?,
                    type_repr(&k)
                );
            }
            match from_object_at(&v, depth + 1)? {
                Some(e) => items.push((k.extract::<String>()?, e)),
                None => return Err(Error::Type(format!("Not a valid literal: {}", v.str()?))),
            }
        }
        return Ok(Some(build::inline_dict(items)));
    }
    Ok(vpy::infer_literal(obj)?.map(|(t, v)| Expr::literal(v, t)))
}

/// Upstream `Expr.from_array()` (`pxt.array()`).
pub fn from_array(obj: &Bound<'_, PyAny>) -> Result<ExprRef> {
    from_array_at(obj, 1)
}

fn from_array_at(obj: &Bound<'_, PyAny>, depth: usize) -> Result<ExprRef> {
    // numpy's limit; also bounds the recursion over nested lists
    if depth > 64 {
        crate::bail!(InvalidArgument, "maximum supported dimension for an ndarray is currently 64");
    }
    let py = obj.py();
    if obj.is_instance(&py.import("numpy")?.getattr("ndarray")?)? {
        if let Some((t, v)) = vpy::infer_literal(obj)? {
            return Ok(Expr::literal(v, t));
        }
    }
    let mut items = Vec::new();
    for el in obj.try_iter()? {
        let el = el?;
        if let Some(e) = as_expr(&el) {
            items.push(e);
        } else if el.is_instance_of::<PyList>() || el.is_instance_of::<PyTuple>() {
            items.push(from_array_at(&el, depth + 1)?);
        } else {
            items.push(literal(&el)?);
        }
    }
    build::inline_array(items)
}

/// The Python object for a new expression; refuses expressions nested more deeply than `MAX_EXPR_DEPTH`, so no
/// Python-visible expression (and nothing built from it) can exhaust the stack in recursive processing.
pub(crate) fn wrap(e: ExprRef) -> Result<PyExpr> {
    e.check_depth()?;
    Ok(PyExpr::new(e))
}

impl PyExpr {
    fn arith(&self, op: ArithOp, other: &Bound<'_, PyAny>, reflected: bool) -> Result<PyExpr> {
        let o = operand(other)?;
        let (a, b) = if reflected { (o, self.e.clone()) } else { (self.e.clone(), o) };
        wrap(build::arithmetic(op, a, b)?)
    }

    fn string(&self, op: StringOp, other: &Bound<'_, PyAny>, reflected: bool) -> Result<PyExpr> {
        let o = operand(other)?;
        let (a, b) = if reflected { (o, self.e.clone()) } else { (self.e.clone(), o) };
        wrap(build::string_op(op, a, b)?)
    }

    fn other_is_string(other: &Bound<'_, PyAny>) -> bool {
        other.is_instance_of::<PyString>() || as_expr(other).is_some_and(|e| e.col_type.is_string())
    }
}

#[pymethods]
impl PyExpr {
    #[getter]
    fn col_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        wrap_type(py, &self.e.col_type)
    }

    #[getter]
    fn _core_type(&self) -> PyColumnType {
        PyColumnType::new(self.e.col_type.clone())
    }

    fn __repr__(&self) -> String {
        display::display(&self.e)
    }

    fn __str__(&self) -> String {
        display::display(&self.e)
    }

    #[pyo3(signature = (inline=true))]
    fn display_str(&self, inline: bool) -> String {
        let _ = inline;
        display::display(&self.e)
    }

    fn default_column_name(&self) -> Option<String> {
        self.e.default_column_name()
    }

    fn equals(&self, other: &Bound<'_, PyAny>) -> bool {
        as_expr(other).is_some_and(|o| self.e.equals(&o))
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        crate::expr::serde::to_json(&self.e).map(|j| j.to_string()).unwrap_or_default().hash(&mut h);
        h.finish()
    }

    /// Upstream `Expr.id`: equal for equal expressions.
    #[getter]
    fn id(&self) -> u64 {
        self.__hash__()
    }

    /// Upstream `Expr.serialize()`: the JSON of `as_dict()`.
    fn serialize(&self) -> Result<String> {
        Ok(crate::expr::serde::to_json(&self.e)?.to_string())
    }

    /// Upstream `Expr.deserialize()`
    #[staticmethod]
    fn deserialize(s: &str) -> Result<PyExpr> {
        let j: serde_json::Value = crate::value::parse_json_bounded(s.as_bytes(), crate::store::MAX_CATALOG_DEPTH)
            .map_err(|e| Error::internal(format!("malformed serialized expression: {e}")))?;
        wrap(crate::expr::serde::from_json(&j)?)
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>> {
        Ok(vpy::json_to_py(py, &crate::expr::serde::to_json(&self.e)?)?)
    }

    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyAny>) -> Result<PyExpr> {
        let s: String = d.py().import("json")?.call_method1("dumps", (d,))?.extract()?;
        let j: serde_json::Value = crate::value::parse_json_bounded(s.as_bytes(), crate::store::MAX_CATALOG_DEPTH)
            .map_err(|e| Error::internal(format!("malformed serialized expression: {e}")))?;
        wrap(crate::expr::serde::from_json(&j)?)
    }

    fn __copy__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __deepcopy__<'py>(slf: Bound<'py, Self>, _memo: &Bound<'py, PyAny>) -> Bound<'py, Self> {
        slf
    }

    fn __bool__(&self) -> Result<bool> {
        Err(Error::Type(format!(
            "Pixeltable expressions cannot be used in conjunction with Python boolean operators (and/or/not)\n{}",
            display::display(&self.e)
        )))
    }

    fn __iter__(&self) -> Result<PyExpr> {
        crate::bail!(
            UnsupportedOperation,
            "Pixeltable expressions are not iterable. Iteration over data is expressed as a query (e.g. t.select(...).collect()) or a view with an iterator: {}",
            display::display(&self.e)
        )
    }

    fn __len__(&self) -> Result<usize> {
        crate::bail!(
            UnsupportedOperation,
            "len() of a Pixeltable expression is undefined; string and json columns support .len(), row counts are t.count(): {}",
            display::display(&self.e)
        )
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> Result<PyExpr> {
        if other.is_none() {
            match op {
                CompareOp::Eq => return wrap(build::is_null(self.e.clone())),
                CompareOp::Ne => return wrap(build::not(build::is_null(self.e.clone()))?),
                _ => {}
            }
        }
        let cmp = match op {
            CompareOp::Lt => CmpOp::Lt,
            CompareOp::Le => CmpOp::Le,
            CompareOp::Eq => CmpOp::Eq,
            CompareOp::Ne => CmpOp::Ne,
            CompareOp::Gt => CmpOp::Gt,
            CompareOp::Ge => CmpOp::Ge,
        };
        wrap(build::comparison(cmp, self.e.clone(), operand(other)?)?)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        if self.e.col_type.is_string() {
            return self.string(StringOp::Concat, other, false);
        }
        self.arith(ArithOp::Add, other, false)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        if self.e.col_type.is_string() {
            return self.string(StringOp::Concat, other, true);
        }
        self.arith(ArithOp::Add, other, true)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::Sub, other, false)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::Sub, other, true)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        if self.e.col_type.is_string() || Self::other_is_string(other) {
            return self.string(StringOp::Repeat, other, false);
        }
        self.arith(ArithOp::Mul, other, false)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        if self.e.col_type.is_string() || other.is_instance_of::<PyString>() {
            return self.string(StringOp::Repeat, other, true);
        }
        self.arith(ArithOp::Mul, other, true)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::Div, other, false)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::Div, other, true)
    }

    fn __mod__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::Mod, other, false)
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::Mod, other, true)
    }

    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::FloorDiv, other, false)
    }

    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        self.arith(ArithOp::FloorDiv, other, true)
    }

    fn __pow__(&self, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> Result<PyExpr> {
        let _ = modulo;
        self.arith(ArithOp::Pow, other, false)
    }

    fn __neg__(&self, py: Python<'_>) -> Result<PyExpr> {
        let minus_one = (-1i64).into_pyobject(py)?.into_any();
        self.arith(ArithOp::Mul, &minus_one, false)
    }

    fn __abs__(&self) -> Result<PyExpr> {
        let f = builtin("pixeltable.functions.math.abs")?;
        wrap(crate::functions::make_builtin_call(f, vec![self.e.clone()], vec![], false)?)
    }

    #[pyo3(signature = (ndigits=None))]
    fn __round__(&self, ndigits: Option<&Bound<'_, PyAny>>) -> Result<PyExpr> {
        let mut args = vec![self.e.clone()];
        if let Some(n) = ndigits.filter(|n| !n.is_none()) {
            args.push(operand(n)?);
        }
        let f = builtin("pixeltable.functions.math.round")?;
        wrap(crate::functions::make_builtin_call(f, args, vec![], false)?)
    }

    fn __and__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        let o = as_expr(other).ok_or_else(|| Error::Type(format!("Other needs to be an expression: {}", type_repr(other))))?;
        wrap(build::logical_binary(LogicalOp::And, self.e.clone(), o)?)
    }

    fn __or__(&self, other: &Bound<'_, PyAny>) -> Result<PyExpr> {
        let o = as_expr(other).ok_or_else(|| Error::Type(format!("Other needs to be an expression: {}", type_repr(other))))?;
        wrap(build::logical_binary(LogicalOp::Or, self.e.clone(), o)?)
    }

    fn __invert__(&self) -> Result<PyExpr> {
        wrap(build::not(self.e.clone())?)
    }

    /// `expr.isin(values)`: `values` is a JSON expression or a constant collection.
    fn isin(&self, value_set: &Bound<'_, PyAny>) -> Result<PyExpr> {
        if let Some(rhs) = as_expr(value_set) {
            return wrap(build::in_expr(self.e.clone(), rhs)?);
        }
        if !self.e.col_type.is_scalar() {
            crate::bail!(UnsupportedOperation, "isin(): only supported for scalar types, not {}", self.e.col_type.repr());
        }
        let iter = value_set.try_iter().map_err(|_| {
            crate::pxt_err!(
                InvalidArgument,
                "isin(): argument must be an Iterable (eg, list, dict, ...), not {}",
                value_set.repr().map(|r| r.to_string()).unwrap_or_default()
            )
        })?;
        let mut kept = Vec::new();
        let mut kept_py = Vec::new();
        for v in iter {
            let v = v?;
            // values of the wrong type are ignored, as upstream does
            if vpy::validate(&v, &self.e.col_type).is_ok() {
                kept.push(vpy::coerce(&v, &self.e.col_type)?);
                kept_py.push(v);
            }
        }
        let display = PyList::new(value_set.py(), kept_py)?.repr()?.to_string();
        wrap(build::in_list(self.e.clone(), kept, display)?)
    }

    /// `expr.astype(type)`: changes the static type (no conversion of values).
    fn astype(&self, new_type: &Bound<'_, PyAny>) -> Result<PyExpr> {
        let t = normalize_type(new_type, false)?;
        wrap(build::type_cast(self.e.clone(), t))
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> Result<PyExpr> {
        if self.e.col_type.is_json() || matches!(self.e.kind, ExprKind::JsonPath { .. }) {
            let el = json_path_elem(index)?;
            return wrap(build::json_path_append(&self.e, el)?);
        }
        if self.e.col_type.is_array() {
            let py = index.py();
            let tuple = match index.cast::<PyTuple>() {
                Ok(t) => t.clone(),
                Err(_) => PyTuple::new(py, [index])?,
            };
            let mut idx = Vec::new();
            for item in tuple.iter() {
                match array_index(&item)? {
                    Some(i) => idx.push(i),
                    None => {
                        return Err(Error::Attribute(format!("Invalid array indices: {}", tuple.repr()?)));
                    }
                }
            }
            return wrap(build::array_slice(self.e.clone(), idx)?);
        }
        Err(Error::Attribute(format!("Type {} is not subscriptable", self.e.col_type.repr())))
    }

    /// `expr.name`: a method or property registered by a UDF (`is_method`/`is_property`) takes precedence; then a
    /// column property (`t.img.localpath`), a builtin method or property of the expression's type (a property such as
    /// `t.img.width` is called implicitly), the `map`/`filter`/`sort` mappers or a field (a JSON path) of a JSON
    /// expression.
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> Result<Bound<'py, PyAny>> {
        if let Some(m) = super::func::udf_attribute(py, &self.e, name)? {
            return Ok(m);
        }
        self.builtin_attribute(py, name)
    }

    /// For a call of a Python function: the (resolved) `Function`; else None.
    #[getter]
    fn r#fn(&self, py: Python<'_>) -> Result<Py<PyAny>> {
        if let crate::expr::ExprKind::FunctionCall(crate::expr::FunctionCall {
            func: crate::functions::FuncRef::Udf(call),
            ..
        }) = &self.e.kind
        {
            return Ok(call.def.py_obj.bind(py).call_method1("_resolved_fn", (call.sig_idx,))?.unbind());
        }
        Ok(py.None())
    }

    /// `expr.apply(fn, col_type=None)`: a call of `fn` (an arbitrary Python callable) on this expression.
    #[pyo3(signature = (fn_, *, col_type=None))]
    fn apply<'py>(slf: &Bound<'py, Self>, fn_: &Bound<'py, PyAny>, col_type: Option<&Bound<'py, PyAny>>) -> Result<Bound<'py, PyAny>> {
        let py = slf.py();
        let arg_type = slf.getattr("col_type")?;
        let f = py
            .import("pixeltable.func._apply")?
            .call_method1("make_applicator", (fn_, col_type.map(|c| c.clone()).unwrap_or_else(|| py.None().into_bound(py)), arg_type))?;
        Ok(f.call1((slf,))?)
    }

    /// For a literal: (True, value); else (False, None).
    fn _literal_value<'py>(&self, py: Python<'py>) -> Result<(bool, Bound<'py, PyAny>)> {
        match self.e.as_literal() {
            Some(v) => Ok((true, vpy::to_py(py, v, &self.e.col_type)?)),
            None => Ok((false, py.None().into_bound(py))),
        }
    }

    /// `col.similarity(...)`: a similarity lookup against an embedding index of the column.
    #[pyo3(signature = (item=None, *, string=None, image=None, audio=None, video=None, document=None, vector=None, idx=None))]
    #[allow(clippy::too_many_arguments)]
    fn similarity<'py>(
        &self,
        py: Python<'py>,
        item: Option<&Bound<'py, PyAny>>,
        string: Option<&Bound<'py, PyAny>>,
        image: Option<&Bound<'py, PyAny>>,
        audio: Option<&Bound<'py, PyAny>>,
        video: Option<&Bound<'py, PyAny>>,
        document: Option<&Bound<'py, PyAny>>,
        vector: Option<&Bound<'py, PyAny>>,
        idx: Option<&str>,
    ) -> Result<PyExpr> {
        let given = |v: Option<&Bound<'py, PyAny>>| v.filter(|v| !v.is_none()).cloned();
        let (string, image, audio) = (given(string), given(image), given(audio));
        let (video, document, vector) = (given(video), given(document), given(vector));
        let item = given(item);
        let named = [
            ("string", string.as_ref()),
            ("image", image.as_ref()),
            ("audio", audio.as_ref()),
            ("video", video.as_ref()),
            ("document", document.as_ref()),
            ("vector", vector.as_ref()),
        ];
        super::indexes::similarity(py, &self.e, item.as_ref(), &named, idx)
    }

    /// `col.recompute()`: `Table.recompute_columns()` for this column.
    #[pyo3(signature = (*, cascade=true, errors_only=false))]
    fn recompute<'py>(slf: &Bound<'py, Self>, cascade: bool, errors_only: bool) -> Result<Bound<'py, PyAny>> {
        let py = slf.py();
        let (e, path_tbl_id) = (slf.get().e.clone(), slf.get().path_tbl_id);
        let Some(c) = e.as_column_ref() else {
            return Err(Error::Attribute(format!("Unknown method (of type {}): recompute", e.col_type.repr())));
        };
        let tbl_id = c.tbl_id;
        if path_tbl_id.is_some_and(|p| p != tbl_id) {
            crate::bail!(UnsupportedOperation, "Cannot recompute column of a base.");
        }
        let is_snapshot = c.tbl_version.is_some()
            || crate::catalog::runtime::with_catalog(py, |cat| Ok(cat.table(tbl_id)?.is_snapshot()))?;
        if is_snapshot {
            crate::bail!(UnsupportedOperation, "Cannot recompute column of a snapshot.");
        }
        let cols = PyList::new(py, [slf])?;
        let d = super::mutate::tbl_recompute_columns(py, &tbl_id.to_string(), None, &cols, None, errors_only, cascade)?;
        Ok(py.import("pixeltable._update_status")?.call_method1("_status_from_core", (d,))?)
    }

    fn _is_column_ref(&self) -> bool {
        self.e.as_column_ref().is_some()
    }

    /// (tbl_id, col_id) of a column reference, else None.
    fn _column_id(&self) -> Option<(String, u32)> {
        self.e.as_column_ref().map(|c| (c.tbl_id.to_string(), c.col_id))
    }
}

impl PyExpr {
    /// `expr.name` for builtins: column properties, builtin methods/properties of the expression's type, JSON
    /// mappers and JSON paths.
    fn builtin_attribute<'py>(&self, py: Python<'py>, name: &str) -> Result<Bound<'py, PyAny>> {
        if name.starts_with('_') {
            return Err(Error::Attribute(format!("'Expr' object has no attribute {}", crate::error::py_repr(name))));
        }
        if let Some(c) = self.e.as_column_ref() {
            if let Some(prop) = ColumnProp::from_name(name).filter(|p| *p != ColumnProp::CellMd) {
                return Ok(Bound::new(py, wrap(column_property(py, &self.e, c, prop)?)?)?.into_any());
            }
        }
        let t = &self.e.col_type;
        if t.is_json() {
            let op = match name {
                "map" => Some(MapperOp::Map),
                "filter" => Some(MapperOp::Filter),
                "sort" => Some(MapperOp::Sort),
                _ => None,
            };
            if let Some(op) = op {
                return Ok(Bound::new(py, PyMapperRef { src: self.e.clone(), op })?.into_any());
            }
        }
        if let Some(f) = crate::functions::lookup_method(t, name) {
            if f.kind == crate::functions::Kind::Property {
                let call = crate::functions::make_builtin_call(f, vec![self.e.clone()], vec![], false)?;
                return Ok(Bound::new(py, wrap(call)?)?.into_any());
            }
            return Ok(Bound::new(py, PyMethodRef { base: self.e.clone(), f, name: name.to_string() })?.into_any());
        }
        if t.is_json() {
            let e = build::json_path_append(&self.e, JsonPathElem::Key(name.to_string()))?;
            return Ok(Bound::new(py, wrap(e)?)?.into_any());
        }
        if let Some(module) = crate::functions::method_module(t) {
            if py.import(module).and_then(|m| m.hasattr(name)).unwrap_or(false) {
                return Err(Error::not_implemented(format!("`{module}.{name}()`")));
            }
        }
        Err(Error::Attribute(format!("Unknown method (of type {}): {name}", t.repr())))
    }
}

/// `col.<prop>` (upstream `ColumnRef.__getattr__()`): checks that the property applies to the column.
fn column_property(py: Python<'_>, col_expr: &ExprRef, c: &crate::expr::ColumnRef, prop: ColumnProp) -> Result<ExprRef> {
    let md = crate::catalog::runtime::with_catalog(py, |cat| {
        cat.table(c.tbl_id)?
            .column(c.col_id)
            .cloned()
            .ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column: {}", c.name))
    })?;
    let name = prop.name();
    let shown = display::display(col_expr);
    if prop.is_cellmd() {
        if !((md.is_computed() || md.col_type.is_media()) && md.stored) {
            crate::bail!(UnsupportedOperation, "{name} only valid for a stored computed or media column: {shown}");
        }
    } else {
        if !md.col_type.is_media() {
            crate::bail!(UnsupportedOperation, "{name} only valid for image/video/audio/document columns: {shown}");
        }
        if md.is_computed() && !md.stored {
            crate::bail!(UnsupportedOperation, "{name} not valid for computed unstored columns: {shown}");
        }
    }
    Ok(build::column_property(col_expr.clone(), prop))
}

/// A JSON path element from a subscript (`'key'`, `'*'`, an int or a slice).
fn json_path_elem(index: &Bound<'_, PyAny>) -> Result<JsonPathElem> {
    if index.is_instance_of::<PyString>() {
        let k: String = index.extract()?;
        return Ok(if k == "*" { JsonPathElem::Wildcard } else { JsonPathElem::Key(k) });
    }
    if index.is_instance_of::<PyInt>() && !index.is_instance_of::<PyBool>() {
        return Ok(JsonPathElem::Index(index.extract()?));
    }
    if let Ok(sl) = index.cast::<PySlice>() {
        let part = |name: &str| -> Result<Option<i64>> {
            let v = sl.getattr(name)?;
            if v.is_none() { Ok(None) } else { Ok(Some(v.extract::<i64>()?)) }
        };
        return Ok(JsonPathElem::Slice(part("start")?, part("stop")?, part("step")?));
    }
    crate::bail!(UnsupportedOperation, "Invalid json list index: {}", index.str()?)
}

/// An array index item (an int or a slice of ints with a nonzero step); None if invalid.
fn array_index(item: &Bound<'_, PyAny>) -> Result<Option<ArrayIndex>> {
    if item.is_instance_of::<PyInt>() && !item.is_instance_of::<PyBool>() {
        return Ok(Some(ArrayIndex::Int(item.extract()?)));
    }
    if let Ok(sl) = item.cast::<PySlice>() {
        let mut parts = [None, None, None];
        for (i, name) in ["start", "stop", "step"].iter().enumerate() {
            let v = sl.getattr(*name)?;
            if v.is_none() {
                continue;
            }
            if !v.is_instance_of::<PyInt>() {
                return Ok(None);
            }
            parts[i] = Some(v.extract::<i64>()?);
        }
        if parts[2] == Some(0) {
            return Ok(None);
        }
        return Ok(Some(ArrayIndex::Slice(parts[0], parts[1], parts[2])));
    }
    Ok(None)
}

fn builtin(qualified: &str) -> Result<&'static crate::functions::BuiltinFn> {
    crate::functions::lookup(qualified).ok_or_else(|| Error::internal(format!("builtin {qualified} is not registered")))
}

fn obj_str(obj: &Bound<'_, PyAny>) -> String {
    obj.str().map(|s| s.to_string()).unwrap_or_default()
}

fn type_str(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type().str().map(|s| s.to_string()).unwrap_or_default()
}

/// Converts call arguments (Python objects) to expressions, with upstream's error for an argument that is not one.
pub fn call_args(
    qualified: &str,
    args: &[Bound<'_, PyAny>],
    kwargs: &[(String, Bound<'_, PyAny>)],
) -> Result<(Vec<ExprRef>, Vec<(String, ExprRef)>)> {
    let mut arg_exprs = Vec::with_capacity(args.len());
    for (i, a) in args.iter().enumerate() {
        match from_object(a)? {
            Some(e) => arg_exprs.push(e),
            None => crate::bail!(
                InvalidExpression,
                "Argument {} in call to {} is not a valid Pixeltable expression",
                i + 1,
                crate::error::py_repr(qualified)
            ),
        }
    }
    let mut kw = Vec::with_capacity(kwargs.len());
    for (k, v) in kwargs {
        match from_object(v)? {
            Some(e) => kw.push((k.clone(), e)),
            None => crate::bail!(
                InvalidExpression,
                "Argument {} in call to {} is not a valid Pixeltable expression",
                crate::error::py_repr(k),
                crate::error::py_repr(qualified)
            ),
        }
    }
    Ok((arg_exprs, kw))
}

fn split_call<'py>(args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> Result<(Vec<Bound<'py, PyAny>>, Vec<(String, Bound<'py, PyAny>)>)> {
    let a: Vec<Bound<'py, PyAny>> = args.iter().collect();
    let mut kw = Vec::new();
    if let Some(kwargs) = kwargs {
        for (k, v) in kwargs.iter() {
            kw.push((k.extract::<String>()?, v));
        }
    }
    Ok((a, kw))
}

/// A call of an aggregate function from Python arguments (upstream `AggregateFunction.__call__`): extracts and
/// checks `order_by` / `group_by`, converts the arguments and builds the call. Python UDAs use this too.
pub fn agg_call_from_py(
    f: crate::functions::agg::AggRef,
    args: Vec<Bound<'_, PyAny>>,
    kwargs: Vec<(String, Bound<'_, PyAny>)>,
) -> Result<ExprRef> {
    let name = f.name().to_string();
    let mut args = args;
    let mut kwargs = kwargs;
    let mut order_by_obj: Option<Bound<'_, PyAny>> = None;
    if let Some(pos) = kwargs.iter().position(|(k, _)| k == "order_by") {
        if f.requires_order_by() {
            crate::bail!(
                UnsupportedOperation,
                "{name}(): order_by invalid, this function requires the first argument to be the ordering expression"
            );
        }
        if !f.allows_window() {
            crate::bail!(UnsupportedOperation, "{name}(): order_by invalid with an aggregate function that does not allow windows");
        }
        order_by_obj = Some(kwargs.remove(pos).1);
    } else if f.requires_order_by() {
        if args.is_empty() {
            crate::bail!(UnsupportedOperation, "{name}(): requires an ordering expression as its first argument");
        }
        let first = args.remove(0);
        if as_expr(&first).is_none() {
            crate::bail!(
                UnsupportedOperation,
                "{name}(): the first argument needs to be a Pixeltable expression, but instead is a {}",
                type_str(&first)
            );
        }
        order_by_obj = Some(first);
    }
    let mut group_by = Vec::new();
    if let Some(pos) = kwargs.iter().position(|(k, _)| k == "group_by") {
        if !f.allows_window() {
            crate::bail!(UnsupportedOperation, "{name}(): group_by invalid with an aggregate function that does not allow windows");
        }
        let g = kwargs.remove(pos).1;
        match as_expr(&g) {
            Some(e) => group_by.push(e),
            None if g.hasattr("_tbl_version_path").unwrap_or(false) || g.hasattr("_id").unwrap_or(false) => {
                return Err(Error::not_implemented("group_by=<table> in a window function"));
            }
            None => crate::bail!(InvalidExpression, "group_by argument needs to be a Pixeltable expression: {}", obj_str(&g)),
        }
    }
    let order_by = match order_by_obj {
        None => None,
        Some(o) => match as_expr(&o) {
            Some(e) => Some(e),
            None => crate::bail!(
                UnsupportedOperation,
                "order_by argument needs to be a Pixeltable expression, but instead is a {}",
                type_str(&o)
            ),
        },
    };
    let (args, kwargs) = call_args(&f.qualified_name(), &args, &kwargs)?;
    crate::functions::agg::make_agg_call(f, args, kwargs, order_by, group_by)
}

/// A method of an expression (`t.c.upper`), called to build the function call (`t.c.upper()`).
#[pyclass(name = "MethodRef", module = "pixeltable", frozen)]
pub struct PyMethodRef {
    base: ExprRef,
    f: &'static crate::functions::BuiltinFn,
    name: String,
}

#[pymethods]
impl PyMethodRef {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(&self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Result<PyExpr> {
        let (a, kw) = split_call(args, kwargs)?;
        let (mut arg_exprs, kw) = call_args(&self.f.qualified_name(), &a, &kw)?;
        arg_exprs.insert(0, self.base.clone());
        wrap(crate::functions::make_builtin_call(self.f, arg_exprs, kw, true)?)
    }

    fn __repr__(&self) -> String {
        format!("{}.{}", display::display(&self.base), self.name)
    }
}

/// `json_expr.map` / `.filter` / `.sort`, called with the element function.
#[pyclass(name = "JsonMapperRef", module = "pixeltable", frozen)]
pub struct PyMapperRef {
    src: ExprRef,
    op: MapperOp,
}

#[pymethods]
impl PyMapperRef {
    #[pyo3(signature = (fn_=None, *, key=None, asc=true))]
    fn __call__(&self, py: Python<'_>, fn_: Option<Bound<'_, PyAny>>, key: Option<Bound<'_, PyAny>>, asc: bool) -> Result<PyExpr> {
        let f = fn_.or(key);
        wrap(json_mapper_call(py, self.op, &self.src, f.as_ref(), asc)?)
    }

    fn __repr__(&self) -> String {
        let name = match self.op {
            MapperOp::Map => "map",
            MapperOp::Filter => "filter",
            MapperOp::Sort => "sort",
        };
        format!("{}.{name}", display::display(&self.src))
    }
}

/// Builds a JSON mapper: calls `f` (a Python callable) once with the relative root to obtain the target expression
/// (upstream `pixeltable.functions.map/filter/sort`). A keyless sort is `json._sort()`.
pub fn json_mapper_call(py: Python<'_>, op: MapperOp, src: &ExprRef, f: Option<&Bound<'_, PyAny>>, asc: bool) -> Result<ExprRef> {
    let Some(f) = f.filter(|f| !f.is_none()) else {
        if op != MapperOp::Sort {
            return Err(Error::Type(format!("{}() missing required argument", if op == MapperOp::Map { "map" } else { "filter" })));
        }
        let sort_fn = builtin("pixeltable.functions.json._sort")?;
        let mut kwargs = Vec::new();
        if !asc {
            kwargs.push(("asc".to_string(), Expr::literal(Value::Bool(false), crate::types::ColumnType::bool(false))));
        }
        return crate::functions::make_builtin_call(sort_fn, vec![src.clone()], kwargs, true);
    };
    let root = build::relative_root(&src.col_type);
    let ExprKind::RelativeRoot { id } = root.kind else { return Err(Error::internal("relative root expected")) };
    let failure = match op {
        MapperOp::Map => "Failed to evaluate map function. (The `fn` argument to `map()` must produce a valid Pixeltable expression.)",
        MapperOp::Filter => {
            "Failed to evaluate filter predicate. (The `predicate` argument to `filter()` must produce a valid Pixeltable expression.)"
        }
        MapperOp::Sort => "Failed to evaluate sort key. (The `key` argument to `sort()` must produce a valid Pixeltable expression.)",
    };
    let target = match f.call1((PyExpr::new(root),)) {
        Ok(result) => match from_object(&result) {
            Ok(Some(e)) => e,
            _ => crate::bail!(UnsupportedOperation, "{failure}"),
        },
        Err(_) => crate::bail!(UnsupportedOperation, "{failure}"),
    };
    let _ = py;
    Ok(build::json_mapper(op, asc, src.clone(), target, id))
}

/// `pixeltable.functions.map/filter/sort(expr, fn, ...)`
#[pyfunction]
#[pyo3(signature = (op, expr, f=None, asc=true))]
pub fn json_mapper(py: Python<'_>, op: &str, expr: &Bound<'_, PyAny>, f: Option<Bound<'_, PyAny>>, asc: bool) -> Result<PyExpr> {
    let op = match op {
        "map" => MapperOp::Map,
        "filter" => MapperOp::Filter,
        _ => MapperOp::Sort,
    };
    let src = from_object(expr)?.ok_or_else(|| crate::pxt_err!(InvalidExpression, "Invalid expression: {}", obj_str(expr)))?;
    wrap(json_mapper_call(py, op, &src, f.as_ref(), asc)?)
}

/// `pixeltable.functions.cast(expr, type)`: the same expression with another static type (no conversion).
#[pyfunction]
pub fn cast(expr: &Bound<'_, PyAny>, target_type: &Bound<'_, PyAny>) -> Result<PyExpr> {
    let e = from_object(expr)?.ok_or_else(|| crate::pxt_err!(InvalidExpression, "Invalid expression: {}", obj_str(expr)))?;
    let t = normalize_type(target_type, true)?;
    wrap(Expr::new(e.kind.clone(), t))
}

/// `Literal(val)` / `Expr.from_object(val)`: None if the object cannot become an expression.
#[pyfunction]
pub fn expr_from_object(obj: &Bound<'_, PyAny>) -> Result<Option<PyExpr>> {
    from_object(obj)?.map(wrap).transpose()
}

/// `pxt.array(elements)`
#[pyfunction]
pub fn array(elements: &Bound<'_, PyAny>) -> Result<PyExpr> {
    wrap(from_array(elements)?)
}

/// A call of the builtin function `qualified_name` (used by the `pixeltable.functions` stubs).
#[pyfunction]
#[pyo3(signature = (qualified_name, *args, **kwargs))]
pub fn call_builtin(
    qualified_name: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> Result<PyExpr> {
    let (a, kw) = split_call(args, kwargs)?;
    if let Some(f) = crate::functions::agg::lookup_builtin_agg(qualified_name) {
        return wrap(agg_call_from_py(f, a, kw)?);
    }
    let f = crate::functions::lookup(qualified_name)
        .ok_or_else(|| crate::pxt_err!(FunctionNotFound, "Function {} not found", crate::error::py_repr(qualified_name)))?;
    let (arg_exprs, kw) = call_args(qualified_name, &a, &kw)?;
    wrap(crate::functions::make_builtin_call(f, arg_exprs, kw, false)?)
}

#[pyfunction]
pub fn builtin_names() -> Vec<String> {
    crate::functions::builtin_names()
}
