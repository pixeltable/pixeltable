//! `pixeltable._core.Udf`: the backend half of a Python function (`pixeltable.func.Function._core_udf()`), and the
//! function registry.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::Arc;

use super::expr::{PyExpr, as_expr, from_object};
use crate::error::{Error, Result, py_repr};
use crate::expr::{Expr, ExprKind, ExprRef};
use crate::functions::signature::Signature;
use crate::functions::udf::{self, UdfDef, UdfImpl};

#[pyclass(name = "Udf", module = "pixeltable._core", frozen)]
pub struct PyUdf {
    pub def: Arc<UdfDef>,
}

fn spec_item<'py>(spec: &Bound<'py, PyDict>, key: &str) -> Result<Option<Bound<'py, PyAny>>> {
    Ok(spec.get_item(key)?.filter(|v| !v.is_none()))
}

/// Converts call arguments to expressions (upstream `Function.__call__()`).
pub fn call_args(
    self_path: Option<&str>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> Result<(Vec<ExprRef>, Vec<(String, ExprRef)>)> {
    let path = self_path.map(py_repr).unwrap_or_else(|| "None".to_string());
    let mut arg_exprs = Vec::new();
    for (i, a) in args.iter().enumerate() {
        match from_object(&a)? {
            Some(e) => arg_exprs.push(e),
            None => crate::bail!(
                InvalidExpression,
                "Argument {} in call to {path} is not a valid Pixeltable expression",
                i + 1
            ),
        }
    }
    let mut kw = Vec::new();
    if let Some(kwargs) = kwargs {
        for (k, v) in kwargs.iter() {
            let name: String = k.extract()?;
            match from_object(&v)? {
                Some(e) => kw.push((name, e)),
                None => crate::bail!(
                    InvalidExpression,
                    "Argument {} in call to {path} is not a valid Pixeltable expression",
                    py_repr(&name)
                ),
            }
        }
    }
    Ok((arg_exprs, kw))
}

#[pymethods]
impl PyUdf {
    /// `spec`: `kind` ('callable', 'template' or 'invalid'), `name`, `display_name`, `self_path`, `signatures`
    /// (dicts, see `Signature::from_py`), `py_fns` / `batch_size` (callable), `templates` (template expressions),
    /// `is_method`, `is_property`, `error_msg` (invalid).
    #[new]
    fn new(py_obj: Bound<'_, PyAny>, spec: &Bound<'_, PyDict>) -> Result<Self> {
        let kind: String = spec_item(spec, "kind")?.ok_or_else(|| Error::internal("udf spec without kind"))?.extract()?;
        let name: Option<String> = spec_item(spec, "name")?.map(|n| n.extract()).transpose()?;
        let display_name: String = spec_item(spec, "display_name")?.map(|n| n.extract()).transpose()?.unwrap_or_default();
        let self_path: Option<String> = spec_item(spec, "self_path")?.map(|n| n.extract()).transpose()?;
        let mut signatures = Vec::new();
        if let Some(sigs) = spec_item(spec, "signatures")? {
            for s in sigs.try_iter()? {
                signatures.push(Signature::from_py(&s?)?);
            }
        }
        let flag = |k: &str| -> Result<bool> { Ok(spec_item(spec, k)?.map(|v| v.is_truthy()).transpose()?.unwrap_or(false)) };
        let imp = match kind.as_str() {
            "callable" => {
                let py_fns: Vec<Py<PyAny>> = match spec_item(spec, "py_fns")? {
                    Some(l) => l.try_iter()?.map(|f| Ok(f?.unbind())).collect::<Result<_>>()?,
                    None => Vec::new(),
                };
                if py_fns.len() != signatures.len() {
                    return Err(Error::internal("a callable function needs one Python callable per signature"));
                }
                let batch_size: Option<usize> = spec_item(spec, "batch_size")?.map(|b| b.extract()).transpose()?;
                let iscoro = py_obj.py().import("inspect")?.getattr("iscoroutinefunction")?;
                let is_async: Vec<bool> =
                    py_fns.iter().map(|f| Ok(iscoro.call1((f.bind(py_obj.py()),))?.is_truthy()?)).collect::<Result<_>>()?;
                UdfImpl::Callable { py_fns, batch_size, is_async, has_resource_pool: flag("has_resource_pool")? }
            }
            "template" => {
                let mut exprs = Vec::new();
                if let Some(ts) = spec_item(spec, "templates")? {
                    for t in ts.try_iter()? {
                        let t = t?;
                        exprs.push(as_expr(&t).ok_or_else(|| Error::internal("template is not an expression"))?);
                    }
                }
                if exprs.len() != signatures.len() {
                    return Err(Error::internal("a template function needs one template per signature"));
                }
                UdfImpl::Template { exprs }
            }
            "query" => {
                let q = spec_item(spec, "query")?.ok_or_else(|| Error::internal("query template without a query"))?;
                let q = q
                    .cast::<super::query::PyQuerySpec>()
                    .map_err(|_| Error::internal("a query template needs a _core.QuerySpec"))?;
                let return_scalar = flag("return_scalar")?;
                UdfImpl::Query { spec: q.get().q.clone(), return_scalar }
            }
            "invalid" => UdfImpl::Invalid {
                error_msg: spec_item(spec, "error_msg")?.map(|m| m.extract()).transpose()?.unwrap_or_default(),
            },
            other => return Err(Error::internal(format!("unknown function kind {other:?}"))),
        };
        let storable = match &imp {
            UdfImpl::Callable { .. } => self_path.is_some(),
            UdfImpl::Template { exprs } => self_path.is_some() || exprs.iter().all(|e| udf::is_storable(e)),
            UdfImpl::Query { spec, .. } => spec.exprs().iter().all(|e| udf::is_storable(e)),
            UdfImpl::Invalid { .. } => false,
        };
        let def = UdfDef::new(
            name,
            display_name,
            self_path,
            signatures,
            imp,
            py_obj.unbind(),
            flag("is_method")?,
            flag("is_property")?,
            storable,
        );
        Ok(PyUdf { def: Arc::new(def) })
    }

    /// `fn(*args, **kwargs)`
    #[pyo3(signature = (*args, **kwargs))]
    fn call(&self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Result<PyExpr> {
        let (a, k) = call_args(self.def.self_path.as_deref(), args, kwargs)?;
        Ok(PyExpr::new(udf::make_call(&self.def, a, k, false)?))
    }

    /// `base.fn(*args, **kwargs)` (a method call)
    #[pyo3(signature = (base, *args, **kwargs))]
    fn call_method(
        &self,
        base: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Result<PyExpr> {
        let base = as_expr(base).ok_or_else(|| Error::internal("method call on a non-expression"))?;
        let (mut a, k) = call_args(self.def.self_path.as_deref(), args, kwargs)?;
        a.insert(0, base);
        Ok(PyExpr::new(udf::make_call(&self.def, a, k, true)?))
    }

    /// `Function.using(**kwargs)`: (signature index, template call) for each signature that accepts the bindings.
    fn using<'py>(&self, py: Python<'py>, kwargs: &Bound<'py, PyDict>) -> Result<Bound<'py, PyList>> {
        let mut bindings = Vec::new();
        for (k, v) in kwargs.iter() {
            let name: String = k.extract()?;
            let e = match from_object(&v)? {
                Some(e) => e,
                None => crate::bail!(
                    UnsupportedOperation,
                    "Expected a constant value for parameter {} in call to .using()",
                    py_repr(&name)
                ),
            };
            bindings.push((name, e));
        }
        let out = udf::bind_template(&self.def, &bindings)?;
        let items: Vec<Bound<'py, PyTuple>> = out
            .into_iter()
            .map(|(i, e)| PyTuple::new(py, [i.into_pyobject(py)?.into_any(), Bound::new(py, PyExpr::new(e))?.into_any()]))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?)
    }

    #[getter]
    fn is_storable(&self) -> bool {
        self.def.storable
    }
}

/// A method of an expression that is a registered UDF (`t.c.increment`), called to build the call.
#[pyclass(name = "UdfMethodRef", module = "pixeltable._core", frozen)]
pub struct PyUdfMethodRef {
    base: ExprRef,
    def: Arc<UdfDef>,
}

#[pymethods]
impl PyUdfMethodRef {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(&self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Result<PyExpr> {
        let (mut a, k) = call_args(self.def.self_path.as_deref(), args, kwargs)?;
        a.insert(0, self.base.clone());
        Ok(PyExpr::new(udf::make_call(&self.def, a, k, true)?))
    }

    fn __repr__(&self) -> String {
        format!("{}.{}", crate::expr::display::display(&self.base), self.def.display_name)
    }
}

/// `expr.name` for a method or property registered for the expression's type: a method reference, or (for a
/// property) the call itself; None if no such method is registered.
pub fn udf_attribute<'py>(py: Python<'py>, base: &ExprRef, name: &str) -> Result<Option<Bound<'py, PyAny>>> {
    let Some(f) = udf::lookup_type_method(py, base.col_type.type_enum() as i64, name) else { return Ok(None) };
    let def = udf::def_of(f.bind(py))?;
    if def.is_property {
        let call = udf::make_call(&def, vec![base.clone()], vec![], true)?;
        return Ok(Some(Bound::new(py, PyExpr::new(call))?.into_any()));
    }
    Ok(Some(Bound::new(py, PyUdfMethodRef { base: base.clone(), def })?.into_any()))
}

/// Registers a module-level function (`FUNCTION_ALREADY_EXISTS` if the name is taken).
#[pyfunction]
#[pyo3(signature = (fqn, name, fn_obj, method_of=None))]
pub fn register_function(fqn: &str, name: &str, fn_obj: Bound<'_, PyAny>, method_of: Option<i64>) -> Result<()> {
    udf::register_function(fqn, name, fn_obj.unbind(), method_of)
}

/// The registered module-level functions (`Function` objects), in registration order.
#[pyfunction]
pub fn registered_functions(py: Python<'_>) -> Vec<Py<PyAny>> {
    udf::registered_functions(py)
}

/// A template parameter: `Variable(name, col_type)`.
#[pyfunction]
pub fn variable(name: &str, col_type: &Bound<'_, PyAny>) -> Result<PyExpr> {
    let t = super::types::core_type(col_type).ok_or_else(|| Error::Type("expected a ColumnType".into()))?;
    Ok(PyExpr::new(Expr::new(ExprKind::Variable(name.to_string()), t)))
}

// ------------------------------------------------------------------------------------------------ iterators

use crate::functions::iterator::{self, IteratorCall, IteratorDef};

/// `pixeltable._core.Iterator`: the backend half of a `pixeltable.func.GeneratingFunction`.
#[pyclass(name = "Iterator", module = "pixeltable._core", frozen)]
pub struct PyIterator {
    pub def: Arc<IteratorDef>,
}

#[pymethods]
impl PyIterator {
    /// `spec`: `fqn`, `signature` (a dict, see `Signature::from_py`; None for an iterator that no longer resolves),
    /// `unstored_cols`, `has_seek`, `error_msg`.
    #[new]
    fn new(py_obj: Bound<'_, PyAny>, spec: &Bound<'_, PyDict>) -> Result<Self> {
        let fqn: String = spec_item(spec, "fqn")?.ok_or_else(|| Error::internal("iterator spec without fqn"))?.extract()?;
        let signature = spec_item(spec, "signature")?.map(|s| Signature::from_py(&s)).transpose()?;
        let unstored_cols: Vec<String> = spec_item(spec, "unstored_cols")?.map(|u| u.extract()).transpose()?.unwrap_or_default();
        let has_seek = spec_item(spec, "has_seek")?.map(|v| v.is_truthy()).transpose()?.unwrap_or(false);
        let error_msg: Option<String> = spec_item(spec, "error_msg")?.map(|m| m.extract()).transpose()?;
        Ok(PyIterator { def: Arc::new(IteratorDef::new(fqn, signature, unstored_cols, has_seek, py_obj.unbind(), error_msg)) })
    }

    /// `iterator(*args, **kwargs)`
    #[pyo3(signature = (*args, **kwargs))]
    fn call(&self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> Result<PyIteratorCall> {
        let (a, k) = call_args(Some(&self.def.fqn), args, kwargs)?;
        Ok(PyIteratorCall { call: Arc::new(iterator::make_call(&self.def, a, k)?) })
    }
}

/// `pixeltable._core.IteratorCall`: a bound iterator call (what `create_view(iterator=...)` receives, inside a
/// `pixeltable.func.GeneratingFunctionCall`).
#[pyclass(name = "IteratorCall", module = "pixeltable._core", frozen)]
pub struct PyIteratorCall {
    pub call: Arc<IteratorCall>,
}

#[pymethods]
impl PyIteratorCall {
    fn display_str(&self) -> String {
        self.call.display()
    }

    /// (folded name, ColumnType) of each output, `pos` first.
    fn outputs<'py>(&self, py: Python<'py>) -> Result<Vec<(String, Bound<'py, PyAny>)>> {
        self.call
            .outputs
            .iter()
            .map(|o| Ok((o.name.clone(), super::types::wrap_type(py, &o.col_type)?)))
            .collect()
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>> {
        Ok(crate::value::py::json_to_py(py, &self.call.to_json()?)?)
    }

    #[getter]
    fn is_valid(&self) -> bool {
        self.call.validation_error.is_none()
    }
}

/// The iterator call passed as `create_view(iterator=...)`: a `GeneratingFunctionCall` (or a `_core.IteratorCall`).
pub fn iterator_call_from_py(obj: &Bound<'_, PyAny>) -> Result<Arc<IteratorCall>> {
    if let Ok(c) = obj.cast::<PyIteratorCall>() {
        return Ok(c.get().call.clone());
    }
    if let Ok(inner) = obj.getattr("_core") {
        if let Ok(c) = inner.cast::<PyIteratorCall>() {
            return Ok(c.get().call.clone());
        }
    }
    Err(Error::Type(format!(
        "iterator must be a call of a @pxt.iterator, not {}",
        obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
    )))
}

/// Upstream `udf.from_table()`: the template of `pxt.udf(table)`: (template expression, parameters as
/// (name, ColumnType)). Data columns of the table and its bases become variables (the parameters); computed columns
/// become their value expressions with the substitutions applied; the template is a dict of all columns, or
/// `return_value` with the same substitutions.
#[pyfunction]
#[pyo3(signature = (tbl_id, return_value=None))]
pub fn table_udf<'py>(
    py: Python<'py>,
    tbl_id: &str,
    return_value: Option<&Bound<'py, PyAny>>,
) -> Result<(PyExpr, Vec<(String, Bound<'py, PyAny>)>)> {
    let id = super::catalog::parse_uuid(tbl_id)?;
    let return_value = match return_value.filter(|r| !r.is_none()) {
        None => None,
        Some(r) => Some(from_object(r)?.ok_or_else(|| Error::Type("return_value must be an expression".into()))?),
    };
    let (expr, params) = crate::catalog::runtime::with_catalog(py, |cat| {
        let mut subst: Vec<((uuid::Uuid, u32), ExprRef)> = Vec::new();
        let mut result: Vec<(String, ExprRef)> = Vec::new();
        let mut params: Vec<(String, crate::types::ColumnType)> = Vec::new();
        for t in cat.base_chain(id).into_iter().rev() {
            for c in cat.table(t)?.columns_at(None)? {
                let key = (c.tbl_id, c.md.id);
                let e = match &c.md.value_expr {
                    Some(j) => {
                        let value = crate::expr::serde::from_json(j)?;
                        udf::substitute(&value, &|x| substitution(&subst, x))?
                    }
                    None => {
                        params.push((c.name.clone(), c.md.col_type.clone()));
                        Expr::new(ExprKind::Variable(c.name.clone()), c.md.col_type.clone())
                    }
                };
                subst.push((key, e.clone()));
                result.push((c.name.clone(), e));
            }
        }
        let expr = match &return_value {
            None => crate::expr::build::inline_dict(result),
            Some(r) => udf::substitute(r, &|x| substitution(&subst, x))?,
        };
        Ok((expr, params))
    })?;
    let params = params
        .into_iter()
        .map(|(n, t)| Ok((n, super::types::wrap_type(py, &t)?)))
        .collect::<Result<Vec<_>>>()?;
    Ok((PyExpr::new(expr), params))
}

fn substitution(subst: &[((uuid::Uuid, u32), ExprRef)], x: &Expr) -> Option<ExprRef> {
    let c = x.as_column_ref()?;
    subst.iter().find(|(k, _)| *k == (c.tbl_id, c.col_id)).map(|(_, e)| e.clone())
}
