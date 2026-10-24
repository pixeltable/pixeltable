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
            "invalid" => UdfImpl::Invalid {
                error_msg: spec_item(spec, "error_msg")?.map(|m| m.extract()).transpose()?.unwrap_or_default(),
            },
            other => return Err(Error::internal(format!("unknown function kind {other:?}"))),
        };
        let storable = match &imp {
            UdfImpl::Callable { .. } => self_path.is_some(),
            UdfImpl::Template { exprs } => self_path.is_some() || exprs.iter().all(|e| udf::is_storable(e)),
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
