//! Python user-defined aggregates (`@pxt.uda`): the `agg::AggregateFunction` implementation behind a
//! `pixeltable.func.AggregateFunction` (created by `_core.Uda`, see `bindings::func`). The aggregation itself is the
//! query engine's (`agg.rs`, `exec`); a state instantiates the `Aggregator` class with the call's init arguments and
//! calls its `update()` and `value()`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::{Arc, OnceLock};

use super::agg::{AggRef, AggState, AggregateFunction};
use super::signature::{ParamKind, Signature};
use crate::error::{Error, Result};
use crate::types::ColumnType;
use crate::value::py as vpy;
use crate::value::{Json, Value};

pub struct UdaFunction {
    pub name: String,
    pub self_path: String,
    pub signatures: Vec<Signature>,
    /// the `Aggregator` class of each signature
    pub agg_classes: Vec<Py<PyAny>>,
    /// names of the `__init__()` parameters (of the first signature; the classes of overloads share them)
    pub init_param_names: Vec<String>,
    pub requires_order_by: bool,
    pub allows_std_agg: bool,
    pub allows_window: bool,
    /// the `pixeltable.func.AggregateFunction`
    pub py_obj: Py<PyAny>,
    descriptor: OnceLock<Json>,
}

impl std::fmt::Debug for UdaFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UdaFunction({})", self.self_path)
    }
}

impl UdaFunction {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        self_path: String,
        signatures: Vec<Signature>,
        agg_classes: Vec<Py<PyAny>>,
        init_param_names: Vec<String>,
        requires_order_by: bool,
        allows_std_agg: bool,
        allows_window: bool,
        py_obj: Py<PyAny>,
    ) -> UdaFunction {
        UdaFunction {
            name,
            self_path,
            signatures,
            agg_classes,
            init_param_names,
            requires_order_by,
            allows_std_agg,
            allows_window,
            py_obj,
            descriptor: OnceLock::new(),
        }
    }

    fn descriptor(&self) -> Result<Json> {
        if let Some(d) = self.descriptor.get() {
            return Ok(d.clone());
        }
        let d = Python::attach(|py| -> Result<Json> {
            let d = self.py_obj.bind(py).call_method0("as_dict")?;
            let s: String = py.import("json")?.call_method1("dumps", (d,))?.extract()?;
            Ok(serde_json::from_str(&s)?)
        })?;
        let _ = self.descriptor.set(d.clone());
        Ok(d)
    }
}

/// The type of the parameter that receives a value (for its conversion to Python).
fn param_type(sig: &Signature, name: Option<&str>, pos: usize) -> ColumnType {
    let p = match name {
        Some(n) => sig.param(n).map(|(_, p)| p),
        None => sig.params.iter().filter(|p| matches!(p.kind, ParamKind::PositionalOnly | ParamKind::PositionalOrKeyword)).nth(pos),
    };
    p.and_then(|p| p.col_type.clone()).unwrap_or_else(|| ColumnType::json(true))
}

impl AggregateFunction for UdaFunction {
    fn name(&self) -> &str {
        &self.name
    }
    fn calls_python(&self) -> bool {
        true
    }
    fn qualified_name(&self) -> String {
        self.self_path.clone()
    }
    fn signatures(&self) -> &[Signature] {
        &self.signatures
    }
    fn init_param_names(&self) -> &[String] {
        &self.init_param_names
    }
    fn requires_order_by(&self) -> bool {
        self.requires_order_by
    }
    fn allows_std_agg(&self) -> bool {
        self.allows_std_agg
    }
    fn allows_window(&self) -> bool {
        self.allows_window
    }
    fn create(&self, sig_idx: usize, init_args: &[(String, Value)]) -> Result<Box<dyn AggState>> {
        let sig = self.signatures.get(sig_idx).ok_or_else(|| Error::internal("UDA signature out of range"))?.clone();
        let obj = Python::attach(|py| -> Result<Py<PyAny>> {
            let kwargs = PyDict::new(py);
            for (k, v) in init_args {
                kwargs.set_item(k, vpy::to_py(py, v, &param_type(&sig, Some(k), 0))?)?;
            }
            Ok(self.agg_classes[sig_idx].bind(py).call((), Some(&kwargs))?.unbind())
        })?;
        Ok(Box::new(UdaState { obj, sig }))
    }
    fn to_json(&self) -> Result<Json> {
        Ok(serde_json::json!({ "uda": self.descriptor()? }))
    }
    fn py_function(&self) -> Option<&Py<PyAny>> {
        Some(&self.py_obj)
    }
}

struct UdaState {
    obj: Py<PyAny>,
    sig: Signature,
}

impl AggState for UdaState {
    fn update(&mut self, args: &[Value], kwargs: &[(String, Value)]) -> Result<()> {
        Python::attach(|py| -> Result<()> {
            let py_args = args
                .iter()
                .enumerate()
                .map(|(i, v)| Ok(vpy::to_py(py, v, &param_type(&self.sig, None, i))?))
                .collect::<Result<Vec<_>>>()?;
            let py_kwargs = PyDict::new(py);
            for (k, v) in kwargs {
                py_kwargs.set_item(k, vpy::to_py(py, v, &param_type(&self.sig, Some(k), 0))?)?;
            }
            self.obj.bind(py).call_method("update", PyTuple::new(py, py_args)?, Some(&py_kwargs))?;
            Ok(())
        })
    }

    fn value(&mut self) -> Result<Value> {
        Python::attach(|py| {
            let v = self.obj.bind(py).call_method0("value")?;
            vpy::coerce(&v, &self.sig.return_type.with_nullable(true))
        })
    }

    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// The aggregate behind a Python `AggregateFunction` object.
pub fn uda_of(obj: &Bound<'_, PyAny>) -> Result<Arc<UdaFunction>> {
    let u = obj.call_method0("_core_udf")?;
    let u = u.cast::<crate::bindings::func::PyUda>().map_err(|_| Error::internal("_core_udf() did not return a _core.Uda"))?;
    Ok(u.get().f.clone())
}

/// Resolves a stored UDA reference (`{"uda": AggregateFunction.as_dict()}`) in the current code.
pub fn resolve(j: &Json) -> Result<AggRef> {
    let descriptor = j.get("uda").ok_or_else(|| Error::internal(format!("malformed UDA reference: {j}")))?;
    Python::attach(|py| {
        let d = vpy::json_to_py(py, descriptor)?;
        let f = py.import("pixeltable.func")?.call_method1("_resolve_function", (d,))?;
        if f.hasattr("error_msg")? {
            let path = f.getattr("self_path")?.extract::<Option<String>>()?.unwrap_or_default();
            let msg: String = f.getattr("error_msg")?.extract()?;
            crate::bail!(InvalidState, "The UDA `{path}` cannot be located, because\n{msg}");
        }
        let u: AggRef = uda_of(&f)?;
        Ok(u)
    })
}
