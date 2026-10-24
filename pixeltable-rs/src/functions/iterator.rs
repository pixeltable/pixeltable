//! Iterator calls (`@pxt.iterator`): the API component views are built on.
//!
//! `pixeltable.func.GeneratingFunction` introspects the iterator (signature, output schema, hooks) and creates an
//! `IteratorDef` through `_core.Iterator`. Calling it (`make_call()`) binds and type-checks the arguments, runs the
//! iterator's `validate()` on the constant arguments, and fixes the outputs: `pos` (the position of the output row
//! within its input row, not stored separately) followed by the iterator's output fields, with folded names.
//!
//! A component view holds one `IteratorCall`; for each base row it evaluates the call's arguments against that row
//! and runs the iterator: `eval_rows()` returns the output rows (values of `outputs[1..]`, in order; the position
//! of a row in the result is its `pos`). Outputs with `is_stored == false` (other than `pos`) are not stored: on
//! read, `seek_row()` repositions a fresh iterator at a `pos` (passing the stored outputs of that row) and
//! returns that row's outputs again. `to_json()`/`from_json()` persist the call; `from_json()` re-binds it against
//! the current code and records why it no longer works if the code changed incompatibly (`validation_error`,
//! surfaced through `check_valid()`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Arc, OnceLock};

use super::signature::{BoundArg, Binding, Signature};
use crate::catalog::path::fold_identifier;
use crate::error::{Error, Result, py_repr};
use crate::expr::ExprRef;
use crate::expr::eval::{RowContext, eval};
use crate::types::ColumnType;
use crate::value::py as vpy;
use crate::value::{Json, Value};

pub const POS_COLUMN_NAME: &str = "pos";

/// An iterator as the backend sees it.
pub struct IteratorDef {
    pub fqn: String,
    /// None for an iterator that no longer resolves
    pub signature: Option<Signature>,
    /// folded names of outputs that are not stored (recomputed with `seek()`)
    pub unstored_cols: Vec<String>,
    pub has_seek: bool,
    /// the `pixeltable.func.GeneratingFunction`
    pub py_obj: Py<PyAny>,
    /// for an iterator that no longer resolves: why
    pub error_msg: Option<String>,
    descriptor: OnceLock<Json>,
}

impl std::fmt::Debug for IteratorDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IteratorDef({})", self.fqn)
    }
}

impl IteratorDef {
    pub fn new(
        fqn: String,
        signature: Option<Signature>,
        unstored_cols: Vec<String>,
        has_seek: bool,
        py_obj: Py<PyAny>,
        error_msg: Option<String>,
    ) -> IteratorDef {
        IteratorDef { fqn, signature, unstored_cols, has_seek, py_obj, error_msg, descriptor: OnceLock::new() }
    }

    /// The unqualified name.
    pub fn name(&self) -> &str {
        self.fqn.rsplit('.').next().unwrap_or(&self.fqn)
    }

    /// `GeneratingFunction.as_dict()`, computed once.
    pub fn descriptor(&self) -> Result<Json> {
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

/// One output column of an iterator call.
#[derive(Debug, Clone, PartialEq)]
pub struct IteratorOutput {
    /// folded name (the view column's name)
    pub name: String,
    /// the field name the iterator yields
    pub orig_name: String,
    pub is_stored: bool,
    pub col_type: ColumnType,
}

impl IteratorOutput {
    pub fn is_pos(&self) -> bool {
        self.orig_name == POS_COLUMN_NAME
    }
}

/// A bound call of an iterator: what a component view is defined by.
#[derive(Debug)]
pub struct IteratorCall {
    pub def: Arc<IteratorDef>,
    pub args: Vec<ExprRef>,
    pub kwargs: Vec<(String, ExprRef)>,
    pub binding: Binding,
    /// `outputs[0]` is `pos`
    pub outputs: Vec<IteratorOutput>,
    /// for a stored call that no longer matches the code: why (upstream `catalog_error_msg()`)
    pub validation_error: Option<String>,
}

/// Upstream `_make_outputs()`: `pos`, then the output schema with folded names.
pub fn make_outputs(schema: &[(String, ColumnType)], unstored_cols: &[String]) -> Result<Vec<IteratorOutput>> {
    let mut outputs = vec![IteratorOutput {
        name: POS_COLUMN_NAME.to_string(),
        orig_name: POS_COLUMN_NAME.to_string(),
        is_stored: false,
        col_type: ColumnType::int(false),
    }];
    for (orig, t) in schema {
        let folded = fold_identifier(orig);
        if outputs[1..].iter().any(|o| o.name == folded) {
            let spellings: Vec<String> =
                schema.iter().filter(|(n, _)| fold_identifier(n) == folded).map(|(n, _)| py_repr(n)).collect();
            crate::bail!(InvalidSchema, "Column names are case-insensitive, but {} were specified", spellings.join(", "));
        }
        if folded == POS_COLUMN_NAME {
            crate::bail!(
                InvalidConfiguration,
                "{} is reserved and cannot be the name of an iterator output.",
                py_repr(POS_COLUMN_NAME)
            );
        }
        let is_stored = !unstored_cols.iter().any(|u| fold_identifier(u) == folded);
        outputs.push(IteratorOutput { name: folded, orig_name: orig.clone(), is_stored, col_type: t.clone() });
    }
    Ok(outputs)
}

fn kwarg_names(kwargs: &[(String, ExprRef)]) -> Vec<String> {
    kwargs.iter().map(|(k, _)| k.clone()).collect()
}

/// The bound arguments as a dict of parameter name -> expression (for the Python hooks).
fn bound_exprs<'py>(
    py: Python<'py>,
    sig: &Signature,
    binding: &Binding,
    args: &[ExprRef],
    kwargs: &[(String, ExprRef)],
) -> Result<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (pi, b) in &binding.bound {
        let e = match b {
            BoundArg::Arg(i) => args[*i].clone(),
            BoundArg::Kwarg(i) => kwargs[*i].1.clone(),
            BoundArg::VarArgs(idxs) => crate::expr::build::inline_list(idxs.iter().map(|i| args[*i].clone()).collect()),
            BoundArg::VarKwargs(idxs) => crate::expr::build::inline_dict(
                idxs.iter().map(|i| (kwargs[*i].0.clone(), kwargs[*i].1.clone())).collect(),
            ),
        };
        d.set_item(&sig.params[*pi].name, crate::bindings::expr::PyExpr::new(e))?;
    }
    Ok(d)
}

/// An output schema returned by Python (a dict of name -> ColumnType).
fn schema_from_py(obj: &Bound<'_, PyAny>) -> Result<Vec<(String, ColumnType)>> {
    let d = obj.cast::<PyDict>().map_err(|_| Error::internal("an output schema must be a dict"))?;
    d.iter()
        .map(|(k, v)| {
            let t = crate::bindings::types::core_type(&v)
                .ok_or_else(|| Error::internal("an output schema maps names to ColumnTypes"))?;
            Ok((k.extract::<String>()?, t))
        })
        .collect()
}

fn bind(def: &IteratorDef, sig: &Signature, args: &[ExprRef], kwargs: &[(String, ExprRef)]) -> Result<Binding> {
    let binding = sig
        .bind(args.len(), &kwarg_names(kwargs))
        .map_err(|msg| crate::pxt_err!(UnsupportedOperation, "Invalid iterator arguments: {msg}"))?;
    sig.validate_args(&binding, args, kwargs, &format!("in iterator `{}`", def.fqn))?;
    Ok(binding)
}

/// `iterator(*args, **kwargs)`
pub fn make_call(def: &Arc<IteratorDef>, args: Vec<ExprRef>, kwargs: Vec<(String, ExprRef)>) -> Result<IteratorCall> {
    let Some(sig) = &def.signature else {
        crate::bail!(
            InvalidConfiguration,
            "The iterator `{}` cannot be used, because\n{}",
            def.fqn,
            def.error_msg.clone().unwrap_or_default()
        );
    };
    let binding = bind(def, sig, &args, &kwargs)?;
    let schema = Python::attach(|py| -> Result<Vec<(String, ColumnType)>> {
        let bound = bound_exprs(py, sig, &binding, &args, &kwargs)?;
        schema_from_py(&def.py_obj.bind(py).call_method1("_check_call", (bound,))?)
    })?;
    let outputs = make_outputs(&schema, &def.unstored_cols)?;
    Ok(IteratorCall { def: def.clone(), args, kwargs, binding, outputs, validation_error: None })
}

impl IteratorCall {
    /// The output columns other than `pos`, in order (the values `eval_rows()` returns).
    pub fn value_outputs(&self) -> &[IteratorOutput] {
        &self.outputs[1..]
    }

    /// `iterator_name(arg1, kw=arg2, ...)`, keyword arguments in name order (upstream `display_str()`).
    pub fn display(&self) -> String {
        use crate::expr::display::display;
        let mut parts: Vec<String> = self.args.iter().map(|a| display(a)).collect();
        let mut kw: Vec<&(String, ExprRef)> = self.kwargs.iter().collect();
        kw.sort_by(|a, b| a.0.cmp(&b.0));
        parts.extend(kw.into_iter().map(|(k, v)| format!("{k}={}", display(v))));
        format!("{}({})", self.def.name(), parts.join(", "))
    }

    /// The expressions the call evaluates against a base row.
    pub fn arg_exprs(&self) -> impl Iterator<Item = &ExprRef> {
        self.args.iter().chain(self.kwargs.iter().map(|(_, e)| e))
    }

    /// Fails with upstream's INVALID_STATE error if the call no longer matches the code; `tbl_display` is e.g.
    /// "View 'v'", `op` e.g. "updated in".
    pub fn check_valid(&self, tbl_display: &str, tbl_name: &str, op: &str) -> Result<()> {
        if let Some(msg) = &self.validation_error {
            crate::bail!(
                InvalidState,
                "Data cannot be {op} the {tbl_display},\nbecause the iterator defined on {} is currently invalid:\n{msg}",
                py_repr(tbl_name)
            );
        }
        Ok(())
    }

    /// The argument values for one base row, as Python keyword arguments.
    fn py_kwargs<'py>(&self, py: Python<'py>, ctx: &dyn RowContext) -> Result<Bound<'py, PyDict>> {
        let sig = self.def.signature.as_ref().ok_or_else(|| Error::internal("call of an invalid iterator"))?;
        let d = PyDict::new(py);
        let val = |e: &ExprRef| -> Result<Bound<'py, PyAny>> { Ok(vpy::to_py(py, &eval(e, ctx)?, &e.col_type)?) };
        for (pi, b) in &self.binding.bound {
            let name = &sig.params[*pi].name;
            match b {
                BoundArg::Arg(i) => d.set_item(name, val(&self.args[*i])?)?,
                BoundArg::Kwarg(i) => d.set_item(name, val(&self.kwargs[*i].1)?)?,
                BoundArg::VarArgs(idxs) => {
                    let items = idxs.iter().map(|i| val(&self.args[*i])).collect::<Result<Vec<_>>>()?;
                    d.set_item(name, PyList::new(py, items)?)?;
                }
                BoundArg::VarKwargs(idxs) => {
                    for i in idxs {
                        d.set_item(&self.kwargs[*i].0, val(&self.kwargs[*i].1)?)?;
                    }
                }
            }
        }
        Ok(d)
    }

    fn convert_item(&self, item: &Bound<'_, PyAny>) -> Result<Vec<Value>> {
        let d = item.cast::<PyDict>().map_err(|_| {
            crate::pxt_err!(
                InvalidDataFormat,
                "Iterator `{}` must yield dicts, but yielded a value of type `{}`",
                self.def.fqn,
                item.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            )
        })?;
        self.value_outputs()
            .iter()
            .map(|o| match d.get_item(&o.orig_name)? {
                Some(v) => vpy::coerce(&v, &o.col_type.with_nullable(true)),
                None => Ok(Value::Null),
            })
            .collect()
    }

    fn start<'py>(&self, py: Python<'py>, ctx: &dyn RowContext) -> Result<Bound<'py, PyAny>> {
        if let Some(msg) = &self.validation_error {
            crate::bail!(InvalidState, "{msg}");
        }
        let kwargs = self.py_kwargs(py, ctx)?;
        Ok(self.def.py_obj.bind(py).call_method1("_start", (kwargs,))?)
    }

    /// Runs the iterator for the base row `ctx`: the values of `value_outputs()` for each output row, in `pos`
    /// order.
    pub fn eval_rows(&self, ctx: &dyn RowContext) -> Result<Vec<Vec<Value>>> {
        Python::attach(|py| {
            let it = self.start(py, ctx)?;
            let mut rows = Vec::new();
            let result = (|| -> Result<()> {
                for item in it.try_iter()? {
                    rows.push(self.convert_item(&item?)?);
                }
                Ok(())
            })();
            close(&it);
            result.map(|_| rows)
        })
    }

    /// The outputs of the row at `pos` for the base row `ctx` (for outputs that are not stored): a fresh iterator
    /// is positioned with `seek(pos, **stored)` (`stored`: the stored output values of that row, by folded name).
    pub fn seek_row(&self, ctx: &dyn RowContext, pos: i64, stored: &[(String, Value)]) -> Result<Vec<Value>> {
        Python::attach(|py| {
            let it = self.start(py, ctx)?;
            let result = (|| -> Result<Vec<Value>> {
                let kwargs = PyDict::new(py);
                for (name, v) in stored {
                    if let Some(o) = self.value_outputs().iter().find(|o| o.name == *name) {
                        kwargs.set_item(&o.orig_name, vpy::to_py(py, v, &o.col_type)?)?;
                    }
                }
                it.call_method("seek", (pos,), Some(&kwargs))?;
                let item = it.call_method0("__next__")?;
                self.convert_item(&item)
            })();
            close(&it);
            result
        })
    }

    pub fn to_json(&self) -> Result<Json> {
        let mut kwargs = serde_json::Map::new();
        for (k, v) in &self.kwargs {
            kwargs.insert(k.clone(), crate::expr::serde::to_json(v)?);
        }
        let mut outputs = serde_json::Map::new();
        for o in &self.outputs {
            outputs.insert(
                o.name.clone(),
                serde_json::json!({ "orig_name": o.orig_name, "is_stored": o.is_stored, "col_type": o.col_type.to_dict() }),
            );
        }
        Ok(serde_json::json!({
            "fn": self.def.descriptor()?,
            "args": self.args.iter().map(|a| crate::expr::serde::to_json(a)).collect::<Result<Vec<_>>>()?,
            "kwargs": Json::Object(kwargs),
            "outputs": Json::Object(outputs),
        }))
    }

    /// Deserializes a stored call (upstream `GeneratingFunctionCall.from_dict()`): the iterator is resolved in the
    /// current code and the stored call re-bound to it; a stored output whose type the current code no longer
    /// produces makes the call invalid.
    pub fn from_json(d: &Json) -> Result<IteratorCall> {
        let bad = || Error::internal(format!("malformed iterator call: {d}"));
        let def = resolve_descriptor(d.get("fn").ok_or_else(bad)?)?;
        let args: Vec<ExprRef> = d
            .get("args")
            .and_then(Json::as_array)
            .ok_or_else(bad)?
            .iter()
            .map(crate::expr::serde::from_json)
            .collect::<Result<_>>()?;
        let kwargs: Vec<(String, ExprRef)> = d
            .get("kwargs")
            .and_then(Json::as_object)
            .ok_or_else(bad)?
            .iter()
            .map(|(k, v)| Ok((k.clone(), crate::expr::serde::from_json(v)?)))
            .collect::<Result<_>>()?;
        let mut outputs = Vec::new();
        for (name, o) in d.get("outputs").and_then(Json::as_object).ok_or_else(bad)? {
            outputs.push(IteratorOutput {
                name: name.clone(),
                orig_name: o.get("orig_name").and_then(Json::as_str).ok_or_else(bad)?.to_string(),
                is_stored: o.get("is_stored").and_then(Json::as_bool).unwrap_or(true),
                col_type: ColumnType::from_dict(o.get("col_type").ok_or_else(bad)?)?,
            });
        }
        let mut call = IteratorCall { def: def.clone(), args, kwargs, binding: Binding::default(), outputs, validation_error: None };
        let Some(sig) = &def.signature else {
            call.validation_error = Some(format!(
                "The iterator `{}` cannot be located, because\n{}",
                def.fqn,
                def.error_msg.clone().unwrap_or_default()
            ));
            return Ok(call);
        };
        match bind(&def, sig, &call.args, &call.kwargs) {
            Ok(b) => call.binding = b,
            Err(_) => {
                let mut parts: Vec<String> = call.args.iter().map(|a| format!("pxt.{}", a.col_type.repr())).collect();
                parts.extend(call.kwargs.iter().map(|(k, a)| format!("{k}: pxt.{}", a.col_type.repr())));
                let call_sig = format!("({}) -> ...", parts.join(", "));
                let code_sig = format!("({}) -> ...", sig.params_str());
                let fqn = &def.fqn;
                call.validation_error = Some(format!(
                    "The signature stored in the database for a call to `{fqn}` no longer\n\
                     matches its signature as currently defined in the code. This probably means that the\n\
                     code for `{fqn}` has changed in a backward-incompatible way.\n\
                     Signature of iterator in the database: {call_sig}\n\
                     Signature of iterator as currently defined in code: {code_sig}"
                ));
                return Ok(call);
            }
        }
        let schema = Python::attach(|py| -> Result<Vec<(String, ColumnType)>> {
            let bound = bound_exprs(py, sig, &call.binding, &call.args, &call.kwargs)?;
            schema_from_py(&def.py_obj.bind(py).call_method1("call_output_schema", (bound,))?)
        })?;
        let mut detail: Option<String> = None;
        for o in call.outputs.iter().filter(|o| !o.is_pos()) {
            match schema.iter().find(|(n, _)| *n == o.orig_name) {
                None => {
                    detail = Some(format!(
                        "The output field {} is no longer present in the output schema.",
                        py_repr(&o.orig_name)
                    ))
                }
                Some((_, t)) if !o.col_type.is_supertype_of(t, false, true) => {
                    detail = Some(format!(
                        "The type of output field {} is incompatible\n(expected `pxt.{}`; got `pxt.{}`).",
                        py_repr(&o.orig_name),
                        o.col_type.repr(),
                        t.repr()
                    ))
                }
                _ => {}
            }
        }
        if let Some(detail) = detail {
            let fqn = &def.fqn;
            call.validation_error = Some(format!(
                "The output schema stored in the database for a call to `{fqn}` no longer\n\
                 matches its output schema as currently defined in the code. This probably means that the\n\
                 code for `{fqn}` has changed in a backward-incompatible way.\n{detail}"
            ));
        }
        Ok(call)
    }
}

fn close(it: &Bound<'_, PyAny>) {
    if let Ok(c) = it.getattr("close") {
        let _ = c.call0();
    }
}

/// The `IteratorDef` behind a Python `GeneratingFunction`.
pub fn def_of(obj: &Bound<'_, PyAny>) -> Result<Arc<IteratorDef>> {
    let it = obj.call_method0("_core_iterator")?;
    let it = it
        .cast::<crate::bindings::func::PyIterator>()
        .map_err(|_| Error::internal("_core_iterator() did not return a _core.Iterator"))?;
    Ok(it.get().def.clone())
}

/// Resolves a serialized iterator (`GeneratingFunction.as_dict()`) in the current code.
pub fn resolve_descriptor(descriptor: &Json) -> Result<Arc<IteratorDef>> {
    Python::attach(|py| {
        let d = vpy::json_to_py(py, descriptor)?;
        let f = py.import("pixeltable.func._iterator")?.call_method1("_resolve_iterator", (d,))?;
        def_of(&f)
    })
}
