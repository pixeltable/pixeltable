//! Calls of Python functions: UDFs (`@pxt.udf`), expression templates (`@pxt.expr_udf`, `Function.using()`,
//! `pxt.udf(table)`).
//!
//! `pixeltable.func` introspects a Python function and creates its `UdfDef` (through `_core.Udf`, see
//! `bindings::func`): the signatures, the callables or template expressions, and the Python `Function` object
//! (`py_obj`), which serializes the function (`as_dict()`) and computes conditional return types. Everything else
//! is here: binding a call to a signature (overload resolution, type checks, the result type) builds a
//! `FunctionCall` node whose `FuncRef::Udf` holds a `UdfCall`; `eval_call()` evaluates it, calling into Python
//! for the function body; `call_from_json()` resolves a stored call against the current code and records why it
//! no longer works if the code changed incompatibly (`UdfCall::validation_error`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::{Arc, Mutex, OnceLock};

use super::FuncRef;
use super::signature::{BoundArg, Binding, Signature};
use crate::error::{Error, Result, py_repr};
use crate::expr::eval::{RowContext, eval};
use crate::expr::{Expr, ExprKind, ExprRef, FunctionCall};
use crate::types::ColumnType;
use crate::value::py as vpy;
use crate::value::{Json, Value};

pub enum UdfImpl {
    /// Python callables, one per signature; `is_async`: per callable, whether it is a coroutine function
    Callable { py_fns: Vec<Py<PyAny>>, batch_size: Option<usize>, is_async: Vec<bool>, has_resource_pool: bool },
    /// template expressions, one per signature; parameters appear as `Variable` nodes
    Template { exprs: Vec<ExprRef> },
    /// a stored function that no longer resolves (`pixeltable.func.InvalidFunction`)
    Invalid { error_msg: String },
}

/// A Python function as the backend sees it.
pub struct UdfDef {
    /// `Function.name` (the default output column name)
    pub name: Option<String>,
    /// how calls are displayed
    pub display_name: String,
    pub self_path: Option<String>,
    pub signatures: Vec<Signature>,
    pub imp: UdfImpl,
    /// the `pixeltable.func.Function` object
    pub py_obj: Py<PyAny>,
    pub is_method: bool,
    pub is_property: bool,
    /// can be stored in a computed column (serialized by reference, or by value without pickled code)
    pub storable: bool,
    descriptor: OnceLock<Json>,
}

impl std::fmt::Debug for UdfDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UdfDef({})", self.display_name)
    }
}

impl UdfDef {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: Option<String>,
        display_name: String,
        self_path: Option<String>,
        signatures: Vec<Signature>,
        imp: UdfImpl,
        py_obj: Py<PyAny>,
        is_method: bool,
        is_property: bool,
        storable: bool,
    ) -> UdfDef {
        UdfDef {
            name,
            display_name,
            self_path,
            signatures,
            imp,
            py_obj,
            is_method,
            is_property,
            storable,
            descriptor: OnceLock::new(),
        }
    }

    pub fn is_polymorphic(&self) -> bool {
        self.signatures.len() > 1
    }

    /// The path used in messages about stored calls.
    fn path_str(&self) -> String {
        self.self_path.clone().unwrap_or_else(|| self.display_name.clone())
    }

    /// The serialized function (`Function.as_dict()`), computed once.
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

/// A bound call of a `UdfDef` (the `FuncRef` of a `FunctionCall` node).
pub struct UdfCall {
    pub def: Arc<UdfDef>,
    /// the selected signature
    pub sig_idx: usize,
    pub binding: Binding,
    /// displayed as `arg0.fn(...)`
    pub is_method_call: bool,
    /// for templates: the template with the arguments substituted (what is evaluated)
    pub instantiated: Option<ExprRef>,
    /// for a stored call that no longer matches the code: why (upstream `catalog_error_msg()`)
    pub validation_error: Option<String>,
}

impl std::fmt::Debug for UdfCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UdfCall({}, sig {})", self.def.display_name, self.sig_idx)
    }
}

impl UdfCall {
    pub fn signature(&self) -> Option<&Signature> {
        self.def.signatures.get(self.sig_idx)
    }

    pub fn to_json(&self) -> Result<Json> {
        Ok(serde_json::json!({ "udf": self.def.descriptor()?, "is_method_call": self.is_method_call }))
    }
}

// ------------------------------------------------------------------------------------------------ binding

fn arg_desc(e: &Expr) -> String {
    match e.as_literal() {
        Some(Value::Null) => "None".to_string(),
        _ => e.col_type.repr(),
    }
}

fn provided_str(args: &[ExprRef], kwargs: &[(String, ExprRef)]) -> String {
    let mut parts: Vec<String> = args.iter().map(|a| arg_desc(a)).collect();
    parts.extend(kwargs.iter().map(|(k, v)| format!("{k}={}", arg_desc(v))));
    parts.join(", ")
}

fn kwarg_names(kwargs: &[(String, ExprRef)]) -> Vec<String> {
    kwargs.iter().map(|(k, _)| k.clone()).collect()
}

fn bound_expr<'a>(b: &BoundArg, args: &'a [ExprRef], kwargs: &'a [(String, ExprRef)]) -> Option<&'a ExprRef> {
    match b {
        BoundArg::Arg(i) => args.get(*i),
        BoundArg::Kwarg(i) => kwargs.get(*i).map(|(_, e)| e),
        _ => None,
    }
}

/// Upstream `Function.validate_call()` for the signature `sig_idx`.
fn validate_call(
    def: &UdfDef,
    sig_idx: usize,
    binding: &Binding,
    args: &[ExprRef],
    kwargs: &[(String, ExprRef)],
) -> Result<()> {
    let sig = &def.signatures[sig_idx];
    let context = format!("in function {}", py_repr(def.name.as_deref().unwrap_or(&def.display_name)));
    sig.validate_args(binding, args, kwargs, &context)?;
    if let UdfImpl::Callable { batch_size: Some(_), .. } = &def.imp {
        for (pi, b) in &binding.bound {
            let p = &sig.params[*pi];
            if p.is_batched || p.kind.is_var() {
                continue;
            }
            if let Some(e) = bound_expr(b, args, kwargs) {
                if !matches!(e.kind, ExprKind::Literal(_) | ExprKind::Variable(_)) {
                    crate::bail!(
                        InvalidArgument,
                        "{}(): parameter {} must be a constant value",
                        def.display_name,
                        p.name
                    );
                }
            }
        }
    }
    Ok(())
}

fn bind_to(def: &UdfDef, sig_idx: usize, args: &[ExprRef], kwargs: &[(String, ExprRef)]) -> Result<Binding> {
    let sig = &def.signatures[sig_idx];
    let binding = sig.bind(args.len(), &kwarg_names(kwargs)).map_err(|msg| {
        crate::pxt_err!(
            InvalidArgument,
            "{msg}; expected ({}), got ({})",
            sig.params_str(),
            provided_str(args, kwargs)
        )
    })?;
    validate_call(def, sig_idx, &binding, args, kwargs)?;
    Ok(binding)
}

/// Upstream `Function._bind_to_matching_signature()`: the first signature (in declaration order) the call binds
/// to; with a single signature, its binding error is raised as is.
pub fn resolve(def: &UdfDef, args: &[ExprRef], kwargs: &[(String, ExprRef)]) -> Result<(usize, Binding)> {
    if let UdfImpl::Invalid { error_msg } = &def.imp {
        crate::bail!(
            InvalidConfiguration,
            "The UDF `{}` cannot be used, because\n{error_msg}",
            def.path_str()
        );
    }
    if def.signatures.len() == 1 {
        return Ok((0, bind_to(def, 0, args, kwargs)?));
    }
    for i in 0..def.signatures.len() {
        if let Ok(b) = bind_to(def, i, args, kwargs) {
            return Ok((i, b));
        }
    }
    crate::bail!(
        UnsupportedOperation,
        "Function {} has no matching signature for arguments",
        py_repr(def.name.as_deref().unwrap_or(&def.display_name))
    )
}

/// The conditional return type of the call (`@fn.conditional_return_type`), if the function has one and its
/// arguments are constants.
fn conditional_return_type(
    def: &UdfDef,
    sig_idx: usize,
    binding: &Binding,
    args: &[ExprRef],
    kwargs: &[(String, ExprRef)],
) -> Result<Option<ColumnType>> {
    Python::attach(|py| {
        let obj = def.py_obj.bind(py);
        match obj.getattr("_conditional_return_type") {
            Ok(crt) if !crt.is_none() => {}
            _ => return Ok(None),
        }
        let sig = &def.signatures[sig_idx];
        let bound = PyDict::new(py);
        for (pi, b) in &binding.bound {
            if let Some(e) = bound_expr(b, args, kwargs) {
                bound.set_item(&sig.params[*pi].name, crate::bindings::expr::PyExpr::new(e.clone()))?;
            }
        }
        let t = obj.call_method1("_call_conditional_return_type", (sig_idx, bound))?;
        if t.is_none() {
            return Ok(None);
        }
        Ok(Some(
            crate::bindings::types::core_type(&t)
                .ok_or_else(|| Error::internal("a conditional return type must be a ColumnType"))?,
        ))
    })
}

/// Substitutes `f(node)` for every node for which it returns a replacement. Function calls whose arguments change
/// are re-bound (their result type may change).
pub fn substitute(e: &ExprRef, f: &dyn Fn(&Expr) -> Option<ExprRef>) -> Result<ExprRef> {
    if let Some(r) = f(e) {
        return Ok(r);
    }
    let children = e.kind.children();
    if children.is_empty() {
        return Ok(e.clone());
    }
    let new_children: Vec<ExprRef> = children.iter().map(|c| substitute(c, f)).collect::<Result<_>>()?;
    if new_children.iter().zip(children.iter()).all(|(a, b)| Arc::ptr_eq(a, b)) {
        return Ok(e.clone());
    }
    if let ExprKind::FunctionCall(fc) = &e.kind {
        let n = fc.args.len();
        let mut it = new_children.into_iter();
        let args: Vec<ExprRef> = (&mut it).take(n).collect();
        let kwargs: Vec<(String, ExprRef)> = fc.kwargs.iter().map(|(k, _)| (k.clone(), it.next().unwrap())).collect();
        return match &fc.func {
            FuncRef::Udf(call) => remake_call(call, args, kwargs),
            FuncRef::Builtin(b) => super::make_call(&b.qualified_name(), args, kwargs),
        };
    }
    Ok(Expr::new(e.kind.with_children(new_children), e.col_type.clone()))
}

/// The template of signature `sig_idx` with the call's arguments (and defaults of unbound parameters) substituted.
fn instantiate(
    def: &UdfDef,
    sig_idx: usize,
    binding: &Binding,
    args: &[ExprRef],
    kwargs: &[(String, ExprRef)],
) -> Result<ExprRef> {
    let UdfImpl::Template { exprs } = &def.imp else {
        return Err(Error::internal("instantiate() of a non-template function"));
    };
    let sig = &def.signatures[sig_idx];
    let mut values: Vec<(String, ExprRef)> = Vec::new();
    for (pi, b) in &binding.bound {
        let name = sig.params[*pi].name.clone();
        let e = match b {
            BoundArg::Arg(i) => args[*i].clone(),
            BoundArg::Kwarg(i) => kwargs[*i].1.clone(),
            BoundArg::VarArgs(idxs) => crate::expr::build::inline_list(idxs.iter().map(|i| args[*i].clone()).collect()),
            BoundArg::VarKwargs(idxs) => crate::expr::build::inline_dict(
                idxs.iter().map(|i| (kwargs[*i].0.clone(), kwargs[*i].1.clone())).collect(),
            ),
        };
        values.push((name, e));
    }
    for p in &sig.params {
        if !values.iter().any(|(n, _)| *n == p.name) {
            if let Some(d) = &p.default {
                values.push((p.name.clone(), d.clone()));
            }
        }
    }
    substitute(&exprs[sig_idx], &|e: &Expr| match &e.kind {
        ExprKind::Variable(name) => values.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone()),
        _ => None,
    })
}

/// Builds the call node for a resolved signature.
fn finish_call(
    def: &Arc<UdfDef>,
    sig_idx: usize,
    binding: Binding,
    args: Vec<ExprRef>,
    kwargs: Vec<(String, ExprRef)>,
    is_method_call: bool,
) -> Result<ExprRef> {
    let (return_type, instantiated) = match &def.imp {
        UdfImpl::Template { .. } => {
            let inst = instantiate(def, sig_idx, &binding, &args, &kwargs)?;
            (inst.col_type.clone(), Some(inst))
        }
        UdfImpl::Callable { .. } => {
            let sig = &def.signatures[sig_idx];
            let rt = conditional_return_type(def, sig_idx, &binding, &args, &kwargs)?
                .unwrap_or_else(|| sig.return_type.clone());
            (sig.call_return_type(&rt, &binding, &args, &kwargs), None)
        }
        UdfImpl::Invalid { .. } => return Err(Error::internal("call of an invalid function")),
    };
    let call = UdfCall { def: def.clone(), sig_idx, binding, is_method_call, instantiated, validation_error: None };
    Ok(Expr::new(ExprKind::FunctionCall(FunctionCall { func: FuncRef::Udf(Arc::new(call)), args, kwargs }), return_type))
}

/// `fn(*args, **kwargs)` (`is_method_call`: `args[0].fn(*args[1:], **kwargs)`).
pub fn make_call(
    def: &Arc<UdfDef>,
    args: Vec<ExprRef>,
    kwargs: Vec<(String, ExprRef)>,
    is_method_call: bool,
) -> Result<ExprRef> {
    let (sig_idx, binding) = resolve(def, &args, &kwargs)?;
    finish_call(def, sig_idx, binding, args, kwargs, is_method_call)
}

/// The same call with new arguments (same signature).
pub fn remake_call(call: &UdfCall, args: Vec<ExprRef>, kwargs: Vec<(String, ExprRef)>) -> Result<ExprRef> {
    let sig = call.signature().ok_or_else(|| Error::internal("remake_call() of an unresolved call"))?;
    let binding = sig
        .bind(args.len(), &kwarg_names(&kwargs))
        .map_err(|msg| crate::pxt_err!(InvalidArgument, "{msg}; expected ({}), got ({})", sig.params_str(), provided_str(&args, &kwargs)))?;
    finish_call(&call.def, call.sig_idx, binding, args, kwargs, call.is_method_call)
}

/// Upstream `Function.using()`: for each signature that accepts the constant bindings, a template call of the
/// function in which the bound parameters are those constants and the others are variables (positional while
/// possible); returns (signature index, template expression, names of the bound parameters).
pub fn bind_template(def: &Arc<UdfDef>, bindings: &[(String, ExprRef)]) -> Result<Vec<(usize, ExprRef)>> {
    let one = |sig_idx: usize| -> Result<ExprRef> {
        let sig = &def.signatures[sig_idx];
        for (k, v) in bindings {
            let Some((_, p)) = sig.param(k) else {
                crate::bail!(UnsupportedOperation, "Unknown parameter: {k}");
            };
            if v.as_literal().is_none() {
                crate::bail!(UnsupportedOperation, "Expected a constant value for parameter {} in call to .using()", py_repr(k));
            }
            let pt = p.col_type.clone().unwrap_or_else(|| ColumnType::json(true));
            if !pt.is_supertype_of(&v.col_type, false, true) {
                crate::bail!(
                    TypeMismatch,
                    "Expected type `{}` for parameter {}; got `{}`",
                    pt.repr(),
                    py_repr(k),
                    v.col_type.repr()
                );
            }
        }
        let mut t_args: Vec<ExprRef> = Vec::new();
        let mut t_kwargs: Vec<(String, ExprRef)> = Vec::new();
        let mut args_ok = true;
        for p in &sig.params {
            if let Some((_, v)) = bindings.iter().find(|(k, _)| *k == p.name) {
                t_kwargs.push((p.name.clone(), v.clone()));
                args_ok = false;
            } else {
                let var = Expr::new(
                    ExprKind::Variable(p.name.clone()),
                    p.col_type.clone().unwrap_or_else(|| ColumnType::json(true)),
                );
                use super::signature::ParamKind::*;
                if args_ok && matches!(p.kind, PositionalOnly | PositionalOrKeyword) {
                    t_args.push(var);
                } else {
                    t_kwargs.push((p.name.clone(), var));
                    args_ok = false;
                }
            }
        }
        let binding = bind_to(def, sig_idx, &t_args, &t_kwargs)?;
        finish_call(def, sig_idx, binding, t_args, t_kwargs, false)
    };
    if def.signatures.len() == 1 {
        return Ok(vec![(0, one(0)?)]);
    }
    let out: Vec<(usize, ExprRef)> = (0..def.signatures.len()).filter_map(|i| one(i).ok().map(|e| (i, e))).collect();
    if out.is_empty() {
        crate::bail!(
            UnsupportedOperation,
            "Function {} has no matching signature for arguments",
            py_repr(def.name.as_deref().unwrap_or(&def.display_name))
        );
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------ evaluation

fn is_batched_arg(sig: &Signature, binding: &Binding, b: &BoundArg) -> bool {
    binding.bound.iter().any(|(pi, x)| x == b && sig.params[*pi].is_batched)
}

/// Evaluates a call of a Python function against a row.
pub fn eval_call(call: &UdfCall, fc: &FunctionCall, e: &Expr, ctx: &dyn RowContext) -> Result<Value> {
    if let Some(msg) = &call.validation_error {
        crate::bail!(InvalidState, "{msg}");
    }
    match &call.def.imp {
        UdfImpl::Template { .. } => {
            let inst = call.instantiated.as_ref().ok_or_else(|| Error::internal("template call without an instance"))?;
            eval(inst, ctx)
        }
        UdfImpl::Invalid { error_msg } => {
            crate::bail!(InvalidState, "The UDF `{}` cannot be located, because\n{error_msg}", call.def.path_str())
        }
        UdfImpl::Callable { py_fns, batch_size, is_async, has_resource_pool } => {
            if *has_resource_pool && !is_async.get(call.sig_idx).copied().unwrap_or(false) {
                crate::bail!(
                    InvalidConfiguration,
                    "{}: resource_pool requires an async function",
                    call.def.display_name
                );
            }
            let sig = call.signature().ok_or_else(|| Error::internal("unresolved call"))?;
            let args: Vec<Value> = fc.args.iter().map(|a| eval(a, ctx)).collect::<Result<_>>()?;
            let kwargs: Vec<Value> = fc.kwargs.iter().map(|(_, a)| eval(a, ctx)).collect::<Result<_>>()?;
            // NULL arguments of non-nullable parameters short-circuit the call
            for (pi, b) in &call.binding.bound {
                let p = &sig.params[*pi];
                if p.col_type.as_ref().is_none_or(|t| t.nullable) {
                    continue;
                }
                let v = match b {
                    BoundArg::Arg(i) => &args[*i],
                    BoundArg::Kwarg(i) => &kwargs[*i],
                    _ => continue,
                };
                if v.is_null() {
                    return Ok(Value::Null);
                }
            }
            let batched = batch_size.is_some();
            Python::attach(|py| -> Result<Value> {
                let py_fn = py_fns[call.sig_idx].bind(py);
                let mut py_args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
                for (i, v) in args.iter().enumerate() {
                    let mut o = vpy::to_py(py, v, &fc.args[i].col_type)?;
                    if batched && is_batched_arg(sig, &call.binding, &BoundArg::Arg(i)) {
                        o = PyList::new(py, [o])?.into_any();
                    }
                    py_args.push(o);
                }
                let py_kwargs = PyDict::new(py);
                for (i, v) in kwargs.iter().enumerate() {
                    let mut o = vpy::to_py(py, v, &fc.kwargs[i].1.col_type)?;
                    if batched && is_batched_arg(sig, &call.binding, &BoundArg::Kwarg(i)) {
                        o = PyList::new(py, [o])?.into_any();
                    }
                    py_kwargs.set_item(&fc.kwargs[i].0, o)?;
                }
                let mut result = py_fn.call(PyTuple::new(py, py_args)?, Some(&py_kwargs))?;
                if is_coroutine(py, &result)? {
                    result = run_coroutine(py, result)?;
                }
                if batched {
                    result = result.get_item(0)?;
                }
                convert_result(&result, &e.col_type)
            })
        }
    }
}

pub fn is_coroutine(py: Python<'_>, obj: &Bound<'_, PyAny>) -> Result<bool> {
    static ISCORO: OnceLock<Py<PyAny>> = OnceLock::new();
    let f = match ISCORO.get() {
        Some(f) => f.bind(py).clone(),
        None => {
            let f = py.import("inspect")?.getattr("iscoroutine")?;
            let _ = ISCORO.set(f.clone().unbind());
            f
        }
    };
    Ok(f.call1((obj,))?.is_truthy()?)
}

/// Runs a coroutine returned by an async UDF to completion.
pub fn run_coroutine<'py>(py: Python<'py>, coro: Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
    Ok(py.import("pixeltable.func._runtime")?.call_method1("run_coroutine", (coro,))?)
}

/// Converts the value a UDF returned to a cell value of the call's type.
pub fn convert_result(obj: &Bound<'_, PyAny>, t: &ColumnType) -> Result<Value> {
    vpy::coerce(obj, &t.with_nullable(true))
}

// ------------------------------------------------------------------------------------------------ display

pub fn display_call(call: &UdfCall, fc: &FunctionCall) -> String {
    use crate::expr::display::display;
    if let (UdfImpl::Template { exprs }, Some(inst)) = (&call.def.imp, &call.instantiated) {
        if matches!(exprs.get(call.sig_idx).map(|e| &e.kind), Some(ExprKind::FunctionCall(_))) {
            return display(inst);
        }
    }
    let skip = usize::from(call.is_method_call && !fc.args.is_empty());
    let mut args: Vec<String> = fc.args.iter().skip(skip).map(|a| display(a)).collect();
    args.extend(fc.kwargs.iter().map(|(k, v)| format!("{k}={}", display(v))));
    let name = if call.def.display_name.is_empty() { "anonymous_fn" } else { &call.def.display_name };
    if skip == 1 {
        format!("{}.{}({})", display(&fc.args[0]), name, args.join(", "))
    } else {
        format!("{}({})", name, args.join(", "))
    }
}

// ------------------------------------------------------------------------------------------------ persistence

pub fn is_udf_json(fn_json: &Json) -> bool {
    fn_json.get("udf").is_some()
}

/// The `UdfDef` behind a Python `Function` object.
pub fn def_of(obj: &Bound<'_, PyAny>) -> Result<Arc<UdfDef>> {
    let udf = obj.call_method0("_core_udf")?;
    let udf = udf.cast::<crate::bindings::func::PyUdf>().map_err(|_| Error::internal("_core_udf() did not return a _core.Udf"))?;
    Ok(udf.get().def.clone())
}

/// Resolves a serialized function (`Function.as_dict()`) to its current definition in code.
pub fn resolve_descriptor(descriptor: &Json) -> Result<Arc<UdfDef>> {
    Python::attach(|py| {
        let d = vpy::json_to_py(py, descriptor)?;
        let f = py.import("pixeltable.func")?.call_method1("_resolve_function", (d,))?;
        def_of(&f)
    })
}

fn signature_error(def: &UdfDef, args: &[ExprRef], kwargs: &[(String, ExprRef)], return_type: &ColumnType) -> String {
    let path = def.path_str();
    let mut parts: Vec<String> = args.iter().map(|a| format!("pxt.{}", a.col_type.repr())).collect();
    parts.extend(kwargs.iter().map(|(k, a)| format!("{k}: pxt.{}", a.col_type.repr())));
    let expr_sig = format!("({}) -> pxt.{}", parts.join(", "), return_type.repr());
    let code_sig = if def.is_polymorphic() {
        format!("{} signatures", def.signatures.len())
    } else {
        def.signatures.first().map(|s| s.display()).unwrap_or_default()
    };
    let note = if def.is_polymorphic() { "any of its signatures" } else { "its signature" };
    format!(
        "The signature stored in the database for a UDF call to `{path}` no longer\n\
         matches {note} as currently defined in the code. This probably means that the\n\
         code for `{path}` has changed in a backward-incompatible way.\n\
         Signature of UDF call in the database: {expr_sig}\n\
         Signature of UDF as currently defined in code: {code_sig}"
    )
}

fn return_type_error(def: &UdfDef, stored: &ColumnType, current: &ColumnType) -> String {
    let path = def.path_str();
    let note = if def.is_polymorphic() { "any of its return types" } else { "its return type" };
    format!(
        "The return type stored in the database for a UDF call to `{path}` no longer\n\
         matches {note} as currently defined in the code. This probably means that the\n\
         code for `{path}` has changed in a backward-incompatible way.\n\
         Return type of UDF call in the database: {}\n\
         Return type of UDF as currently defined in code: {}",
        stored.repr(),
        current.repr()
    )
}

fn invalid_call(
    def: Arc<UdfDef>,
    args: Vec<ExprRef>,
    kwargs: Vec<(String, ExprRef)>,
    is_method_call: bool,
    col_type: ColumnType,
    msg: String,
) -> ExprRef {
    let call = UdfCall {
        def,
        sig_idx: 0,
        binding: Binding::default(),
        is_method_call,
        instantiated: None,
        validation_error: Some(msg),
    };
    Expr::new(ExprKind::FunctionCall(FunctionCall { func: FuncRef::Udf(Arc::new(call)), args, kwargs }), col_type)
}

/// Deserializes a stored call of a Python function (upstream `FunctionCall._from_dict()`): the function is
/// resolved in the current code and the stored call pattern re-bound to its signatures; if that fails, the call
/// keeps its stored type and records a validation error (it can no longer be evaluated).
pub fn call_from_json(
    fn_json: &Json,
    args: Vec<ExprRef>,
    kwargs: Vec<(String, ExprRef)>,
    col_type: ColumnType,
) -> Result<ExprRef> {
    let descriptor = fn_json.get("udf").ok_or_else(|| Error::internal("not a UDF reference"))?;
    let is_method_call = fn_json.get("is_method_call").and_then(Json::as_bool).unwrap_or(false);
    let def = resolve_descriptor(descriptor)?;
    if let UdfImpl::Invalid { error_msg } = &def.imp {
        let msg = format!("The UDF `{}` cannot be located, because\n{error_msg}", def.path_str());
        return Ok(invalid_call(def, args, kwargs, is_method_call, col_type, msg));
    }
    let (sig_idx, binding) = match resolve(&def, &args, &kwargs) {
        Ok(r) => r,
        Err(_) => {
            let msg = signature_error(&def, &args, &kwargs, &col_type);
            return Ok(invalid_call(def, args, kwargs, is_method_call, col_type, msg));
        }
    };
    match finish_call(&def, sig_idx, binding, args.clone(), kwargs.clone(), is_method_call) {
        Ok(e) => {
            if !col_type.is_supertype_of(&e.col_type, true, true) {
                let msg = return_type_error(&def, &col_type, &e.col_type);
                return Ok(invalid_call(def, args, kwargs, is_method_call, col_type, msg));
            }
            // the call keeps its stored type
            Ok(Expr::new(e.kind.clone(), col_type))
        }
        Err(Error::Python(err)) if Python::attach(|py| err.is_instance_of::<pyo3::exceptions::PyImportError>(py)) => {
            let import_msg = Python::attach(|py| err.value(py).str().map(|s| s.to_string()).unwrap_or_default());
            let msg = format!(
                "A UDF call to `{}` could not be fully resolved, because a module required\n\
                 by the UDF could not be imported:\n{import_msg}",
                def.path_str()
            );
            Ok(invalid_call(def, args, kwargs, is_method_call, col_type, msg))
        }
        Err(e) => Err(e),
    }
}

/// The validation errors of the Python function calls in `e` (stored calls that no longer match the code).
pub fn validation_errors(e: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    e.visit(&mut |x| {
        if let ExprKind::FunctionCall(FunctionCall { func: FuncRef::Udf(call), .. }) = &x.kind {
            if let Some(msg) = &call.validation_error {
                out.push(msg.clone());
            }
        }
    });
    out
}

/// Fails if `e` calls a function that cannot be persisted (upstream `Expr.validate_storable()`); `context` names
/// what would store it (e.g. "Computed column 'c'").
pub fn validate_storable(e: &Expr, context: &str) -> Result<()> {
    let mut err: Option<Error> = None;
    e.visit(&mut |x| {
        if err.is_some() {
            return;
        }
        if let ExprKind::FunctionCall(FunctionCall { func: FuncRef::Udf(call), .. }) = &x.kind {
            if let UdfImpl::Invalid { error_msg } = &call.def.imp {
                err = Some(crate::pxt_err!(
                    UnsupportedOperation,
                    "{context} uses a UDF this process cannot resolve: {error_msg}"
                ));
            } else if !call.def.storable {
                err = Some(crate::pxt_err!(
                    UnsupportedOperation,
                    "{context} uses `{}()`, which was created with `.apply()` or defined as a local function. \
                     Define it as a module-level `@pxt.udf` and use that instead.\n\
                     For details, see: https://docs.pixeltable.com/platform/udfs-in-pixeltable",
                    call.def.display_name
                ));
            }
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// True if every Python function call in `e` can be persisted.
pub fn is_storable(e: &Expr) -> bool {
    validate_storable(e, "").is_ok()
}

// ------------------------------------------------------------------------------------------------ registry

/// Module-level functions by fully-qualified name, and the methods/properties registered per type (upstream
/// `FunctionRegistry`). Holds the Python `Function` objects; process-wide (survives `reset_db()`).
#[derive(Default)]
struct Registry {
    module_fns: Vec<(String, Py<PyAny>)>,
    /// (type enum, name) -> function
    type_methods: Vec<((i64, String), Py<PyAny>)>,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Registry::default()))
}

/// Registers a module function; `method_of` is the type enum of its first parameter if it is a method or
/// property.
pub fn register_function(fqn: &str, name: &str, obj: Py<PyAny>, method_of: Option<i64>) -> Result<()> {
    let mut r = registry().lock().map_err(|_| Error::internal("function registry poisoned"))?;
    if r.module_fns.iter().any(|(n, _)| n == fqn) {
        crate::bail!(FunctionAlreadyExists, "A UDF with that name already exists: {fqn}");
    }
    if let Some(t) = method_of {
        if r.type_methods.iter().any(|((te, n), _)| *te == t && n == name) {
            let type_name = format!("{:?}", crate::types::TypeEnum::from_i64(t).ok_or_else(|| Error::internal("bad type enum"))?)
                .to_ascii_uppercase();
            crate::bail!(FunctionAlreadyExists, "Duplicate method name for type Type.{type_name}: {name}");
        }
        r.type_methods.push(((t, name.to_string()), Python::attach(|py| obj.clone_ref(py))));
    }
    r.module_fns.push((fqn.to_string(), obj));
    Ok(())
}

/// The method or property `name` registered for values of type enum `t` (an Int also finds Float methods).
pub fn lookup_type_method(py: Python<'_>, t: i64, name: &str) -> Option<Py<PyAny>> {
    let r = registry().lock().ok()?;
    let find = |te: i64| r.type_methods.iter().find(|((x, n), _)| *x == te && n == name).map(|(_, f)| f.clone_ref(py));
    let int = crate::types::TypeEnum::Int as i64;
    let float = crate::types::TypeEnum::Float as i64;
    find(t).or_else(|| if t == int { find(float) } else { None })
}

pub fn registered_functions(py: Python<'_>) -> Vec<Py<PyAny>> {
    registry().lock().map(|r| r.module_fns.iter().map(|(_, f)| f.clone_ref(py)).collect()).unwrap_or_default()
}
