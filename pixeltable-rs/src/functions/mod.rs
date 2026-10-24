//! Builtin functions and the function-reference type used by `FunctionCall` nodes.
//!
//! A builtin is a static `Builtin` entry in one of the per-module tables (`string::FUNCTIONS`, `math::FUNCTIONS`,
//! ...), registered under its qualified name (`pixeltable.functions.string.upper`). The Python stub module exposes
//! it as `upper = _builtin('pixeltable.functions.string.upper')`, which builds a `FunctionCall` node through
//! `make_call()`.
//!
//! To add a builtin: write `fn return_type(&[ColumnType]) -> Result<ColumnType>` and `fn eval(&[Value]) ->
//! Result<Value>`, add a `Builtin` entry to the module's `FUNCTIONS`, and add the one-line Python stub.
//!
//! Python UDFs are `FuncRef::Udf`: the node holds the Python callable and a descriptor (module + qualname, or
//! pickled bytes for local functions) that Python resolves when the expression is deserialized. Evaluation calls
//! back into Python; that is the only place where builtin-free evaluation leaves Rust.

pub mod math;
pub mod signature;
pub mod string;
pub mod udf;

use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result};
use crate::expr::{Expr, ExprKind, ExprRef, FunctionCall};
use crate::types::ColumnType;
use crate::value::{Json, Value};

pub struct Builtin {
    /// unqualified name, e.g. `upper`
    pub name: &'static str,
    /// Python module that exposes it, e.g. `pixeltable.functions.string`
    pub module: &'static str,
    /// parameter names, in order
    pub params: &'static [&'static str],
    /// number of leading parameters that are required
    pub required: usize,
    /// if true, any NULL argument makes the result NULL without calling `eval` (non-nullable parameters)
    pub strict: bool,
    pub return_type: fn(&[ColumnType]) -> Result<ColumnType>,
    pub eval: fn(&[Value]) -> Result<Value>,
}

impl std::fmt::Debug for Builtin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Builtin({}.{})", self.module, self.name)
    }
}

impl Builtin {
    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.module, self.name)
    }
}

fn registry() -> &'static HashMap<String, &'static Builtin> {
    static REGISTRY: OnceLock<HashMap<String, &'static Builtin>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();
        for table in [string::FUNCTIONS, math::FUNCTIONS] {
            for b in table {
                m.insert(b.qualified_name(), b);
            }
        }
        m
    })
}

pub fn lookup(qualified_name: &str) -> Option<&'static Builtin> {
    registry().get(qualified_name).copied()
}

pub fn builtin_names() -> Vec<String> {
    let mut names: Vec<String> = registry().keys().cloned().collect();
    names.sort();
    names
}

#[derive(Debug, Clone)]
pub enum FuncRef {
    Builtin(&'static Builtin),
    /// a call of a Python function (see `udf`)
    Udf(Arc<udf::UdfCall>),
}

impl FuncRef {
    /// The name used as the default output column name (upstream `Function.name`).
    pub fn name(&self) -> &str {
        match self {
            FuncRef::Builtin(b) => b.name,
            FuncRef::Udf(u) => u.def.name.as_deref().unwrap_or(""),
        }
    }

    pub fn to_json(&self) -> Result<Json> {
        Ok(match self {
            FuncRef::Builtin(b) => json!({ "builtin": b.qualified_name() }),
            FuncRef::Udf(u) => u.to_json()?,
        })
    }

    pub fn from_json(j: &Json) -> Result<FuncRef> {
        if let Some(name) = j.get("builtin").and_then(Json::as_str) {
            return lookup(name)
                .map(FuncRef::Builtin)
                .ok_or_else(|| crate::pxt_err!(FunctionNotFound, "Function {} not found", crate::error::py_repr(name)));
        }
        Err(Error::internal(format!("malformed function reference: {j}")))
    }

    pub fn call(&self, args: &[Value], kwargs: &[(String, Value)], _result_type: &ColumnType) -> Result<Value> {
        match self {
            FuncRef::Builtin(b) => {
                if !kwargs.is_empty() {
                    return Err(Error::internal("builtin calls are bound positionally"));
                }
                if b.strict && args.iter().any(Value::is_null) {
                    return Ok(Value::Null);
                }
                (b.eval)(args)
            }
            FuncRef::Udf(_) => Err(Error::internal("Python function calls are evaluated by `udf::eval_call()`")),
        }
    }
}

/// Evaluates a function call node against a row.
pub fn eval_call(fc: &FunctionCall, e: &Expr, ctx: &dyn crate::expr::eval::RowContext) -> Result<Value> {
    if let FuncRef::Udf(call) = &fc.func {
        return udf::eval_call(call, fc, e, ctx);
    }
    let args: Vec<Value> = fc.args.iter().map(|a| crate::expr::eval::eval(a, ctx)).collect::<Result<_>>()?;
    let kwargs: Vec<(String, Value)> =
        fc.kwargs.iter().map(|(k, a)| Ok((k.clone(), crate::expr::eval::eval(a, ctx)?))).collect::<Result<_>>()?;
    fc.func.call(&args, &kwargs, &e.col_type)
}

/// `str()` of a function call node.
pub fn display_call(fc: &FunctionCall) -> String {
    if let FuncRef::Udf(call) = &fc.func {
        return udf::display_call(call, fc);
    }
    let mut args: Vec<String> = fc.args.iter().map(|a| crate::expr::display::display(a)).collect();
    args.extend(fc.kwargs.iter().map(|(k, v)| format!("{k}={}", crate::expr::display::display(v))));
    format!("{}({})", fc.func.name(), args.join(", "))
}

/// Builds a call of the builtin `qualified_name`, binding keyword arguments to positions.
pub fn make_call(qualified_name: &str, args: Vec<ExprRef>, kwargs: Vec<(String, ExprRef)>) -> Result<ExprRef> {
    let b = lookup(qualified_name)
        .ok_or_else(|| crate::pxt_err!(FunctionNotFound, "Function {} not found", crate::error::py_repr(qualified_name)))?;
    if args.len() > b.params.len() {
        crate::bail!(
            InvalidArgument,
            "{}() takes at most {} positional argument(s) but {} were given",
            b.name,
            b.params.len(),
            args.len()
        );
    }
    let mut bound: Vec<Option<ExprRef>> = vec![None; b.params.len()];
    for (i, a) in args.into_iter().enumerate() {
        bound[i] = Some(a);
    }
    for (k, v) in kwargs {
        let pos = b
            .params
            .iter()
            .position(|p| *p == k)
            .ok_or_else(|| crate::pxt_err!(InvalidArgument, "{}() got an unexpected keyword argument {}", b.name, crate::error::py_repr(&k)))?;
        if bound[pos].is_some() {
            crate::bail!(InvalidArgument, "{}() got multiple values for argument {}", b.name, crate::error::py_repr(&k));
        }
        bound[pos] = Some(v);
    }
    let n_bound = bound.iter().take_while(|a| a.is_some()).count();
    if n_bound < b.required || bound[n_bound..].iter().any(Option::is_some) {
        let missing = b.params.iter().zip(bound.iter()).find(|(_, a)| a.is_none()).map(|(p, _)| *p).unwrap_or("?");
        crate::bail!(InvalidArgument, "{}() missing required argument: {}", b.name, crate::error::py_repr(missing));
    }
    let args: Vec<ExprRef> = bound.into_iter().flatten().collect();
    let arg_types: Vec<ColumnType> = args.iter().map(|a| a.col_type.clone()).collect();
    let mut ret = (b.return_type)(&arg_types)?;
    if b.strict && arg_types.iter().any(|t| t.nullable) {
        ret = ret.with_nullable(true);
    }
    Ok(Expr::new(ExprKind::FunctionCall(FunctionCall { func: FuncRef::Builtin(b), args, kwargs: vec![] }), ret))
}

/// Checks that an argument has the expected kind of type (helper for `return_type` functions).
pub fn expect_arg(fn_name: &str, param: &str, t: &ColumnType, ok: bool, expected: &str) -> Result<()> {
    if !ok {
        crate::bail!(
            TypeMismatch,
            "{fn_name}(): parameter {} expects a value of type `{expected}`, but got `{}`",
            crate::error::py_repr(param),
            t.repr()
        );
    }
    Ok(())
}
