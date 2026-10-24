//! Builtin functions and the function-reference type used by `FunctionCall` nodes.
//!
//! A builtin is a static `Builtin` entry (name, signature in upstream's notation, kind, evaluator) in one of the
//! per-module tables (`string::FUNCTIONS`, `math::FUNCTIONS`, ...), registered under its qualified name
//! (`pixeltable.functions.string.upper`) by `registry()`. The generated Python stub `upper = _fn(__name__,
//! 'upper')` resolves to it automatically, and `t.c.upper()` works when the entry is a method of the value's type.
//!
//! To add a builtin: write `fn eval(&Args) -> Result<Value>` (and, for a conditional return type,
//! `fn ret(&CallTypes) -> Result<ColumnType>`), add `Builtin::new(name, signature, eval)` (with `.method()`,
//! `.property()`, `.sql()`, `.ret(f)` as needed) to the module's `FUNCTIONS`, rebuild. Binding, argument type
//! checks, NULL handling (a NULL argument for a non-nullable parameter yields NULL without calling `eval`) and the
//! return type's nullability follow upstream (`signature.rs`).
//!
//! Aggregate functions are `FuncRef::Agg` (see `agg.rs`); Python UDFs are `FuncRef::Udf`: the node holds the
//! Python callable and a descriptor (module + qualname, or pickled bytes) that Python resolves when the expression
//! is deserialized. Evaluation calls back into Python; that is the only place where evaluation leaves Rust.

pub mod agg;
pub mod array;
pub mod date;
pub mod json;
pub mod math;
pub mod signature;
pub mod string;
pub mod timestamp;
pub mod uuid;

use pyo3::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result, py_repr};
use crate::expr::{Expr, ExprKind, ExprRef, FunctionCall};
use crate::types::ColumnType;
use crate::value::{Json, Value};
use signature::Signature;

/// How a builtin is reachable from an expression of its first parameter's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// only as `module.fn(...)`
    Function,
    /// also as `expr.fn(...)`
    Method,
    /// also as `expr.fn`, called implicitly
    Property,
}

pub struct Builtin {
    /// registered (unqualified) name, e.g. `upper`
    pub name: &'static str,
    /// the name shown in `str(expr)` when it differs from `name`
    pub display_name: Option<&'static str>,
    /// signature in upstream's notation, see `signature::Signature::parse`
    pub sig: &'static str,
    pub kind: Kind,
    /// true if upstream evaluates the function in SQL (matters for which queries upstream accepts)
    pub sql: bool,
    /// conditional return type (upstream `@fn.conditional_return_type`); default: the signature's return type
    pub ret: Option<fn(&CallTypes<'_>) -> Result<ColumnType>>,
    pub eval: fn(&Args<'_>) -> Result<Value>,
}

impl Builtin {
    pub const fn new(name: &'static str, sig: &'static str, eval: fn(&Args<'_>) -> Result<Value>) -> Builtin {
        Builtin { name, display_name: None, sig, kind: Kind::Function, sql: false, ret: None, eval }
    }
    pub const fn method(self) -> Builtin {
        Builtin { kind: Kind::Method, ..self }
    }
    pub const fn property(self) -> Builtin {
        Builtin { kind: Kind::Property, ..self }
    }
    pub const fn sql(self) -> Builtin {
        Builtin { sql: true, ..self }
    }
    pub const fn ret(self, f: fn(&CallTypes<'_>) -> Result<ColumnType>) -> Builtin {
        Builtin { ret: Some(f), ..self }
    }
    pub const fn display(self, name: &'static str) -> Builtin {
        Builtin { display_name: Some(name), ..self }
    }
}

/// A registered builtin: the static entry, its module and its parsed signature.
pub struct BuiltinFn {
    pub module: &'static str,
    pub b: &'static Builtin,
    pub sig: Signature,
}

impl std::fmt::Debug for BuiltinFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Builtin({}.{})", self.module, self.b.name)
    }
}

impl BuiltinFn {
    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.module, self.b.name)
    }
    pub fn display_name(&self) -> &'static str {
        self.b.display_name.unwrap_or(self.b.name)
    }
}

/// The bound argument values of a builtin call.
pub struct Args<'a> {
    /// one value per non-variadic parameter (defaults filled in)
    pub vals: &'a [Value],
    pub var_args: &'a [Value],
    pub var_kwargs: &'a [(String, Value)],
}

fn arg_err(i: usize, what: &str, v: &Value) -> Error {
    Error::internal(format!("argument {i}: expected {what}, got {}", v.py_type_name()))
}

impl Args<'_> {
    pub fn get(&self, i: usize) -> &Value {
        self.vals.get(i).unwrap_or(&Value::Null)
    }
    pub fn is_null(&self, i: usize) -> bool {
        self.get(i).is_null()
    }
    pub fn str(&self, i: usize) -> Result<&str> {
        match self.get(i) {
            Value::String(s) | Value::Media(s) => Ok(s),
            Value::Json(Json::String(s)) => Ok(s),
            v => Err(arg_err(i, "a string", v)),
        }
    }
    pub fn opt_str(&self, i: usize) -> Result<Option<&str>> {
        if self.is_null(i) { Ok(None) } else { self.str(i).map(Some) }
    }
    pub fn int(&self, i: usize) -> Result<i64> {
        match self.get(i).json_scalar() {
            Value::Int(n) => Ok(n),
            Value::Bool(b) => Ok(b as i64),
            v => Err(arg_err(i, "an int", &v)),
        }
    }
    pub fn opt_int(&self, i: usize) -> Result<Option<i64>> {
        if self.is_null(i) { Ok(None) } else { self.int(i).map(Some) }
    }
    pub fn float(&self, i: usize) -> Result<f64> {
        self.get(i).json_scalar().as_f64().ok_or_else(|| arg_err(i, "a number", self.get(i)))
    }
    pub fn opt_float(&self, i: usize) -> Result<Option<f64>> {
        if self.is_null(i) { Ok(None) } else { self.float(i).map(Some) }
    }
    pub fn bool(&self, i: usize) -> Result<bool> {
        match self.get(i).json_scalar() {
            Value::Bool(b) => Ok(b),
            v => Err(arg_err(i, "a bool", &v)),
        }
    }
    /// The argument as a JSON value (None for NULL).
    pub fn json(&self, i: usize) -> Result<Option<Json>> {
        Ok(match self.get(i) {
            Value::Null => None,
            v => Some(crate::value::value_to_json(v)?),
        })
    }
    pub fn timestamp(&self, i: usize) -> Result<i64> {
        match self.get(i) {
            Value::Timestamp(t) => Ok(*t),
            v => Err(arg_err(i, "a timestamp", v)),
        }
    }
    pub fn date(&self, i: usize) -> Result<i32> {
        match self.get(i) {
            Value::Date(d) => Ok(*d),
            v => Err(arg_err(i, "a date", v)),
        }
    }
}

/// The bound argument expressions of a call, for conditional return types.
pub struct CallTypes<'a> {
    pub args: &'a [Option<ExprRef>],
    pub var_args: &'a [ExprRef],
}

impl CallTypes<'_> {
    pub fn ty(&self, i: usize) -> Option<&ColumnType> {
        self.args.get(i).and_then(|a| a.as_ref()).map(|e| &e.col_type)
    }
    pub fn literal(&self, i: usize) -> Option<&Value> {
        self.args.get(i).and_then(|a| a.as_ref()).and_then(|e| e.as_literal())
    }
}

fn builtin_tables() -> Vec<(&'static str, &'static [Builtin])> {
    vec![
        (string::MODULE, string::FUNCTIONS),
        (math::MODULE, math::FUNCTIONS),
        (timestamp::MODULE, timestamp::FUNCTIONS),
        (date::MODULE, date::FUNCTIONS),
        (json::MODULE, json::FUNCTIONS),
        (uuid::MODULE, uuid::FUNCTIONS),
        (array::MODULE, array::FUNCTIONS),
    ]
}

fn registry() -> &'static HashMap<String, &'static BuiltinFn> {
    static REGISTRY: OnceLock<HashMap<String, &'static BuiltinFn>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();
        for (module, table) in builtin_tables() {
            for b in table {
                let sig = Signature::parse(b.sig).unwrap_or_else(|e| panic!("{e:?}"));
                let f: &'static BuiltinFn = Box::leak(Box::new(BuiltinFn { module, b, sig }));
                m.insert(f.qualified_name(), f);
            }
        }
        m
    })
}

pub fn lookup(qualified_name: &str) -> Option<&'static BuiltinFn> {
    registry().get(qualified_name).copied()
}

/// The builtin that is a method (or property) named `name` of values of type `t`.
pub fn lookup_method(t: &ColumnType, name: &str) -> Option<&'static BuiltinFn> {
    let module = method_module(t)?;
    lookup(&format!("{module}.{name}")).filter(|f| f.b.kind != Kind::Function)
}

/// The function module whose methods apply to values of type `t`.
pub fn method_module(t: &ColumnType) -> Option<&'static str> {
    use crate::types::TypeKind::*;
    Some(match t.kind {
        String => "pixeltable.functions.string",
        Int | Float => "pixeltable.functions.math",
        Timestamp => "pixeltable.functions.timestamp",
        Date => "pixeltable.functions.date",
        Uuid => "pixeltable.functions.uuid",
        Json(_) => "pixeltable.functions.json",
        Array { .. } => "pixeltable.functions.array",
        Image { .. } => "pixeltable.functions.image",
        Video => "pixeltable.functions.video",
        Audio => "pixeltable.functions.audio",
        Document { .. } => "pixeltable.functions.document",
        _ => return None,
    })
}

/// Qualified names of all builtins (scalar and aggregate).
pub fn builtin_names() -> Vec<String> {
    let mut names: Vec<String> = registry().keys().cloned().collect();
    names.extend(agg::builtin_agg_names());
    names.sort();
    names
}

/// A Python UDF referenced from an expression.
pub struct UdfRef {
    pub name: String,
    /// how to find the callable again after a reload (produced and resolved by `pixeltable.func`)
    pub descriptor: Json,
    pub callable: Py<PyAny>,
}

impl std::fmt::Debug for UdfRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UdfRef({})", self.name)
    }
}

#[derive(Debug, Clone)]
pub enum FuncRef {
    Builtin(&'static BuiltinFn),
    Udf(Arc<UdfRef>),
    Agg(agg::AggRef),
}

impl FuncRef {
    /// The name shown in `str(expr)` and used as the default column name.
    pub fn name(&self) -> &str {
        match self {
            FuncRef::Builtin(b) => b.display_name(),
            FuncRef::Udf(u) => &u.name,
            FuncRef::Agg(a) => a.name(),
        }
    }

    /// True if upstream can evaluate a call of this function in SQL (given SQL-expressible arguments).
    pub fn is_sql(&self, arg_types: &[ColumnType]) -> bool {
        match self {
            FuncRef::Builtin(b) => b.b.sql,
            FuncRef::Udf(_) => false,
            FuncRef::Agg(a) => a.is_sql(arg_types),
        }
    }

    pub fn to_json(&self) -> Result<Json> {
        Ok(match self {
            FuncRef::Builtin(b) => json!({ "builtin": b.qualified_name() }),
            FuncRef::Udf(u) => json!({ "udf": u.descriptor, "name": u.name }),
            FuncRef::Agg(a) => a.to_json()?,
        })
    }

    pub fn from_json(j: &Json) -> Result<FuncRef> {
        if let Some(name) = j.get("builtin").and_then(Json::as_str) {
            if let Some(f) = lookup(name) {
                return Ok(FuncRef::Builtin(f));
            }
            return Err(crate::pxt_err!(FunctionNotFound, "Function {} not found", py_repr(name)));
        }
        if let Some(name) = j.get("builtin_agg").and_then(Json::as_str) {
            return agg::lookup_builtin_agg(name)
                .map(FuncRef::Agg)
                .ok_or_else(|| crate::pxt_err!(FunctionNotFound, "Function {} not found", py_repr(name)));
        }
        if j.get("udf").is_some() {
            return Err(Error::not_implemented("resolving a stored Python UDF"));
        }
        if j.get("uda").is_some() {
            return agg::resolve_uda(j).map(FuncRef::Agg);
        }
        Err(Error::internal(format!("malformed function reference: {j}")))
    }

    /// Calls a scalar function with evaluated arguments (as given in the call: positional, then keyword).
    pub fn call(&self, args: Vec<Value>, kwargs: Vec<(String, Value)>, _result_type: &ColumnType) -> Result<Value> {
        match self {
            FuncRef::Builtin(b) => call_builtin(b, args, kwargs),
            FuncRef::Udf(u) => Err(Error::not_implemented(format!("calling the Python UDF `{}`", u.name))),
            FuncRef::Agg(a) => Err(Error::internal(format!("aggregate function {} evaluated outside of an aggregation", a.name()))),
        }
    }
}

/// Binds evaluated arguments to `sig` (positionally, then by name), filling in defaults. Returns None if a
/// non-nullable parameter got a NULL argument (the call then yields NULL without running the function).
pub fn bind_values(
    sig: &Signature,
    args: Vec<Value>,
    kwargs: Vec<(String, Value)>,
) -> Result<Option<(Vec<Value>, Vec<Value>, Vec<(String, Value)>)>> {
    let idx: Vec<usize> = (0..args.len()).collect();
    let kw_idx: Vec<(String, usize)> = kwargs.iter().enumerate().map(|(i, (k, _))| (k.clone(), args.len() + i)).collect();
    let bound = sig.bind(&idx, &kw_idx).map_err(|m| Error::internal(format!("binding failed at evaluation: {m}")))?;
    let mut all: Vec<Value> = args;
    all.extend(kwargs.into_iter().map(|(_, v)| v));
    let mut vals = Vec::with_capacity(sig.params.len());
    for (p, a) in sig.params.iter().zip(bound.args.iter()) {
        if p.kind.is_var() {
            vals.push(Value::Null);
            continue;
        }
        match a {
            Some(i) => {
                let v = std::mem::replace(&mut all[*i], Value::Null);
                if v.is_null() && !p.col_type.nullable {
                    return Ok(None);
                }
                vals.push(v);
            }
            None => vals.push(p.default.clone().unwrap_or(Value::Null)),
        }
    }
    let var_args = bound.var_args.iter().map(|i| std::mem::replace(&mut all[*i], Value::Null)).collect();
    let var_kwargs = bound.var_kwargs.iter().map(|(k, i)| (k.clone(), std::mem::replace(&mut all[*i], Value::Null))).collect();
    Ok(Some((vals, var_args, var_kwargs)))
}

fn call_builtin(f: &BuiltinFn, args: Vec<Value>, kwargs: Vec<(String, Value)>) -> Result<Value> {
    match bind_values(&f.sig, args, kwargs)? {
        None => Ok(Value::Null),
        Some((vals, var_args, var_kwargs)) => (f.b.eval)(&Args { vals: &vals, var_args: &var_args, var_kwargs: &var_kwargs }),
    }
}

/// Builds a call of the builtin `qualified_name` (scalar or aggregate).
pub fn make_call(qualified_name: &str, args: Vec<ExprRef>, kwargs: Vec<(String, ExprRef)>) -> Result<ExprRef> {
    if let Some(f) = lookup(qualified_name) {
        return make_builtin_call(f, args, kwargs, false);
    }
    if let Some(a) = agg::lookup_builtin_agg(qualified_name) {
        return agg::make_agg_call(a, args, kwargs, None, Vec::new());
    }
    Err(crate::pxt_err!(FunctionNotFound, "Function {} not found", py_repr(qualified_name)))
}

/// Builds a call of a builtin: binds and type-checks the arguments and computes the return type.
pub fn make_builtin_call(
    f: &'static BuiltinFn,
    args: Vec<ExprRef>,
    kwargs: Vec<(String, ExprRef)>,
    is_method_call: bool,
) -> Result<ExprRef> {
    let bound = f.sig.bind_exprs(&args, &kwargs)?;
    f.sig.validate_args(&bound, f.b.name)?;
    let ret = match f.b.ret {
        Some(rf) => rf(&CallTypes { args: &bound.args, var_args: &bound.var_args })?,
        None => f.sig.return_type.clone(),
    };
    let ret = f.sig.call_return_type(ret, &bound);
    let mut fc = FunctionCall::new(FuncRef::Builtin(f), args, kwargs);
    fc.is_method_call = is_method_call;
    Ok(Expr::new(ExprKind::FunctionCall(fc), ret))
}

/// Checks that an argument has the expected kind of type (helper for conditional return types).
pub fn expect_arg(fn_name: &str, param: &str, t: &ColumnType, ok: bool, expected: &str) -> Result<()> {
    if !ok {
        crate::bail!(
            TypeMismatch,
            "{fn_name}(): parameter {} expects a value of type `{expected}`, but got `{}`",
            py_repr(param),
            t.repr()
        );
    }
    Ok(())
}
