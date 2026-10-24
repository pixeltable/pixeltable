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
//! The image/video/audio builtins (`image.rs`, `video.rs`) are older positional `PosBuiltin` entries (parameter
//! names, a return-type function over the positional argument types, strict NULL handling); they are registered
//! next to the signature-based ones and bound with a synthesized signature.
//!
//! Aggregate functions are `FuncRef::Agg` (see `agg.rs`). Calls of Python functions (UDFs, expression templates)
//! are `FuncRef::Udf` (see `udf.rs`): binding, typing and evaluation happen there; evaluation calls back into
//! Python for the function body, the only place where evaluation leaves Rust.

pub mod agg;
pub mod array;
pub mod date;
pub mod image;
pub mod image_iterators;
pub mod iterator;
pub mod json;
pub mod math;
pub mod signature;
pub mod string;
pub mod timestamp;
pub mod udf;
pub mod uuid;
pub mod video_iterators;
pub mod video;

use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result, py_repr};
use crate::expr::{Expr, ExprKind, ExprRef, FunctionCall};
use crate::types::ColumnType;
use crate::value::{Json, Value};
use signature::{BoundArg, Param, ParamKind, Signature};

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

/// A positional builtin (`image.rs`, `video.rs`).
pub struct PosBuiltin {
    /// unqualified name, e.g. `rotate`
    pub name: &'static str,
    /// Python module that exposes it, e.g. `pixeltable.functions.image`
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

impl std::fmt::Debug for PosBuiltin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PosBuiltin({}.{})", self.module, self.name)
    }
}

impl PosBuiltin {
    /// A signature for binding: untyped parameters (`return_type` checks the types), optional ones defaulting to None.
    fn signature(&self) -> Signature {
        let params = self
            .params
            .iter()
            .enumerate()
            .map(|(i, name)| Param {
                name: name.to_string(),
                kind: ParamKind::PositionalOrKeyword,
                col_type: Some(ColumnType::invalid(true)),
                default: (i >= self.required).then(|| Expr::literal(Value::Null, ColumnType::invalid(true))),
                is_batched: false,
            })
            .collect();
        Signature { params, return_type: ColumnType::invalid(true), is_batched: false }
    }
}

pub enum BuiltinImpl {
    Sig(&'static Builtin),
    Pos(&'static PosBuiltin),
}

/// A registered builtin: its module, kind, parsed (or synthesized) signature and implementation.
pub struct BuiltinFn {
    pub module: &'static str,
    pub name: &'static str,
    pub kind: Kind,
    pub sql: bool,
    pub sig: Signature,
    pub imp: BuiltinImpl,
}

impl std::fmt::Debug for BuiltinFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Builtin({}.{})", self.module, self.name)
    }
}

impl BuiltinFn {
    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.module, self.name)
    }
    pub fn display_name(&self) -> &'static str {
        match &self.imp {
            BuiltinImpl::Sig(b) => b.display_name.unwrap_or(b.name),
            BuiltinImpl::Pos(b) => b.name,
        }
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
        let mut add = |f: BuiltinFn| {
            let f: &'static BuiltinFn = Box::leak(Box::new(f));
            m.insert(f.qualified_name(), f);
        };
        for (module, table) in builtin_tables() {
            for b in table {
                let sig = Signature::parse(b.sig).unwrap_or_else(|e| panic!("{e:?}"));
                add(BuiltinFn { module, name: b.name, kind: b.kind, sql: b.sql, sig, imp: BuiltinImpl::Sig(b) });
            }
        }
        for table in [image::FUNCTIONS, video::VIDEO_FUNCTIONS, video::AUDIO_FUNCTIONS] {
            for b in table {
                // every function of the media modules is a method of its first argument's type
                let kind = if b.module == image::MODULE && image::PROPERTIES.contains(&b.name) { Kind::Property } else { Kind::Method };
                add(BuiltinFn { module: b.module, name: b.name, kind, sql: false, sig: b.signature(), imp: BuiltinImpl::Pos(b) });
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
    lookup(&format!("{module}.{name}")).filter(|f| f.kind != Kind::Function)
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

#[derive(Debug, Clone)]
pub enum FuncRef {
    Builtin(&'static BuiltinFn),
    /// a call of a Python function (see `udf`)
    Udf(Arc<udf::UdfCall>),
    Agg(agg::AggRef),
}

impl FuncRef {
    /// The name shown in `str(expr)` and used as the default column name (upstream `Function.name`).
    pub fn name(&self) -> &str {
        match self {
            FuncRef::Builtin(b) => b.display_name(),
            FuncRef::Udf(u) => u.def.name.as_deref().unwrap_or(""),
            FuncRef::Agg(a) => a.name(),
        }
    }

    /// True if upstream can evaluate a call of this function in SQL (given SQL-expressible arguments).
    pub fn is_sql(&self, arg_types: &[ColumnType]) -> bool {
        match self {
            FuncRef::Builtin(b) => b.sql,
            FuncRef::Udf(_) => false,
            FuncRef::Agg(a) => a.is_sql(arg_types),
        }
    }

    pub fn to_json(&self) -> Result<Json> {
        Ok(match self {
            FuncRef::Builtin(b) => json!({ "builtin": b.qualified_name() }),
            FuncRef::Udf(u) => u.to_json()?,
            FuncRef::Agg(a) => a.to_json()?,
        })
    }

    /// Resolves a stored builtin or aggregate (stored Python function calls go through `udf::call_from_json()`).
    pub fn from_json(j: &Json) -> Result<FuncRef> {
        if let Some(name) = j.get("builtin").and_then(Json::as_str) {
            return lookup(name)
                .map(FuncRef::Builtin)
                .ok_or_else(|| crate::pxt_err!(FunctionNotFound, "Function {} not found", py_repr(name)));
        }
        if let Some(name) = j.get("builtin_agg").and_then(Json::as_str) {
            return agg::lookup_builtin_agg(name)
                .map(FuncRef::Agg)
                .ok_or_else(|| crate::pxt_err!(FunctionNotFound, "Function {} not found", py_repr(name)));
        }
        if j.get("uda").is_some() {
            return agg::resolve_uda(j).map(FuncRef::Agg);
        }
        Err(Error::internal(format!("malformed function reference: {j}")))
    }
}

/// Evaluates a function call node against a row.
pub fn eval_call(fc: &FunctionCall, e: &Expr, ctx: &dyn crate::expr::eval::RowContext) -> Result<Value> {
    use crate::expr::eval::eval;
    let f = match &fc.func {
        FuncRef::Udf(call) => return udf::eval_call(call, fc, e, ctx),
        FuncRef::Agg(a) => {
            return Err(Error::internal(format!("aggregate function {} evaluated outside of an aggregation", a.name())));
        }
        FuncRef::Builtin(f) => f,
    };
    let args: Vec<Value> = fc.args.iter().map(|a| eval(a, ctx)).collect::<Result<_>>()?;
    let kwargs: Vec<(String, Value)> = fc.kwargs.iter().map(|(k, a)| Ok((k.clone(), eval(a, ctx)?))).collect::<Result<_>>()?;
    call_builtin(f, args, kwargs)
}

/// Calls a builtin with evaluated arguments (as given in the call: positional, then keyword).
pub fn call_builtin(f: &BuiltinFn, args: Vec<Value>, kwargs: Vec<(String, Value)>) -> Result<Value> {
    match &f.imp {
        BuiltinImpl::Pos(b) => {
            if !kwargs.is_empty() {
                return Err(Error::internal("positional builtin calls are bound positionally"));
            }
            if b.strict && args.iter().any(Value::is_null) {
                return Ok(Value::Null);
            }
            (b.eval)(&args)
        }
        BuiltinImpl::Sig(b) => match bind_values(&f.sig, args, kwargs)? {
            None => Ok(Value::Null),
            Some((vals, var_args, var_kwargs)) => (b.eval)(&Args { vals: &vals, var_args: &var_args, var_kwargs: &var_kwargs }),
        },
    }
}

/// Binds evaluated arguments to `sig` (positionally, then by name), filling in defaults. Returns None if a
/// non-nullable parameter got a NULL argument (the call then yields NULL without running the function).
pub fn bind_values(
    sig: &Signature,
    args: Vec<Value>,
    kwargs: Vec<(String, Value)>,
) -> Result<Option<(Vec<Value>, Vec<Value>, Vec<(String, Value)>)>> {
    let names: Vec<String> = kwargs.iter().map(|(k, _)| k.clone()).collect();
    let binding = sig.bind(args.len(), &names).map_err(|m| Error::internal(format!("binding failed at evaluation: {m}")))?;
    let mut args: Vec<Value> = args;
    let mut kwargs: Vec<(String, Value)> = kwargs;
    let mut vals = Vec::with_capacity(sig.params.len());
    let mut var_args = Vec::new();
    let mut var_kwargs = Vec::new();
    for (pi, p) in sig.params.iter().enumerate() {
        let v = match binding.get(pi) {
            Some(BoundArg::Arg(i)) => std::mem::replace(&mut args[*i], Value::Null),
            Some(BoundArg::Kwarg(i)) => std::mem::replace(&mut kwargs[*i].1, Value::Null),
            Some(BoundArg::VarArgs(idx)) => {
                var_args = idx.iter().map(|i| std::mem::replace(&mut args[*i], Value::Null)).collect();
                vals.push(Value::Null);
                continue;
            }
            Some(BoundArg::VarKwargs(idx)) => {
                var_kwargs = idx.iter().map(|i| (kwargs[*i].0.clone(), std::mem::replace(&mut kwargs[*i].1, Value::Null))).collect();
                vals.push(Value::Null);
                continue;
            }
            None => {
                vals.push(p.default.as_ref().and_then(|d| d.as_literal().cloned()).unwrap_or(Value::Null));
                continue;
            }
        };
        if v.is_null() && p.col_type.as_ref().is_some_and(|t| !t.nullable) {
            return Ok(None);
        }
        vals.push(v);
    }
    Ok(Some((vals, var_args, var_kwargs)))
}

/// `str()` of a function call node.
pub fn display_call(fc: &FunctionCall) -> String {
    use crate::expr::display::{display, print_list};
    if let FuncRef::Udf(call) = &fc.func {
        return udf::display_call(call, fc);
    }
    let skip = usize::from(fc.is_method_call && !fc.args.is_empty());
    let mut args: Vec<String> = fc.args.iter().skip(skip).map(|a| display(a)).collect();
    args.extend(fc.kwargs.iter().map(|(k, v)| format!("{k}={}", display(v))));
    if !fc.order_by.is_empty() {
        let ob = print_list(&fc.order_by);
        let requires = matches!(&fc.func, FuncRef::Agg(a) if a.requires_order_by());
        if requires {
            args.insert(0, ob);
        } else {
            args.push(format!("order_by={ob}"));
        }
    }
    if !fc.group_by.is_empty() {
        args.push(format!("group_by={}", print_list(&fc.group_by)));
    }
    if skip == 1 {
        format!("{}.{}({})", display(&fc.args[0]), fc.func.name(), args.join(", "))
    } else {
        format!("{}({})", fc.func.name(), args.join(", "))
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
    let binding = f.sig.bind_exprs(&args, &kwargs)?;
    let (args, kwargs, ret) = match &f.imp {
        BuiltinImpl::Sig(b) => {
            f.sig.validate_args(&binding, &args, &kwargs, &format!("in function {}", py_repr(f.name)))?;
            let ret = match b.ret {
                Some(rf) => {
                    let bound = f.sig.bound_exprs(&binding, &args, &kwargs);
                    let var_args: Vec<ExprRef> = f
                        .sig
                        .params
                        .iter()
                        .enumerate()
                        .find_map(|(pi, _)| match binding.get(pi) {
                            Some(BoundArg::VarArgs(idx)) => Some(idx.iter().map(|i| args[*i].clone()).collect()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    rf(&CallTypes { args: &bound, var_args: &var_args })?
                }
                None => f.sig.return_type.clone(),
            };
            let ret = f.sig.call_return_type(&ret, &binding, &args, &kwargs);
            (args, kwargs, ret)
        }
        BuiltinImpl::Pos(b) => {
            // positional builtins take the bound arguments up to the first omitted optional parameter
            let bound = f.sig.bound_exprs(&binding, &args, &kwargs);
            let n = bound.iter().take_while(|a| a.is_some()).count();
            if let Some(pi) = bound[n..].iter().position(Option::is_some) {
                crate::bail!(
                    InvalidArgument,
                    "{}() requires argument {} when {} is given",
                    b.name,
                    py_repr(b.params[n]),
                    py_repr(b.params[n + pi])
                );
            }
            let args: Vec<ExprRef> = bound.into_iter().flatten().collect();
            let arg_types: Vec<ColumnType> = args.iter().map(|a| a.col_type.clone()).collect();
            let mut ret = (b.return_type)(&arg_types)?;
            if b.strict && arg_types.iter().any(|t| t.nullable) {
                ret = ret.with_nullable(true);
            }
            (args, Vec::new(), ret)
        }
    };
    let mut fc = FunctionCall::new(FuncRef::Builtin(f), args, kwargs);
    fc.is_method_call = is_method_call;
    Ok(Expr::new(ExprKind::FunctionCall(fc), ret))
}

/// Checks that an argument has the expected kind of type (helper for return-type functions).
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
