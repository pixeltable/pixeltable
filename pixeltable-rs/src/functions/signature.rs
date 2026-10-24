//! Signatures of Python functions (UDFs, UDAs, iterators) and the binding of call arguments to them.
//!
//! `pixeltable.func` introspects the Python callable (type hints, defaults, parameter kinds) and hands the result to
//! the backend (`Signature::from_py`); everything that happens when a function is called (binding arguments to
//! parameters, type checking, overload resolution, the result type) happens here. `bind()` reproduces Python's
//! `inspect.Signature.bind()`, including its error messages, which upstream surfaces verbatim.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::json;

use crate::error::{Error, Result, py_repr};
use crate::expr::ExprRef;
use crate::types::ColumnType;
use crate::value::Json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    PositionalOnly,
    PositionalOrKeyword,
    VarPositional,
    KeywordOnly,
    VarKeyword,
}

impl ParamKind {
    /// The `inspect.Parameter` kind name.
    pub fn name(self) -> &'static str {
        match self {
            ParamKind::PositionalOnly => "POSITIONAL_ONLY",
            ParamKind::PositionalOrKeyword => "POSITIONAL_OR_KEYWORD",
            ParamKind::VarPositional => "VAR_POSITIONAL",
            ParamKind::KeywordOnly => "KEYWORD_ONLY",
            ParamKind::VarKeyword => "VAR_KEYWORD",
        }
    }

    pub fn from_name(name: &str) -> Option<ParamKind> {
        Some(match name {
            "POSITIONAL_ONLY" => ParamKind::PositionalOnly,
            "POSITIONAL_OR_KEYWORD" => ParamKind::PositionalOrKeyword,
            "VAR_POSITIONAL" => ParamKind::VarPositional,
            "KEYWORD_ONLY" => ParamKind::KeywordOnly,
            "VAR_KEYWORD" => ParamKind::VarKeyword,
            _ => return None,
        })
    }

    pub fn is_var(self) -> bool {
        matches!(self, ParamKind::VarPositional | ParamKind::VarKeyword)
    }
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    /// None for variadic parameters
    pub col_type: Option<ColumnType>,
    /// a constant (Literal) expression
    pub default: Option<ExprRef>,
    pub is_batched: bool,
}

impl Param {
    pub fn has_default(&self) -> bool {
        self.default.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub params: Vec<Param>,
    pub return_type: ColumnType,
    pub is_batched: bool,
}

/// Where the value of a parameter comes from in a call: indices into the call's positional args or kwargs.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundArg {
    Arg(usize),
    Kwarg(usize),
    VarArgs(Vec<usize>),
    VarKwargs(Vec<usize>),
}

/// The result of binding a call to a signature: (parameter index, source) for each bound parameter, in parameter
/// order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Binding {
    pub bound: Vec<(usize, BoundArg)>,
}

impl Binding {
    pub fn get(&self, param_idx: usize) -> Option<&BoundArg> {
        self.bound.iter().find(|(i, _)| *i == param_idx).map(|(_, b)| b)
    }
}

impl Signature {
    pub fn param(&self, name: &str) -> Option<(usize, &Param)> {
        self.params.iter().enumerate().find(|(_, p)| p.name == name)
    }

    pub fn required_params(&self) -> impl Iterator<Item = &Param> {
        self.params.iter().filter(|p| !p.has_default())
    }

    /// `inspect.Signature.bind(*args, **kwargs)` for `n_args` positional arguments and the given keyword names;
    /// the error is the TypeError message Python produces.
    pub fn bind(&self, n_args: usize, kwarg_names: &[String]) -> std::result::Result<Binding, String> {
        let mut bound: Vec<(usize, BoundArg)> = Vec::new();
        // remaining kwargs, in order
        let mut kwargs: Vec<(usize, &String)> = kwarg_names.iter().enumerate().collect();
        let in_kwargs = |kwargs: &Vec<(usize, &String)>, name: &str| kwargs.iter().any(|(_, k)| *k == name);
        let mut params = self.params.iter().enumerate().peekable();
        let mut params_ex: Option<(usize, &Param)> = None;
        let mut pos_only_in_kwargs: Vec<&str> = Vec::new();
        let mut arg_idx = 0usize;
        loop {
            if arg_idx >= n_args {
                // no more positional arguments
                let Some((pi, p)) = params.next() else { break };
                if p.kind == ParamKind::VarPositional {
                    break;
                } else if in_kwargs(&kwargs, &p.name) {
                    if p.kind == ParamKind::PositionalOnly {
                        if !p.has_default() {
                            return Err(format!("missing a required positional-only argument: {}", py_repr(&p.name)));
                        }
                        pos_only_in_kwargs.push(&p.name);
                        continue;
                    }
                    params_ex = Some((pi, p));
                    break;
                } else if p.kind == ParamKind::VarKeyword || p.has_default() {
                    params_ex = Some((pi, p));
                    break;
                } else {
                    let argtype = if p.kind == ParamKind::KeywordOnly { " keyword-only" } else { "" };
                    return Err(format!("missing a required{argtype} argument: {}", py_repr(&p.name)));
                }
            } else {
                let Some((pi, p)) = params.next() else { return Err("too many positional arguments".into()) };
                if matches!(p.kind, ParamKind::VarKeyword | ParamKind::KeywordOnly) {
                    return Err("too many positional arguments".into());
                }
                if p.kind == ParamKind::VarPositional {
                    bound.push((pi, BoundArg::VarArgs((arg_idx..n_args).collect())));
                    break;
                }
                if in_kwargs(&kwargs, &p.name) && p.kind != ParamKind::PositionalOnly {
                    return Err(format!("multiple values for argument {}", py_repr(&p.name)));
                }
                bound.push((pi, BoundArg::Arg(arg_idx)));
                arg_idx += 1;
            }
        }
        let mut kwargs_param: Option<usize> = None;
        for (pi, p) in params_ex.into_iter().chain(params) {
            if p.kind == ParamKind::VarKeyword {
                kwargs_param = Some(pi);
                continue;
            }
            if p.kind == ParamKind::VarPositional {
                continue;
            }
            match kwargs.iter().position(|(_, k)| **k == p.name) {
                Some(pos) => {
                    let (ki, _) = kwargs.remove(pos);
                    bound.push((pi, BoundArg::Kwarg(ki)));
                }
                None => {
                    if !p.has_default() {
                        return Err(format!("missing a required argument: {}", py_repr(&p.name)));
                    }
                }
            }
        }
        if !kwargs.is_empty() {
            if let Some(pi) = kwargs_param {
                bound.push((pi, BoundArg::VarKwargs(kwargs.iter().map(|(i, _)| *i).collect())));
            } else if !pos_only_in_kwargs.is_empty() {
                return Err(format!(
                    "got some positional-only arguments passed as keyword arguments: {}",
                    py_repr(&pos_only_in_kwargs.join(", "))
                ));
            } else {
                return Err(format!("got an unexpected keyword argument {}", py_repr(kwargs[0].1)));
            }
        }
        bound.sort_by_key(|(pi, _)| *pi);
        Ok(Binding { bound })
    }

    /// Upstream `Signature.validate_args()`: each bound argument must fit its parameter's type (a nullable argument
    /// is fine for a non-nullable parameter: the call then short-circuits NULLs).
    pub fn validate_args(
        &self,
        binding: &Binding,
        args: &[ExprRef],
        kwargs: &[(String, ExprRef)],
        context: &str,
    ) -> Result<()> {
        let context = if context.is_empty() { String::new() } else { format!(" ({context})") };
        for (pi, b) in &binding.bound {
            let p = &self.params[*pi];
            let Some(param_type) = &p.col_type else { continue };
            let arg = match b {
                BoundArg::Arg(i) => &args[*i],
                BoundArg::Kwarg(i) => &kwargs[*i].1,
                _ => continue,
            };
            if !arg_matches(param_type, &arg.col_type) {
                crate::bail!(
                    TypeMismatch,
                    "Parameter {}{context}: argument type {} does not match parameter type {}",
                    py_repr(&p.name),
                    arg.col_type.repr(),
                    param_type.repr()
                );
            }
        }
        Ok(())
    }

    /// The result type of a call, given the declared (or conditional) return type: non-nullable results become
    /// nullable when a nullable argument is bound to a non-nullable parameter (that call yields NULL).
    pub fn call_return_type(
        &self,
        return_type: &ColumnType,
        binding: &Binding,
        args: &[ExprRef],
        kwargs: &[(String, ExprRef)],
    ) -> ColumnType {
        if return_type.nullable {
            return return_type.clone();
        }
        for (pi, b) in &binding.bound {
            let p = &self.params[*pi];
            let Some(param_type) = &p.col_type else { continue };
            let arg = match b {
                BoundArg::Arg(i) => &args[*i],
                BoundArg::Kwarg(i) => &kwargs[*i].1,
                _ => continue,
            };
            if arg.col_type.nullable && !param_type.nullable {
                return return_type.with_nullable(true);
            }
        }
        return_type.clone()
    }

    /// The parameter list as upstream's `Signature.params_str()` renders it.
    pub fn params_str(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut pending_pos_only_marker = false;
        let mut needs_kw_only_marker = true;
        for p in &self.params {
            if p.kind == ParamKind::PositionalOnly {
                pending_pos_only_marker = true;
            } else if pending_pos_only_marker {
                parts.push("/".into());
                pending_pos_only_marker = false;
            }
            if p.kind == ParamKind::VarPositional {
                needs_kw_only_marker = false;
            } else if p.kind == ParamKind::KeywordOnly && needs_kw_only_marker {
                parts.push("*".into());
                needs_kw_only_marker = false;
            }
            match p.kind {
                ParamKind::VarPositional => parts.push(format!("*{}", p.name)),
                ParamKind::VarKeyword => parts.push(format!("**{}", p.name)),
                _ => parts.push(format!(
                    "{}: pxt.{}",
                    p.name,
                    p.col_type.as_ref().map(|t| t.repr()).unwrap_or_default()
                )),
            }
        }
        if pending_pos_only_marker {
            parts.push("/".into());
        }
        parts.join(", ")
    }

    /// `str(signature)`
    pub fn display(&self) -> String {
        format!("({}) -> pxt.{}", self.params_str(), self.return_type.repr())
    }

    /// Upstream `Signature.is_consistent_with()`: a call to `self` can safely be served by `other`.
    pub fn is_consistent_with(&self, other: &Signature) -> bool {
        if !self.return_type.is_supertype_of(&other.return_type, true, true) {
            return false;
        }
        for p in &self.params {
            let Some((_, q)) = other.param(&p.name) else { return false };
            if p.kind != q.kind || p.col_type.is_none() != q.col_type.is_none() {
                return false;
            }
            if let (Some(pt), Some(qt)) = (&p.col_type, &q.col_type) {
                if !qt.is_supertype_of(pt, true, true) {
                    return false;
                }
            }
        }
        other.required_params().all(|q| self.param(&q.name).is_some())
    }

    /// Upstream `Signature.as_dict()`.
    pub fn to_json(&self) -> Result<Json> {
        let params = self
            .params
            .iter()
            .map(|p| {
                Ok(json!({
                    "name": p.name,
                    "col_type": p.col_type.as_ref().map(|t| t.to_dict()),
                    "kind": p.kind.name(),
                    "is_batched": p.is_batched,
                    "default": p.default.as_ref().map(|d| crate::expr::serde::to_json(d)).transpose()?,
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(json!({ "return_type": self.return_type.to_dict(), "parameters": params, "is_batched": self.is_batched }))
    }

    pub fn from_json(d: &Json) -> Result<Signature> {
        let bad = || Error::internal(format!("malformed signature: {d}"));
        let return_type = ColumnType::from_dict(d.get("return_type").ok_or_else(bad)?)?;
        let mut params = Vec::new();
        for p in d.get("parameters").and_then(Json::as_array).ok_or_else(bad)? {
            params.push(Param {
                name: p.get("name").and_then(Json::as_str).ok_or_else(bad)?.to_string(),
                kind: p.get("kind").and_then(Json::as_str).and_then(ParamKind::from_name).ok_or_else(bad)?,
                col_type: match p.get("col_type") {
                    None | Some(Json::Null) => None,
                    Some(t) => Some(ColumnType::from_dict(t)?),
                },
                default: match p.get("default") {
                    None | Some(Json::Null) => None,
                    Some(e) => Some(crate::expr::serde::from_json(e)?),
                },
                is_batched: p.get("is_batched").and_then(Json::as_bool).unwrap_or(false),
            });
        }
        Ok(Signature { params, return_type, is_batched: d.get("is_batched").and_then(Json::as_bool).unwrap_or(false) })
    }

    /// A signature produced by `pixeltable.func.Signature._to_core()`: a dict with `parameters` (dicts with `name`,
    /// `kind`, `col_type`, `default` (a Literal expression or None) and `is_batched`), `return_type` and
    /// `is_batched`.
    pub fn from_py(d: &Bound<'_, PyAny>) -> Result<Signature> {
        let d = d.cast::<PyDict>().map_err(|_| Error::internal("signature spec must be a dict"))?;
        let get = |k: &str| -> Result<Bound<'_, PyAny>> {
            d.get_item(k)?.ok_or_else(|| Error::internal(format!("signature spec without {k:?}")))
        };
        let return_type = crate::bindings::types::core_type(&get("return_type")?)
            .ok_or_else(|| Error::internal("signature return type is not a ColumnType"))?;
        let mut params = Vec::new();
        let plist = get("parameters")?;
        let plist = plist.cast::<PyList>().map_err(|_| Error::internal("signature parameters must be a list"))?;
        for p in plist.iter() {
            let p = p.cast::<PyDict>().map_err(|_| Error::internal("parameter spec must be a dict"))?;
            let item = |k: &str| -> Result<Bound<'_, PyAny>> {
                p.get_item(k)?.ok_or_else(|| Error::internal(format!("parameter spec without {k:?}")))
            };
            let col_type = item("col_type")?;
            let default = item("default")?;
            params.push(Param {
                name: item("name")?.extract()?,
                kind: ParamKind::from_name(&item("kind")?.extract::<String>()?)
                    .ok_or_else(|| Error::internal("invalid parameter kind"))?,
                col_type: if col_type.is_none() { None } else { crate::bindings::types::core_type(&col_type) },
                default: if default.is_none() { None } else { crate::bindings::expr::as_expr(&default) },
                is_batched: item("is_batched")?.extract()?,
            });
        }
        Ok(Signature { params, return_type, is_batched: get("is_batched")?.extract()? })
    }
}

// ------------------------------------------------------------------------------------- builtin signatures (text)

impl Signature {
    /// Parses a builtin's signature in upstream's notation: `(name: Type [= default], ..., *args, **kwargs) -> Type`.
    /// `/` and `*` markers work as in Python. Types: `String`, `Int`, `Float`, `Bool`, `Timestamp`, `Date`, `UUID`,
    /// `Binary`, `Json`, `Json[(T1, T2)]`, `Json[(T, ...)]`, `Array`, `Array[<dtype>]`, `Image`, `Video`, `Audio`,
    /// `Document`, `Any` (accepts every type), each optionally followed by `| None`. Defaults: `None`, `True`,
    /// `False`, integers, floats, quoted strings.
    pub fn parse(s: &str) -> Result<Signature> {
        let bad = |what: &str| Error::internal(format!("malformed builtin signature ({what}): {s}"));
        let s = s.trim();
        let open = s.find('(').ok_or_else(|| bad("no '('"))?;
        let mut depth = 0i32;
        let mut close = None;
        for (i, c) in s.char_indices().skip(open) {
            match c {
                '(' | '[' => depth += 1,
                ')' | ']' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.ok_or_else(|| bad("no ')'"))?;
        let params_str = &s[open + 1..close];
        let ret_str = s[close + 1..].trim().strip_prefix("->").ok_or_else(|| bad("no return type"))?.trim();
        let return_type = parse_type(ret_str).ok_or_else(|| bad("return type"))?;
        let mut params: Vec<Param> = Vec::new();
        let mut kw_only = false;
        for raw in split_top_level(params_str) {
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            if raw == "/" {
                for p in params.iter_mut() {
                    p.kind = ParamKind::PositionalOnly;
                }
                continue;
            }
            if raw == "*" {
                kw_only = true;
                continue;
            }
            let (decl, default) = match raw.split_once('=') {
                Some((d, v)) => (d.trim(), Some(parse_default(v.trim()).ok_or_else(|| bad("default"))?)),
                None => (raw, None),
            };
            let (name, type_str) = match decl.split_once(':') {
                Some((n, t)) => (n.trim(), Some(t.trim())),
                None => (decl, None),
            };
            let (name, kind) = if let Some(n) = name.strip_prefix("**") {
                (n, ParamKind::VarKeyword)
            } else if let Some(n) = name.strip_prefix('*') {
                kw_only = true;
                (n, ParamKind::VarPositional)
            } else if kw_only {
                (name, ParamKind::KeywordOnly)
            } else {
                (name, ParamKind::PositionalOrKeyword)
            };
            let col_type = if kind.is_var() {
                None
            } else {
                Some(match type_str {
                    Some(t) => parse_type(t).ok_or_else(|| bad("parameter type"))?,
                    None => ColumnType::invalid(true),
                })
            };
            params.push(Param { name: name.to_string(), kind, col_type, default, is_batched: false });
        }
        Ok(Signature { params, return_type, is_batched: false })
    }

    /// Binds a call's argument expressions, raising upstream's `INVALID_ARGUMENT` error for a binding failure.
    pub fn bind_exprs(&self, args: &[ExprRef], kwargs: &[(String, ExprRef)]) -> Result<Binding> {
        let names: Vec<String> = kwargs.iter().map(|(k, _)| k.clone()).collect();
        self.bind(args.len(), &names).map_err(|cause| {
            let desc = |e: &ExprRef| -> String {
                if matches!(e.as_literal(), Some(crate::value::Value::Null)) { "None".into() } else { e.col_type.repr() }
            };
            let mut provided: Vec<String> = args.iter().map(desc).collect();
            provided.extend(kwargs.iter().map(|(k, v)| format!("{k}={}", desc(v))));
            crate::pxt_err!(InvalidArgument, "{cause}; expected ({}), got ({})", self.params_str(), provided.join(", "))
        })
    }

    /// The argument expression bound to each parameter (None: not given), for conditional return types.
    pub fn bound_exprs(&self, binding: &Binding, args: &[ExprRef], kwargs: &[(String, ExprRef)]) -> Vec<Option<ExprRef>> {
        (0..self.params.len())
            .map(|pi| match binding.get(pi) {
                Some(BoundArg::Arg(i)) => Some(args[*i].clone()),
                Some(BoundArg::Kwarg(i)) => Some(kwargs[*i].1.clone()),
                _ => None,
            })
            .collect()
    }
}

/// True if an argument of type `arg` can be passed to a parameter of type `param` (an `Any` parameter, i.e.
/// `Invalid`, accepts everything).
pub fn arg_matches(param: &ColumnType, arg: &ColumnType) -> bool {
    if param.is_invalid() {
        return true;
    }
    param.is_supertype_of(arg, true, false) || (arg.is_json() && param.is_scalar()) || arg.is_invalid()
}

/// Splits on commas that are not nested in brackets or quotes.
fn split_top_level(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None => match c {
                '\'' | '"' => quote = Some(c),
                '[' | '(' | '{' => depth += 1,
                ']' | ')' | '}' => depth -= 1,
                ',' if depth == 0 => {
                    out.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            },
        }
    }
    out.push(&s[start..]);
    out
}

/// Parses a type in upstream's repr notation (see `Signature::parse`).
pub fn parse_type(s: &str) -> Option<ColumnType> {
    use crate::types::{DType, JsonSchema, JsonSpec, TypeKind};
    let s = s.trim();
    let (base, nullable) = match s.strip_suffix("| None") {
        Some(b) => (b.trim(), true),
        None => (s, false),
    };
    let kind = match base {
        "Any" => TypeKind::Invalid,
        "String" => TypeKind::String,
        "Int" => TypeKind::Int,
        "Float" => TypeKind::Float,
        "Bool" => TypeKind::Bool,
        "Timestamp" => TypeKind::Timestamp,
        "Date" => TypeKind::Date,
        "UUID" => TypeKind::Uuid,
        "Binary" => TypeKind::Binary,
        "Json" => TypeKind::Json(None),
        "Array" => TypeKind::Array { shape: None, dtype: None },
        "Image" => TypeKind::Image { width: None, height: None, mode: None },
        "Video" => TypeKind::Video,
        "Audio" => TypeKind::Audio,
        "Document" => TypeKind::Document { formats: None },
        other => {
            if let Some(inner) = other.strip_prefix("Json[(").and_then(|r| r.strip_suffix(")]")) {
                // a tuple type `Json[(T1, T2)]` or a list type `Json[(T, ...)]`
                let mut items: Vec<ColumnType> = Vec::new();
                let mut variadic = None;
                for part in split_top_level(inner) {
                    let part = part.trim();
                    if part.is_empty() {
                        continue;
                    }
                    if part == "..." {
                        variadic = Some(items.pop()?);
                    } else {
                        items.push(parse_type(part)?);
                    }
                }
                let schema = JsonSchema { spec: JsonSpec::List(items), variadic, optional_keys: Default::default() };
                TypeKind::Json(Some(Box::new(schema)))
            } else {
                let inner = other.strip_prefix("Array[")?.strip_suffix(']')?;
                TypeKind::Array { shape: None, dtype: Some(DType::from_name(inner.trim())?) }
            }
        }
    };
    Some(ColumnType::new(kind, nullable))
}

/// A default value as a literal expression.
fn parse_default(s: &str) -> Option<ExprRef> {
    use crate::value::Value;
    let (v, t) = match s {
        "None" => (Value::Null, ColumnType::invalid(true)),
        "True" => (Value::Bool(true), ColumnType::bool(false)),
        "False" => (Value::Bool(false), ColumnType::bool(false)),
        _ => {
            if let Some(q) = s.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')) {
                (Value::String(q.replace("\\n", "\n").replace("\\t", "\t")), ColumnType::string(false))
            } else if let Ok(i) = s.parse::<i64>() {
                (Value::Int(i), ColumnType::int(false))
            } else {
                (Value::Float(s.parse::<f64>().ok()?), ColumnType::float(false))
            }
        }
    };
    Some(crate::expr::Expr::literal(v, t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(params: &[(&str, ParamKind, bool)]) -> Signature {
        Signature {
            params: params
                .iter()
                .map(|(n, k, has_default)| Param {
                    name: n.to_string(),
                    kind: *k,
                    col_type: if k.is_var() { None } else { Some(ColumnType::int(false)) },
                    default: if *has_default {
                        Some(crate::expr::Expr::literal(crate::value::Value::Int(0), ColumnType::int(false)))
                    } else {
                        None
                    },
                    is_batched: false,
                })
                .collect(),
            return_type: ColumnType::int(false),
            is_batched: false,
        }
    }

    fn kw(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_builtin_signatures() {
        assert!(!crate::functions::builtin_names().is_empty());
        let t = parse_type("Json[(String | None, ...)] | None").unwrap();
        assert_eq!(t.repr(), "Json[(String | None, ...)] | None");
        let s = Signature::parse("(self: String, substr: String, case: Bool = True) -> Bool").unwrap();
        assert_eq!(s.params_str(), "self: pxt.String, substr: pxt.String, case: pxt.Bool");
        assert_eq!(s.bind(1, &[]).unwrap_err(), "missing a required argument: 'substr'");
        let v = Signature::parse("(template: String, *args, **kwargs) -> String").unwrap();
        assert!(v.params[1].kind == ParamKind::VarPositional && v.params[1].col_type.is_none());
    }

    #[test]
    fn bind_messages() {
        use ParamKind::*;
        let pos_only = sig(&[("a", PositionalOnly, false), ("b", PositionalOnly, false)]);
        assert_eq!(pos_only.bind(0, &[]).unwrap_err(), "missing a required argument: 'a'");
        assert_eq!(pos_only.bind(1, &[]).unwrap_err(), "missing a required argument: 'b'");
        assert_eq!(pos_only.bind(0, &kw(&["a"])).unwrap_err(), "missing a required positional-only argument: 'a'");
        let kw_only = sig(&[("a", KeywordOnly, false), ("b", KeywordOnly, false)]);
        assert_eq!(kw_only.bind(1, &[]).unwrap_err(), "too many positional arguments");
        assert_eq!(kw_only.bind(0, &kw(&["x"])).unwrap_err(), "missing a required keyword-only argument: 'a'");
        let pk = sig(&[("a", PositionalOrKeyword, false), ("b", PositionalOrKeyword, false)]);
        assert_eq!(pk.bind(1, &kw(&["a"])).unwrap_err(), "multiple values for argument 'a'");
        assert_eq!(pk.bind(0, &kw(&["a", "x"])).unwrap_err(), "missing a required argument: 'b'");
        let var_pos = sig(&[("args", VarPositional, false)]);
        assert_eq!(var_pos.bind(1, &kw(&["x"])).unwrap_err(), "got an unexpected keyword argument 'x'");
        let b = var_pos.bind(2, &[]).unwrap();
        assert_eq!(b.bound, vec![(0, BoundArg::VarArgs(vec![0, 1]))]);
        let defaults = sig(&[("a", PositionalOrKeyword, false), ("b", PositionalOrKeyword, true)]);
        let b = defaults.bind(0, &kw(&["a"])).unwrap();
        assert_eq!(b.bound, vec![(0, BoundArg::Kwarg(0))]);
    }
}
