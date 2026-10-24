//! Function signatures: parameters (name, type, kind, default) and a return type, binding of call arguments with
//! Python's `inspect.Signature.bind()` semantics, and upstream's argument validation and return-type rules.
//!
//! Builtins declare their signature as text in upstream's notation, e.g.
//! `(self: String, substr: String, case: Bool = True) -> Bool` (see `Signature::parse`). Python UDFs and UDAs can
//! build a `Signature` directly from their type hints and reuse `bind()`, `validate_args()` and
//! `call_return_type()`.

use crate::error::{Error, Result};
use crate::expr::ExprRef;
use crate::types::{ColumnType, DType, TypeKind};
use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamKind {
    PositionalOnly,
    PositionalOrKeyword,
    KeywordOnly,
    VarPositional,
    VarKeyword,
}

impl ParamKind {
    pub fn is_var(self) -> bool {
        matches!(self, ParamKind::VarPositional | ParamKind::VarKeyword)
    }
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    /// for variadic parameters: the element type; `Invalid` accepts any type
    pub col_type: ColumnType,
    pub kind: ParamKind,
    pub default: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub params: Vec<Param>,
    pub return_type: ColumnType,
}

/// Arguments bound to the parameters of a signature.
#[derive(Debug, Clone)]
pub struct Bound<T> {
    /// one entry per parameter; None: not given (the default applies), always None for variadic parameters
    pub args: Vec<Option<T>>,
    pub var_args: Vec<T>,
    pub var_kwargs: Vec<(String, T)>,
}

impl Signature {
    /// Parses `(name: Type [= default], ..., *args, **kwargs) -> Type`. `/` and `*` markers work as in Python.
    /// Types: `String`, `Int`, `Float`, `Bool`, `Timestamp`, `Date`, `UUID`, `Binary`, `Json`, `Array`,
    /// `Array[<dtype>]`, `Image`, `Video`, `Audio`, `Document`, `Any` (accepts every type), each optionally
    /// followed by `| None`. Defaults: `None`, `True`, `False`, integers, floats, quoted strings.
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
            let col_type = match type_str {
                Some(t) => parse_type(t).ok_or_else(|| bad("parameter type"))?,
                None => ColumnType::invalid(true),
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
            params.push(Param { name: name.to_string(), col_type, kind, default });
        }
        Ok(Signature { params, return_type })
    }

    pub fn param(&self, name: &str) -> Option<(usize, &Param)> {
        self.params.iter().enumerate().find(|(_, p)| p.name == name)
    }

    pub fn has_var_args(&self) -> bool {
        self.params.iter().any(|p| p.kind.is_var())
    }

    /// Python's `Signature.bind()`; the error is the `TypeError` message.
    pub fn bind<T: Clone>(&self, args: &[T], kwargs: &[(String, T)]) -> std::result::Result<Bound<T>, String> {
        let n = self.params.len();
        let mut bound: Vec<Option<T>> = vec![None; n];
        let mut var_args = Vec::new();
        let mut var_kwargs = Vec::new();
        let mut pos = self
            .params
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.kind, ParamKind::PositionalOnly | ParamKind::PositionalOrKeyword))
            .map(|(i, _)| i);
        let has_var_pos = self.params.iter().any(|p| p.kind == ParamKind::VarPositional);
        let has_var_kw = self.params.iter().any(|p| p.kind == ParamKind::VarKeyword);
        for a in args {
            match pos.next() {
                Some(i) => bound[i] = Some(a.clone()),
                None if has_var_pos => var_args.push(a.clone()),
                None => return Err("too many positional arguments".into()),
            }
        }
        for (k, v) in kwargs {
            match self.params.iter().position(|p| p.name == *k && !p.kind.is_var()) {
                Some(i) if self.params[i].kind != ParamKind::PositionalOnly => {
                    if bound[i].is_some() {
                        return Err(format!("multiple values for argument {}", crate::error::py_repr(k)));
                    }
                    bound[i] = Some(v.clone());
                }
                _ if has_var_kw => var_kwargs.push((k.clone(), v.clone())),
                Some(_) => {
                    return Err(format!(
                        "got some positional-only arguments passed as keyword arguments: {}",
                        crate::error::py_repr(k)
                    ));
                }
                None => return Err(format!("got an unexpected keyword argument {}", crate::error::py_repr(k))),
            }
        }
        for (i, p) in self.params.iter().enumerate() {
            if bound[i].is_none() && p.default.is_none() && !p.kind.is_var() {
                let argtype = match p.kind {
                    ParamKind::KeywordOnly => " keyword-only",
                    ParamKind::PositionalOnly => " positional-only",
                    _ => "",
                };
                return Err(format!("missing a required{argtype} argument: {}", crate::error::py_repr(&p.name)));
            }
        }
        Ok(Bound { args: bound, var_args, var_kwargs })
    }

    /// Upstream `Signature.params_str()`.
    pub fn params_str(&self) -> String {
        let mut out: Vec<String> = Vec::new();
        let mut pending_pos_only = false;
        let mut needs_kw_marker = true;
        for p in &self.params {
            if p.kind == ParamKind::PositionalOnly {
                pending_pos_only = true;
            } else if pending_pos_only {
                out.push("/".into());
                pending_pos_only = false;
            }
            if p.kind == ParamKind::VarPositional {
                needs_kw_marker = false;
            } else if p.kind == ParamKind::KeywordOnly && needs_kw_marker {
                out.push("*".into());
                needs_kw_marker = false;
            }
            match p.kind {
                ParamKind::VarPositional => out.push(format!("*{}", p.name)),
                ParamKind::VarKeyword => out.push(format!("**{}", p.name)),
                _ => out.push(format!("{}: pxt.{}", p.name, p.col_type.repr())),
            }
        }
        if pending_pos_only {
            out.push("/".into());
        }
        out.join(", ")
    }

    /// Binds expression arguments, raising upstream's `INVALID_ARGUMENT` error for a binding failure.
    pub fn bind_exprs(&self, args: &[ExprRef], kwargs: &[(String, ExprRef)]) -> Result<Bound<ExprRef>> {
        self.bind(args, kwargs).map_err(|cause| {
            let desc = |e: &ExprRef| -> String {
                if matches!(e.as_literal(), Some(Value::Null)) { "None".into() } else { e.col_type.repr() }
            };
            let mut provided: Vec<String> = args.iter().map(desc).collect();
            provided.extend(kwargs.iter().map(|(k, v)| format!("{k}={}", desc(v))));
            crate::pxt_err!(InvalidArgument, "{cause}; expected ({}), got ({})", self.params_str(), provided.join(", "))
        })
    }

    /// Upstream `Signature.validate_args()`: every argument's type must be compatible with its parameter's type
    /// (a non-nullable parameter accepts a nullable argument; a JSON argument is accepted for a scalar parameter).
    pub fn validate_args(&self, bound: &Bound<ExprRef>, fn_name: &str) -> Result<()> {
        for (p, a) in self.params.iter().zip(bound.args.iter()) {
            let Some(a) = a else { continue };
            if !arg_matches(&p.col_type, &a.col_type) {
                crate::bail!(
                    TypeMismatch,
                    "Parameter {} (in function {}): argument type {} does not match parameter type {}",
                    crate::error::py_repr(&p.name),
                    crate::error::py_repr(fn_name),
                    a.col_type.repr(),
                    p.col_type.repr()
                );
            }
        }
        Ok(())
    }

    /// Upstream `Function.call_return_type()`: a non-nullable return type becomes nullable when a nullable argument
    /// is passed to a non-nullable parameter (such calls return None without calling the function).
    pub fn call_return_type(&self, ret: ColumnType, bound: &Bound<ExprRef>) -> ColumnType {
        if ret.nullable {
            return ret;
        }
        for (p, a) in self.params.iter().zip(bound.args.iter()) {
            if let Some(a) = a {
                if a.col_type.nullable && !p.col_type.nullable {
                    return ret.with_nullable(true);
                }
            }
        }
        ret
    }
}

/// True if an argument of type `arg` can be passed to a parameter of type `param`.
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
                let schema = crate::types::JsonSchema {
                    spec: crate::types::JsonSpec::List(items),
                    variadic,
                    optional_keys: Default::default(),
                };
                TypeKind::Json(Some(Box::new(schema)))
            } else {
                let inner = other.strip_prefix("Array[")?.strip_suffix(']')?;
                TypeKind::Array { shape: None, dtype: Some(DType::from_name(inner.trim())?) }
            }
        }
    };
    Some(ColumnType::new(kind, nullable))
}

fn parse_default(s: &str) -> Option<Value> {
    Some(match s {
        "None" => Value::Null,
        "True" => Value::Bool(true),
        "False" => Value::Bool(false),
        _ => {
            if let Some(q) = s.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')) {
                Value::String(q.replace("\\n", "\n").replace("\\t", "\t"))
            } else if let Ok(i) = s.parse::<i64>() {
                Value::Int(i)
            } else {
                Value::Float(s.parse::<f64>().ok()?)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_builtin_signatures_parse() {
        assert!(!crate::functions::builtin_names().is_empty());
        let t = parse_type("Json[(String | None, ...)] | None").unwrap();
        assert_eq!(t.repr(), "Json[(String | None, ...)] | None");
    }

    #[test]
    fn parse_and_bind() {
        let sig = Signature::parse("(self: String, substr: String, case: Bool = True) -> Bool").unwrap();
        assert_eq!(sig.params.len(), 3);
        assert_eq!(sig.params_str(), "self: pxt.String, substr: pxt.String, case: pxt.Bool");
        let b = sig.bind(&[1, 2], &[]).unwrap();
        assert_eq!(b.args, vec![Some(1), Some(2), None]);
        assert_eq!(sig.bind(&[1], &[]).unwrap_err(), "missing a required argument: 'substr'");
        assert_eq!(sig.bind(&[1, 2, 3, 4], &[]).unwrap_err(), "too many positional arguments");
        assert_eq!(sig.bind(&[1, 2], &[("x".into(), 3)]).unwrap_err(), "got an unexpected keyword argument 'x'");
        assert_eq!(sig.bind(&[1, 2], &[("substr".into(), 3)]).unwrap_err(), "multiple values for argument 'substr'");
        let v = Signature::parse("(template: String, *args, **kwargs) -> String").unwrap();
        let b = v.bind(&[1, 2, 3], &[("a".into(), 4)]).unwrap();
        assert_eq!(b.var_args, vec![2, 3]);
        assert_eq!(b.var_kwargs, vec![("a".to_string(), 4)]);
        let k = Signature::parse("(self: Json | None, *, asc: Bool = True) -> Json").unwrap();
        assert_eq!(k.params[1].kind, ParamKind::KeywordOnly);
        assert!(k.params[0].col_type.nullable);
    }
}
