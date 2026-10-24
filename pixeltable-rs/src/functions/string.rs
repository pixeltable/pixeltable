//! `pixeltable.functions.string`: Python `str` semantics on code points.

use fancy_regex::Regex;

use super::{Builtin, CallTypes};
use crate::error::{Error, Result};
use crate::types::ColumnType;
use crate::value::{Json, Value, py_str};

pub const MODULE: &str = "pixeltable.functions.string";

fn s(v: String) -> Result<Value> {
    Ok(Value::String(v))
}

fn b(v: bool) -> Result<Value> {
    Ok(Value::Bool(v))
}

fn json_strs(items: Vec<String>) -> Result<Value> {
    Ok(Value::Json(Json::Array(items.into_iter().map(Json::String).collect())))
}

/// Python's `str.isspace()` for one character.
pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\x1c'..='\x1f')
}

fn is_cased(c: char) -> bool {
    c.is_lowercase() || c.is_uppercase() || is_titlecase(c)
}

fn is_titlecase(c: char) -> bool {
    matches!(c, 'ǅ' | 'ǈ' | 'ǋ' | 'ǲ') || ('\u{1f88}'..='\u{1faf}').contains(&c) && !c.is_lowercase()
}

/// The title-case form of a character (Python uses the Unicode titlecase mapping for capitalize()/title()).
fn to_title(c: char) -> String {
    match c {
        'ǆ' | 'ǅ' | 'Ǆ' => "ǅ".into(),
        'ǉ' | 'ǈ' | 'Ǉ' => "ǈ".into(),
        'ǌ' | 'ǋ' | 'Ǌ' => "ǋ".into(),
        'ǳ' | 'ǲ' | 'Ǳ' => "ǲ".into(),
        c => c.to_uppercase().collect(),
    }
}

/// Resolves Python slice bounds `[start:end]` against a sequence of length `n` (no step).
fn slice_bounds(n: usize, start: Option<i64>, end: Option<i64>) -> (usize, usize) {
    let n = n as i64;
    let norm = |v: i64| -> i64 { if v < 0 { (v + n).max(0) } else { v.min(n) } };
    let st = start.map_or(0, norm);
    let en = end.map_or(n, norm);
    (st as usize, en.max(st) as usize)
}

/// `self[start:stop:step]` on code points.
pub fn py_slice<T: Clone>(items: &[T], start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Result<Vec<T>> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(Error::py_exc("ValueError", "slice step cannot be zero"));
    }
    let n = items.len() as i64;
    let mut out = Vec::new();
    if step > 0 {
        let norm = |v: i64| -> i64 { if v < 0 { (v + n).max(0) } else { v.min(n) } };
        let mut i = start.map_or(0, norm);
        let e = stop.map_or(n, norm);
        while i < e {
            out.push(items[i as usize].clone());
            i += step;
        }
    } else {
        let norm = |v: i64| -> i64 { if v < 0 { (v + n).max(-1) } else { v.min(n - 1) } };
        let mut i = start.map_or(n - 1, norm);
        let e = stop.map_or(-1, norm);
        while i > e {
            out.push(items[i as usize].clone());
            i += step;
        }
    }
    Ok(out)
}

fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

fn find_sub(hay: &[char], needle: &[char], from_right: bool) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    let range = 0..=(hay.len() - needle.len());
    if from_right {
        range.rev().find(|&i| hay[i..i + needle.len()] == *needle)
    } else {
        range.into_iter().find(|&i| hay[i..i + needle.len()] == *needle)
    }
}

/// Python's `str.find()` / `rfind()` with slice bounds; -1 if absent.
fn py_find(s: &str, sub: &str, start: Option<i64>, end: Option<i64>, from_right: bool) -> i64 {
    let cs = chars(s);
    let (st, en) = slice_bounds(cs.len(), start, end);
    if st > cs.len() {
        return -1;
    }
    match find_sub(&cs[st..en], &chars(sub), from_right) {
        Some(i) => (st + i) as i64,
        None => -1,
    }
}

fn pad_str(s: &str, width: i64, fill: &str, how: char) -> Result<String> {
    let fc: Vec<char> = fill.chars().collect();
    if fc.len() != 1 {
        return Err(Error::py_exc("TypeError", "The fill character must be exactly one character long"));
    }
    let fc = fc[0];
    let n = s.chars().count() as i64;
    if width <= n {
        return Ok(s.to_string());
    }
    let marg = width - n;
    let left = match how {
        'l' => 0,
        'r' => marg,
        _ => marg / 2 + (marg & width & 1),
    };
    let right = marg - left;
    let mut out = String::new();
    out.extend(std::iter::repeat_n(fc, left as usize));
    out.push_str(s);
    out.extend(std::iter::repeat_n(fc, right as usize));
    Ok(out)
}

fn strip_chars(s: &str, chars: Option<&str>, left: bool, right: bool) -> String {
    let pred = |c: char| match chars {
        Some(set) => set.contains(c),
        None => is_py_space(c),
    };
    let mut r: &str = s;
    if left {
        r = r.trim_start_matches(pred);
    }
    if right {
        r = r.trim_end_matches(pred);
    }
    r.to_string()
}

/// Python's `str.split()` / `rsplit()`.
pub fn py_split(s: &str, sep: Option<&str>, maxsplit: i64, from_right: bool) -> Result<Vec<String>> {
    let max = if maxsplit < 0 { usize::MAX } else { maxsplit as usize };
    match sep {
        Some("") => Err(Error::py_exc("ValueError", "empty separator")),
        Some(sep) => {
            if from_right {
                let mut parts: Vec<String> = s.rsplitn(max.saturating_add(1), sep).map(str::to_string).collect();
                parts.reverse();
                Ok(parts)
            } else {
                Ok(s.splitn(max.saturating_add(1), sep).map(str::to_string).collect())
            }
        }
        None => {
            let cs: Vec<char> = s.chars().collect();
            let mut out: Vec<String> = Vec::new();
            if !from_right {
                let mut i = 0;
                while i < cs.len() {
                    while i < cs.len() && is_py_space(cs[i]) {
                        i += 1;
                    }
                    if i >= cs.len() {
                        break;
                    }
                    if out.len() == max {
                        out.push(cs[i..].iter().collect());
                        break;
                    }
                    let st = i;
                    while i < cs.len() && !is_py_space(cs[i]) {
                        i += 1;
                    }
                    out.push(cs[st..i].iter().collect());
                }
            } else {
                let mut i = cs.len() as i64 - 1;
                while i >= 0 {
                    while i >= 0 && is_py_space(cs[i as usize]) {
                        i -= 1;
                    }
                    if i < 0 {
                        break;
                    }
                    if out.len() == max {
                        out.push(cs[..=i as usize].iter().collect());
                        break;
                    }
                    let en = i as usize + 1;
                    while i >= 0 && !is_py_space(cs[i as usize]) {
                        i -= 1;
                    }
                    out.push(cs[(i + 1) as usize..en].iter().collect());
                }
                out.reverse();
            }
            Ok(out)
        }
    }
}

fn splitlines(s: &str, keepends: bool) -> Vec<String> {
    let is_break = |c: char| matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}');
    let cs: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut st = 0;
    let mut i = 0;
    while i < cs.len() {
        if is_break(cs[i]) {
            let mut eol = i + 1;
            if cs[i] == '\r' && i + 1 < cs.len() && cs[i + 1] == '\n' {
                eol += 1;
            }
            let end = if keepends { eol } else { i };
            out.push(cs[st..end].iter().collect());
            st = eol;
            i = eol;
        } else {
            i += 1;
        }
    }
    if st < cs.len() {
        out.push(cs[st..].iter().collect());
    }
    out
}

fn title(s: &str) -> String {
    let mut out = String::new();
    let mut prev_cased = false;
    for c in s.chars() {
        if is_cased(c) {
            if prev_cased {
                out.extend(c.to_lowercase());
            } else {
                out.push_str(&to_title(c));
            }
            prev_cased = true;
        } else {
            out.push(c);
            prev_cased = false;
        }
    }
    out
}

fn capitalize(s: &str) -> String {
    let mut it = s.chars();
    match it.next() {
        None => String::new(),
        Some(first) => {
            let mut out = to_title(first);
            for c in it {
                out.extend(c.to_lowercase());
            }
            out
        }
    }
}

fn swapcase(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_uppercase() {
            out.extend(c.to_lowercase());
        } else if c.is_lowercase() {
            out.extend(c.to_uppercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn is_lower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_uppercase() || is_titlecase(c) {
            return false;
        }
        if c.is_lowercase() {
            cased = true;
        }
    }
    cased
}

fn is_upper(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if c.is_lowercase() || is_titlecase(c) {
            return false;
        }
        if c.is_uppercase() {
            cased = true;
        }
    }
    cased
}

fn is_title(s: &str) -> bool {
    let mut cased = false;
    let mut prev_cased = false;
    for c in s.chars() {
        if c.is_uppercase() || is_titlecase(c) {
            if prev_cased {
                return false;
            }
            prev_cased = true;
            cased = true;
        } else if c.is_lowercase() {
            if !prev_cased {
                return false;
            }
            prev_cased = true;
            cased = true;
        } else {
            prev_cased = false;
        }
    }
    cased
}

fn is_identifier(s: &str) -> bool {
    let mut it = s.chars();
    match it.next() {
        Some(c) if c == '_' || c.is_alphabetic() => it.all(|c| c == '_' || c.is_alphanumeric()),
        _ => false,
    }
}

fn zfill(s: &str, width: i64) -> String {
    let n = s.chars().count() as i64;
    if width <= n {
        return s.to_string();
    }
    let fill = "0".repeat((width - n) as usize);
    match s.chars().next() {
        Some(c @ ('+' | '-')) => format!("{c}{fill}{}", &s[1..]),
        _ => format!("{fill}{s}"),
    }
}

// ------------------------------------------------------------------------------------------------ regex

/// Compiles a Python `re` pattern with `re` flags (IGNORECASE=2, MULTILINE=8, DOTALL=16, VERBOSE=64).
pub fn py_regex(pattern: &str, flags: i64) -> Result<Regex> {
    let mut inline = String::new();
    if flags & 2 != 0 {
        inline.push('i');
    }
    if flags & 8 != 0 {
        inline.push('m');
    }
    if flags & 16 != 0 {
        inline.push('s');
    }
    if flags & 64 != 0 {
        inline.push('x');
    }
    let full = if inline.is_empty() { pattern.to_string() } else { format!("(?{inline}){pattern}") };
    Regex::new(&full).map_err(|e| Error::py_exc("re.error", format!("{e}")))
}

fn re_err(e: fancy_regex::Error) -> Error {
    Error::py_exc("RuntimeError", format!("regular expression evaluation failed: {e}"))
}

fn findall(s: &str, pattern: &str, flags: i64) -> Result<Vec<Json>> {
    let re = py_regex(pattern, flags)?;
    let mut out = Vec::new();
    for caps in re.captures_iter(s) {
        let caps = caps.map_err(re_err)?;
        let n = caps.len();
        if n == 1 {
            out.push(Json::String(caps.get(0).map_or("", |m| m.as_str()).to_string()));
        } else if n == 2 {
            out.push(Json::String(caps.get(1).map_or("", |m| m.as_str()).to_string()));
        } else {
            out.push(Json::Array((1..n).map(|i| Json::String(caps.get(i).map_or("", |m| m.as_str()).to_string())).collect()));
        }
    }
    Ok(out)
}

/// Python `re.sub()` replacement syntax (`\1`, `\g<1>`, `\g<name>`) converted to the regex crate's (`${1}`).
fn convert_repl(repl: &str) -> String {
    let mut out = String::new();
    let cs: Vec<char> = repl.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c == '$' {
            out.push_str("$$");
        } else if c == '\\' && i + 1 < cs.len() {
            let d = cs[i + 1];
            if d.is_ascii_digit() {
                let mut j = i + 1;
                let mut num = String::new();
                while j < cs.len() && cs[j].is_ascii_digit() && num.len() < 2 {
                    num.push(cs[j]);
                    j += 1;
                }
                out.push_str(&format!("${{{num}}}"));
                i = j;
                continue;
            } else if d == 'g' && i + 2 < cs.len() && cs[i + 2] == '<' {
                if let Some(close) = cs[i + 3..].iter().position(|&x| x == '>') {
                    let name: String = cs[i + 3..i + 3 + close].iter().collect();
                    out.push_str(&format!("${{{name}}}"));
                    i += 4 + close;
                    continue;
                }
            } else {
                let esc = match d {
                    'n' => Some('\n'),
                    't' => Some('\t'),
                    'r' => Some('\r'),
                    '\\' => Some('\\'),
                    _ => None,
                };
                if let Some(e) = esc {
                    out.push(e);
                    i += 2;
                    continue;
                }
            }
            out.push(c);
        } else {
            out.push(c);
        }
        i += 1;
    }
    out
}

// ------------------------------------------------------------------------------------------------ format

/// Python's `str.format()` for the common cases: `{}`, `{0}`, `{name}`, `{0[1]}`, `{x.attr}` (dict keys),
/// conversions `!s` / `!r`, and format specs (fill, align, sign, width, grouping, precision, type).
pub fn py_format(template: &str, args: &[Value], kwargs: &[(String, Value)]) -> Result<String> {
    let cs: Vec<char> = template.chars().collect();
    let mut out = String::new();
    let mut auto_idx = 0usize;
    let mut i = 0;
    let fmt_err = |m: &str| Error::py_exc("ValueError", m.to_string());
    while i < cs.len() {
        let c = cs[i];
        if c == '{' {
            if i + 1 < cs.len() && cs[i + 1] == '{' {
                out.push('{');
                i += 2;
                continue;
            }
            let close = cs[i..].iter().position(|&x| x == '}').ok_or_else(|| fmt_err("expected '}' before end of string"))?;
            let field: String = cs[i + 1..i + close].iter().collect();
            i += close + 1;
            let (field, spec) = match field.split_once(':') {
                Some((f, s)) => (f.to_string(), s.to_string()),
                None => (field, String::new()),
            };
            let (field, conv) = match field.split_once('!') {
                Some((f, c)) => (f.to_string(), Some(c.to_string())),
                None => (field, None),
            };
            // first component: index or name; then .attr / [key] accesses
            let name_end = field.find(['.', '[']).unwrap_or(field.len());
            let (head, mut rest) = field.split_at(name_end);
            let mut val: Value = if head.is_empty() {
                let v = args.get(auto_idx).cloned().ok_or_else(|| Error::py_exc("IndexError", format!("Replacement index {auto_idx} out of range for positional args tuple")))?;
                auto_idx += 1;
                v
            } else if let Ok(idx) = head.parse::<usize>() {
                args.get(idx).cloned().ok_or_else(|| Error::py_exc("IndexError", format!("Replacement index {idx} out of range for positional args tuple")))?
            } else {
                kwargs
                    .iter()
                    .find(|(k, _)| k == head)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| Error::py_exc("KeyError", crate::error::py_repr(head)))?
            };
            while !rest.is_empty() {
                let (key, remaining) = if let Some(r) = rest.strip_prefix('.') {
                    let end = r.find(['.', '[']).unwrap_or(r.len());
                    (r[..end].to_string(), &r[end..])
                } else if let Some(r) = rest.strip_prefix('[') {
                    let end = r.find(']').ok_or_else(|| fmt_err("Missing ']' in format string"))?;
                    (r[..end].to_string(), &r[end + 1..])
                } else {
                    return Err(fmt_err("Only '.' or '[' may follow ']' in format field specifier"));
                };
                rest = remaining;
                let j = crate::value::value_to_json(&val)?;
                let next = match (&j, key.parse::<usize>()) {
                    (Json::Array(a), Ok(idx)) => a.get(idx).cloned().ok_or_else(|| Error::py_exc("IndexError", "list index out of range"))?,
                    (Json::Object(m), _) => m.get(&key).cloned().ok_or_else(|| Error::py_exc("KeyError", crate::error::py_repr(&key)))?,
                    _ => return Err(Error::py_exc("TypeError", format!("'{}' object is not subscriptable", val.py_type_name()))),
                };
                val = Value::Json(next).json_scalar();
            }
            let val = match conv.as_deref() {
                Some("r") => Value::String(crate::value::py_repr_value(&val)),
                Some("s") | Some("a") => Value::String(py_str(&val)),
                Some(_) => return Err(fmt_err("Unknown conversion specifier")),
                None => val,
            };
            out.push_str(&apply_spec(&val, &spec)?);
        } else if c == '}' {
            if i + 1 < cs.len() && cs[i + 1] == '}' {
                out.push('}');
                i += 2;
                continue;
            }
            return Err(fmt_err("Single '}' encountered in format string"));
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}

/// Applies a format spec `[[fill]align][sign][#][0][width][,][.precision][type]` to a value.
fn apply_spec(val: &Value, spec: &str) -> Result<String> {
    if spec.is_empty() {
        return Ok(match val {
            Value::String(s) => s.clone(),
            v => py_str(v),
        });
    }
    let cs: Vec<char> = spec.chars().collect();
    let mut i = 0;
    let mut fill = ' ';
    let mut align: Option<char> = None;
    if cs.len() >= 2 && matches!(cs[1], '<' | '>' | '^' | '=') {
        fill = cs[0];
        align = Some(cs[1]);
        i = 2;
    } else if !cs.is_empty() && matches!(cs[0], '<' | '>' | '^' | '=') {
        align = Some(cs[0]);
        i = 1;
    }
    let mut sign = '-';
    if i < cs.len() && matches!(cs[i], '+' | '-' | ' ') {
        sign = cs[i];
        i += 1;
    }
    if i < cs.len() && cs[i] == '#' {
        i += 1;
    }
    if i < cs.len() && cs[i] == '0' {
        if align.is_none() {
            fill = '0';
            align = Some('=');
        }
        i += 1;
    }
    let mut width = String::new();
    while i < cs.len() && cs[i].is_ascii_digit() {
        width.push(cs[i]);
        i += 1;
    }
    let mut grouping: Option<char> = None;
    if i < cs.len() && (cs[i] == ',' || cs[i] == '_') {
        grouping = Some(cs[i]);
        i += 1;
    }
    let mut precision: Option<usize> = None;
    if i < cs.len() && cs[i] == '.' {
        i += 1;
        let mut p = String::new();
        while i < cs.len() && cs[i].is_ascii_digit() {
            p.push(cs[i]);
            i += 1;
        }
        precision = p.parse().ok();
    }
    let ty: Option<char> = cs.get(i).copied();
    let v = val.json_scalar();
    let (mut body, negative, numeric) = match (&v, ty) {
        (Value::String(s), None | Some('s')) => {
            let s = match precision {
                Some(p) => s.chars().take(p).collect(),
                None => s.clone(),
            };
            (s, false, false)
        }
        (Value::Int(n), None | Some('d') | Some('n')) => (n.unsigned_abs().to_string(), *n < 0, true),
        (Value::Bool(bv), None | Some('d')) => ((*bv as i64).to_string(), false, true),
        (Value::Int(_) | Value::Float(_) | Value::Bool(_), Some(t @ ('f' | 'F' | 'e' | 'E' | '%' | 'g' | 'G'))) => {
            let x = v.as_f64().unwrap_or(f64::NAN);
            let p = precision.unwrap_or(6);
            let s = match t {
                'f' | 'F' => format!("{:.*}", p, x.abs()),
                '%' => format!("{:.*}%", p, x.abs() * 100.0),
                'e' | 'E' => {
                    let s = format!("{:.*e}", p, x.abs());
                    let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
                    let e: i32 = e.parse().unwrap_or(0);
                    let s = format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
                    if t == 'E' { s.to_uppercase() } else { s }
                }
                _ => {
                    let r = crate::value::py_float_repr(super::math::py_round(x.abs(), p.max(1) as i64));
                    r.trim_end_matches(".0").to_string()
                }
            };
            (s, x < 0.0, true)
        }
        (Value::Float(x), None) => {
            let s = match precision {
                Some(p) => {
                    let r = format!("{:.*e}", p.saturating_sub(1), x.abs());
                    r.parse::<f64>().map(crate::value::py_float_repr).unwrap_or(r)
                }
                None => crate::value::py_float_repr(x.abs()),
            };
            (s, *x < 0.0, true)
        }
        (Value::Int(n), Some('x')) => (format!("{:x}", n.unsigned_abs()), *n < 0, true),
        (Value::Int(n), Some('X')) => (format!("{:X}", n.unsigned_abs()), *n < 0, true),
        (Value::Int(n), Some('o')) => (format!("{:o}", n.unsigned_abs()), *n < 0, true),
        (Value::Int(n), Some('b')) => (format!("{:b}", n.unsigned_abs()), *n < 0, true),
        (other, None) => (py_str(other), false, false),
        (other, Some(t)) => {
            return Err(Error::py_exc(
                "ValueError",
                format!("Unknown format code '{t}' for object of type '{}'", other.py_type_name()),
            ));
        }
    };
    if let (Some(g), true) = (grouping, numeric) {
        let (int_part, frac) = match body.find('.') {
            Some(p) => (body[..p].to_string(), body[p..].to_string()),
            None => (body.clone(), String::new()),
        };
        let digits: Vec<char> = int_part.chars().collect();
        let mut grouped = String::new();
        for (k, d) in digits.iter().enumerate() {
            if k > 0 && (digits.len() - k) % 3 == 0 {
                grouped.push(g);
            }
            grouped.push(*d);
        }
        body = grouped + &frac;
    }
    let sign_str = if numeric {
        if negative {
            "-"
        } else if sign == '+' {
            "+"
        } else if sign == ' ' {
            " "
        } else {
            ""
        }
    } else {
        ""
    };
    let width: usize = width.parse().unwrap_or(0);
    let content_len = sign_str.chars().count() + body.chars().count();
    let align = align.unwrap_or(if numeric { '>' } else { '<' });
    if content_len >= width {
        return Ok(format!("{sign_str}{body}"));
    }
    let pad = width - content_len;
    let fills = |n: usize| -> String { std::iter::repeat_n(fill, n).collect() };
    Ok(match align {
        '<' => format!("{sign_str}{body}{}", fills(pad)),
        '>' => format!("{}{sign_str}{body}", fills(pad)),
        '=' => format!("{sign_str}{}{body}", fills(pad)),
        _ => format!("{}{sign_str}{body}{}", fills(pad / 2), fills(pad - pad / 2)),
    })
}

// ------------------------------------------------------------------------------------------------ textwrap

/// Python's `textwrap` whitespace (`_whitespace`).
const TW_WHITESPACE: &str = "\t\n\x0b\x0c\r ";

/// `textwrap.TextWrapper.wordsep_re` (with `break_on_hyphens`): whitespace runs, em-dashes between words, and words
/// split after the hyphens of hyphenated words.
fn wordsep_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        let ws = r"[\t\n\x0b\x0c\r ]";
        let nws = r"[^\t\n\x0b\x0c\r ]";
        let wp = r#"[\w!"'&.,?]"#;
        let lt = r"[^\d\W]";
        let pattern = format!(
            r"({ws}+|(?<={wp})-{{2,}}(?=\w)|{nws}+?(?:-(?:(?<={lt}{{2}}-)|(?<={lt}-{lt}-))(?={lt}-?{lt})|(?={ws}|\z)|(?<={wp})(?=-{{2,}}\w)))"
        );
        Regex::new(&pattern).unwrap_or_else(|e| panic!("textwrap pattern: {e}"))
    })
}

/// `textwrap.wrap(text, width)` with the default options: tabs expanded, whitespace replaced by spaces, words
/// split at hyphens, long words broken (filling the rest of the current line), whitespace dropped at line ends.
fn wrap_text(text: &str, width: i64) -> Result<Vec<String>> {
    if width <= 0 {
        return Err(Error::py_exc("ValueError", format!("invalid width {width} (must be > 0)")));
    }
    let width = width as usize;
    // expandtabs(8), then each whitespace character becomes a space
    let mut expanded = String::with_capacity(text.len());
    let mut col = 0usize;
    for c in text.chars() {
        match c {
            '\t' => {
                let n = 8 - col % 8;
                expanded.extend(std::iter::repeat_n(' ', n));
                col += n;
            }
            '\n' | '\r' => {
                expanded.push(' ');
                col = 0;
            }
            c if TW_WHITESPACE.contains(c) => {
                expanded.push(' ');
                col += 1;
            }
            c => {
                expanded.push(c);
                col += 1;
            }
        }
    }
    // the chunks (re.split with the capturing pattern, empty pieces dropped)
    let mut chunks: Vec<String> = Vec::new();
    let mut last = 0;
    for m in wordsep_re().find_iter(&expanded) {
        let m = m.map_err(re_err)?;
        if m.start() > last {
            chunks.push(expanded[last..m.start()].to_string());
        }
        if !m.as_str().is_empty() {
            chunks.push(m.as_str().to_string());
        }
        last = m.end();
    }
    if last < expanded.len() {
        chunks.push(expanded[last..].to_string());
    }
    let len = |c: &str| c.chars().count();
    let is_blank = |c: &str| c.chars().all(is_py_space);
    chunks.reverse();
    let mut lines = Vec::new();
    while !chunks.is_empty() {
        let mut cur_line: Vec<String> = Vec::new();
        let mut cur_len = 0usize;
        if !lines.is_empty() && chunks.last().is_some_and(|c| is_blank(c)) {
            chunks.pop();
        }
        while let Some(c) = chunks.last() {
            let l = len(c);
            if cur_len + l > width {
                break;
            }
            cur_len += l;
            cur_line.push(chunks.pop().expect("chunk"));
        }
        if chunks.last().is_some_and(|c| len(c) > width) {
            // `_handle_long_word()`: the head of the chunk fills the line, preferably up to a hyphen
            let space_left = width.saturating_sub(cur_len);
            if space_left > 0 {
                let chunk = chunks.pop().expect("chunk");
                let chars: Vec<char> = chunk.chars().collect();
                let mut end = space_left;
                if chars.len() > space_left {
                    if let Some(h) = chars[..space_left].iter().rposition(|c| *c == '-') {
                        if h > 0 && chars[..h].iter().any(|c| *c != '-') {
                            end = h + 1;
                        }
                    }
                }
                let end = end.min(chars.len());
                cur_line.push(chars[..end].iter().collect());
                chunks.push(chars[end..].iter().collect());
            } else if cur_line.is_empty() {
                cur_line.push(chunks.pop().expect("chunk"));
            }
        }
        if cur_line.last().is_some_and(|c| is_blank(c)) {
            cur_line.pop();
        }
        if !cur_line.is_empty() {
            lines.push(cur_line.concat());
        }
    }
    Ok(lines)
}

fn format_ret(_: &CallTypes<'_>) -> Result<ColumnType> {
    Ok(ColumnType::string(false))
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin::new("capitalize", "(self: String) -> String", |a| s(capitalize(a.str(0)?))).method().sql(),
    Builtin::new("casefold", "(self: String) -> String", |a| s(a.str(0)?.to_lowercase().replace('ß', "ss"))).method(),
    Builtin::new("center", "(self: String, width: Int, fillchar: String = ' ') -> String", |a| {
        s(pad_str(a.str(0)?, a.int(1)?, a.str(2)?, 'c')?)
    })
    .method(),
    Builtin::new("contains", "(self: String, substr: String, case: Bool = True) -> Bool", |a| {
        let (st, sub) = (a.str(0)?, a.str(1)?);
        b(if a.bool(2)? { st.contains(sub) } else { st.to_lowercase().contains(&sub.to_lowercase()) })
    })
    .method()
    .sql(),
    Builtin::new("contains_re", "(self: String, pattern: String, flags: Int = 0) -> Bool", |a| {
        b(py_regex(a.str(1)?, a.int(2)?)?.is_match(a.str(0)?).map_err(re_err)?)
    })
    .method(),
    Builtin::new("count", "(self: String, pattern: String, flags: Int = 0) -> Int", |a| {
        Ok(Value::Int(findall(a.str(0)?, a.str(1)?, a.int(2)?)?.len() as i64))
    })
    .method(),
    Builtin::new("endswith", "(self: String, substr: String) -> Bool", |a| b(a.str(0)?.ends_with(a.str(1)?))).method().sql(),
    Builtin::new("fill", "(self: String, width: Int, **kwargs) -> String", |a| s(wrap_text(a.str(0)?, a.int(1)?)?.join("\n"))).method(),
    Builtin::new("find", "(self: String, substr: String, start: Int = 0, end: Int | None = None) -> Int", |a| {
        Ok(Value::Int(py_find(a.str(0)?, a.str(1)?, Some(a.int(2)?), a.opt_int(3)?, false)))
    })
    .method()
    .sql(),
    Builtin::new("findall", "(self: String, pattern: String, flags: Int = 0) -> Json[(Json, ...)]", |a| {
        Ok(Value::Json(Json::Array(findall(a.str(0)?, a.str(1)?, a.int(2)?)?)))
    })
    .method(),
    Builtin::new("format", "(self: String, *args, **kwargs) -> String", |a| s(py_format(a.str(0)?, a.var_args, a.var_kwargs)?))
        .method()
        .ret(format_ret),
    Builtin::new("fullmatch", "(self: String, pattern: String, case: Bool = True, flags: Int = 0) -> Bool", |a| {
        let flags = if a.bool(2)? { a.int(3)? } else { a.int(3)? | 2 };
        b(py_regex(&format!("(?:{})\\z", a.str(1)?), flags)?
            .find(a.str(0)?)
            .map_err(re_err)?
            .is_some_and(|m| m.start() == 0))
    })
    .method(),
    Builtin::new("index", "(self: String, substr: String, start: Int = 0, end: Int | None = None) -> Int", |a| {
        match py_find(a.str(0)?, a.str(1)?, Some(a.int(2)?), a.opt_int(3)?, false) {
            -1 => Err(Error::py_exc("ValueError", "substring not found")),
            i => Ok(Value::Int(i)),
        }
    })
    .method(),
    Builtin::new("isalnum", "(self: String) -> Bool", |a| {
        let st = a.str(0)?;
        b(!st.is_empty() && st.chars().all(char::is_alphanumeric))
    })
    .method(),
    Builtin::new("isalpha", "(self: String) -> Bool", |a| {
        let st = a.str(0)?;
        b(!st.is_empty() && st.chars().all(char::is_alphabetic))
    })
    .method(),
    Builtin::new("isascii", "(self: String) -> Bool", |a| b(a.str(0)?.is_ascii())).method(),
    Builtin::new("isdecimal", "(self: String) -> Bool", |a| {
        let st = a.str(0)?;
        b(!st.is_empty() && st.chars().all(|c| c.to_digit(10).is_some() || (c.is_numeric() && !c.is_ascii() && c.to_digit(10).is_some())))
    })
    .method(),
    Builtin::new("isdigit", "(self: String) -> Bool", |a| {
        let st = a.str(0)?;
        b(!st.is_empty() && st.chars().all(|c| c.is_ascii_digit() || matches!(c, '²' | '³' | '¹' | '⁰'..='⁹' | '₀'..='₉')))
    })
    .method(),
    Builtin::new("isidentifier", "(self: String) -> Bool", |a| b(is_identifier(a.str(0)?))).method(),
    Builtin::new("islower", "(self: String) -> Bool", |a| b(is_lower(a.str(0)?))).method(),
    Builtin::new("isnumeric", "(self: String) -> Bool", |a| {
        let st = a.str(0)?;
        b(!st.is_empty() && st.chars().all(char::is_numeric))
    })
    .method(),
    Builtin::new("isupper", "(self: String) -> Bool", |a| b(is_upper(a.str(0)?))).method(),
    Builtin::new("istitle", "(self: String) -> Bool", |a| b(is_title(a.str(0)?))).method(),
    Builtin::new("isspace", "(self: String) -> Bool", |a| {
        let st = a.str(0)?;
        b(!st.is_empty() && st.chars().all(is_py_space))
    })
    .method(),
    Builtin::new("join", "(sep: String, elements: Json[(Json, ...)]) -> String", |a| {
        let sep = a.str(0)?;
        let items = match a.json(1)? {
            Some(Json::Array(items)) => items,
            _ => return Err(Error::py_exc("TypeError", "can only join an iterable")),
        };
        let mut parts = Vec::with_capacity(items.len());
        for (i, it) in items.iter().enumerate() {
            match it {
                Json::String(st) => parts.push(st.clone()),
                other => {
                    return Err(Error::py_exc(
                        "TypeError",
                        format!("sequence item {i}: expected str instance, {} found", Value::Json(other.clone()).py_type_name()),
                    ));
                }
            }
        }
        s(parts.join(sep))
    }),
    Builtin::new("len", "(self: String) -> Int", |a| Ok(Value::Int(a.str(0)?.chars().count() as i64))).method().sql(),
    Builtin::new("ljust", "(self: String, width: Int, fillchar: String = ' ') -> String", |a| {
        s(pad_str(a.str(0)?, a.int(1)?, a.str(2)?, 'l')?)
    })
    .method(),
    Builtin::new("lower", "(self: String) -> String", |a| s(a.str(0)?.to_lowercase())).method().sql(),
    Builtin::new("lstrip", "(self: String, chars: String | None = None) -> String", |a| {
        s(strip_chars(a.str(0)?, a.opt_str(1)?, true, false))
    })
    .method()
    .sql(),
    Builtin::new("match", "(self: String, pattern: String, case: Bool = True, flags: Int = 0) -> Bool", |a| {
        let flags = if a.bool(2)? { a.int(3)? } else { a.int(3)? | 2 };
        b(py_regex(&format!("\\A(?:{})", a.str(1)?), flags)?.is_match(a.str(0)?).map_err(re_err)?)
    })
    .method(),
    Builtin::new("normalize", "(self: String, form: String) -> String", |a| {
        use unicode_normalization::UnicodeNormalization;
        let st = a.str(0)?;
        s(match a.str(1)? {
            "NFC" => st.nfc().collect(),
            "NFD" => st.nfd().collect(),
            "NFKC" => st.nfkc().collect(),
            "NFKD" => st.nfkd().collect(),
            _ => return Err(Error::py_exc("ValueError", "invalid normalization form")),
        })
    })
    .method(),
    Builtin::new("pad", "(self: String, width: Int, side: String = 'left', fillchar: String = ' ') -> String", |a| {
        let how = match a.str(2)? {
            "left" => 'l',
            "right" => 'r',
            "both" => 'c',
            side => return Err(Error::py_exc("ValueError", format!("Invalid side: {side}"))),
        };
        s(pad_str(a.str(0)?, a.int(1)?, a.str(3)?, how)?)
    })
    .method(),
    Builtin::new("partition", "(self: String, sep: String = ' ') -> Json[(String, String, String)]", |a| {
        let (st, sep) = (a.str(0)?, a.str(1)?);
        json_strs(match st.find(sep) {
            None => vec![st.to_string(), String::new(), String::new()],
            Some(i) => vec![st[..i].to_string(), sep.to_string(), st[i + sep.len()..].to_string()],
        })
    })
    .method(),
    Builtin::new("removeprefix", "(self: String, prefix: String) -> String", |a| {
        let (st, p) = (a.str(0)?, a.str(1)?);
        s(st.strip_prefix(p).unwrap_or(st).to_string())
    })
    .method()
    .sql(),
    Builtin::new("removesuffix", "(self: String, suffix: String) -> String", |a| {
        let (st, p) = (a.str(0)?, a.str(1)?);
        s(st.strip_suffix(p).unwrap_or(st).to_string())
    })
    .method()
    .sql(),
    Builtin::new("repeat", "(self: String, n: Int) -> String", |a| s(a.str(0)?.repeat(a.int(1)?.max(0) as usize))).method().sql(),
    Builtin::new("replace", "(self: String, substr: String, repl: String, n: Int | None = None) -> String", |a| {
        let (st, sub, repl) = (a.str(0)?, a.str(1)?, a.str(2)?);
        s(match a.opt_int(3)?.filter(|n| *n > 0) {
            None => st.replace(sub, repl),
            Some(n) => st.replacen(sub, repl, n as usize),
        })
    })
    .method()
    .sql(),
    Builtin::new("replace_re", "(self: String, pattern: String, repl: String, n: Int | None = None, flags: Int = 0) -> String", |a| {
        let re = py_regex(a.str(1)?, a.int(4)?)?;
        let repl = convert_repl(a.str(2)?);
        let limit = a.opt_int(3)?.unwrap_or(0).max(0) as usize;
        let st = a.str(0)?;
        let mut out = String::new();
        let mut last = 0;
        let mut count = 0;
        for caps in re.captures_iter(st) {
            let caps = caps.map_err(re_err)?;
            if limit > 0 && count >= limit {
                break;
            }
            let m = caps.get(0).expect("group 0");
            out.push_str(&st[last..m.start()]);
            caps.expand(&repl, &mut out);
            last = m.end();
            count += 1;
        }
        out.push_str(&st[last..]);
        s(out)
    })
    .method(),
    Builtin::new("reverse", "(self: String) -> String", |a| s(a.str(0)?.chars().rev().collect())).method().sql(),
    Builtin::new("rfind", "(self: String, substr: String, start: Int | None = 0, end: Int | None = None) -> Int", |a| {
        Ok(Value::Int(py_find(a.str(0)?, a.str(1)?, a.opt_int(2)?, a.opt_int(3)?, true)))
    })
    .method(),
    Builtin::new("rindex", "(self: String, substr: String, start: Int | None = 0, end: Int | None = None) -> Int", |a| {
        match py_find(a.str(0)?, a.str(1)?, a.opt_int(2)?, a.opt_int(3)?, true) {
            -1 => Err(Error::py_exc("ValueError", "substring not found")),
            i => Ok(Value::Int(i)),
        }
    })
    .method(),
    Builtin::new("rjust", "(self: String, width: Int, fillchar: String = ' ') -> String", |a| {
        s(pad_str(a.str(0)?, a.int(1)?, a.str(2)?, 'r')?)
    })
    .method(),
    Builtin::new("rpartition", "(self: String, sep: String = ' ') -> Json[(String, String, String)]", |a| {
        let (st, sep) = (a.str(0)?, a.str(1)?);
        json_strs(match st.rfind(sep) {
            None => vec![st.to_string(), String::new(), String::new()],
            Some(i) => vec![st[..i].to_string(), sep.to_string(), st[i + sep.len()..].to_string()],
        })
    })
    .method(),
    Builtin::new("rsplit", "(self: String, sep: String | None = None, maxsplit: Int = -1) -> Json[(String, ...)]", |a| {
        json_strs(py_split(a.str(0)?, a.opt_str(1)?, a.int(2)?, true)?)
    })
    .method(),
    Builtin::new("rstrip", "(self: String, chars: String | None = None) -> String", |a| {
        s(strip_chars(a.str(0)?, a.opt_str(1)?, false, true))
    })
    .method()
    .sql(),
    Builtin::new("slice", "(self: String, start: Int | None = None, stop: Int | None = None, step: Int | None = None) -> String", |a| {
        s(py_slice(&chars(a.str(0)?), a.opt_int(1)?, a.opt_int(2)?, a.opt_int(3)?)?.into_iter().collect())
    })
    .method()
    .sql(),
    Builtin::new(
        "slice_replace",
        "(self: String, start: Int | None = None, stop: Int | None = None, repl: String | None = None) -> String",
        |a| {
            let cs = chars(a.str(0)?);
            let (st, en) = slice_bounds(cs.len(), a.opt_int(1)?, a.opt_int(2)?);
            let head: String = cs[..st].iter().collect();
            let tail: String = cs[en.max(st)..].iter().collect();
            s(format!("{head}{}{tail}", a.opt_str(3)?.unwrap_or("")))
        },
    )
    .method(),
    Builtin::new("split", "(self: String, sep: String | None = None, maxsplit: Int = -1) -> Json[(String, ...)]", |a| {
        json_strs(py_split(a.str(0)?, a.opt_str(1)?, a.int(2)?, false)?)
    })
    .method(),
    Builtin::new("splitlines", "(self: String, keepends: Bool = False) -> Json[(String, ...)]", |a| {
        json_strs(splitlines(a.str(0)?, a.bool(1)?))
    })
    .method(),
    Builtin::new("startswith", "(self: String, substr: String) -> Bool", |a| b(a.str(0)?.starts_with(a.str(1)?))).method().sql(),
    Builtin::new("strip", "(self: String, chars: String | None = None) -> String", |a| {
        s(strip_chars(a.str(0)?, a.opt_str(1)?, true, true))
    })
    .method()
    .sql(),
    Builtin::new("swapcase", "(self: String) -> String", |a| s(swapcase(a.str(0)?))).method(),
    Builtin::new("title", "(self: String) -> String", |a| s(title(a.str(0)?))).method(),
    Builtin::new("upper", "(self: String) -> String", |a| s(a.str(0)?.to_uppercase())).method().sql(),
    Builtin::new("wrap", "(self: String, width: Int, **kwargs) -> Json[(String, ...)]", |a| json_strs(wrap_text(a.str(0)?, a.int(1)?)?))
        .method(),
    Builtin::new("zfill", "(self: String, width: Int) -> String", |a| s(zfill(a.str(0)?, a.int(1)?))).method(),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_semantics() {
        assert_eq!(title("they're bill's"), "They'Re Bill'S");
        assert_eq!(capitalize("hELLO wORLD"), "Hello world");
        assert_eq!(pad_str("ab", 7, "*", 'c').unwrap(), "***ab**");
        assert_eq!(pad_str("abc", 8, "*", 'c').unwrap(), "**abc***");
        assert_eq!(zfill("-42", 6), "-00042");
        assert_eq!(py_split(" a  b ", None, 1, false).unwrap(), vec!["a", "b "]);
        assert_eq!(py_split(" a  b ", None, 1, true).unwrap(), vec![" a", "b"]);
        assert_eq!(py_split("a,b,,c", Some(","), 2, true).unwrap(), vec!["a,b", "", "c"]);
        assert_eq!(splitlines("ab\ncd\r\nef\r", true), vec!["ab\n", "cd\r\n", "ef\r"]);
        assert_eq!(wrap_text("The quick brown fox jumps over the lazy dog", 10).unwrap(), vec!["The quick", "brown fox", "jumps over", "the lazy", "dog"]);
        let args = vec![Value::String("a".into()), Value::Int(3)];
        assert_eq!(py_format("{0}/{1}", &args, &[]).unwrap(), "a/3");
        assert_eq!(py_format("{}-{}", &args, &[]).unwrap(), "a-3");
        assert_eq!(py_format("{:>6.2f}|{:,}", &[Value::Float(3.14159), Value::Int(1234567)], &[]).unwrap(), "  3.14|1,234,567");
        assert_eq!(py_find("hello", "l", None, None, true), 3);
        assert_eq!(py_slice(&chars("abcdef"), None, None, Some(-2)).unwrap().iter().collect::<String>(), "fdb");
    }
}
