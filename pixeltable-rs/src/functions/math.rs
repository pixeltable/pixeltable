//! `pixeltable.functions.math`

use super::{Args, Builtin};
use crate::error::{Error, Result};
use crate::value::Value;

pub const MODULE: &str = "pixeltable.functions.math";

/// Python's `round(x, digits)` for floats (round half to even on the exact binary value).
pub fn py_round(x: f64, digits: i64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if digits >= 0 {
        let prec = digits.min(340) as usize;
        format!("{x:.prec$}").parse::<f64>().unwrap_or(x)
    } else {
        let p = 10f64.powi((-digits).min(308) as i32);
        (x / p).round_ties_even() * p
    }
}

fn f(x: f64) -> Result<Value> {
    Ok(Value::Float(x))
}

fn math_domain() -> Error {
    Error::py_exc("ValueError", "math domain error")
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin::new("abs", "(self: Float) -> Float", |a| f(a.float(0)?.abs())).method().sql(),
    Builtin::new("ceil", "(self: Float) -> Float", |a| f(a.float(0)?.ceil())).method().sql(),
    Builtin::new("floor", "(self: Float) -> Float", |a| f(a.float(0)?.floor())).method().sql(),
    Builtin::new("round", "(self: Float, digits: Int | None = None) -> Float", |a: &Args<'_>| {
        let x = a.float(0)?;
        f(match a.opt_int(1)? {
            None => x.round_ties_even(),
            Some(d) => py_round(x, d),
        })
    })
    .method()
    .sql(),
    Builtin::new("pow", "(self: Float, other: Float) -> Float", |a| f(a.float(0)?.powf(a.float(1)?))).method().sql(),
    Builtin::new("sqrt", "(self: Float) -> Float", |a| {
        let x = a.float(0)?;
        if x < 0.0 { Err(math_domain()) } else { f(x.sqrt()) }
    })
    .method()
    .sql(),
    Builtin::new("exp", "(self: Float) -> Float", |a| f(a.float(0)?.exp())).method().sql(),
    Builtin::new("log", "(self: Float) -> Float", |a| {
        let x = a.float(0)?;
        if x <= 0.0 { Err(math_domain()) } else { f(x.ln()) }
    })
    .method()
    .sql(),
    Builtin::new("log10", "(self: Float) -> Float", |a| {
        let x = a.float(0)?;
        if x <= 0.0 { Err(math_domain()) } else { f(x.log10()) }
    })
    .method()
    .sql(),
    Builtin::new("bitwise_and", "(self: Int, other: Int) -> Int", |a| Ok(Value::Int(a.int(0)? & a.int(1)?))).method().sql(),
    Builtin::new("bitwise_or", "(self: Int, other: Int) -> Int", |a| Ok(Value::Int(a.int(0)? | a.int(1)?))).method().sql(),
    Builtin::new("bitwise_xor", "(self: Int, other: Int) -> Int", |a| Ok(Value::Int(a.int(0)? ^ a.int(1)?))).method().sql(),
    Builtin::new("to_int", "(self: Float) -> Int", |a| {
        let x = a.float(0)?;
        if !x.is_finite() {
            return Err(Error::py_exc(
                if x.is_nan() { "ValueError" } else { "OverflowError" },
                if x.is_nan() { "cannot convert float NaN to integer" } else { "cannot convert float infinity to integer" },
            ));
        }
        Ok(Value::Int(x.trunc() as i64))
    })
    .method()
    .sql(),
    Builtin::new("to_float", "(self: Int) -> Float", |a| f(a.float(0)?)).method().sql(),
];

#[cfg(test)]
mod tests {
    use super::py_round;

    #[test]
    fn rounding() {
        assert_eq!(py_round(2.5, 0), 2.0);
        assert_eq!(py_round(3.5, 0), 4.0);
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round(1234.5, -2), 1200.0);
    }
}
