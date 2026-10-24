//! `pixeltable.functions.math`

use super::{Builtin, expect_arg};
use crate::error::{Error, Result};
use crate::types::ColumnType;
use crate::value::Value;

const MODULE: &str = "pixeltable.functions.math";

fn numeric_to_float(name: &'static str) -> impl Fn(&[ColumnType]) -> Result<ColumnType> {
    move |args: &[ColumnType]| {
        let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
        expect_arg(name, "self", t, t.is_numeric(), "Float")?;
        Ok(ColumnType::float(false))
    }
}

fn abs_type(args: &[ColumnType]) -> Result<ColumnType> {
    numeric_to_float("abs")(args)
}

fn round_type(args: &[ColumnType]) -> Result<ColumnType> {
    if let Some(d) = args.get(1) {
        expect_arg("round", "digits", d, d.is_int(), "Int")?;
    }
    numeric_to_float("round")(args)
}

fn float_arg(args: &[Value]) -> Result<f64> {
    args.first().and_then(Value::as_f64).ok_or_else(|| Error::internal("expected a numeric argument"))
}

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

pub static FUNCTIONS: &[Builtin] = &[
    Builtin {
        name: "abs",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: abs_type,
        eval: |args| Ok(Value::Float(float_arg(args)?.abs())),
    },
    Builtin {
        name: "round",
        module: MODULE,
        params: &["self", "digits"],
        required: 1,
        strict: true,
        return_type: round_type,
        eval: |args| {
            let digits = args.get(1).and_then(Value::as_i64).unwrap_or(0);
            Ok(Value::Float(py_round(float_arg(args)?, digits)))
        },
    },
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
