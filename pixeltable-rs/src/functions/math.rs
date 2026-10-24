//! `pixeltable.functions.math`

use super::{Builtin, expect_arg};
use crate::error::{Error, Result};
use crate::types::ColumnType;
use crate::value::Value;

const MODULE: &str = "pixeltable.functions.math";

fn numeric_same_type(args: &[ColumnType]) -> Result<ColumnType> {
    let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
    expect_arg("abs", "self", t, t.is_numeric(), "Int | Float")?;
    Ok(t.with_nullable(false))
}

pub static FUNCTIONS: &[Builtin] = &[Builtin {
    name: "abs",
    module: MODULE,
    params: &["self"],
    required: 1,
    strict: true,
    return_type: numeric_same_type,
    eval: |args| match args.first() {
        Some(Value::Int(i)) => {
            i.checked_abs().map(Value::Int).ok_or_else(|| crate::pxt_err!(UnsupportedOperation, "integer out of range in abs()"))
        }
        Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
        _ => Err(Error::internal("abs() expects a numeric argument")),
    },
}];
