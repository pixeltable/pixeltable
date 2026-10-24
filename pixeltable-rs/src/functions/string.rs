//! `pixeltable.functions.string`

use super::{Builtin, expect_arg};
use crate::error::{Error, Result};
use crate::types::ColumnType;
use crate::value::Value;

const MODULE: &str = "pixeltable.functions.string";

fn string_to_string(args: &[ColumnType]) -> Result<ColumnType> {
    let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
    expect_arg("string function", "self", t, t.is_string(), "String")?;
    Ok(ColumnType::string(false))
}

fn string_to_int(args: &[ColumnType]) -> Result<ColumnType> {
    let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
    expect_arg("len", "self", t, t.is_string(), "String")?;
    Ok(ColumnType::int(false))
}

fn str_arg(args: &[Value]) -> Result<&str> {
    match args.first() {
        Some(Value::String(s)) => Ok(s),
        _ => Err(Error::internal("expected a string argument")),
    }
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin {
        name: "upper",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: string_to_string,
        eval: |args| Ok(Value::String(str_arg(args)?.to_uppercase())),
    },
    Builtin {
        name: "lower",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: string_to_string,
        eval: |args| Ok(Value::String(str_arg(args)?.to_lowercase())),
    },
    Builtin {
        name: "len",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: string_to_int,
        eval: |args| Ok(Value::Int(str_arg(args)?.chars().count() as i64)),
    },
];
