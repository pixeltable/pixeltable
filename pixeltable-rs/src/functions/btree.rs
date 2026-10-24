//! `pixeltable.index.btree`: the value expression of a B-tree index's value column on a string column.

use super::{Builtin, expect_arg};
use crate::error::Result;
use crate::types::ColumnType;
use crate::value::Value;

pub const MODULE: &str = "pixeltable.index.btree";

/// Upstream `BtreeIndex.MAX_STRING_LEN`: an index holds string values truncated to this many characters.
pub const MAX_STRING_LEN: usize = 256;

/// `s` truncated to `MAX_STRING_LEN` characters.
pub fn truncate(v: &Value) -> Value {
    match v {
        Value::String(s) if s.chars().count() > MAX_STRING_LEN => Value::String(s.chars().take(MAX_STRING_LEN).collect()),
        v => v.clone(),
    }
}

pub static FUNCTIONS: &[Builtin] = &[Builtin {
    name: "str_filter",
    module: MODULE,
    params: &["s"],
    required: 1,
    strict: true,
    return_type: |args: &[ColumnType]| -> Result<ColumnType> {
        let t = args.first().cloned().unwrap_or_else(|| ColumnType::string(true));
        expect_arg("str_filter", "s", &t, t.is_string(), "String")?;
        Ok(ColumnType::string(true))
    },
    eval: |args| Ok(args.first().map(truncate).unwrap_or(Value::Null)),
}];
