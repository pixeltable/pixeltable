//! `pixeltable.index.btree`: the value expression of a B-tree index's value column on a string column.

use super::Builtin;
use crate::value::Value;

pub const MODULE: &str = "pixeltable.index.btree";

/// Upstream `BtreeIndex.MAX_STRING_LEN`: an index holds string values truncated to this many characters.
pub const MAX_STRING_LEN: usize = 256;

/// `v` with a string truncated to `MAX_STRING_LEN` characters.
pub fn truncate(v: &Value) -> Value {
    match v {
        Value::String(s) if s.chars().count() > MAX_STRING_LEN => Value::String(s.chars().take(MAX_STRING_LEN).collect()),
        v => v.clone(),
    }
}

pub static FUNCTIONS: &[Builtin] = &[Builtin::new("str_filter", "(s: String | None) -> String | None", |a| Ok(truncate(a.get(0))))];
