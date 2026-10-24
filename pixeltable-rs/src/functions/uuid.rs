//! `pixeltable.functions.uuid`

use super::Builtin;
use crate::value::Value;

pub const MODULE: &str = "pixeltable.functions.uuid";

pub static FUNCTIONS: &[Builtin] = &[
    Builtin::new("uuid4", "() -> UUID", |_| Ok(Value::Uuid(uuid::Uuid::new_v4()))).sql(),
    Builtin::new("uuid7", "() -> UUID", |_| Ok(Value::Uuid(uuid::Uuid::now_v7()))),
    Builtin::new("to_string", "(u: UUID) -> String", |a| match a.get(0) {
        Value::Uuid(u) => Ok(Value::String(u.hyphenated().to_string())),
        v => Ok(Value::String(crate::value::py_str(v))),
    })
    .method()
    .sql(),
    Builtin::new("hex", "(u: UUID) -> String", |a| match a.get(0) {
        Value::Uuid(u) => Ok(Value::String(u.simple().to_string())),
        v => Ok(Value::String(crate::value::py_str(v))),
    })
    .method()
    .sql(),
];
