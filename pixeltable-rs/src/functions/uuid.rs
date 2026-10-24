//! `pixeltable.functions.uuid`

use super::{Builtin, expect_arg};
use crate::error::{Error, Result};
use crate::types::{ColumnType, TypeKind};
use crate::value::Value;

const MODULE: &str = "pixeltable.functions.uuid";

fn no_args_to_uuid(_: &[ColumnType]) -> Result<ColumnType> {
    Ok(ColumnType::new(TypeKind::Uuid, false))
}

fn uuid_to_string(name: &'static str) -> impl Fn(&[ColumnType]) -> Result<ColumnType> {
    move |args: &[ColumnType]| {
        let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
        expect_arg(name, "u", t, matches!(t.kind, TypeKind::Uuid), "UUID")?;
        Ok(ColumnType::string(false))
    }
}

fn uuid_arg(args: &[Value]) -> Result<::uuid::Uuid> {
    match args.first() {
        Some(Value::Uuid(u)) => Ok(*u),
        _ => Err(Error::internal("expected a UUID argument")),
    }
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin {
        name: "uuid4",
        module: MODULE,
        params: &[],
        required: 0,
        strict: true,
        return_type: no_args_to_uuid,
        eval: |_| Ok(Value::Uuid(uuid::Uuid::new_v4())),
    },
    Builtin {
        name: "uuid7",
        module: MODULE,
        params: &[],
        required: 0,
        strict: true,
        return_type: no_args_to_uuid,
        eval: |_| Ok(Value::Uuid(uuid::Uuid::now_v7())),
    },
    Builtin {
        name: "to_string",
        module: MODULE,
        params: &["u"],
        required: 1,
        strict: true,
        return_type: |args| uuid_to_string("to_string")(args),
        eval: |args| Ok(Value::String(uuid_arg(args)?.hyphenated().to_string())),
    },
    Builtin {
        name: "hex",
        module: MODULE,
        params: &["u"],
        required: 1,
        strict: true,
        return_type: |args| uuid_to_string("hex")(args),
        eval: |args| Ok(Value::String(uuid_arg(args)?.simple().to_string())),
    },
];
