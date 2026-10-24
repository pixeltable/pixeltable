//! `pixeltable.functions.timestamp`: calendar fields are those of the default time zone (`value::time`).

use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};

use super::{Args, Builtin, CallTypes};
use crate::error::{Error, Result};
use crate::types::{ColumnType, JsonSchema, JsonSpec, TypeKind};
use crate::value::time::{from_local, isoformat, offset_str, to_local};
use crate::value::{Json, Value, days_from_date};

pub const MODULE: &str = "pixeltable.functions.timestamp";

fn int(v: i64) -> Result<Value> {
    Ok(Value::Int(v))
}

fn local(a: &Args<'_>) -> Result<chrono::DateTime<chrono_tz::Tz>> {
    Ok(to_local(a.timestamp(0)?))
}

pub fn value_err(msg: impl Into<String>) -> Error {
    Error::py_exc("ValueError", msg)
}

/// `IsoCalendar` (upstream TypedDict): `Json[{'year': Int, 'week': Int, 'weekday': Int}]`.
pub fn iso_calendar_type(_: &CallTypes<'_>) -> Result<ColumnType> {
    let mut fields = indexmap::IndexMap::new();
    for k in ["year", "week", "weekday"] {
        fields.insert(k.to_string(), ColumnType::int(false));
    }
    let schema = JsonSchema { spec: JsonSpec::Dict(fields), variadic: None, optional_keys: Default::default() };
    Ok(ColumnType::new(TypeKind::Json(Some(Box::new(schema))), false))
}

pub fn iso_calendar(d: NaiveDate) -> Result<Value> {
    let w = d.iso_week();
    let mut m = serde_json::Map::new();
    m.insert("year".into(), Json::from(w.year()));
    m.insert("week".into(), Json::from(w.week()));
    m.insert("weekday".into(), Json::from(d.weekday().number_from_monday()));
    Ok(Value::Json(Json::Object(m)))
}

/// Translates a Python `strftime` format to chrono's (`%f` is microseconds in Python).
pub fn chrono_format(fmt: &str) -> String {
    fmt.replace("%f", "%6f")
}

/// Python's `isoformat(sep, timespec)` of a local date-time.
fn isoformat_spec(dt: &chrono::DateTime<chrono_tz::Tz>, sep: &str, timespec: &str) -> Result<String> {
    use chrono::Offset;
    let sep = sep.chars().next().unwrap_or('T');
    let off = offset_str(dt.offset().fix().local_minus_utc());
    let date = format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day());
    let us = dt.nanosecond() / 1000;
    let time = match timespec {
        "auto" => return Ok(isoformat(dt, sep)),
        "hours" => format!("{:02}", dt.hour()),
        "minutes" => format!("{:02}:{:02}", dt.hour(), dt.minute()),
        "seconds" => format!("{:02}:{:02}:{:02}", dt.hour(), dt.minute(), dt.second()),
        "milliseconds" => format!("{:02}:{:02}:{:02}.{:03}", dt.hour(), dt.minute(), dt.second(), us / 1000),
        "microseconds" => format!("{:02}:{:02}:{:02}.{:06}", dt.hour(), dt.minute(), dt.second(), us),
        _ => return Err(value_err("Unknown timespec value")),
    };
    Ok(format!("{date}{sep}{time}{off}"))
}

fn make_naive(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64, us: i64) -> Result<NaiveDateTime> {
    let date = NaiveDate::from_ymd_opt(y as i32, mo as u32, d as u32).ok_or_else(|| value_err("day is out of range for month"))?;
    let time = NaiveTime::from_hms_micro_opt(h as u32, mi as u32, s as u32, us as u32)
        .ok_or_else(|| value_err("hour, minute, second or microsecond out of range"))?;
    Ok(NaiveDateTime::new(date, time))
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin::new("year", "(self: Timestamp) -> Int", |a| int(local(a)?.year() as i64)).property().sql(),
    Builtin::new("month", "(self: Timestamp) -> Int", |a| int(local(a)?.month() as i64)).property().sql(),
    Builtin::new("day", "(self: Timestamp) -> Int", |a| int(local(a)?.day() as i64)).property().sql(),
    Builtin::new("hour", "(self: Timestamp) -> Int", |a| int(local(a)?.hour() as i64)).property().sql(),
    Builtin::new("minute", "(self: Timestamp) -> Int", |a| int(local(a)?.minute() as i64)).property().sql(),
    Builtin::new("second", "(self: Timestamp) -> Int", |a| int(local(a)?.second() as i64)).property().sql(),
    Builtin::new("microsecond", "(self: Timestamp) -> Int", |a| int((local(a)?.nanosecond() / 1000) as i64)).property().sql(),
    Builtin::new("date", "(self: Timestamp) -> Date", |a| {
        Ok(Value::Date(days_from_date(local(a)?.date_naive()).ok_or_else(|| value_err("date out of range"))?))
    })
    .method()
    .sql(),
    Builtin::new("astimezone", "(self: Timestamp, tz: String) -> Timestamp", |a| {
        a.str(1)?.parse::<chrono_tz::Tz>().map_err(|_| Error::py_exc("ZoneInfoNotFoundError", format!("'No time zone found with key {}'", a.str(1).unwrap_or(""))))?;
        Ok(Value::Timestamp(a.timestamp(0)?))
    })
    .method(),
    Builtin::new("weekday", "(self: Timestamp) -> Int", |a| int(local(a)?.weekday().num_days_from_monday() as i64)).method().sql(),
    Builtin::new("isoweekday", "(self: Timestamp) -> Int", |a| int(local(a)?.weekday().number_from_monday() as i64)).method().sql(),
    Builtin::new("isocalendar", "(self: Timestamp) -> Json", |a| iso_calendar(local(a)?.date_naive())).method().ret(iso_calendar_type),
    Builtin::new("isoformat", "(self: Timestamp, sep: String = 'T', timespec: String = 'auto') -> String", |a| {
        Ok(Value::String(isoformat_spec(&local(a)?, a.str(1)?, a.str(2)?)?))
    })
    .method(),
    Builtin::new("strftime", "(self: Timestamp, format: String) -> String", |a| {
        use std::fmt::Write;
        let mut out = String::new();
        write!(out, "{}", local(a)?.format(&chrono_format(a.str(1)?))).map_err(|_| value_err("Invalid format string"))?;
        Ok(Value::String(out))
    })
    .method(),
    Builtin::new(
        "make_timestamp",
        "(year: Int, month: Int, day: Int, hour: Int = 0, minute: Int = 0, second: Int = 0, microsecond: Int = 0) -> Timestamp",
        |a| {
            let naive = make_naive(a.int(0)?, a.int(1)?, a.int(2)?, a.int(3)?, a.int(4)?, a.int(5)?, a.int(6)?)?;
            Ok(Value::Timestamp(from_local(naive)))
        },
    )
    .method()
    .sql(),
    Builtin::new(
        "replace",
        "(self: Timestamp, year: Int | None = None, month: Int | None = None, day: Int | None = None, hour: Int | None = None, minute: Int | None = None, second: Int | None = None, microsecond: Int | None = None) -> Timestamp",
        |a| {
            let dt = local(a)?;
            let pick = |i: usize, cur: i64| -> Result<i64> { Ok(a.opt_int(i)?.unwrap_or(cur)) };
            let naive = make_naive(
                pick(1, dt.year() as i64)?,
                pick(2, dt.month() as i64)?,
                pick(3, dt.day() as i64)?,
                pick(4, dt.hour() as i64)?,
                pick(5, dt.minute() as i64)?,
                pick(6, dt.second() as i64)?,
                pick(7, (dt.nanosecond() / 1000) as i64)?,
            )?;
            Ok(Value::Timestamp(from_local(naive)))
        },
    )
    .method(),
    Builtin::new("strip_timezone", "(self: Timestamp) -> Timestamp", |a| Ok(Value::Timestamp(a.timestamp(0)?))).method(),
    Builtin::new("toordinal", "(self: Timestamp) -> Int", |a| int(local(a)?.date_naive().num_days_from_ce() as i64)).method(),
    Builtin::new("posix_timestamp", "(self: Timestamp) -> Float", |a| Ok(Value::Float(a.timestamp(0)? as f64 / 1e6))).method(),
];
