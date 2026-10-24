//! `pixeltable.functions.date`

use chrono::{Datelike, NaiveDate};

use super::timestamp::{chrono_format, iso_calendar, iso_calendar_type, value_err};
use super::{Args, Builtin};
use crate::error::Result;
use crate::value::time::date_of_days;
use crate::value::{Value, days_from_date};

pub const MODULE: &str = "pixeltable.functions.date";

fn int(v: i64) -> Result<Value> {
    Ok(Value::Int(v))
}

fn d(a: &Args<'_>) -> Result<NaiveDate> {
    Ok(date_of_days(a.date(0)?))
}

fn date_val(nd: NaiveDate) -> Result<Value> {
    Ok(Value::Date(days_from_date(nd).ok_or_else(|| value_err("date value out of range"))?))
}

fn make(y: i64, m: i64, day: i64) -> Result<NaiveDate> {
    if !(1..=9999).contains(&y) {
        return Err(value_err(format!("year {y} is out of range")));
    }
    if !(1..=12).contains(&m) {
        return Err(value_err("month must be in 1..12"));
    }
    NaiveDate::from_ymd_opt(y as i32, m as u32, day as u32).ok_or_else(|| value_err("day is out of range for month"))
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin::new("year", "(self: Date) -> Int", |a| int(d(a)?.year() as i64)).property().sql(),
    Builtin::new("month", "(self: Date) -> Int", |a| int(d(a)?.month() as i64)).property().sql(),
    Builtin::new("day", "(self: Date) -> Int", |a| int(d(a)?.day() as i64)).property().sql(),
    Builtin::new("make_date", "(year: Int, month: Int, day: Int) -> Date", |a| date_val(make(a.int(0)?, a.int(1)?, a.int(2)?)?))
        .method()
        .sql(),
    Builtin::new("weekday", "(self: Date) -> Int", |a| int(d(a)?.weekday().num_days_from_monday() as i64)).method().sql(),
    Builtin::new("isoweekday", "(self: Date) -> Int", |a| int(d(a)?.weekday().number_from_monday() as i64)).method().sql(),
    Builtin::new("isocalendar", "(self: Date) -> Json", |a| iso_calendar(d(a)?)).method().ret(iso_calendar_type),
    Builtin::new("isoformat", "(self: Date, sep: String = 'T', timespec: String = 'auto') -> String", |a| {
        Ok(Value::String(d(a)?.format("%Y-%m-%d").to_string()))
    })
    .method(),
    Builtin::new("toordinal", "(self: Date) -> Int", |a| int(d(a)?.num_days_from_ce() as i64)).method(),
    Builtin::new("strftime", "(self: Date, format: String) -> String", |a| {
        use std::fmt::Write;
        let mut out = String::new();
        let dt = d(a)?.and_hms_opt(0, 0, 0).unwrap_or_default();
        write!(out, "{}", dt.format(&chrono_format(a.str(1)?))).map_err(|_| value_err("Invalid format string"))?;
        Ok(Value::String(out))
    })
    .method(),
    Builtin::new("add_days", "(self: Date, n: Int) -> Date", |a| {
        let nd = d(a)?.checked_add_signed(chrono::Duration::days(a.int(1)?)).ok_or_else(|| value_err("date value out of range"))?;
        date_val(nd)
    })
    .method(),
    Builtin::new("replace", "(self: Date, year: Int | None = None, month: Int | None = None, day: Int | None = None) -> Date", |a| {
        let cur = d(a)?;
        date_val(make(
            a.opt_int(1)?.unwrap_or(cur.year() as i64),
            a.opt_int(2)?.unwrap_or(cur.month() as i64),
            a.opt_int(3)?.unwrap_or(cur.day() as i64),
        )?)
    })
    .method(),
];
