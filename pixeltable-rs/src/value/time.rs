//! Timestamps in the default time zone, computed in Rust (chrono-tz), for builtins and string conversions.
//!
//! Timestamps are stored as UTC microseconds; the calendar fields upstream exposes (`year`, `hour`, `str()`, ...)
//! are those of the default time zone (`value::py::default_tz_name()`), as in upstream.

use std::sync::OnceLock;

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Offset, TimeZone, Timelike};
use chrono_tz::Tz;

pub fn default_tz() -> Tz {
    static TZ: OnceLock<Tz> = OnceLock::new();
    *TZ.get_or_init(|| super::py::default_tz_name().parse::<Tz>().unwrap_or(chrono_tz::UTC))
}

/// A stored timestamp as a date-time in the default time zone.
pub fn to_local(micros: i64) -> DateTime<Tz> {
    DateTime::from_timestamp_micros(micros).unwrap_or_default().with_timezone(&default_tz())
}

/// Microseconds of a naive local date-time in the default time zone (the earlier instant if ambiguous, shifted
/// forward over a gap).
pub fn from_local(naive: NaiveDateTime) -> i64 {
    let tz = default_tz();
    match tz.from_local_datetime(&naive) {
        chrono::LocalResult::Single(dt) => dt.timestamp_micros(),
        chrono::LocalResult::Ambiguous(a, _) => a.timestamp_micros(),
        chrono::LocalResult::None => {
            let shifted = naive + chrono::Duration::hours(1);
            tz.from_local_datetime(&shifted).earliest().map_or(naive.and_utc().timestamp_micros(), |d| d.timestamp_micros())
        }
    }
}

/// `+HH:MM` (with `:SS` if needed), as Python's `isoformat()` renders a UTC offset.
pub fn offset_str(secs: i32) -> String {
    let sign = if secs < 0 { '-' } else { '+' };
    let a = secs.abs();
    let (h, m, s) = (a / 3600, (a % 3600) / 60, a % 60);
    if s != 0 { format!("{sign}{h:02}:{m:02}:{s:02}") } else { format!("{sign}{h:02}:{m:02}") }
}

/// Python's `datetime.isoformat(sep)` of an aware date-time.
pub fn isoformat<T: TimeZone>(dt: &DateTime<T>, sep: char) -> String {
    let us = dt.nanosecond() / 1000;
    let frac = if us != 0 { format!(".{us:06}") } else { String::new() };
    format!(
        "{:04}-{:02}-{:02}{sep}{:02}:{:02}:{:02}{frac}{}",
        dt.year(),
        dt.month(),
        dt.day(),
        dt.hour(),
        dt.minute(),
        dt.second(),
        offset_str(dt.offset().fix().local_minus_utc())
    )
}

/// `str(datetime)` of a stored timestamp (as returned by a query, in the default time zone).
pub fn py_str_timestamp(micros: i64) -> String {
    isoformat(&to_local(micros), ' ')
}

pub fn date_of_days(days: i32) -> NaiveDate {
    super::date_from_days(days).unwrap_or_default()
}
