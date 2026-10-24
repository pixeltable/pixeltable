//! `pixeltable.functions.video` and `pixeltable.functions.audio`: container metadata, read with PyAV.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{Builtin, expect_arg};
use crate::error::{Error, Result};
use crate::media;
use crate::types::ColumnType;
use crate::value::py as vpy;
use crate::value::Value;

pub const VIDEO_MODULE: &str = "pixeltable.functions.video";
pub const AUDIO_MODULE: &str = "pixeltable.functions.audio";

fn opt_float<'py>(py: Python<'py>, v: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if v.is_none() { Ok(v) } else { Ok(py.import("builtins")?.getattr("float")?.call1((v,))?) }
}

/// Upstream `util.__get_stream_metadata()`.
fn stream_metadata<'py>(py: Python<'py>, stream: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let stream_type: String = stream.getattr("type")?.extract()?;
    d.set_item("type", &stream_type)?;
    if stream_type != "audio" && stream_type != "video" {
        return Ok(d);
    }
    let cc = stream.getattr("codec_context")?;
    let codec_tag: String = cc
        .getattr("codec_tag")?
        .call_method1("encode", ("unicode-escape",))?
        .call_method1("decode", ("utf-8",))?
        .extract()?;
    let duration = stream.getattr("duration")?;
    let time_base = stream.getattr("time_base")?;
    let duration_seconds = if duration.is_none() || time_base.is_none() {
        py.None().into_bound(py)
    } else {
        opt_float(py, duration.mul(&time_base)?)?
    };
    let cc_md = PyDict::new(py);
    cc_md.set_item("name", cc.getattr("name")?)?;
    cc_md.set_item("codec_tag", codec_tag)?;
    cc_md.set_item("profile", cc.getattr("profile")?)?;
    d.set_item("duration", &duration)?;
    d.set_item("time_base", opt_float(py, time_base)?)?;
    d.set_item("duration_seconds", duration_seconds)?;
    d.set_item("frames", stream.getattr("frames")?)?;
    d.set_item("metadata", stream.getattr("metadata")?)?;
    if stream_type == "audio" {
        let channels = cc.getattr("channels")?;
        let channels = if channels.is_none() { channels } else { py.import("builtins")?.getattr("int")?.call1((channels,))? };
        cc_md.set_item("channels", channels)?;
    } else {
        cc_md.set_item("pix_fmt", cc.getattr("pix_fmt").unwrap_or_else(|_| py.None().into_bound(py)))?;
        d.set_item("width", stream.getattr("width")?)?;
        d.set_item("height", stream.getattr("height")?)?;
        d.set_item("average_rate", opt_float(py, stream.getattr("average_rate")?)?)?;
        d.set_item("base_rate", opt_float(py, stream.getattr("base_rate")?)?)?;
        d.set_item("guessed_rate", opt_float(py, stream.getattr("guessed_rate")?)?)?;
    }
    d.set_item("codec_context", cc_md)?;
    Ok(d)
}

/// Upstream `util.get_metadata()`: metadata of a media container.
fn container_metadata(args: &[Value]) -> Result<Value> {
    let stored = match args.first() {
        Some(Value::Media(s)) => s.clone(),
        Some(Value::String(s)) => s.clone(),
        _ => return Err(Error::internal("expected a media file argument")),
    };
    Python::attach(|py| {
        let path = media::local_path(py, &stored)?;
        let container = py.import("av")?.call_method1("open", (path.to_string_lossy().to_string(),))?;
        let result = (|| -> PyResult<Bound<'_, PyDict>> {
            let streams = pyo3::types::PyList::empty(py);
            for s in container.getattr("streams")?.try_iter()? {
                streams.append(stream_metadata(py, &s?)?)?;
            }
            let d = PyDict::new(py);
            match container.getattr("bit_exact") {
                Ok(v) => d.set_item("bit_exact", v)?,
                Err(_) => d.set_item("bit_exact", false)?,
            }
            d.set_item("bit_rate", container.getattr("bit_rate")?)?;
            d.set_item("size", container.getattr("size")?)?;
            d.set_item("metadata", container.getattr("metadata")?)?;
            d.set_item("streams", streams)?;
            Ok(d)
        })();
        let _ = container.call_method0("close");
        Ok(Value::Json(vpy::json_from_py(result?.as_any())?))
    })
}

fn metadata_type(param: &'static str, ok: fn(&ColumnType) -> bool, expected: &'static str) -> impl Fn(&[ColumnType]) -> Result<ColumnType> {
    move |args: &[ColumnType]| {
        let t = args.first().ok_or_else(|| Error::internal("missing argument"))?;
        expect_arg("get_metadata", param, t, ok(t), expected)?;
        Ok(ColumnType::json(false))
    }
}

pub static VIDEO_FUNCTIONS: &[Builtin] = &[Builtin {
    name: "get_metadata",
    module: VIDEO_MODULE,
    params: &["video"],
    required: 1,
    strict: true,
    return_type: |args| metadata_type("video", ColumnType::is_video, "Video")(args),
    eval: container_metadata,
}];

pub static AUDIO_FUNCTIONS: &[Builtin] = &[Builtin {
    name: "get_metadata",
    module: AUDIO_MODULE,
    params: &["audio"],
    required: 1,
    strict: true,
    return_type: |args| metadata_type("audio", ColumnType::is_audio, "Audio")(args),
    eval: container_metadata,
}];
