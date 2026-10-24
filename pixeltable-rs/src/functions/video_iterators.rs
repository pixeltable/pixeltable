//! Builtin iterators over videos: `pixeltable.functions.video.frame_iterator` and `legacy_frame_iterator`
//! (upstream `pixeltable/functions/video/iterators.py`). Decoding goes through PyAV (media I/O); the frame
//! selection logic is here.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::iterator::{BuiltinIterator, IteratorCall, RowSource};
use super::signature::{Param, ParamKind};
use crate::error::{Error, Result};
use crate::expr::Expr;
use crate::types::{ColumnType, JsonSchema, JsonSpec, TypeKind};
use crate::value::{Json, Value};

pub static FRAME_ITERATOR: BuiltinIterator = BuiltinIterator {
    fqn: "pixeltable.functions.video.frame_iterator",
    params: frame_params,
    unstored_cols: &["frame"],
    output_schema: frame_schema,
    validate: validate_frame_args,
    start: start_frame_iterator,
};

pub static LEGACY_FRAME_ITERATOR: BuiltinIterator = BuiltinIterator {
    fqn: "pixeltable.functions.video.legacy_frame_iterator",
    params: frame_params,
    unstored_cols: &["frame"],
    output_schema: legacy_frame_schema,
    validate: validate_frame_args,
    start: start_legacy_frame_iterator,
};

fn param(name: &str, kind: ParamKind, t: ColumnType, default: Option<Value>) -> Param {
    Param {
        name: name.to_string(),
        kind,
        default: default.map(|v| {
            let lt = if v.is_null() { ColumnType::invalid(true) } else { t.with_nullable(false) };
            Expr::literal(v, lt)
        }),
        col_type: Some(t),
        is_batched: false,
    }
}

/// `(video: pxt.Video, *, fps: float | None = None, num_frames: int | None = None, keyframes_only: bool = False)`
fn frame_params() -> Vec<Param> {
    vec![
        param("video", ParamKind::PositionalOrKeyword, ColumnType::new(TypeKind::Video, false), None),
        param("fps", ParamKind::KeywordOnly, ColumnType::float(true), Some(Value::Null)),
        param("num_frames", ParamKind::KeywordOnly, ColumnType::int(true), Some(Value::Null)),
        param("keyframes_only", ParamKind::KeywordOnly, ColumnType::bool(false), Some(Value::Bool(false))),
    ]
}

fn image_type() -> ColumnType {
    ColumnType::new(TypeKind::Image { width: None, height: None, mode: None }, false)
}

fn frame_attrs_type() -> ColumnType {
    let mut fields = indexmap::IndexMap::new();
    fields.insert("index".to_string(), ColumnType::int(false));
    fields.insert("pts".to_string(), ColumnType::int(false));
    fields.insert("dts".to_string(), ColumnType::int(true));
    fields.insert("time".to_string(), ColumnType::float(false));
    fields.insert("is_corrupt".to_string(), ColumnType::bool(false));
    fields.insert("key_frame".to_string(), ColumnType::bool(false));
    fields.insert("pict_type".to_string(), ColumnType::int(false));
    fields.insert("interlaced_frame".to_string(), ColumnType::bool(false));
    let schema = JsonSchema { spec: JsonSpec::Dict(fields), variadic: None, optional_keys: Default::default() };
    ColumnType::new(TypeKind::Json(Some(Box::new(schema))), false)
}

fn frame_schema(_args: &[(String, Value)]) -> Result<Vec<(String, ColumnType)>> {
    Ok(vec![("frame".to_string(), image_type()), ("frame_attrs".to_string(), frame_attrs_type())])
}

fn legacy_frame_schema(_args: &[(String, Value)]) -> Result<Vec<(String, ColumnType)>> {
    Ok(vec![
        ("frame".to_string(), image_type()),
        ("frame_idx".to_string(), ColumnType::int(false)),
        ("pos_msec".to_string(), ColumnType::float(false)),
        ("pos_frame".to_string(), ColumnType::int(false)),
    ])
}

fn arg<'a>(args: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    args.iter().find(|(n, _)| n == name).map(|(_, v)| v).filter(|v| !v.is_null())
}

fn validate_frame_args(args: &[(String, Value)]) -> Result<()> {
    let fps = arg(args, "fps");
    let num_frames = arg(args, "num_frames");
    let keyframes_only = matches!(arg(args, "keyframes_only"), Some(Value::Bool(true)));
    if usize::from(fps.is_some()) + usize::from(num_frames.is_some()) + usize::from(keyframes_only) > 1 {
        crate::bail!(UnsupportedOperation, "At most one of `fps`, `num_frames` or `keyframes_only` may be specified");
    }
    if let Some(f) = fps {
        if f.as_f64().is_none_or(|f| f <= 0.0) {
            crate::bail!(InvalidArgument, "`fps` must be a positive number");
        }
    }
    Ok(())
}

/// The frame selection state machine of upstream's `frame_iterator`.
struct FrameSource {
    container: Py<PyAny>,
    stream: Py<PyAny>,
    time_base: f64,
    video_start_time: f64,
    keyframes_only: bool,
    extraction_step: Option<f64>,
    next_extraction_time: Option<f64>,
    pos: i64,
    video_idx: i64,
    cur_frame: Option<Py<PyAny>>,
}

fn frame_time(py: Python<'_>, frame: &Py<PyAny>, time_base: f64) -> Result<f64> {
    let pts: i64 = frame.bind(py).getattr("pts")?.extract()?;
    Ok(pts as f64 * time_base)
}

impl FrameSource {
    fn open(py: Python<'_>, args: &[(String, Value)]) -> Result<FrameSource> {
        let video = match arg(args, "video") {
            Some(Value::Media(s)) | Some(Value::String(s)) => s.clone(),
            _ => return Err(Error::internal("frame_iterator() without a video")),
        };
        let path = crate::media::local_path(py, &video)?;
        let container = py.import("av")?.call_method1("open", (path.to_string_lossy().to_string(),))?;
        let stream = container.getattr("streams")?.getattr("video")?.get_item(0)?;
        let tb = stream.getattr("time_base")?;
        let time_base: f64 = tb.extract()?;
        let start_time = stream.getattr("start_time")?;
        let start_pts: i64 = if start_time.is_none() { 0 } else { start_time.extract()? };
        let video_start_time = start_pts as f64 * time_base;
        let duration = stream.getattr("duration")?;
        let video_duration: Option<f64> = if !duration.is_none() {
            Some(duration.extract::<i64>()? as f64 * time_base)
        } else {
            let field = stream.getattr("metadata")?.call_method1("get", ("DURATION",))?;
            if field.is_none() {
                None
            } else {
                Some(py.import("pandas")?.call_method1("to_timedelta", (field,))?.call_method0("total_seconds")?.extract()?)
            }
        };
        let fps = arg(args, "fps").and_then(Value::as_f64);
        let num_frames = arg(args, "num_frames").and_then(Value::as_i64);
        let keyframes_only = matches!(arg(args, "keyframes_only"), Some(Value::Bool(true)));
        if video_duration.is_none() && num_frames.is_some() {
            crate::bail!(UnsupportedOperation, "Could not determine duration of video: {video}");
        }
        let (extraction_step, next_extraction_time) = if let Some(n) = num_frames {
            let step = (video_duration.unwrap_or(0.0) - video_start_time) / n as f64;
            (Some(step), Some(video_start_time + step / 2.0))
        } else if let Some(f) = fps {
            (Some(1.0 / f), Some(video_start_time))
        } else {
            (None, None)
        };
        let mut src = FrameSource {
            container: container.unbind(),
            stream: stream.unbind(),
            time_base,
            video_start_time,
            keyframes_only,
            extraction_step,
            next_extraction_time,
            pos: 0,
            video_idx: 0,
            cur_frame: None,
        };
        src.cur_frame = src.next_frame(py)?;
        Ok(src)
    }

    fn next_frame(&self, py: Python<'_>) -> Result<Option<Py<PyAny>>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("video", 0)?;
        let decoder = self.container.bind(py).call_method("decode", (), Some(&kwargs))?;
        match decoder.call_method0("__next__") {
            Ok(f) => Ok(Some(f.unbind())),
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) || e.is_instance_of::<pyo3::exceptions::PyEOFError>(py) => {
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The next selected frame: (PIL image, frame attributes) .
    fn next_selected(&mut self, py: Python<'_>) -> Result<Option<(Py<PyAny>, serde_json::Map<String, Json>)>> {
        loop {
            let Some(cur) = self.cur_frame.take() else { return Ok(None) };
            let next = self.next_frame(py)?;
            let curb = cur.bind(py);
            if self.keyframes_only && !curb.getattr("key_frame")?.is_truthy()? {
                self.cur_frame = next;
                self.video_idx += 1;
                continue;
            }
            let pts: i64 = curb.getattr("pts")?.extract()?;
            let cur_time = pts as f64 * self.time_base;
            if let (Some(_), Some(target)) = (self.extraction_step, self.next_extraction_time) {
                match &next {
                    None => {
                        if cur_time < target {
                            return Ok(None);
                        }
                    }
                    Some(n) => {
                        let next_time = frame_time(py, n, self.time_base)?;
                        if next_time - target < target - cur_time {
                            self.cur_frame = next;
                            self.video_idx += 1;
                            continue;
                        }
                    }
                }
            }
            let image = curb.call_method0("to_image")?;
            let dts = curb.getattr("dts")?;
            let mut attrs = serde_json::Map::new();
            attrs.insert("index".into(), Json::from(self.video_idx));
            attrs.insert("pts".into(), Json::from(pts));
            attrs.insert("dts".into(), if dts.is_none() { Json::Null } else { Json::from(dts.extract::<i64>()?) });
            attrs.insert("time".into(), Json::from(cur_time));
            attrs.insert("is_corrupt".into(), Json::from(curb.getattr("is_corrupt")?.is_truthy()?));
            attrs.insert("key_frame".into(), Json::from(curb.getattr("key_frame")?.is_truthy()?));
            attrs.insert("pict_type".into(), Json::from(curb.getattr("pict_type")?.extract::<i64>()?));
            attrs.insert("interlaced_frame".into(), Json::from(curb.getattr("interlaced_frame")?.is_truthy()?));
            self.cur_frame = next;
            self.video_idx += 1;
            self.pos += 1;
            if let (Some(step), Some(t)) = (self.extraction_step, self.next_extraction_time.as_mut()) {
                *t += step;
            }
            return Ok(Some((image.unbind(), attrs)));
        }
    }

    /// Positions the source at `pos`, given that row's frame index and time.
    fn seek_to(&mut self, py: Python<'_>, pos: i64, video_idx: i64, seek_time: f64) -> Result<()> {
        if self.pos == pos {
            return Ok(());
        }
        self.pos = pos;
        self.video_idx = video_idx;
        let offset = ((seek_time - self.video_start_time) / self.time_base).floor() as i64;
        let kwargs = PyDict::new(py);
        kwargs.set_item("backward", true)?;
        kwargs.set_item("stream", self.stream.bind(py))?;
        self.container.bind(py).call_method("seek", (offset,), Some(&kwargs))?;
        self.cur_frame = self.next_frame(py)?;
        while let Some(f) = &self.cur_frame {
            if frame_time(py, f, self.time_base)? >= seek_time - 1e-3 {
                break;
            }
            self.cur_frame = self.next_frame(py)?;
        }
        Ok(())
    }

    fn close_container(&mut self, py: Python<'_>) {
        let _ = self.container.bind(py).call_method0("close");
    }
}

fn start_frame_iterator(py: Python<'_>, args: &[(String, Value)]) -> Result<Box<dyn RowSource>> {
    Ok(Box::new(FrameIter(FrameSource::open(py, args)?)))
}

fn start_legacy_frame_iterator(py: Python<'_>, args: &[(String, Value)]) -> Result<Box<dyn RowSource>> {
    Ok(Box::new(LegacyFrameIter(FrameSource::open(py, args)?)))
}

fn stored<'a>(stored: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    stored.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

struct FrameIter(FrameSource);

impl RowSource for FrameIter {
    fn next_row(&mut self, py: Python<'_>, call: &IteratorCall) -> Result<Option<Vec<Value>>> {
        let Some((image, attrs)) = self.0.next_selected(py)? else { return Ok(None) };
        let attrs = Value::Json(Json::Object(attrs));
        Ok(Some(
            call.value_outputs()
                .iter()
                .map(|o| match o.orig_name.as_str() {
                    "frame" => Value::PyObj(std::sync::Arc::new(image.clone_ref(py))),
                    "frame_attrs" => attrs.clone(),
                    _ => Value::Null,
                })
                .collect(),
        ))
    }

    fn seek(&mut self, py: Python<'_>, _call: &IteratorCall, pos: i64, stored_vals: &[(String, Value)]) -> Result<()> {
        let Some(Value::Json(attrs)) = stored(stored_vals, "frame_attrs") else {
            return Err(Error::internal("frame_iterator seek() without frame_attrs"));
        };
        let idx = attrs.get("index").and_then(Json::as_i64).unwrap_or(0);
        let time = attrs.get("time").and_then(Json::as_f64).unwrap_or(0.0);
        self.0.seek_to(py, pos, idx, time)
    }

    fn close(&mut self, py: Python<'_>) {
        self.0.close_container(py);
    }
}

struct LegacyFrameIter(FrameSource);

impl RowSource for LegacyFrameIter {
    fn next_row(&mut self, py: Python<'_>, call: &IteratorCall) -> Result<Option<Vec<Value>>> {
        let Some((image, attrs)) = self.0.next_selected(py)? else { return Ok(None) };
        let time = attrs.get("time").and_then(Json::as_f64).unwrap_or(0.0);
        let index = attrs.get("index").and_then(Json::as_i64).unwrap_or(0);
        Ok(Some(
            call.value_outputs()
                .iter()
                .map(|o| match o.orig_name.as_str() {
                    "frame" => Value::PyObj(std::sync::Arc::new(image.clone_ref(py))),
                    // frame_idx == pos, which the source has already advanced
                    "frame_idx" => Value::Int(self.0.pos - 1),
                    "pos_msec" => Value::Float((time - self.0.video_start_time) * 1000.0),
                    "pos_frame" => Value::Int(index),
                    _ => Value::Null,
                })
                .collect(),
        ))
    }

    fn seek(&mut self, py: Python<'_>, _call: &IteratorCall, pos: i64, stored_vals: &[(String, Value)]) -> Result<()> {
        let idx = stored(stored_vals, "pos_frame").and_then(Value::as_i64).unwrap_or(0);
        let msec = stored(stored_vals, "pos_msec").and_then(Value::as_f64).unwrap_or(0.0);
        self.0.seek_to(py, pos, idx, msec / 1000.0)
    }

    fn close(&mut self, py: Python<'_>) {
        self.0.close_container(py);
    }
}
