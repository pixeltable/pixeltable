//! `pixeltable.functions.image`: image functions, evaluated with PIL (the pixel operations are PIL's, as upstream).
//!
//! Image arguments arrive as stored images (`Value::Media`, opened from their file) or in-flight PIL images
//! (`Value::PyObj`); image results are in-flight PIL images, which the store writes to the media directory when the
//! value is stored.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::Arc;

use super::{Builtin, CallTypes};
use crate::error::{Error, Result};
use crate::media;
use crate::types::{ColumnType, TypeKind};
use crate::value::py as vpy;
use crate::value::{Json, Value};

pub const MODULE: &str = "pixeltable.functions.image";

// ------------------------------------------------------------------------------------------------ types

/// The image type of the call's `self` argument.
fn self_image(ct: &CallTypes<'_>) -> (Option<u64>, Option<u64>, Option<String>, bool) {
    match ct.ty(0) {
        Some(ColumnType { kind: TypeKind::Image { width, height, mode }, nullable }) => (*width, *height, mode.clone(), *nullable),
        Some(t) => (None, None, None, t.nullable),
        None => (None, None, None, false),
    }
}

fn image(width: Option<u64>, height: Option<u64>, mode: Option<String>, nullable: bool) -> ColumnType {
    ColumnType::new(TypeKind::Image { width, height, mode }, nullable)
}

/// The integers of a literal list argument.
fn literal_ints(ct: &CallTypes<'_>, i: usize) -> Option<Vec<i64>> {
    let items = match ct.literal(i)? {
        Value::Json(Json::Array(items)) => items.clone(),
        _ => return None,
    };
    items.iter().map(|x| x.as_i64()).collect()
}

/// upstream `convert.conditional_return_type`: the input's size, the given mode
fn convert_type(ct: &CallTypes<'_>) -> Result<ColumnType> {
    let (w, h, _, nullable) = self_image(ct);
    let mode = match ct.literal(1) {
        Some(Value::String(m)) => Some(m.clone()),
        _ => None,
    };
    Ok(image(w, h, mode, nullable))
}

/// upstream `crop.conditional_return_type`: the box's size if it is a literal, the input's mode
fn crop_type(ct: &CallTypes<'_>) -> Result<ColumnType> {
    let (_, _, mode, nullable) = self_image(ct);
    match literal_ints(ct, 1).as_deref() {
        Some([x0, y0, x1, y1]) if x1 >= x0 && y1 >= y0 => Ok(image(Some((x1 - x0) as u64), Some((y1 - y0) as u64), mode, nullable)),
        _ => Ok(image(None, None, mode, nullable)),
    }
}

/// upstream `getchannel.conditional_return_type`: the input's size, mode `L`
fn getchannel_type(ct: &CallTypes<'_>) -> Result<ColumnType> {
    let (w, h, _, nullable) = self_image(ct);
    Ok(image(w, h, Some("L".to_string()), nullable))
}

/// upstream `resize.conditional_return_type`: the given size, the input's mode
fn resize_type(ct: &CallTypes<'_>) -> Result<ColumnType> {
    let (_, _, mode, nullable) = self_image(ct);
    match literal_ints(ct, 1).as_deref() {
        Some([w, h]) if *w >= 0 && *h >= 0 => Ok(image(Some(*w as u64), Some(*h as u64), mode, nullable)),
        _ => Ok(image(None, None, mode, nullable)),
    }
}

/// upstream conditional return type of `rotate()`, `effect_spread()`, `transpose()`: the input's type
fn same_type(ct: &CallTypes<'_>) -> Result<ColumnType> {
    Ok(ct.ty(0).cloned().unwrap_or_else(|| image(None, None, None, false)))
}

// ------------------------------------------------------------------------------------------------ arguments

fn img<'py>(py: Python<'py>, args: &[Value], i: usize) -> Result<Bound<'py, PyAny>> {
    media::open_image(py, args.get(i).ok_or_else(|| Error::internal("missing image argument"))?)
}

/// An optional argument as a Python object (None when absent or NULL).
fn opt<'py>(py: Python<'py>, args: &[Value], i: usize) -> Result<Bound<'py, PyAny>> {
    match args.get(i) {
        None | Some(Value::Null) => Ok(py.None().into_bound(py)),
        Some(v) => arg(py, v),
    }
}

fn arg<'py>(py: Python<'py>, v: &Value) -> Result<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Media(_) | Value::PyObj(_) => media::open_image(py, v)?,
        Value::Json(j) => vpy::json_to_py(py, j)?,
        other => vpy::to_py(py, other, &ColumnType::invalid(true))?,
    })
}

fn req<'py>(py: Python<'py>, args: &[Value], i: usize) -> Result<Bound<'py, PyAny>> {
    arg(py, args.get(i).ok_or_else(|| Error::internal("missing argument"))?)
}

/// A list argument as a tuple (PIL wants tuples for boxes and sizes).
fn tuple_arg<'py>(py: Python<'py>, args: &[Value], i: usize) -> Result<Bound<'py, PyAny>> {
    let v = req(py, args, i)?;
    Ok(PyTuple::new(py, v.try_iter()?.collect::<PyResult<Vec<_>>>()?)?.into_any())
}

// ------------------------------------------------------------------------------------------------ results

fn image_result(obj: Bound<'_, PyAny>) -> Result<Value> {
    Ok(Value::PyObj(Arc::new(obj.unbind())))
}

fn json_result(obj: &Bound<'_, PyAny>) -> Result<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    Ok(Value::Json(vpy::json_from_py(obj)?))
}

fn with_py(f: impl for<'py> FnOnce(Python<'py>) -> Result<Value>) -> Result<Value> {
    Python::attach(f)
}

/// `self.<method>(*rest)` returning an image.
fn method_image(args: &[Value], method: &str, rest: impl for<'py> FnOnce(Python<'py>) -> Result<Vec<Bound<'py, PyAny>>>) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?;
        let rest = PyTuple::new(py, rest(py)?)?;
        image_result(im.call_method1(method, rest)?)
    })
}

/// `self.<method>(*rest)` returning a JSON-convertible value.
fn method_json(args: &[Value], method: &str, rest: impl for<'py> FnOnce(Python<'py>) -> Result<Vec<Bound<'py, PyAny>>>) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?;
        let rest = PyTuple::new(py, rest(py)?)?;
        json_result(&im.call_method1(method, rest)?)
    })
}

fn none(_: Python<'_>) -> Result<Vec<Bound<'_, PyAny>>> {
    Ok(vec![])
}

// ------------------------------------------------------------------------------------------------ functions

fn b64_encode(args: &[Value]) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?;
        let format = match args.get(1) {
            Some(Value::String(s)) => s.clone(),
            _ => "png".to_string(),
        };
        let buf = py.import("io")?.call_method0("BytesIO")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("format", format)?;
        im.call_method("save", (&buf,), Some(&kwargs))?;
        let bytes: Vec<u8> = buf.call_method0("getvalue")?.extract()?;
        use base64::Engine;
        Ok(Value::String(base64::engine::general_purpose::STANDARD.encode(bytes)))
    })
}

fn pil_fn(name: &str, args: &[Value], n_images: usize, extra: Option<usize>) -> Result<Value> {
    with_py(|py| {
        let mut call_args = Vec::new();
        for i in 0..n_images {
            call_args.push(img(py, args, i)?);
        }
        if let Some(i) = extra {
            call_args.push(req(py, args, i)?);
        }
        image_result(media::pil_image(py)?.call_method1(name, PyTuple::new(py, call_args)?)?)
    })
}

fn get_metadata(args: &[Value]) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?;
        let d = PyDict::new(py);
        d.set_item("width", im.getattr("width")?)?;
        d.set_item("height", im.getattr("height")?)?;
        d.set_item("mode", im.getattr("mode")?)?;
        d.set_item("bits", im.getattr("bits").unwrap_or_else(|_| py.None().into_bound(py)))?;
        d.set_item("format", im.getattr("format")?)?;
        json_result(d.as_any())
    })
}

fn getbbox(args: &[Value]) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("alpha_only", !matches!(args.get(1), Some(Value::Bool(false))))?;
        json_result(&im.call_method("getbbox", (), Some(&kwargs))?)
    })
}

fn quantize(args: &[Value]) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?;
        let call_args = PyTuple::new(py, [req(py, args, 1)?, opt(py, args, 2)?, req(py, args, 3)?, opt(py, args, 4)?, req(py, args, 5)?])?;
        image_result(im.call_method1("quantize", call_args)?)
    })
}

fn thumbnail(args: &[Value]) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?.call_method0("copy")?;
        let size = tuple_arg(py, args, 1)?;
        let resample = media::pil_image(py)?.getattr("Resampling")?.call1((req(py, args, 2)?,))?;
        im.call_method1("thumbnail", (size, resample, opt(py, args, 3)?))?;
        image_result(im)
    })
}

fn attr_int(args: &[Value], name: &str) -> Result<Value> {
    with_py(|py| Ok(Value::Int(img(py, args, 0)?.getattr(name)?.extract()?)))
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin::new("b64_encode", "(img: Image, image_format: String = 'png') -> String", |a| b64_encode(a.vals)).method(),
    Builtin::new("alpha_composite", "(im1: Image, im2: Image) -> Image", |a| pil_fn("alpha_composite", a.vals, 2, None)).method(),
    Builtin::new("blend", "(im1: Image, im2: Image, alpha: Float) -> Image", |a| pil_fn("blend", a.vals, 2, Some(2))).method(),
    Builtin::new("composite", "(image1: Image, image2: Image, mask: Image) -> Image", |a| pil_fn("composite", a.vals, 3, None))
        .method(),
    Builtin::new("convert", "(self: Image, mode: String) -> Image", |a| {
        method_image(a.vals, "convert", |py| Ok(vec![req(py, a.vals, 1)?]))
    })
    .method()
    .ret(convert_type),
    Builtin::new("crop", "(self: Image, box: Json[(Int, Int, Int, Int)]) -> Image", |a| {
        method_image(a.vals, "crop", |py| Ok(vec![tuple_arg(py, a.vals, 1)?]))
    })
    .method()
    .ret(crop_type),
    Builtin::new("getchannel", "(self: Image, channel: Int) -> Image", |a| {
        method_image(a.vals, "getchannel", |py| Ok(vec![req(py, a.vals, 1)?]))
    })
    .method()
    .ret(getchannel_type),
    Builtin::new("get_metadata", "(self: Image) -> Json", |a| get_metadata(a.vals)).method(),
    Builtin::new("point", "(self: Image, lut: Json[(Int, ...)], mode: String | None = None) -> Image", |a| {
        method_image(a.vals, "point", |py| Ok(vec![req(py, a.vals, 1)?, opt(py, a.vals, 2)?]))
    })
    .method(),
    Builtin::new("resize", "(self: Image, size: Json[(Int, Int)]) -> Image", |a| {
        method_image(a.vals, "resize", |py| Ok(vec![tuple_arg(py, a.vals, 1)?]))
    })
    .method()
    .ret(resize_type),
    Builtin::new("rotate", "(self: Image, angle: Int) -> Image", |a| method_image(a.vals, "rotate", |py| Ok(vec![req(py, a.vals, 1)?])))
        .method()
        .ret(same_type),
    Builtin::new("effect_spread", "(self: Image, distance: Int) -> Image", |a| {
        method_image(a.vals, "effect_spread", |py| Ok(vec![req(py, a.vals, 1)?]))
    })
    .method()
    .ret(same_type),
    Builtin::new("transpose", "(self: Image, method: Int) -> Image", |a| {
        method_image(a.vals, "transpose", |py| Ok(vec![req(py, a.vals, 1)?]))
    })
    .method()
    .ret(same_type),
    Builtin::new("entropy", "(self: Image, mask: Image | None = None, extrema: Json[(Json, ...)] | None = None) -> Float", |a| {
        with_py(|py| {
            let im = img(py, a.vals, 0)?;
            Ok(Value::Float(im.call_method1("entropy", (opt(py, a.vals, 1)?, opt(py, a.vals, 2)?))?.extract()?))
        })
    })
    .method(),
    Builtin::new("getbands", "(self: Image) -> Json[(String, ...)]", |a| method_json(a.vals, "getbands", none)).method(),
    Builtin::new("getbbox", "(self: Image, *, alpha_only: Bool = True) -> Json[(Int, Int, Int, Int)] | None", |a| getbbox(a.vals))
        .method(),
    Builtin::new("getcolors", "(self: Image, maxcolors: Int = 256) -> Json[(Json[(Int, Int)], ...)]", |a| {
        method_json(a.vals, "getcolors", |py| Ok(vec![req(py, a.vals, 1)?]))
    })
    .method(),
    Builtin::new("getextrema", "(self: Image) -> Json[(Int, Int)]", |a| method_json(a.vals, "getextrema", none)).method(),
    Builtin::new("getpalette", "(self: Image, mode: String | None = None) -> Json[(Int, ...)] | None", |a| {
        method_json(a.vals, "getpalette", |py| Ok(vec![opt(py, a.vals, 1)?]))
    })
    .method(),
    Builtin::new("getpixel", "(self: Image, xy: Json[(Json, ...)]) -> Json[(Int)]", |a| {
        method_json(a.vals, "getpixel", |py| Ok(vec![tuple_arg(py, a.vals, 1)?]))
    })
    .method(),
    Builtin::new("getprojection", "(self: Image) -> Json[(Json[(Int, ...)], Json[(Int, ...)])]", |a| {
        method_json(a.vals, "getprojection", none)
    })
    .method(),
    Builtin::new(
        "histogram",
        "(self: Image, mask: Image | None = None, extrema: Json[(Json, ...)] | None = None) -> Json[(Int, ...)]",
        |a| method_json(a.vals, "histogram", |py| Ok(vec![opt(py, a.vals, 1)?, opt(py, a.vals, 2)?])),
    )
    .method(),
    // the `dither` default is `PIL.Image.Dither.FLOYDSTEINBERG`
    Builtin::new(
        "quantize",
        "(self: Image, colors: Int = 256, method: Int | None = None, kmeans: Int = 0, palette: Image | None = None, dither: Int = 3) -> Image",
        |a| quantize(a.vals),
    )
    .method(),
    Builtin::new("reduce", "(self: Image, factor: Int, box: Json[(Int, Int, Int, Int)] | None = None) -> Image", |a| {
        method_image(a.vals, "reduce", |py| {
            let bx = match a.get(2) {
                Value::Null => py.None().into_bound(py),
                _ => tuple_arg(py, a.vals, 2)?,
            };
            Ok(vec![req(py, a.vals, 1)?, bx])
        })
    })
    .method(),
    // the `resample` default is `PIL.Image.Resampling.LANCZOS`
    Builtin::new("thumbnail", "(self: Image, size: Json[(Int, Int)], resample: Int = 1, reducing_gap: Float | None = 2.0) -> Image", |a| {
        thumbnail(a.vals)
    })
    .method(),
    Builtin::new("width", "(self: Image) -> Int", |a| attr_int(a.vals, "width")).property(),
    Builtin::new("height", "(self: Image) -> Int", |a| attr_int(a.vals, "height")).property(),
    Builtin::new("mode", "(self: Image) -> String", |a| with_py(|py| Ok(Value::String(img(py, a.vals, 0)?.getattr("mode")?.extract()?))))
        .property(),
    Builtin::new("size", "(self: Image) -> Json[(Int, Int)]", |a| {
        with_py(|py| {
            let (w, h): (i64, i64) = img(py, a.vals, 0)?.getattr("size")?.extract()?;
            Ok(Value::Json(Json::Array(vec![Json::from(w), Json::from(h)])))
        })
    })
    .property(),
];
