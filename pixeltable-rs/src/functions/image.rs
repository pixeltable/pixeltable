//! `pixeltable.functions.image`: image functions, evaluated with PIL (the pixel operations are PIL's, as upstream).
//!
//! Image arguments arrive as stored images (`Value::Media`, opened from their file) or in-flight PIL images
//! (`Value::PyObj`); image results are in-flight PIL images, which the store writes to the media directory when the
//! value is stored.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::Arc;

use super::{PosBuiltin as Builtin, expect_arg};
use crate::error::{Error, Result};
use crate::media;
use crate::types::{ColumnType, TypeKind};
use crate::value::py as vpy;
use crate::value::{Json, Value};

pub const MODULE: &str = "pixeltable.functions.image";

/// Functions that are properties of image expressions (`t.img.width`, no call).
pub static PROPERTIES: &[&str] = &["width", "height", "mode", "size"];

// ------------------------------------------------------------------------------------------------ types

fn image_type(nullable: bool) -> ColumnType {
    ColumnType::new(TypeKind::Image { width: None, height: None, mode: None }, nullable)
}

fn check_image(fn_name: &str, param: &str, t: Option<&ColumnType>) -> Result<()> {
    let t = t.ok_or_else(|| Error::internal("missing argument"))?;
    expect_arg(fn_name, param, t, t.is_image(), "Image")
}

fn check_self(name: &str, args: &[ColumnType]) -> Result<()> {
    check_image(name, "self", args.first())
}

macro_rules! ret {
    ($name:literal, $t:expr) => {
        |args: &[ColumnType]| -> Result<ColumnType> {
            check_self($name, args)?;
            Ok($t)
        }
    };
}

/// Same type as the input image (rotate, effect_spread, transpose).
fn same_image(args: &[ColumnType]) -> Result<ColumnType> {
    check_self("image function", args)?;
    Ok(args[0].with_nullable(false))
}

/// An image of the input's mode and unknown size.
fn image_of_mode(args: &[ColumnType]) -> Result<ColumnType> {
    check_self("image function", args)?;
    let mode = match &args[0].kind {
        TypeKind::Image { mode, .. } => mode.clone(),
        _ => None,
    };
    Ok(ColumnType::new(TypeKind::Image { width: None, height: None, mode }, false))
}

fn any_image(args: &[ColumnType]) -> Result<ColumnType> {
    check_self("image function", args)?;
    Ok(image_type(false))
}

fn two_images(args: &[ColumnType]) -> Result<ColumnType> {
    check_image("image function", "im1", args.first())?;
    check_image("image function", "im2", args.get(1))?;
    Ok(image_type(false))
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
        let colors = match args.get(1) {
            Some(Value::Int(n)) => *n,
            _ => 256,
        };
        let kmeans = match args.get(3) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        let dither = match args.get(5) {
            None | Some(Value::Null) => media::pil_image(py)?.getattr("Dither")?.getattr("FLOYDSTEINBERG")?,
            Some(v) => arg(py, v)?,
        };
        let call_args = PyTuple::new(
            py,
            [colors.into_pyobject(py)?.into_any(), opt(py, args, 2)?, kmeans.into_pyobject(py)?.into_any(), opt(py, args, 4)?, dither],
        )?;
        image_result(im.call_method1("quantize", call_args)?)
    })
}

fn thumbnail(args: &[Value]) -> Result<Value> {
    with_py(|py| {
        let im = img(py, args, 0)?.call_method0("copy")?;
        let size = tuple_arg(py, args, 1)?;
        let resampling = media::pil_image(py)?.getattr("Resampling")?;
        let resample = match args.get(2) {
            None | Some(Value::Null) => resampling.getattr("LANCZOS")?,
            Some(v) => resampling.call1((arg(py, v)?,))?,
        };
        let reducing_gap = match args.get(3) {
            None => 2.0f64.into_pyobject(py)?.into_any(),
            Some(v) => opt(py, std::slice::from_ref(v), 0)?,
        };
        im.call_method1("thumbnail", (size, resample, reducing_gap))?;
        image_result(im)
    })
}

fn attr_int(args: &[Value], name: &str) -> Result<Value> {
    with_py(|py| Ok(Value::Int(img(py, args, 0)?.getattr(name)?.extract()?)))
}

pub static FUNCTIONS: &[Builtin] = &[
    Builtin {
        name: "b64_encode",
        module: MODULE,
        params: &["img", "image_format"],
        required: 1,
        strict: true,
        return_type: |args| {
            check_image("b64_encode", "img", args.first())?;
            Ok(ColumnType::string(false))
        },
        eval: b64_encode,
    },
    Builtin {
        name: "alpha_composite",
        module: MODULE,
        params: &["im1", "im2"],
        required: 2,
        strict: true,
        return_type: two_images,
        eval: |args| pil_fn("alpha_composite", args, 2, None),
    },
    Builtin {
        name: "blend",
        module: MODULE,
        params: &["im1", "im2", "alpha"],
        required: 3,
        strict: true,
        return_type: two_images,
        eval: |args| pil_fn("blend", args, 2, Some(2)),
    },
    Builtin {
        name: "composite",
        module: MODULE,
        params: &["image1", "image2", "mask"],
        required: 3,
        strict: true,
        return_type: |args| {
            check_image("composite", "image1", args.first())?;
            check_image("composite", "image2", args.get(1))?;
            check_image("composite", "mask", args.get(2))?;
            Ok(image_type(false))
        },
        eval: |args| pil_fn("composite", args, 3, None),
    },
    Builtin {
        name: "convert",
        module: MODULE,
        params: &["self", "mode"],
        required: 2,
        strict: true,
        return_type: any_image,
        eval: |args| method_image(args, "convert", |py| Ok(vec![req(py, args, 1)?])),
    },
    Builtin {
        name: "crop",
        module: MODULE,
        params: &["self", "box"],
        required: 2,
        strict: true,
        return_type: image_of_mode,
        eval: |args| method_image(args, "crop", |py| Ok(vec![tuple_arg(py, args, 1)?])),
    },
    Builtin {
        name: "getchannel",
        module: MODULE,
        params: &["self", "channel"],
        required: 2,
        strict: true,
        return_type: ret!("getchannel", ColumnType::new(TypeKind::Image { width: None, height: None, mode: Some("L".into()) }, false)),
        eval: |args| method_image(args, "getchannel", |py| Ok(vec![req(py, args, 1)?])),
    },
    Builtin {
        name: "get_metadata",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("get_metadata", ColumnType::json(false)),
        eval: get_metadata,
    },
    Builtin {
        name: "point",
        module: MODULE,
        params: &["self", "lut", "mode"],
        required: 2,
        strict: true,
        return_type: any_image,
        eval: |args| method_image(args, "point", |py| Ok(vec![req(py, args, 1)?, opt(py, args, 2)?])),
    },
    Builtin {
        name: "resize",
        module: MODULE,
        params: &["self", "size"],
        required: 2,
        strict: true,
        return_type: image_of_mode,
        eval: |args| method_image(args, "resize", |py| Ok(vec![tuple_arg(py, args, 1)?])),
    },
    Builtin {
        name: "rotate",
        module: MODULE,
        params: &["self", "angle"],
        required: 2,
        strict: true,
        return_type: same_image,
        eval: |args| method_image(args, "rotate", |py| Ok(vec![req(py, args, 1)?])),
    },
    Builtin {
        name: "effect_spread",
        module: MODULE,
        params: &["self", "distance"],
        required: 2,
        strict: true,
        return_type: same_image,
        eval: |args| method_image(args, "effect_spread", |py| Ok(vec![req(py, args, 1)?])),
    },
    Builtin {
        name: "transpose",
        module: MODULE,
        params: &["self", "method"],
        required: 2,
        strict: true,
        return_type: same_image,
        eval: |args| method_image(args, "transpose", |py| Ok(vec![req(py, args, 1)?])),
    },
    Builtin {
        name: "entropy",
        module: MODULE,
        params: &["self", "mask", "extrema"],
        required: 1,
        strict: false,
        return_type: ret!("entropy", ColumnType::float(false)),
        eval: |args| {
            if matches!(args.first(), None | Some(Value::Null)) {
                return Ok(Value::Null);
            }
            with_py(|py| {
                let im = img(py, args, 0)?;
                Ok(Value::Float(im.call_method1("entropy", (opt(py, args, 1)?, opt(py, args, 2)?))?.extract()?))
            })
        },
    },
    Builtin {
        name: "getbands",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("getbands", ColumnType::json(false)),
        eval: |args| method_json(args, "getbands", none),
    },
    Builtin {
        name: "getbbox",
        module: MODULE,
        params: &["self", "alpha_only"],
        required: 1,
        strict: true,
        return_type: ret!("getbbox", ColumnType::json(true)),
        eval: getbbox,
    },
    Builtin {
        name: "getcolors",
        module: MODULE,
        params: &["self", "maxcolors"],
        required: 1,
        strict: true,
        return_type: ret!("getcolors", ColumnType::json(false)),
        eval: |args| {
            method_json(args, "getcolors", |py| {
                Ok(vec![match args.get(1) {
                    Some(v) => arg(py, v)?,
                    None => 256i64.into_pyobject(py)?.into_any(),
                }])
            })
        },
    },
    Builtin {
        name: "getextrema",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("getextrema", ColumnType::json(false)),
        eval: |args| method_json(args, "getextrema", none),
    },
    Builtin {
        name: "getpalette",
        module: MODULE,
        params: &["self", "mode"],
        required: 1,
        strict: false,
        return_type: ret!("getpalette", ColumnType::json(true)),
        eval: |args| {
            if matches!(args.first(), None | Some(Value::Null)) {
                return Ok(Value::Null);
            }
            method_json(args, "getpalette", |py| Ok(vec![opt(py, args, 1)?]))
        },
    },
    Builtin {
        name: "getpixel",
        module: MODULE,
        params: &["self", "xy"],
        required: 2,
        strict: true,
        return_type: ret!("getpixel", ColumnType::json(false)),
        eval: |args| method_json(args, "getpixel", |py| Ok(vec![tuple_arg(py, args, 1)?])),
    },
    Builtin {
        name: "getprojection",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("getprojection", ColumnType::json(false)),
        eval: |args| method_json(args, "getprojection", none),
    },
    Builtin {
        name: "histogram",
        module: MODULE,
        params: &["self", "mask", "extrema"],
        required: 1,
        strict: false,
        return_type: ret!("histogram", ColumnType::json(false)),
        eval: |args| {
            if matches!(args.first(), None | Some(Value::Null)) {
                return Ok(Value::Null);
            }
            method_json(args, "histogram", |py| Ok(vec![opt(py, args, 1)?, opt(py, args, 2)?]))
        },
    },
    Builtin {
        name: "quantize",
        module: MODULE,
        params: &["self", "colors", "method", "kmeans", "palette", "dither"],
        required: 1,
        strict: false,
        return_type: ret!("quantize", ColumnType::new(TypeKind::Image { width: None, height: None, mode: Some("P".into()) }, false)),
        eval: |args| {
            if matches!(args.first(), None | Some(Value::Null)) {
                return Ok(Value::Null);
            }
            quantize(args)
        },
    },
    Builtin {
        name: "reduce",
        module: MODULE,
        params: &["self", "factor", "box"],
        required: 2,
        strict: false,
        return_type: image_of_mode,
        eval: |args| {
            if matches!(args.first(), None | Some(Value::Null)) {
                return Ok(Value::Null);
            }
            method_image(args, "reduce", |py| {
                let bx = match args.get(2) {
                    None | Some(Value::Null) => py.None().into_bound(py),
                    Some(_) => tuple_arg(py, args, 2)?,
                };
                Ok(vec![req(py, args, 1)?, bx])
            })
        },
    },
    Builtin {
        name: "thumbnail",
        module: MODULE,
        params: &["self", "size", "resample", "reducing_gap"],
        required: 2,
        strict: false,
        return_type: image_of_mode,
        eval: |args| {
            if matches!(args.first(), None | Some(Value::Null)) {
                return Ok(Value::Null);
            }
            thumbnail(args)
        },
    },
    Builtin {
        name: "width",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("width", ColumnType::int(false)),
        eval: |args| attr_int(args, "width"),
    },
    Builtin {
        name: "height",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("height", ColumnType::int(false)),
        eval: |args| attr_int(args, "height"),
    },
    Builtin {
        name: "mode",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("mode", ColumnType::string(false)),
        eval: |args| with_py(|py| Ok(Value::String(img(py, args, 0)?.getattr("mode")?.extract()?))),
    },
    Builtin {
        name: "size",
        module: MODULE,
        params: &["self"],
        required: 1,
        strict: true,
        return_type: ret!("size", ColumnType::json(false)),
        eval: |args| {
            with_py(|py| {
                let (w, h): (i64, i64) = img(py, args, 0)?.getattr("size")?.extract()?;
                Ok(Value::Json(Json::Array(vec![Json::from(w), Json::from(h)])))
            })
        },
    },
];
