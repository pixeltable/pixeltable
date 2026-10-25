//! Builtin iterators over images: `pixeltable.functions.image.tile_iterator` (upstream
//! `pixeltable/functions/image.py`); cropping goes through PIL.

use pyo3::prelude::*;

use super::iterator::{BuiltinIterator, IteratorCall, RowSource};
use super::signature::{Param, ParamKind};
use crate::error::{Error, Result};
use crate::expr::Expr;
use crate::types::{ColumnType, JsonSchema, JsonSpec, TypeKind};
use crate::value::{Json, Value};

pub static TILE_ITERATOR: BuiltinIterator = BuiltinIterator {
    fqn: "pixeltable.functions.image.tile_iterator",
    defined_at: "pixeltable.functions.image.tile_iterator",
    params: tile_params,
    unstored_cols: &["tile"],
    output_schema: tile_schema,
    validate: validate_tile_args,
    start: start_tile_iterator,
};

fn int_tuple(n: usize) -> ColumnType {
    let schema =
        JsonSchema { spec: JsonSpec::List(vec![ColumnType::int(false); n]), variadic: None, optional_keys: Default::default() };
    ColumnType::new(TypeKind::Json(Some(Box::new(schema))), false)
}

/// `(image: pxt.Image, tile_size: tuple[int, int], *, overlap: tuple[int, int] = (0, 0))`
fn tile_params() -> Vec<Param> {
    let image = ColumnType::new(TypeKind::Image { width: None, height: None, mode: None }, false);
    vec![
        Param { name: "image".into(), kind: ParamKind::PositionalOrKeyword, col_type: Some(image), default: None, is_batched: false },
        Param {
            name: "tile_size".into(),
            kind: ParamKind::PositionalOrKeyword,
            col_type: Some(int_tuple(2)),
            default: None,
            is_batched: false,
        },
        Param {
            name: "overlap".into(),
            kind: ParamKind::KeywordOnly,
            col_type: Some(int_tuple(2)),
            default: Some(Expr::literal(Value::Json(serde_json::json!([0, 0])), int_tuple(2))),
            is_batched: false,
        },
    ]
}

fn tile_schema(_args: &[(String, Value)]) -> Result<Vec<(String, ColumnType)>> {
    Ok(vec![
        ("tile".to_string(), ColumnType::new(TypeKind::Image { width: None, height: None, mode: None }, false)),
        ("tile_coord".to_string(), int_tuple(2)),
        ("tile_box".to_string(), int_tuple(4)),
    ])
}

fn pair(args: &[(String, Value)], name: &str) -> Option<(i64, i64)> {
    match args.iter().find(|(n, _)| n == name).map(|(_, v)| v) {
        Some(Value::Json(Json::Array(a))) if a.len() == 2 => Some((a[0].as_i64()?, a[1].as_i64()?)),
        _ => None,
    }
}

fn pair_str(p: (i64, i64)) -> String {
    format!("[{}, {}]", p.0, p.1)
}

fn validate_tile_args(args: &[(String, Value)]) -> Result<()> {
    let overlap = pair(args, "overlap").unwrap_or((0, 0));
    let Some(tile_size) = pair(args, "tile_size") else { return Ok(()) };
    if tile_size.0 <= 0 || tile_size.1 <= 0 {
        crate::bail!(InvalidArgument, "`tile_size` dimensions must be positive; got {}", pair_str(tile_size));
    }
    if overlap.0 < 0 || overlap.1 < 0 {
        crate::bail!(InvalidArgument, "`overlap` dimensions must be non-negative; got {}", pair_str(overlap));
    }
    if overlap.0 >= tile_size.0 || overlap.1 >= tile_size.1 {
        crate::bail!(
            UnsupportedOperation,
            "`overlap` dimensions {} are not strictly smaller than `tile_size` {}",
            pair_str(overlap),
            pair_str(tile_size)
        );
    }
    Ok(())
}

struct TileIter {
    image: Py<PyAny>,
    tile_size: (i64, i64),
    overlap: (i64, i64),
    xlen: i64,
    ylen: i64,
    i: i64,
    j: i64,
}

fn start_tile_iterator(py: Python<'_>, args: &[(String, Value)]) -> Result<Box<dyn RowSource>> {
    let image_val = args.iter().find(|(n, _)| n == "image").map(|(_, v)| v.clone()).unwrap_or(Value::Null);
    let image = crate::media::open_image(py, &image_val)?;
    image.call_method0("load")?;
    let (width, height): (i64, i64) = image.getattr("size")?.extract()?;
    let tile_size = pair(args, "tile_size").ok_or_else(|| Error::internal("tile_iterator() without tile_size"))?;
    let overlap = pair(args, "overlap").unwrap_or((0, 0));
    let xlen = (width - overlap.0 - 1).div_euclid(tile_size.0 - overlap.0) + 1;
    let ylen = (height - overlap.1 - 1).div_euclid(tile_size.1 - overlap.1) + 1;
    Ok(Box::new(TileIter { image: image.unbind(), tile_size, overlap, xlen, ylen, i: 0, j: 0 }))
}

impl RowSource for TileIter {
    fn next_row(&mut self, py: Python<'_>, call: &IteratorCall) -> Result<Option<Vec<Value>>> {
        if self.j >= self.ylen {
            return Ok(None);
        }
        let x1 = self.i * (self.tile_size.0 - self.overlap.0);
        let y1 = self.j * (self.tile_size.1 - self.overlap.1);
        let x2 = x1 + self.tile_size.0;
        let y2 = y1 + self.tile_size.1;
        let tile = self.image.bind(py).call_method1("crop", ((x1, y1, x2, y2),))?;
        let coord = Value::Json(serde_json::json!([self.i, self.j]));
        let bx = Value::Json(serde_json::json!([x1, y1, x2, y2]));
        self.i += 1;
        if self.i >= self.xlen {
            self.i = 0;
            self.j += 1;
        }
        Ok(Some(
            call.value_outputs()
                .iter()
                .map(|o| match o.orig_name.as_str() {
                    "tile" => Value::PyObj(std::sync::Arc::new(tile.clone().unbind())),
                    "tile_coord" => coord.clone(),
                    "tile_box" => bx.clone(),
                    _ => Value::Null,
                })
                .collect(),
        ))
    }

    fn seek(&mut self, _py: Python<'_>, _call: &IteratorCall, pos: i64, _stored: &[(String, Value)]) -> Result<()> {
        self.j = pos.div_euclid(self.xlen);
        self.i = pos.rem_euclid(self.xlen);
        Ok(())
    }

    fn close(&mut self, _py: Python<'_>) {}
}
