//! `pixeltable.functions.array`

use super::Builtin;
use crate::value::{Json, Value};

pub const MODULE: &str = "pixeltable.functions.array";

/// `ndarray.tolist()`: nested lists of Python scalars.
pub fn array_to_json(a: &crate::value::NdArray) -> Json {
    fn build(flat: &[Json], shape: &[usize]) -> Json {
        match shape.split_first() {
            None => flat.first().cloned().unwrap_or(Json::Null),
            Some((&n, rest)) => {
                let stride: usize = rest.iter().product();
                Json::Array((0..n).map(|i| build(&flat[i * stride..(i + 1) * stride], rest)).collect())
            }
        }
    }
    let flat: Vec<Json> = crate::value::array_elements(a).iter().map(|v| crate::value::value_to_json(v).unwrap_or(Json::Null)).collect();
    if a.shape.is_empty() {
        return flat.into_iter().next().unwrap_or(Json::Null);
    }
    build(&flat, &a.shape)
}

pub static FUNCTIONS: &[Builtin] = &[Builtin::new("to_list", "(self: Array) -> Json", |a| match a.get(0) {
    Value::Array(arr) => Ok(Value::Json(array_to_json(arr))),
    _ => Ok(Value::Null),
})
.method()];
