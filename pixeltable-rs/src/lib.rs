//! `pxtcore`: the Rust backend of pixeltable-rs, exposed to Python as `pixeltable._core`.
//!
//! See ARCHITECTURE.md for a map. This file only declares the modules and the Python module entry point.

pub mod bindings;
pub mod catalog;
pub mod config;
pub mod error;
pub mod exec;
pub mod expr;
pub mod functions;
pub mod media;
pub mod store;
pub mod types;
pub mod value;

use pyo3::prelude::*;

#[pymodule(name = "_core")]
fn pxtcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    bindings::register(m)
}
