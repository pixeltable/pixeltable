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

// The module runs without the GIL on free-threaded CPython (3.13t/3.14t), which it declares: shared Rust state is
// behind mutexes or atomics, a mutex that can be held while detached from the interpreter is waited for detached
// (`MutexExt::lock_py_attached`), Python objects are held as `Py` (thread-safe reference counts), cached type objects are
// `PyOnceLock`s, and every pyclass is frozen (interior state behind a mutex). See ARCHITECTURE.md, "Concurrency of user
// code".
#[pymodule(name = "_core", gil_used = false)]
fn pxtcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    bindings::register(m)
}
