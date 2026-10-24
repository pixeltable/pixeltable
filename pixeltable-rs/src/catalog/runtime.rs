//! The process-wide catalog instance.
//!
//! The catalog is created lazily (from `$PIXELTABLE_HOME`) on first use, dropped by `reload()` (the next use
//! reloads it from disk) and wiped by `reset()`.
//!
//! Locking rule: the mutex is held only for catalog reads and writes, never while user Python code runs (UDFs,
//! iterators, user generators). Operations that run user code take a snapshot (`Arc`s of metadata and row data),
//! release the lock, compute, and re-acquire it to commit. The lock is acquired with `lock_py_attached`, which
//! detaches from the interpreter while waiting, so a thread holding the lock can always re-attach.

use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use std::collections::HashSet;
use std::sync::RwLock;

use super::Catalog;
use crate::error::{Error, Result};
use crate::store::Store;

static CATALOG: Mutex<Option<Catalog>> = Mutex::new(None);
static RESERVED_NAMES: RwLock<Option<HashSet<String>>> = RwLock::new(None);

/// Runs `f` with exclusive access to the catalog, loading it first if needed.
pub fn with_catalog<R>(py: Python<'_>, f: impl FnOnce(&mut Catalog) -> Result<R>) -> Result<R> {
    let mut guard = CATALOG.lock_py_attached(py);
    if guard.is_none() {
        *guard = Some(Catalog::load(Store::open_default()?)?);
    }
    match guard.as_mut() {
        Some(cat) => f(cat),
        None => Err(Error::internal("catalog failed to initialize")),
    }
}

/// Drops all in-memory state; the next operation reloads from disk.
pub fn reload(py: Python<'_>) {
    let mut guard = CATALOG.lock_py_attached(py);
    *guard = None;
}

/// Deletes all data of the current database and starts with an empty catalog.
pub fn reset(py: Python<'_>) -> Result<()> {
    let mut guard = CATALOG.lock_py_attached(py);
    *guard = None;
    let store = Store::open_default()?;
    store.wipe()?;
    *guard = Some(Catalog::load(store)?);
    Ok(())
}

/// Registers the attribute names of the Python table classes, which cannot be used as column names.
pub fn set_reserved_names(names: impl IntoIterator<Item = String>) {
    if let Ok(mut guard) = RESERVED_NAMES.write() {
        *guard = Some(names.into_iter().map(|n| n.to_ascii_lowercase()).collect());
    }
}

const PYTHON_KEYWORDS: [&str; 32] = [
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield",
];

/// True if `folded_name` is reserved (a table attribute or a Python keyword).
pub fn is_reserved_name(folded_name: &str) -> bool {
    if PYTHON_KEYWORDS.contains(&folded_name) {
        return true;
    }
    RESERVED_NAMES.read().ok().and_then(|g| g.as_ref().map(|s| s.contains(folded_name))).unwrap_or(false)
}
