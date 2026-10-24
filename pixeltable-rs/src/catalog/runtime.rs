//! The process-wide catalog instance.
//!
//! The catalog is created lazily (from `$PIXELTABLE_HOME`) on first use, dropped by `reload()` (the next use
//! reloads it from disk) and wiped by `reset()`.
//!
//! Locking rule: the mutex is held only for catalog reads and writes, never while user Python code runs (UDFs,
//! iterators, user generators). Operations that run user code take a snapshot (`Arc`s of metadata and row data),
//! release the lock, compute, and re-acquire it to commit. The lock is acquired with `lock_py_attached`, which
//! detaches from the interpreter while waiting, so a thread holding the lock can always re-attach.
//!
//! Processes sharing a store: each operation also holds the store's file lock (`Store::lock()`), and first compares
//! the stamp of `catalog.json` with that of the in-memory catalog; if another process has committed since, the
//! catalog is reloaded before the operation runs (`Catalog::reload()`, which keeps the cached history and rows of a
//! table that the other process only appended to).

use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::RwLock;

use super::Catalog;
use crate::error::{Error, Result};
use crate::store::Store;

static CATALOG: Mutex<Option<Catalog>> = Mutex::new(None);
static RESERVED_NAMES: RwLock<Option<HashSet<String>>> = RwLock::new(None);

thread_local! {
    /// true while this thread holds the catalog lock (in `with_catalog`)
    static HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True while this thread holds the catalog lock.
pub fn lock_held() -> bool {
    HELD.with(|h| h.get())
}

struct HeldGuard;

impl Drop for HeldGuard {
    fn drop(&mut self) {
        HELD.with(|h| h.set(false));
    }
}

/// Runs `f` with exclusive access to the catalog (in this process and, through the store's file lock, among all
/// processes sharing the store), loading or reloading it first if needed. `f` is atomic: if it fails or panics, its
/// changes are rolled back in memory and on disk (see `Catalog::checkpoint()`); a panic becomes an InternalError.
pub fn with_catalog<R>(py: Python<'_>, f: impl FnOnce(&mut Catalog) -> Result<R>) -> Result<R> {
    if HELD.with(|h| h.get()) {
        // Python code called while this thread holds the lock (e.g. resolving a stored UDF) re-entered the catalog;
        // the mutex is not reentrant, so fail instead of deadlocking
        return Err(Error::internal("the catalog was accessed re-entrantly (from Python code run under its lock)"));
    }
    let mut guard = CATALOG.lock_py_attached(py);
    HELD.with(|h| h.set(true));
    let _reset = HeldGuard;
    let store = match guard.as_ref() {
        Some(cat) => cat.store().clone(),
        None => Store::open_default()?,
    };
    let _store_lock = py.detach(|| store.lock())?;
    let on_disk = py.detach(|| store.read_stamp())?;
    match guard.take() {
        Some(cat) if Some(cat.stamp()) == on_disk => *guard = Some(cat),
        // another process committed: reload, keeping what is still valid of the cached history and rows
        Some(cat) => *guard = Some(py.detach(|| Catalog::reload(store, cat))?),
        None => *guard = Some(py.detach(|| Catalog::load(store, true))?),
    }
    let Some(cat) = guard.as_mut() else { return Err(Error::internal("catalog failed to initialize")) };
    let (result, discard) = run_atomic(cat, f);
    if discard {
        // the in-memory state may no longer match the store: reload it on the next operation
        *guard = None;
    }
    result
}

/// Runs `f` between `Catalog::checkpoint()` and `Catalog::finish()`, turning a panic into an InternalError. Also
/// returns whether the in-memory catalog must be discarded (after a panic, or if the rollback failed).
fn run_atomic<R>(cat: &mut Catalog, f: impl FnOnce(&mut Catalog) -> Result<R>) -> (Result<R>, bool) {
    let checkpoint = cat.checkpoint();
    let (result, panicked) = match catch_unwind(AssertUnwindSafe(|| f(cat))) {
        Ok(result) => (result, false),
        Err(payload) => (Err(Error::internal(format!("internal error: {}", panic_message(payload.as_ref())))), true),
    };
    let finished = cat.finish(checkpoint, result.is_ok());
    let discard = panicked || finished.is_err();
    match (result, finished) {
        (Ok(_), Err(rollback_err)) => (Err(rollback_err), discard),
        (result, _) => (result, discard),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Drops all in-memory state; the next operation reloads from disk.
pub fn reload(py: Python<'_>) {
    let mut guard = CATALOG.lock_py_attached(py);
    *guard = None;
    crate::functions::udf::reset_load_state();
}

/// Deletes all data of the current database and starts with an empty catalog.
pub fn reset(py: Python<'_>) -> Result<()> {
    let mut guard = CATALOG.lock_py_attached(py);
    *guard = None;
    crate::functions::udf::reset_load_state();
    let store = Store::open_default()?;
    let _store_lock = py.detach(|| store.lock())?;
    *guard = Some(py.detach(|| {
        store.wipe()?;
        Catalog::load(store, false)
    })?);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::IfExists;
    use crate::catalog::path::Path;
    use crate::error::ErrorCode;

    #[test]
    fn panic_rolls_back_memory_and_disk() -> Result<()> {
        let tmp = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test-tmp");
        let dir = tmp.join(uuid::Uuid::new_v4().to_string());
        let store = Store::open(&dir, "test")?;
        let mut cat = Catalog::load(store.clone(), false)?;
        let d1 = Path::parse("d1", false, false)?;
        let (result, discard) = run_atomic(&mut cat, |cat| {
            cat.create_dir(&d1, IfExists::Error, false)?;
            if cat.dir_at(&d1).is_some() {
                panic!("boom");
            }
            Ok(())
        });
        match result {
            Err(Error::Pxt { code: ErrorCode::InternalError, message }) => assert!(message.contains("boom")),
            other => panic!("expected an InternalError, got {other:?}"),
        }
        assert!(discard);
        assert!(cat.dir_at(&d1).is_none());
        assert!(Catalog::load(store.clone(), false)?.dir_at(&d1).is_none());
        assert_eq!(store.read_stamp()?, Some(cat.stamp()));
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }
}
