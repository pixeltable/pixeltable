//! Data parallelism (rayon) for the pure-Rust parts of execution: evaluating expressions over many rows, sorting.
//!
//! Results are always in row order (rayon's indexed collect), so output and the error reported (the first failing
//! row's) are the same as with sequential evaluation. Only expressions that run no user code are evaluated in
//! parallel (`is_pure()`): calls of Python functions must have been made ahead (`exec::prefetch`), since rayon's
//! threads would run them concurrently and out of order. The calling thread is detached from the interpreter while
//! rayon works (`detached()`), so that a worker that needs Python (media helpers) can attach.

use rayon::prelude::*;
use std::cmp::Ordering;

use crate::expr::{ColumnProp, ColumnRef, Expr, ExprKind};
use crate::functions::FuncRef;
use crate::functions::udf::UdfImpl;

/// The fewest rows worth spreading over threads.
pub const MIN_PARALLEL_ROWS: usize = 4096;

fn parallel(n: usize) -> bool {
    n >= MIN_PARALLEL_ROWS && pool().current_num_threads() > 1
}

/// The thread pool of this process. Not rayon's global pool: after a fork, the child has none of the parent's threads,
/// and the global pool would wait for them forever; a child builds its own pool.
fn pool() -> std::sync::Arc<rayon::ThreadPool> {
    static POOL: parking_lot::Mutex<Option<(u32, std::sync::Arc<rayon::ThreadPool>)>> = parking_lot::Mutex::new(None);
    let mut guard = POOL.lock();
    let pid = std::process::id();
    match guard.as_ref() {
        Some((p, pool)) if *p == pid => pool.clone(),
        _ => {
            if let Some(stale) = guard.take() {
                // the parent's pool: its threads do not exist here, so it must not be dropped (that would wait for them)
                std::mem::forget(stale);
            }
            let pool = std::sync::Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .thread_name(|i| format!("pixeltable-par-{i}"))
                    .stack_size(16 << 20)
                    .build()
                    .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("a one-thread pool")),
            );
            *guard = Some((pid, pool.clone()));
            pool
        }
    }
}

/// Runs `f` (which uses rayon) in this process's pool, detached from the interpreter if this thread is attached to it
/// (pool threads may need to attach).
pub fn detached<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    let pool = pool();
    // SAFETY: PyGILState_Check() may be called without holding the GIL
    if unsafe { pyo3::ffi::PyGILState_Check() } == 1 {
        pyo3::Python::attach(|py| py.detach(|| pool.install(f)))
    } else {
        pool.install(f)
    }
}

/// `(0..n).map(f).collect()`, in parallel for many rows (`f` must be pure: see `is_pure()`). Row 0 is evaluated
/// first, alone, so that values an expression computes once (the query embedding of a similarity) are computed once.
pub fn map_rows<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if !parallel(n) {
        return (0..n).map(f).collect();
    }
    let first = f(0);
    detached(|| {
        let mut out = Vec::with_capacity(n);
        out.push(first);
        out.par_extend((1..n).into_par_iter().map(&f));
        out
    })
}

/// `(0..n).map(f).collect()`, or the error of the first row (in row order) that fails. With `pure` (`f` runs no user
/// code, see `is_pure()`), fixed chunks of rows are evaluated in parallel for many rows, each up to its first failure;
/// else the rows are evaluated in order in this thread up to the first failure, so that evaluation that runs user code
/// makes no call a sequential evaluation would not make.
pub fn try_map_rows<T: Send>(
    n: usize,
    pure: bool,
    f: impl Fn(usize) -> crate::error::Result<T> + Sync + Send,
) -> crate::error::Result<Vec<T>> {
    let run = |range: std::ops::Range<usize>| -> (Vec<T>, Option<crate::error::Error>) {
        let mut out = Vec::with_capacity(range.len());
        for i in range {
            match f(i) {
                Ok(v) => out.push(v),
                Err(err) => return (out, Some(err)),
            }
        }
        (out, None)
    };
    if !pure || !parallel(n) {
        let (out, failed) = run(0..n);
        return failed.map_or(Ok(out), Err);
    }
    let mut out = Vec::with_capacity(n);
    for (part, failed) in map_chunks(n, run) {
        if let Some(err) = failed {
            return Err(err);
        }
        out.extend(part);
    }
    Ok(out)
}

/// Calls `f(i, &mut items[i])` for every item, in parallel for many items (`f` must be pure), and returns the results
/// in order.
pub fn map_mut<T: Send, R: Send>(items: &mut [T], f: impl Fn(usize, &mut T) -> R + Sync + Send) -> Vec<R> {
    if !parallel(items.len()) {
        return items.iter_mut().enumerate().map(|(i, x)| f(i, x)).collect();
    }
    let (first, rest) = items.split_at_mut(1);
    let r0 = f(0, &mut first[0]);
    detached(|| {
        let mut out = Vec::with_capacity(rest.len() + 1);
        out.push(r0);
        out.par_extend(rest.par_iter_mut().enumerate().map(|(i, x)| f(i + 1, x)));
        out
    })
}

/// True if `n` items are worth spreading over threads.
pub fn worth_parallel(n: usize) -> bool {
    parallel(n)
}

/// True on a thread of a rayon pool (data-parallel work), where user code must never run: its calls would run
/// concurrently and out of row order, ignoring `max_concurrency`.
pub fn on_parallel_thread() -> bool {
    rayon::current_thread_index().is_some()
}

/// Rows per chunk of chunked parallel work: fixed, so that results never depend on the number of threads.
pub const CHUNK_ROWS: usize = 16 * 1024;

/// `f` over the chunks `[0, CHUNK_ROWS)`, `[CHUNK_ROWS, 2 * CHUNK_ROWS)`, ... of `0..n`, in parallel (`f` must be
/// pure); the results in chunk order.
pub fn map_chunks<T: Send>(n: usize, f: impl Fn(std::ops::Range<usize>) -> T + Sync + Send) -> Vec<T> {
    let chunks: Vec<std::ops::Range<usize>> = (0..n).step_by(CHUNK_ROWS).map(|s| s..(s + CHUNK_ROWS).min(n)).collect();
    if chunks.len() <= 1 || !parallel(n) {
        return chunks.into_iter().map(f).collect();
    }
    detached(|| chunks.into_par_iter().map(&f).collect())
}

/// A stable sort, in parallel for many elements.
pub fn sort_by<T: Send>(v: &mut [T], cmp: impl Fn(&T, &T) -> Ordering + Sync) {
    if parallel(v.len()) {
        detached(|| v.par_sort_by(&cmp));
    } else {
        v.sort_by(&cmp);
    }
}

/// True if evaluating `e` runs no user code and no Python-backed builtin, so that rows can be evaluated on several
/// threads: calls of Python functions are allowed only where `prefetched(node)` (their outcomes are looked up), column
/// reads only where `col_pure(col)` (e.g. not iterator outputs, which run the iterator again), aggregate calls only as
/// precomputed values (they are never evaluated per row).
pub fn is_pure(e: &Expr, col_pure: &dyn Fn(&ColumnRef) -> bool, prefetched: &dyn Fn(&Expr) -> bool) -> bool {
    let children = |e: &Expr| e.kind.children().into_iter().all(|c| is_pure(c, col_pure, prefetched));
    match &e.kind {
        ExprKind::ColumnRef(c) => col_pure(c),
        ExprKind::ColumnProperty { col, prop } => *prop != ColumnProp::LocalPath && is_pure(col, col_pure, prefetched),
        ExprKind::Variable(_) => false,
        ExprKind::FunctionCall(fc) => match &fc.func {
            FuncRef::Builtin(b) => !b.calls_python() && children(e),
            FuncRef::Agg(_) => true,
            FuncRef::Udf(call) => match (&call.def.imp, &call.instantiated) {
                (UdfImpl::Template { .. }, Some(inst)) => is_pure(inst, col_pure, prefetched),
                (UdfImpl::Callable { .. }, _) => prefetched(e),
                _ => false,
            },
        },
        // the embedding of the query item is computed once per expression, before rows are evaluated in parallel
        // (`exec::query::init_similarities()`); if that failed, rows are evaluated in order, and the first reports it
        ExprKind::Similarity(sim) => sim.cache.get().is_some() && is_pure(&sim.col, col_pure, prefetched),
        _ => children(e),
    }
}
