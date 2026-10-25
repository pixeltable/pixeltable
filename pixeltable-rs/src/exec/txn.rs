//! Consistency of mutations that compute rows without the catalog lock.
//!
//! Most mutations run in one `with_catalog` section and are atomic. Those that may run user code (UDFs, UDAs,
//! iterators) must not hold the catalog lock while it runs: they plan under the lock (capturing a snapshot), compute
//! without it and commit under it. Two mechanisms make their result what it would be had they run alone, as
//! upstream's table locks make it:
//! - commit validation: a plan records the data and schema versions of every table it read (`ReadSet`;
//!   `TableMd::data_version` advances with every change of a table's rows, in any process sharing the store); a commit
//!   that finds one changed fails with SERIALIZATION_FAILURE, and `run()` plans and computes the operation again
//!   (against the new state), up to `MAX_RETRIES` times;
//! - writer locks (`lock()`): in this process, the operations on one tree of tables (a base table and the views over
//!   it, keyed by the root base table) wait for each other, so that they are rarely redone and a long operation is not
//!   starved by short ones. An operation that rewrites rows or changes a schema holds its tree exclusively from plan
//!   to commit; one that only appends rows (insert) or runs in one catalog-lock section holds it shared. A waiting
//!   exclusive holder blocks new shared holders.
//!
//! An operation started by user code that an operation runs (on a UDF worker thread, on the event-loop thread, or on
//! a thread that already holds another tree's lock) never waits for a writer lock: it takes the lock only if it is
//! free and else runs without it, relying on validation alone, so that an operation never waits for user code that
//! waits for it. Other processes are not excluded: validation catches their commits.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::thread::ThreadId;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use uuid::Uuid;

use super::scan::ElemInfo;
use crate::catalog::Catalog;
use crate::catalog::runtime::with_catalog;
use crate::error::{Error, ErrorCode, Result, py_repr};
use crate::expr::{Expr, ExprKind};
use crate::functions::FuncRef;
use crate::functions::udf::UdfImpl;

// ------------------------------------------------------------------------------------------------ read sets

/// The tables a plan read, with their data and schema versions then.
#[derive(Debug, Clone, Default)]
pub struct ReadSet(Vec<(Uuid, u64, i64)>);

impl ReadSet {
    pub fn of(cat: &Catalog, tbl_ids: impl IntoIterator<Item = Uuid>) -> Result<ReadSet> {
        let mut rs = ReadSet::default();
        for id in tbl_ids {
            rs.add(cat, id)?;
        }
        Ok(rs)
    }

    pub fn add(&mut self, cat: &Catalog, tbl_id: Uuid) -> Result<()> {
        if !self.0.iter().any(|(id, _, _)| *id == tbl_id) {
            let md = cat.table(tbl_id)?;
            self.0.push((tbl_id, md.data_version, md.current_schema_version));
        }
        Ok(())
    }

    /// Fails with SERIALIZATION_FAILURE (`run()` then redoes the operation) if the rows or the schema of a table
    /// changed since the set was captured; `what` names the operation for the message ("the update of 't'").
    pub fn check(&self, cat: &Catalog, what: &str) -> Result<()> {
        for (id, data_version, schema_version) in &self.0 {
            let current = cat.table(*id).ok().map(|md| (md.data_version, md.current_schema_version));
            if current != Some((*data_version, *schema_version)) {
                return Err(conflict(what));
            }
        }
        Ok(())
    }
}

/// The error of a commit that found what its operation read changed (`run()` retries on it).
pub fn conflict(what: &str) -> Error {
    Error::new(ErrorCode::SerializationFailure, format!("The data read by {what} changed while it was being computed; retry"))
}

pub fn is_conflict(err: &Error) -> bool {
    err.code() == Some(ErrorCode::SerializationFailure)
}

// ------------------------------------------------------------------------------------------------ user code

/// True if evaluating `e` against rows of the path `infos` may run user code (a Python function, aggregate or
/// iterator, also through the unstored computed columns and iterator outputs it reads), which must not run under
/// the catalog lock.
pub fn runs_user_code(e: &Expr, infos: &[&ElemInfo]) -> bool {
    let mut found = false;
    e.visit(&mut |n| {
        if found {
            return;
        }
        found = match &n.kind {
            ExprKind::FunctionCall(fc) => match &fc.func {
                FuncRef::Builtin(_) => false,
                FuncRef::Agg(a) => a.py_function().is_some(),
                FuncRef::Udf(call) => match (&call.def.imp, &call.instantiated) {
                    (UdfImpl::Template { .. }, Some(inst)) => runs_user_code(inst, infos),
                    _ => true,
                },
            },
            // the embedding of the query item
            ExprKind::Similarity(_) => true,
            ExprKind::ColumnRef(c) => infos.iter().filter(|i| i.tbl_id == c.tbl_id).any(|i| {
                let unstored_output = i.iterator.as_ref().is_some_and(|it| {
                    it.call.def.builtin.is_none() && it.outputs.iter().any(|(col, _, stored)| *col == c.col_id && !stored)
                });
                unstored_output || i.unstored.get(&c.col_id).is_some_and(|ue| runs_user_code(ue, infos))
            }),
            _ => false,
        };
    });
    found
}

// ------------------------------------------------------------------------------------------------ writer locks

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Shared,
    Exclusive,
}

#[derive(Default)]
struct TreeLock {
    shared: usize,
    exclusive: Option<ThreadId>,
    waiting_exclusive: usize,
}

struct Locks {
    pid: u32,
    trees: HashMap<Uuid, TreeLock>,
    /// per tree: commits made in this process by operations that did not hold the tree's lock (see `Attempts`)
    unlocked_commits: HashMap<Uuid, u64>,
}

fn locks() -> &'static (Mutex<Locks>, Condvar) {
    static LOCKS: OnceLock<(Mutex<Locks>, Condvar)> = OnceLock::new();
    LOCKS.get_or_init(|| {
        let l = Locks { pid: std::process::id(), trees: HashMap::new(), unlocked_commits: HashMap::new() };
        (Mutex::new(l), Condvar::new())
    })
}

/// The lock state of this process (a forked child starts with none: the parent's holders do not exist in it).
fn state() -> parking_lot::MutexGuard<'static, Locks> {
    let mut g = locks().0.lock();
    let pid = std::process::id();
    if g.pid != pid {
        *g = Locks { pid, trees: HashMap::new(), unlocked_commits: HashMap::new() };
        HELD.with(|h| h.borrow_mut().clear());
    }
    g
}

thread_local! {
    /// the writer locks this thread holds: (root, mode, depth)
    static HELD: RefCell<Vec<(Uuid, Mode, u32)>> = const { RefCell::new(Vec::new()) };
}

/// The writer lock an operation holds (or runs without) until it is dropped.
pub struct WriterGuard {
    root: Uuid,
    /// the mode held; None: the operation runs without the lock
    held: Option<Mode>,
    /// true if this thread already held the lock for an operation that is still running (this one was started by user
    /// code that operation runs)
    reentered: bool,
}

impl WriterGuard {
    /// True if no other operation of this process writes the tree while this guard lives.
    pub fn exclusive(&self) -> bool {
        self.held == Some(Mode::Exclusive) && !self.reentered
    }

    /// Records a commit of the operation (one that did not take the lock itself is noted, see `Attempts`).
    pub fn committed(&self) {
        if self.held.is_none() || self.reentered {
            *state().unlocked_commits.entry(self.root).or_default() += 1;
        }
    }
}

impl Drop for WriterGuard {
    fn drop(&mut self) {
        let Some(mode) = self.held else { return };
        let last = HELD.with(|h| {
            let mut h = h.borrow_mut();
            match h.iter().position(|(r, _, _)| *r == self.root) {
                Some(i) if h[i].2 > 1 => {
                    h[i].2 -= 1;
                    false
                }
                Some(i) => {
                    h.remove(i);
                    true
                }
                None => false,
            }
        });
        if !last {
            return;
        }
        let mut st = state();
        if let Some(t) = st.trees.get_mut(&self.root) {
            match mode {
                Mode::Shared => t.shared = t.shared.saturating_sub(1),
                Mode::Exclusive => t.exclusive = None,
            }
            if t.shared == 0 && t.exclusive.is_none() && t.waiting_exclusive == 0 {
                st.trees.remove(&self.root);
            }
        }
        drop(st);
        locks().1.notify_all();
    }
}

/// True if this thread runs on behalf of an operation that waits for it (see the module comment).
fn nested() -> bool {
    super::dispatch::in_worker()
        || super::dispatch::in_loop_thread()
        || crate::catalog::runtime::lock_held()
        || HELD.with(|h| !h.borrow().is_empty())
}

fn can_take(t: &TreeLock, mode: Mode, wait_for_waiters: bool) -> bool {
    match mode {
        Mode::Shared => t.exclusive.is_none() && (!wait_for_waiters || t.waiting_exclusive == 0),
        Mode::Exclusive => t.exclusive.is_none() && t.shared == 0,
    }
}

const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(50);

/// Takes the writer lock of the tree whose root base table is `root` (see the module comment): waits for it, or,
/// for a nested operation, takes it only if it is free.
pub fn lock(py: Python<'_>, root: Uuid, mode: Mode) -> Result<WriterGuard> {
    // re-entry: a thread that holds the tree's lock (an operation called by user code run inline by an operation)
    let reentered = HELD.with(|h| {
        let mut h = h.borrow_mut();
        match h.iter_mut().find(|(r, _, _)| *r == root) {
            Some(e) if e.1 == Mode::Exclusive || mode == Mode::Shared => {
                e.2 += 1;
                Some(Some(e.1))
            }
            // a shared holder cannot upgrade (two of them would wait for each other)
            Some(_) => Some(None),
            None => None,
        }
    });
    if let Some(held) = reentered {
        return Ok(WriterGuard { root, held, reentered: true });
    }
    let me = std::thread::current().id();
    let take = |t: &mut TreeLock| match mode {
        Mode::Shared => t.shared += 1,
        Mode::Exclusive => t.exclusive = Some(me),
    };
    if nested() {
        let mut st = state();
        let t = st.trees.entry(root).or_default();
        let held = if can_take(t, mode, false) {
            take(t);
            Some(mode)
        } else {
            None
        };
        drop(st);
        if held.is_some() {
            HELD.with(|h| h.borrow_mut().push((root, mode, 1)));
        }
        return Ok(WriterGuard { root, held, reentered: false });
    }
    py.detach(|| -> Result<()> {
        let (_, cv) = locks();
        let mut st = state();
        let mut waiting = false;
        loop {
            let t = st.trees.entry(root).or_default();
            if can_take(t, mode, true) {
                take(t);
                if waiting {
                    t.waiting_exclusive -= 1;
                }
                return Ok(());
            }
            if mode == Mode::Exclusive && !waiting {
                t.waiting_exclusive += 1;
                waiting = true;
            }
            cv.wait_for(&mut st, SIGNAL_CHECK_INTERVAL);
            // a KeyboardInterrupt delivered to the main thread while it waits
            if let Err(e) = parking_lot::MutexGuard::unlocked(&mut st, || Python::attach(|py| py.check_signals())) {
                if waiting {
                    if let Some(t) = st.trees.get_mut(&root) {
                        t.waiting_exclusive -= 1;
                    }
                    drop(st);
                    cv.notify_all();
                }
                return Err(e.into());
            }
        }
    })?;
    HELD.with(|h| h.borrow_mut().push((root, mode, 1)));
    Ok(WriterGuard { root, held: Some(mode), reentered: false })
}

/// The root base table of the tree that table `tbl_id` belongs to (bases never change, so it is looked up once); the
/// table itself if it does not exist (the operation then fails with its own error).
pub fn root_of(py: Python<'_>, tbl_id: Uuid) -> Result<Uuid> {
    static ROOTS: OnceLock<Mutex<HashMap<Uuid, Uuid>>> = OnceLock::new();
    let roots = ROOTS.get_or_init(Default::default);
    if let Some(r) = roots.lock().get(&tbl_id) {
        return Ok(*r);
    }
    let root = with_catalog(py, |cat| Ok(cat.table(tbl_id).is_ok().then(|| cat.base_chain(tbl_id).last().copied()).flatten()))?;
    match root {
        Some(r) => {
            roots.lock().insert(tbl_id, r);
            Ok(r)
        }
        None => Ok(tbl_id),
    }
}

/// Takes the writer lock of table `tbl_id`'s tree (see `lock()`).
pub fn lock_table(py: Python<'_>, tbl_id: Uuid, mode: Mode) -> Result<WriterGuard> {
    let root = root_of(py, tbl_id)?;
    lock(py, root, mode)
}

/// Runs a mutation of table `tbl_id`'s tree in one catalog-lock section, holding its writer lock in `mode`.
pub fn write<R>(py: Python<'_>, tbl_id: Uuid, mode: Mode, f: impl FnOnce(&mut Catalog) -> Result<R>) -> Result<R> {
    let guard = lock_table(py, tbl_id, mode)?;
    let r = with_catalog(py, f)?;
    guard.committed();
    Ok(r)
}

// ------------------------------------------------------------------------------------------------ retries

/// How often an operation is planned and computed again after its commit found what it read changed. Conflicts
/// come from other processes (this process's writers wait for each other), whose operations each commit once, so a
/// redo rarely conflicts again; the limit stops an operation that never gets to commit, as upstream's retry limit
/// does.
pub const MAX_RETRIES: u32 = 100;

/// The retries of one operation.
pub struct Attempts<'g> {
    guard: &'g WriterGuard,
    conflicts: u32,
    /// conflicts caused by operations of this process that ran without the tree's lock while this operation held it
    /// exclusively: operations started by the user code it runs, which a redo would repeat
    nested_conflicts: u32,
    unlocked_at_plan: u64,
}

impl<'g> Attempts<'g> {
    pub fn new(guard: &'g WriterGuard) -> Attempts<'g> {
        Attempts { guard, conflicts: 0, nested_conflicts: 0, unlocked_at_plan: 0 }
    }

    /// Called before each plan.
    pub fn planned(&mut self) {
        self.unlocked_at_plan = state().unlocked_commits.get(&self.guard.root).copied().unwrap_or(0);
    }

    /// Called when a commit failed with `err`: Ok to plan again (after a short random delay), or the error to raise.
    pub fn conflict(&mut self, py: Python<'_>, err: Error, tbl_name: &str) -> Result<()> {
        if !is_conflict(&err) {
            return Err(err);
        }
        if self.guard.exclusive() && state().unlocked_commits.get(&self.guard.root).copied().unwrap_or(0) != self.unlocked_at_plan {
            self.nested_conflicts += 1;
            if self.nested_conflicts >= 3 {
                crate::bail!(
                    ConcurrentModification,
                    "Table {} was changed by an operation started by the user code it computes with; the operation cannot complete",
                    py_repr(tbl_name)
                );
            }
        }
        self.conflicts += 1;
        if self.conflicts > MAX_RETRIES {
            crate::bail!(SerializationFailure, "Serialization retry limit ({MAX_RETRIES}) exceeded");
        }
        // a random delay that grows with the conflicts, so that competing processes do not keep colliding
        let max_us = 200u64 << self.conflicts.min(8);
        let delay = Duration::from_micros(super::sample::random_u64() % max_us);
        py.detach(|| std::thread::sleep(delay));
        py.check_signals()?;
        Ok(())
    }
}

/// What a plan step decided.
pub enum Step<P, R> {
    /// the operation completed in the plan's catalog-lock section (it runs no user code, or there is nothing to do)
    Done(R),
    /// the plan, to compute without the lock
    Compute(P),
}

/// Runs a mutation of table `tbl_id`'s tree that computes without the catalog lock: holding the tree's writer lock in
/// `mode`, `plan` (under the catalog lock) either completes the operation or returns a plan, `compute` runs without
/// the catalog lock (detached from the interpreter) and `commit` (under the catalog lock) writes the result; if the
/// commit finds what the plan read changed (it fails with SERIALIZATION_FAILURE, see `ReadSet::check()`), the
/// operation is planned and computed again.
pub fn run<P, O, R>(
    py: Python<'_>,
    tbl_id: Uuid,
    mode: Mode,
    mut plan: impl FnMut(&mut Catalog) -> Result<Step<P, R>>,
    compute: impl Fn(&P) -> Result<O> + Sync,
    mut commit: impl FnMut(&mut Catalog, &P, O) -> Result<R>,
) -> Result<R>
where
    P: Sync,
    O: Send,
{
    let guard = lock_table(py, tbl_id, mode)?;
    let mut attempts = Attempts::new(&guard);
    let mut tbl_name: Option<String> = None;
    loop {
        attempts.planned();
        let step = with_catalog(py, |cat| {
            if tbl_name.is_none() {
                tbl_name = cat.table(tbl_id).ok().map(|md| md.name.clone());
            }
            plan(cat)
        })?;
        let p = match step {
            Step::Done(r) => {
                guard.committed();
                return Ok(r);
            }
            Step::Compute(p) => p,
        };
        let out = py.detach(|| compute(&p))?;
        match with_catalog(py, |cat| commit(cat, &p, out)) {
            Ok(r) => {
                guard.committed();
                return Ok(r);
            }
            Err(e) => attempts.conflict(py, e, tbl_name.as_deref().unwrap_or_default())?,
        }
    }
}
