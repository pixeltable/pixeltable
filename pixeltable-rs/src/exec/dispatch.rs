//! Running the calls of Python functions (UDFs) for many rows at once.
//!
//! `run()` takes the calls of a set of independent call nodes (`NodeCalls`: one bound function call, the rows that
//! need a call and their argument values) and returns one outcome per row, placed by row:
//! - calls of a sync function run on a pool of worker threads (Rust threads attached to the interpreter), while the
//!   dispatching thread waits detached from it, so calls that release the GIL (I/O, numpy, PIL, torch, network)
//!   overlap; a probe of the first calls measures whether they do (`Probe`), and if they do not (pure-Python CPU work
//!   holds the GIL), the dispatching thread runs the remaining calls itself, one after the other, as sequential
//!   execution would;
//! - calls of an async function run as tasks on one persistent event loop (`pixeltable.func._runtime`), all in
//!   flight together up to the function's concurrency limit, and for functions in a resource pool at the pool's
//!   request rate, with retries of failed calls that look retryable;
//! - a batched function gets batches of up to `batch_size` rows; several batches are in flight at once.
//!
//! Outcomes are identical to sequential execution: each row gets its own value or error (a failed batch fails each of
//! its rows). With `abort_on_error`, calls for rows after the first failed row are not started (the caller evaluates
//! rows in order and stops at the first failure, so it never needs them; if it does, it calls the function itself).
//! An interrupt (a `BaseException` that is not an `Exception`, such as a `KeyboardInterrupt` raised by a call or
//! delivered to the main thread while it waits) cancels the outstanding work and is returned at once.
//!
//! Calls run sequentially in the dispatching thread when parallel execution is off (`udf_threads` = 1), when the
//! function declares `max_concurrency=1`, and when the dispatching thread is itself a worker or holds the catalog lock
//! (nested evaluation from inside a UDF; dispatching to the pool from there could deadlock).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::error::{Error, Result};
use crate::functions::signature::BoundArg;
use crate::functions::udf::{self, UdfCall, UdfImpl};
use crate::types::ColumnType;
use crate::value::Value;
use crate::value::py as vpy;

// ------------------------------------------------------------------------------------------------ configuration

static UDF_THREADS: AtomicUsize = AtomicUsize::new(0);
static ASYNC_CONCURRENCY: AtomicUsize = AtomicUsize::new(0);

/// The default number of worker threads: 4 per CPU, between 4 and 32 (calls that wait on I/O need more threads than
/// there are cores; calls that compute need no more than the cores).
fn default_udf_threads() -> usize {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (4 * cpus).clamp(4, 32)
}

/// The number of worker threads that run calls of sync UDFs (`PIXELTABLE_UDF_THREADS`, config `udf_threads`);
/// 1 runs every call in the thread that evaluates the expression, in row order.
pub fn udf_threads() -> usize {
    match UDF_THREADS.load(Ordering::Relaxed) {
        0 => {
            let n = crate::config::get_int("pixeltable", "udf_threads")
                .filter(|n| *n >= 1)
                .map(|n| n.min(1024) as usize)
                .unwrap_or_else(default_udf_threads);
            UDF_THREADS.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

/// The default number of calls of one async UDF in flight at once (`PIXELTABLE_UDF_ASYNC_CONCURRENCY`, config
/// `udf_async_concurrency`, default 1024); a function's `max_concurrency` overrides it.
pub fn async_concurrency() -> usize {
    match ASYNC_CONCURRENCY.load(Ordering::Relaxed) {
        0 => {
            let n = crate::config::get_int("pixeltable", "udf_async_concurrency")
                .filter(|n| *n >= 1)
                .map(|n| n.min(1 << 20) as usize)
                .unwrap_or(1024);
            ASYNC_CONCURRENCY.store(n, Ordering::Relaxed);
            n
        }
        n => n,
    }
}

/// Overrides the configuration for this process (tests, benchmarks); None leaves a value as it is.
pub fn configure(threads: Option<usize>, async_limit: Option<usize>) {
    if let Some(n) = threads {
        UDF_THREADS.store(n.max(1), Ordering::Relaxed);
    }
    if let Some(n) = async_limit {
        ASYNC_CONCURRENCY.store(n.max(1), Ordering::Relaxed);
    }
}

// ------------------------------------------------------------------------------------------------ worker pool

type Job = Box<dyn FnOnce() + Send + 'static>;

struct PoolState {
    queue: VecDeque<Job>,
    threads: usize,
    idle: usize,
}

struct Pool {
    state: Mutex<PoolState>,
    cv: Condvar,
    pid: u32,
}

thread_local! {
    /// true on the pool's worker threads
    static IN_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// True on a worker thread of the UDF pool.
pub fn in_worker() -> bool {
    IN_WORKER.with(|w| w.get())
}

fn pool() -> Arc<Pool> {
    static POOL: Mutex<Option<Arc<Pool>>> = Mutex::new(None);
    let mut guard = POOL.lock();
    let pid = std::process::id();
    match guard.as_ref() {
        // the worker threads of a parent process do not exist in a forked child: start over
        Some(p) if p.pid == pid => p.clone(),
        _ => {
            let p = Arc::new(Pool {
                state: Mutex::new(PoolState { queue: VecDeque::new(), threads: 0, idle: 0 }),
                cv: Condvar::new(),
                pid,
            });
            *guard = Some(p.clone());
            p
        }
    }
}

impl Pool {
    /// Queues `job`; starts another worker thread if none is idle and fewer than `max_threads` exist.
    fn spawn(self: &Arc<Self>, job: Job, max_threads: usize) {
        let mut st = self.state.lock();
        st.queue.push_back(job);
        if st.idle >= st.queue.len() {
            self.cv.notify_one();
            return;
        }
        if st.threads < max_threads {
            st.threads += 1;
            let pool = self.clone();
            // a UDF may evaluate expressions itself (a query), which recurses as deep as on the main thread
            let started = std::thread::Builder::new()
                .name("pixeltable-udf".into())
                .stack_size(16 << 20)
                .spawn(move || worker_main(pool));
            if started.is_err() {
                st.threads -= 1;
            }
        }
        self.cv.notify_one();
    }
}

fn worker_main(pool: Arc<Pool>) {
    // Give this thread a Python thread state for its whole life (released, so it does not hold the GIL): without it,
    // every `Python::attach` would create and destroy one, which is slow and loses thread-local Python state.
    // SAFETY: the interpreter is initialized (the pool is started from code called by Python); the state stays
    // registered until the thread ends.
    unsafe {
        pyo3::ffi::PyGILState_Ensure();
        pyo3::ffi::PyEval_SaveThread();
    }
    IN_WORKER.with(|w| w.set(true));
    loop {
        let job = {
            let mut st = pool.state.lock();
            loop {
                if let Some(j) = st.queue.pop_front() {
                    break j;
                }
                st.idle += 1;
                pool.cv.wait(&mut st);
                st.idle -= 1;
            }
        };
        job();
    }
}

// ------------------------------------------------------------------------------------------------ resource pools

enum PoolKind {
    /// `request-rate:<endpoint>[:<model>]`: at most one request per `secs_per_request`
    RequestRate { secs_per_request: f64 },
    /// `rate-limits:<...>`: limits reported by the provider (upstream `RateLimitsInfo`, which pixeltable-rs does not
    /// have); calls are limited only by the concurrency limit, and retried
    RateLimits,
}

/// Process-wide state of a resource pool (upstream's `RequestRateScheduler` / `RateLimitsScheduler`), shared by every
/// function that names it.
struct ResourcePool {
    kind: PoolKind,
    /// the earliest time the next request may start
    next_slot: Mutex<Instant>,
}

impl ResourcePool {
    fn max_retries(&self) -> u32 {
        match self.kind {
            PoolKind::RequestRate { .. } => 3,
            PoolKind::RateLimits => 10,
        }
    }

    /// The delay before retry number `attempt` (0-based) of a call that failed retryably, upstream's
    /// `_compute_retry_delay()` / `exponential_backoff()`.
    fn retry_delay(&self, attempt: u32, retry_after: Option<f64>) -> f64 {
        match self.kind {
            PoolKind::RequestRate { .. } => match retry_after {
                Some(r) if r > 0.0 => r.clamp(1.0, 60.0),
                _ => exponential_backoff(attempt, 60.0),
            },
            PoolKind::RateLimits => retry_after.unwrap_or_else(|| exponential_backoff(attempt, 16.0)),
        }
    }

    /// Takes the next request slot if it is due; else returns how long until it is.
    fn try_acquire(&self, now: Instant) -> std::result::Result<(), Duration> {
        let PoolKind::RequestRate { secs_per_request } = self.kind else { return Ok(()) };
        let mut next = self.next_slot.lock();
        if *next > now {
            return Err(*next - now);
        }
        *next = now + Duration::from_secs_f64(secs_per_request);
        Ok(())
    }
}

fn exponential_backoff(attempt: u32, max_delay: f64) -> f64 {
    let basic = 2f64.powi(attempt.min(30) as i32).min(max_delay) / 2.0;
    let jitter = (super::sample::random_u64() >> 11) as f64 / (1u64 << 53) as f64;
    basic + jitter * basic
}

/// The pool named `name` (None for a name that is not a known kind of pool: no limits apply).
fn resource_pool(name: &str) -> Option<Arc<ResourcePool>> {
    static POOLS: OnceLock<Mutex<HashMap<String, Arc<ResourcePool>>>> = OnceLock::new();
    let mut pools = POOLS.get_or_init(Default::default).lock();
    if let Some(p) = pools.get(name) {
        return Some(p.clone());
    }
    let parts: Vec<&str> = name.split(':').collect();
    let kind = match parts.as_slice() {
        ["request-rate", endpoint] => request_rate(crate::config::get_int(endpoint, "rate_limit")),
        ["request-rate", endpoint, model] => {
            request_rate(crate::config::get_int(&format!("{endpoint}.rate_limits"), model))
        }
        ["rate-limits", ..] => PoolKind::RateLimits,
        _ => return None,
    };
    let p = Arc::new(ResourcePool { kind, next_slot: Mutex::new(Instant::now()) });
    pools.insert(name.to_string(), p.clone());
    Some(p)
}

/// A request-rate pool with `rpm` requests per minute (upstream's default: 600).
fn request_rate(rpm: Option<i64>) -> PoolKind {
    let rpm = rpm.filter(|r| *r > 0).unwrap_or(600) as f64;
    PoolKind::RequestRate { secs_per_request: 60.0 / rpm }
}

// ------------------------------------------------------------------------------------------------ the work

/// The calls of one call node: `call` is the bound function call, `rows` the rows that need a call (ascending), and
/// `args`/`kwargs` hold the argument values of row `rows[k]` at `k * arg_types.len()` resp. `k * kwarg_types.len()`
/// (in the order of the call's positional and keyword arguments).
pub struct NodeCalls {
    pub call: Arc<UdfCall>,
    pub arg_types: Vec<ColumnType>,
    pub kwarg_names: Vec<String>,
    pub kwarg_types: Vec<ColumnType>,
    pub result_type: ColumnType,
    /// the call's resource pool, if its function has one (`Function._call_resource_pool()`)
    pub resource_pool: Option<String>,
    pub rows: Vec<usize>,
    pub args: Vec<Value>,
    pub kwargs: Vec<Value>,
}

/// Per node, the outcome of the call of each row (by row index; `None`: no call, or not called because an earlier
/// row failed and the caller aborts).
pub type Outcomes = [Vec<Option<Result<Value>>>];

/// Which arguments of a batched function get a list of values (the others get the first row's value, as upstream's
/// `create_batch_kwargs()`).
struct BatchArgs {
    args: Vec<bool>,
    kwargs: Vec<bool>,
}

/// A node being executed; shared with the worker threads and the event loop.
struct Node {
    calls: NodeCalls,
    /// rows per Python invocation (a unit; 1 unless the function is batched) and the number of units
    unit_size: usize,
    num_units: usize,
    batch: Option<BatchArgs>,
    is_async: bool,
    max_concurrency: Option<usize>,
    /// the next unit to start (shared by the workers and the dispatching thread)
    next: AtomicUsize,
    /// set when the probe finds that calls do not overlap: the workers stop taking units
    serial: AtomicBool,
    /// when the first unit started (the probe measures from here)
    first_start: OnceLock<Instant>,
}

impl Node {
    fn new(calls: NodeCalls) -> Result<Node> {
        let UdfImpl::Callable { batch_size, is_async, max_concurrency, .. } = &calls.call.def.imp else {
            return Err(Error::internal("only calls of Python callables are dispatched"));
        };
        let sig = calls.call.signature().ok_or_else(|| Error::internal("unresolved call"))?;
        let binding = &calls.call.binding;
        let batched = |b: BoundArg| binding.bound.iter().any(|(pi, x)| *x == b && sig.params[*pi].is_batched);
        let batch = batch_size.map(|_| BatchArgs {
            args: (0..calls.arg_types.len()).map(|i| batched(BoundArg::Arg(i))).collect(),
            kwargs: (0..calls.kwarg_types.len()).map(|i| batched(BoundArg::Kwarg(i))).collect(),
        });
        let size = batch_size.unwrap_or(1).max(1);
        Ok(Node {
            unit_size: size,
            num_units: calls.rows.len().div_ceil(size),
            batch,
            is_async: is_async.get(calls.call.sig_idx).copied().unwrap_or(false),
            max_concurrency: *max_concurrency,
            calls,
            next: AtomicUsize::new(0),
            serial: AtomicBool::new(false),
            first_start: OnceLock::new(),
        })
    }

    /// Unit `u`: `[start, end)` into `calls.rows`.
    fn unit(&self, u: usize) -> (usize, usize) {
        let s = u * self.unit_size;
        (s, (s + self.unit_size).min(self.calls.rows.len()))
    }

    fn first_row(&self, u: usize) -> usize {
        self.calls.rows[u * self.unit_size]
    }

    fn py_fn<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>> {
        match &self.calls.call.def.imp {
            UdfImpl::Callable { py_fns, .. } => Ok(py_fns[self.calls.call.sig_idx].bind(py).clone()),
            _ => Err(Error::internal("only calls of Python callables are dispatched")),
        }
    }

    /// The Python arguments of unit `[s, e)`.
    fn py_args<'py>(&self, py: Python<'py>, s: usize, e: usize) -> Result<(Bound<'py, PyTuple>, Option<Bound<'py, PyDict>>)> {
        let c = &self.calls;
        let (na, nk) = (c.arg_types.len(), c.kwarg_types.len());
        let mut args = Vec::with_capacity(na);
        for (i, t) in c.arg_types.iter().enumerate() {
            let batched = self.batch.as_ref().is_some_and(|b| b.args[i]);
            args.push(arg_value(py, batched, s, e, |k| &c.args[k * na + i], t)?);
        }
        let kwargs = if nk == 0 {
            None
        } else {
            let d = PyDict::new(py);
            for (i, (name, t)) in c.kwarg_names.iter().zip(&c.kwarg_types).enumerate() {
                let batched = self.batch.as_ref().is_some_and(|b| b.kwargs[i]);
                d.set_item(name, arg_value(py, batched, s, e, |k| &c.kwargs[k * nk + i], t)?)?;
            }
            Some(d)
        };
        Ok((PyTuple::new(py, args)?, kwargs))
    }

    /// Calls the function for unit `u` (sync; a coroutine it returns is run to completion).
    fn run_unit(&self, py: Python<'_>, u: usize) -> Vec<Result<Value>> {
        let (s, e) = self.unit(u);
        let outcome = (|| -> Result<Bound<'_, PyAny>> {
            let (args, kwargs) = self.py_args(py, s, e)?;
            let mut r = self.py_fn(py)?.call(args, kwargs.as_ref())?;
            if udf::is_coroutine(py, &r)? {
                r = udf::run_coroutine(py, r)?;
            }
            Ok(r)
        })();
        self.unit_results(py, outcome, s, e)
    }

    /// Calls the function for unit `u` (async): the coroutine to run, or the outcomes if the call did not return one.
    fn start_unit<'py>(&self, py: Python<'py>, u: usize) -> std::result::Result<Bound<'py, PyAny>, Vec<Result<Value>>> {
        let (s, e) = self.unit(u);
        let started = (|| -> Result<Bound<'py, PyAny>> {
            let (args, kwargs) = self.py_args(py, s, e)?;
            Ok(self.py_fn(py)?.call(args, kwargs.as_ref())?)
        })();
        match started {
            Ok(r) if udf::is_coroutine(py, &r).unwrap_or(false) => Ok(r),
            other => Err(self.unit_results(py, other, s, e)),
        }
    }

    /// The per-row outcomes of unit `[s, e)` from what the call returned (a batched function returns one value per
    /// row; a failed call fails every row of the unit).
    fn unit_results(&self, py: Python<'_>, outcome: Result<Bound<'_, PyAny>>, s: usize, e: usize) -> Vec<Result<Value>> {
        let t = &self.calls.result_type;
        let n = e - s;
        let obj = match outcome {
            Ok(obj) => obj,
            Err(err) => return replicate(py, err, n),
        };
        if self.batch.is_none() {
            return vec![udf::convert_result(&obj, t)];
        }
        match obj.len() {
            Ok(len) if len == n => (0..n).map(|i| obj.get_item(i).map_err(Error::from).and_then(|x| udf::convert_result(&x, t))).collect(),
            Ok(len) => replicate(
                py,
                Error::py_exc(
                    "ValueError",
                    format!("{}: a batched call returned {len} results for {n} rows", self.calls.call.def.display_name),
                ),
                n,
            ),
            Err(err) => replicate(py, err.into(), n),
        }
    }
}

/// The Python value of an argument for unit `[s, e)`: the list of the rows' values for a batched parameter, else the
/// first row's value.
fn arg_value<'a, 'py>(
    py: Python<'py>,
    batched: bool,
    s: usize,
    e: usize,
    vals: impl Fn(usize) -> &'a Value,
    t: &ColumnType,
) -> Result<Bound<'py, PyAny>> {
    if batched {
        let items = (s..e).map(|k| vpy::to_py(py, vals(k), t)).collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, items)?.into_any())
    } else {
        Ok(vpy::to_py(py, vals(s), t)?)
    }
}

fn replicate(py: Python<'_>, err: Error, n: usize) -> Vec<Result<Value>> {
    let mut out: Vec<Result<Value>> = (1..n).map(|_| Err(err.clone_ref(py))).collect();
    out.push(Err(err));
    out
}

// ------------------------------------------------------------------------------------------------ dispatch

enum Msg {
    Done { node: usize, unit: usize, results: Vec<Result<Value>>, busy: Duration },
    Skipped { node: usize, unit: usize },
}

/// Called on the event loop's thread when a task finishes: `done((node, unit), task)`.
#[pyclass(frozen, module = "pixeltable._core")]
pub struct AsyncDone {
    tx: Sender<Msg>,
    nodes: Arc<Vec<Arc<Node>>>,
}

#[pymethods]
impl AsyncDone {
    fn __call__(&self, py: Python<'_>, token: (usize, usize), task: Bound<'_, PyAny>) {
        let (node, unit) = token;
        // a cancelled task belongs to a dispatch that is gone
        if task.call_method0("cancelled").and_then(|c| c.is_truthy()).unwrap_or(true) {
            return;
        }
        let n = &self.nodes[node];
        let (s, e) = n.unit(unit);
        let outcome = task.call_method0("result").map_err(Error::from);
        let results = n.unit_results(py, outcome, s, e);
        let _ = self.tx.send(Msg::Done { node, unit, results, busy: Duration::ZERO });
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// the dispatching thread runs the calls, in row order
    Inline,
    /// worker threads run the calls
    Parallel,
    /// tasks on the event loop
    Async,
}

/// Measures whether the calls of a parallel node overlap: the node starts with `PROBE_WORKERS` workers; after `target`
/// calls, the sum of their durations (each measured while its worker is attached) over the time since the first call
/// started tells whether they ran at the same time. Below 1.5, the calls hold the GIL and run no faster in parallel:
/// the dispatching thread runs the rest; else the node gets all its workers (`helpers`).
struct Probe {
    busy: Duration,
    completed: usize,
    target: usize,
    decided: bool,
    helpers: usize,
}

/// The workers a parallel node starts with, until the probe has decided.
const PROBE_WORKERS: usize = 2;

struct AsyncState {
    limit: usize,
    in_flight: usize,
    pool: Option<Arc<ResourcePool>>,
    attempts: Vec<u32>,
    /// units to retry, and when
    retries: Vec<(Instant, usize)>,
}

const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(50);
/// Calls that take less than this run in the dispatching thread (see `Dispatch::run_inline_slice()`).
const PARALLEL_MIN_CALL: Duration = Duration::from_micros(50);
const INLINE_SLICE: Duration = Duration::from_millis(20);

struct Dispatch<'o> {
    nodes: Arc<Vec<Arc<Node>>>,
    modes: Vec<Mode>,
    outcomes: &'o mut Outcomes,
    /// per node: units not finished
    remaining: Vec<usize>,
    probes: Vec<Option<Probe>>,
    /// parallel nodes: the calls the dispatching thread ran before workers were started, and their duration
    measured: Vec<(usize, Duration)>,
    asyncs: Vec<Option<AsyncState>>,
    tx: Sender<Msg>,
    rx: Mutex<Receiver<Msg>>,
    abort: bool,
    /// the first row with a failed call (`usize::MAX`: none); calls for later rows are not started
    abort_row: Arc<AtomicUsize>,
    /// set when the dispatch ends early: workers stop taking units
    stop: Arc<AtomicBool>,
    group: Option<Py<PyAny>>,
    done_cb: Option<Py<AsyncDone>>,
    last_signal_check: Instant,
}

/// Runs the calls of `work` (independent nodes) and stores their outcomes in `outcomes` (`outcomes[i][row]` for node
/// `i`, which must have room for every row of the node). `abort_on_error`: calls for rows after the first failed row
/// (or `abort_row`, a row that already failed) are not started. Returns an error only for an interrupt.
pub fn run(work: Vec<NodeCalls>, outcomes: &mut Outcomes, abort_on_error: bool, abort_row: usize) -> Result<()> {
    if work.iter().all(|w| w.rows.is_empty()) {
        return Ok(());
    }
    Python::attach(|py| {
        let mut d = Dispatch::new(py, work, outcomes, abort_on_error, abort_row)?;
        let result = d.run(py);
        d.stop.store(true, Ordering::SeqCst);
        if result.is_err() {
            d.cancel_async(py);
        }
        result
    })
}

impl<'o> Dispatch<'o> {
    fn new(py: Python<'_>, work: Vec<NodeCalls>, outcomes: &'o mut Outcomes, abort: bool, abort_row: usize) -> Result<Dispatch<'o>> {
        let threads = udf_threads();
        let nested = in_worker() || crate::catalog::runtime::lock_held();
        let in_loop = work.iter().any(|w| matches!(&w.call.def.imp, UdfImpl::Callable { is_async, .. } if is_async.iter().any(|a| *a)))
            && py.import("pixeltable.func._runtime")?.call_method0("in_loop_thread")?.is_truthy()?;
        let nodes: Vec<Arc<Node>> = work.into_iter().map(|w| Node::new(w).map(Arc::new)).collect::<Result<_>>()?;
        let mut modes = Vec::with_capacity(nodes.len());
        let mut asyncs = Vec::with_capacity(nodes.len());
        for n in &nodes {
            let mode = if n.is_async && !in_loop {
                Mode::Async
            } else if n.is_async || threads <= 1 || nested || n.max_concurrency == Some(1) || n.num_units <= 1 {
                Mode::Inline
            } else {
                Mode::Parallel
            };
            modes.push(mode);
            asyncs.push(if mode == Mode::Async {
                let pool = n.calls.resource_pool.as_deref().and_then(resource_pool);
                Some(AsyncState {
                    limit: n.max_concurrency.unwrap_or_else(async_concurrency).max(1),
                    in_flight: 0,
                    pool,
                    attempts: vec![0; n.num_units],
                    retries: Vec::new(),
                })
            } else {
                None
            });
        }
        let (tx, rx) = channel();
        Ok(Dispatch {
            outcomes,
            remaining: nodes.iter().map(|n| n.num_units).collect(),
            probes: nodes.iter().map(|_| None).collect(),
            measured: nodes.iter().map(|_| (0, Duration::ZERO)).collect(),
            nodes: Arc::new(nodes),
            modes,
            asyncs,
            tx,
            rx: Mutex::new(rx),
            abort,
            abort_row: Arc::new(AtomicUsize::new(if abort { abort_row } else { usize::MAX })),
            stop: Arc::new(AtomicBool::new(false)),
            group: None,
            done_cb: None,
            last_signal_check: Instant::now(),
        })
    }

    fn run(&mut self, py: Python<'_>) -> Result<()> {
        while self.remaining.iter().any(|r| *r > 0) {
            let next_timer = self.submit_async(py)?;
            let ran_inline = self.run_inline_slice(py)?;
            // collect finished calls: without blocking if there is inline work left, else wait detached
            let wait = if ran_inline {
                Duration::ZERO
            } else {
                next_timer.map_or(SIGNAL_CHECK_INTERVAL, |t| t.min(SIGNAL_CHECK_INTERVAL))
            };
            let mut msg = if wait.is_zero() {
                self.rx.lock().try_recv().ok()
            } else {
                let rx = &self.rx;
                match py.detach(|| rx.lock().recv_timeout(wait)) {
                    Ok(m) => Some(m),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => return Err(Error::internal("UDF dispatch channel closed")),
                }
            };
            while let Some(m) = msg {
                self.on_msg(py, m)?;
                msg = self.rx.lock().try_recv().ok();
            }
            if self.last_signal_check.elapsed() >= SIGNAL_CHECK_INTERVAL {
                self.last_signal_check = Instant::now();
                // a KeyboardInterrupt delivered to the main thread while it waited
                py.check_signals()?;
            }
        }
        Ok(())
    }

    /// Starts the worker jobs of parallel node `ni`: as many as it can use, up to the pool size.
    fn start_workers(&mut self, ni: usize) {
        let threads = udf_threads();
        let node = &self.nodes[ni];
        let helpers = threads.min(node.max_concurrency.unwrap_or(usize::MAX)).min(node.num_units).max(1);
        let first = helpers.min(PROBE_WORKERS);
        self.probes[ni] = Some(Probe { busy: Duration::ZERO, completed: 0, target: node.num_units.min(4 * first), decided: false, helpers });
        self.spawn_workers(ni, first);
    }

    fn spawn_workers(&self, ni: usize, count: usize) {
        let threads = udf_threads();
        let pool = pool();
        for _ in 0..count {
            let job = WorkerJob {
                ni,
                node: self.nodes[ni].clone(),
                tx: self.tx.clone(),
                stop: self.stop.clone(),
                abort_row: self.abort_row.clone(),
                pool: pool.clone(),
                max_threads: threads,
            };
            pool.spawn(Box::new(move || job.run()), threads);
        }
    }

    /// Runs units in the dispatching thread for up to one time slice: those of inline nodes, of parallel nodes whose
    /// calls are still being measured (a call shorter than `PARALLEL_MIN_CALL` would gain less from a worker than the
    /// handoff costs, so workers start only for slower calls), and of parallel nodes the probe made serial. False if
    /// there were none to run.
    fn run_inline_slice(&mut self, py: Python<'_>) -> Result<bool> {
        let slice_start = Instant::now();
        let mut ran = false;
        for ni in 0..self.nodes.len() {
            let node = self.nodes[ni].clone();
            let measuring = self.modes[ni] == Mode::Parallel && self.probes[ni].is_none();
            let serial = self.modes[ni] == Mode::Inline || (self.modes[ni] == Mode::Parallel && node.serial.load(Ordering::Relaxed));
            if !measuring && !serial {
                continue;
            }
            let mut since_check = 0usize;
            let mut check_start = Instant::now();
            loop {
                let u = node.next.fetch_add(1, Ordering::SeqCst);
                if u >= node.num_units {
                    break;
                }
                ran = true;
                if node.first_row(u) > self.abort_row.load(Ordering::Relaxed) {
                    self.finish_unit(ni, u, None);
                    continue;
                }
                let results = node.run_unit(py, u);
                self.on_results(py, ni, u, results)?;
                since_check += 1;
                // the clock is read every few calls (a short call costs less than reading it twice)
                if since_check < if measuring { 2 } else { 16 } {
                    continue;
                }
                let now = Instant::now();
                if measuring {
                    let m = &mut self.measured[ni];
                    m.0 += since_check;
                    m.1 += now - check_start;
                    if m.1 >= PARALLEL_MIN_CALL * m.0 as u32 && node.next.load(Ordering::SeqCst) < node.num_units {
                        self.start_workers(ni);
                        break;
                    }
                }
                since_check = 0;
                check_start = now;
                if now - slice_start >= INLINE_SLICE {
                    return Ok(true);
                }
            }
        }
        Ok(ran)
    }

    /// Starts async units up to each node's limit and pool rate, and due retries. Returns the time until the next unit
    /// may start, if one is waiting on a timer.
    fn submit_async(&mut self, py: Python<'_>) -> Result<Option<Duration>> {
        enum Next {
            Stop,
            Skip(usize),
            Start(usize),
        }
        let mut items: Vec<Bound<'_, PyAny>> = Vec::new();
        let mut next_timer: Option<Duration> = None;
        let now = Instant::now();
        for ni in 0..self.nodes.len() {
            if self.modes[ni] != Mode::Async {
                continue;
            }
            let node = self.nodes[ni].clone();
            loop {
                let next = {
                    let st = self.asyncs[ni].as_mut().expect("async state");
                    let due = st.retries.iter().position(|(t, _)| *t <= now);
                    let u = node.next.load(Ordering::SeqCst);
                    if st.in_flight >= st.limit {
                        Next::Stop
                    } else if due.is_none() && !st.retries.is_empty() {
                        // while a retry is pending, the pool starts no new requests (upstream retries the failed request
                        // before anything else)
                        Next::Stop
                    } else if due.is_none() && u >= node.num_units {
                        Next::Stop
                    } else if due.is_none() && node.first_row(u) > self.abort_row.load(Ordering::SeqCst) {
                        node.next.store(u + 1, Ordering::SeqCst);
                        Next::Skip(u)
                    } else if let Some(Err(wait)) = st.pool.as_ref().map(|p| p.try_acquire(now)) {
                        next_timer = Some(next_timer.map_or(wait, |t| t.min(wait)));
                        Next::Stop
                    } else {
                        match due {
                            Some(i) => Next::Start(st.retries.remove(i).1),
                            None => {
                                node.next.store(u + 1, Ordering::SeqCst);
                                Next::Start(u)
                            }
                        }
                    }
                };
                match next {
                    Next::Stop => break,
                    Next::Skip(u) => self.finish_unit(ni, u, None),
                    Next::Start(u) => match node.start_unit(py, u) {
                        Ok(coro) => {
                            self.asyncs[ni].as_mut().expect("async state").in_flight += 1;
                            items.push(PyTuple::new(py, [PyTuple::new(py, [ni, u])?.into_any(), coro])?.into_any());
                        }
                        Err(results) => self.on_results(py, ni, u, results)?,
                    },
                }
            }
        }
        if !items.is_empty() {
            let rt = py.import("pixeltable.func._runtime")?;
            if self.group.is_none() {
                self.group = Some(rt.call_method0("new_group")?.unbind());
                let cb = AsyncDone { tx: self.tx.clone(), nodes: self.nodes.clone() };
                self.done_cb = Some(Py::new(py, cb)?);
            }
            let (group, cb) = (self.group.as_ref().expect("group"), self.done_cb.as_ref().expect("callback"));
            rt.call_method1("submit", (PyList::new(py, items)?, cb.bind(py), group.bind(py)))?;
        }
        // pending retries are timers
        for st in self.asyncs.iter().flatten() {
            if let Some(wait) = st.retries.iter().map(|(t, _)| t.saturating_duration_since(now)).min() {
                next_timer = Some(next_timer.map_or(wait, |t| t.min(wait)));
            }
        }
        Ok(next_timer)
    }

    fn on_msg(&mut self, py: Python<'_>, msg: Msg) -> Result<()> {
        match msg {
            Msg::Skipped { node, unit } => {
                self.finish_unit(node, unit, None);
                Ok(())
            }
            Msg::Done { node, unit, results, busy } => {
                let mut ramp_up = 0;
                if let Some(p) = self.probes[node].as_mut() {
                    p.busy += busy;
                    p.completed += 1;
                    if !p.decided && p.completed >= p.target {
                        p.decided = true;
                        let started = self.nodes[node].first_start.get().copied().unwrap_or_else(Instant::now);
                        let overlap = p.busy.as_secs_f64() / started.elapsed().as_secs_f64().max(1e-9);
                        if overlap < 1.5 && p.helpers > 1 {
                            self.nodes[node].serial.store(true, Ordering::SeqCst);
                        } else {
                            ramp_up = p.helpers.saturating_sub(PROBE_WORKERS.min(p.helpers));
                        }
                    }
                }
                if ramp_up > 0 && self.nodes[node].next.load(Ordering::SeqCst) < self.nodes[node].num_units {
                    self.spawn_workers(node, ramp_up);
                }
                if self.modes[node] == Mode::Async {
                    let st = self.asyncs[node].as_mut().expect("async state");
                    st.in_flight -= 1;
                    if self.should_retry(py, node, unit, &results)? {
                        return Ok(());
                    }
                }
                self.on_results(py, node, unit, results)
            }
        }
    }

    /// For a failed call of a function in a resource pool: schedules a retry if the error looks retryable and the
    /// pool's retry limit is not reached.
    fn should_retry(&mut self, py: Python<'_>, node: usize, unit: usize, results: &[Result<Value>]) -> Result<bool> {
        let st = self.asyncs[node].as_mut().expect("async state");
        let Some(pool) = st.pool.clone() else { return Ok(false) };
        let Some(Err(Error::Python(err))) = results.first() else { return Ok(false) };
        if !err.is_instance_of::<pyo3::exceptions::PyException>(py) || st.attempts[unit] >= pool.max_retries() {
            return Ok(false);
        }
        let info = py.import("pixeltable.func._runtime")?.call_method1("retry_info", (err.value(py),))?;
        let (retryable, retry_after): (bool, Option<f64>) = info.extract()?;
        if !retryable {
            return Ok(false);
        }
        let delay = pool.retry_delay(st.attempts[unit], retry_after);
        st.attempts[unit] += 1;
        st.retries.push((Instant::now() + Duration::from_secs_f64(delay.max(0.0)), unit));
        Ok(true)
    }

    /// Records the outcomes of unit `u` of node `ni`; an interrupt ends the dispatch.
    fn on_results(&mut self, py: Python<'_>, ni: usize, u: usize, results: Vec<Result<Value>>) -> Result<()> {
        if let Some(pos) = results.iter().position(|r| matches!(r, Err(e) if is_interrupt(py, e))) {
            let mut results = results;
            return Err(match results.swap_remove(pos) {
                Err(e) => e,
                Ok(_) => Error::internal("interrupt expected"),
            });
        }
        if self.abort {
            let (s, _) = self.nodes[ni].unit(u);
            if let Some(k) = results.iter().position(|r| r.is_err()) {
                self.abort_row.fetch_min(self.nodes[ni].calls.rows[s + k], Ordering::SeqCst);
            }
        }
        self.finish_unit(ni, u, Some(results));
        Ok(())
    }

    fn finish_unit(&mut self, ni: usize, u: usize, results: Option<Vec<Result<Value>>>) {
        let (s, _) = self.nodes[ni].unit(u);
        if let Some(results) = results {
            let rows = &self.nodes[ni].calls.rows;
            for (k, r) in results.into_iter().enumerate() {
                self.outcomes[ni][rows[s + k]] = Some(r);
            }
        }
        self.remaining[ni] -= 1;
    }

    fn cancel_async(&self, py: Python<'_>) {
        if let Some(group) = &self.group {
            if let Ok(rt) = py.import("pixeltable.func._runtime") {
                let _ = rt.call_method1("cancel", (group.bind(py),));
            }
        }
    }
}

/// A BaseException that is not an Exception (KeyboardInterrupt, SystemExit): never recorded as a cell error.
fn is_interrupt(py: Python<'_>, err: &Error) -> bool {
    matches!(err, Error::Python(e) if !e.is_instance_of::<pyo3::exceptions::PyException>(py))
}

/// A worker's job for a parallel node: takes units for up to a time slice, then queues itself again (so that the jobs
/// of several nodes share the workers), until there are none left, the probe made the node serial, or the dispatch
/// stopped.
struct WorkerJob {
    ni: usize,
    node: Arc<Node>,
    tx: Sender<Msg>,
    stop: Arc<AtomicBool>,
    abort_row: Arc<AtomicUsize>,
    pool: Arc<Pool>,
    max_threads: usize,
}

const WORKER_SLICE: Duration = Duration::from_millis(5);

impl WorkerJob {
    fn run(self) {
        let slice_start = Instant::now();
        loop {
            if self.stop.load(Ordering::SeqCst) || self.node.serial.load(Ordering::SeqCst) {
                return;
            }
            let u = self.node.next.fetch_add(1, Ordering::SeqCst);
            if u >= self.node.num_units {
                return;
            }
            let msg = if self.node.first_row(u) > self.abort_row.load(Ordering::SeqCst) {
                Msg::Skipped { node: self.ni, unit: u }
            } else {
                Python::attach(|py| {
                    let t0 = Instant::now();
                    self.node.first_start.get_or_init(|| t0);
                    let results = self.node.run_unit(py, u);
                    Msg::Done { node: self.ni, unit: u, results, busy: t0.elapsed() }
                })
            };
            if self.tx.send(msg).is_err() {
                return;
            }
            if slice_start.elapsed() >= WORKER_SLICE {
                break;
            }
        }
        if self.node.next.load(Ordering::SeqCst) < self.node.num_units {
            let (pool, max_threads) = (self.pool.clone(), self.max_threads);
            pool.spawn(Box::new(move || self.run()), max_threads);
        }
    }
}
