//! Batched evaluation, phase A: the calls of Python functions in the expressions evaluated over a batch of rows are
//! made before the rows are evaluated, all calls of a dependency level together (`dispatch::run()`).
//!
//! `prefetch()` collects the dispatchable call nodes of the expressions: calls of Python callables, including those in
//! the value expressions of unstored computed columns the expressions read, but not those inside the target of a
//! JSON mapper (evaluated once per element) or the arguments of an aggregate (evaluated by the aggregation). Each node
//! gets a level: 0 if its arguments contain no other such node, else one more than the highest level among them.
//! Level by level, `prefetch()` evaluates the arguments of each node for every row (evaluation finds the outcomes of
//! lower levels) and dispatches the calls of the whole level together, so independent calls (different columns or
//! sub-expressions) run concurrently. Phase B is the caller's row-at-a-time evaluation: contexts that carry a
//! `CallSlot` hand each call's outcome to `expr::eval` through `RowContext::call_result()`.
//!
//! As upstream evaluates every sub-expression of a row, a call is made even where `&`/`|` would not need it; its
//! outcome (also an error) is consulted only if evaluation needs it. With `abort_on_error`, arguments and calls for
//! rows after the first failed row are skipped: the caller stops at the first failing row, and evaluates (calls)
//! whatever it still needs itself.

use std::collections::HashMap;
use std::sync::Arc;

use super::dispatch::{self, NodeCalls};
use crate::error::Result;
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ColumnRef, Expr, ExprKind, ExprRef, FunctionCall};
use crate::functions::FuncRef;
use crate::functions::signature::BoundArg;
use crate::functions::udf::{self, UdfCall, UdfImpl};
use crate::value::Value;

/// The outcomes of the prefetched calls of a batch: per call node (by address), one per row (None: not called).
#[derive(Default)]
pub struct CallResults {
    /// (node address, outcomes); a batch has few call nodes, so a linear search is the fastest lookup
    nodes: Vec<(usize, Vec<Option<Result<Value>>>)>,
}

impl CallResults {
    /// True if the calls of node `e` were made.
    pub fn contains(&self, e: &Expr) -> bool {
        let key = node_key(e);
        self.nodes.iter().any(|(k, _)| *k == key)
    }
}

/// What a row context needs to find prefetched outcomes: the batch's results and the row's index in the batch.
#[derive(Clone, Copy)]
pub struct CallSlot<'a> {
    pub results: &'a CallResults,
    pub row: usize,
}

impl CallSlot<'_> {
    /// The outcome of call node `e` for this row, if it was made.
    pub fn get(&self, e: &Expr) -> Option<Result<Value>> {
        let key = node_key(e);
        let (_, outcomes) = self.results.nodes.iter().find(|(k, _)| *k == key)?;
        let slot = outcomes.get(self.row)?.as_ref()?;
        Some(match slot {
            Ok(v) => Ok(v.clone()),
            Err(err) => Err(err.duplicate()),
        })
    }
}

fn node_key(e: &Expr) -> usize {
    e as *const Expr as usize
}

/// A call of a Python callable that can be made ahead of evaluation; None for anything else (builtins, templates,
/// query functions, calls that are not valid any more, or a resource pool on a sync function, which fails at
/// evaluation).
fn dispatchable(e: &Expr) -> Option<(&FunctionCall, &Arc<UdfCall>)> {
    let ExprKind::FunctionCall(fc) = &e.kind else { return None };
    let FuncRef::Udf(call) = &fc.func else { return None };
    if call.validation_error.is_some() {
        return None;
    }
    match &call.def.imp {
        UdfImpl::Callable { has_resource_pool, is_async, .. } => {
            if *has_resource_pool && !is_async.get(call.sig_idx).copied().unwrap_or(false) {
                return None;
            }
            Some((fc, call))
        }
        _ => None,
    }
}

/// True if `e` reads a mapper's element or a template parameter (it cannot be evaluated per row).
fn reads_scope(e: &Expr) -> bool {
    let mut found = false;
    e.visit(&mut |x| found |= matches!(x.kind, ExprKind::RelativeRoot { .. } | ExprKind::Variable(_)));
    found
}

struct Collector<'a> {
    derived: &'a dyn Fn(&ColumnRef) -> Option<ExprRef>,
    /// the dispatchable nodes and their levels
    nodes: Vec<(ExprRef, usize)>,
    index: HashMap<usize, usize>,
    /// the highest node level in the value expression of a derived column (memo)
    derived_levels: HashMap<(uuid::Uuid, crate::catalog::table::ColId), Option<usize>>,
}

impl Collector<'_> {
    /// The highest level of a dispatchable node in `e` (None: there is none), registering the nodes.
    fn walk(&mut self, e: &ExprRef) -> Option<usize> {
        match &e.kind {
            ExprKind::ColumnRef(c) => {
                let key = (c.tbl_id, c.col_id);
                if let Some(l) = self.derived_levels.get(&key) {
                    return *l;
                }
                let level = match (self.derived)(c) {
                    Some(x) => {
                        // a placeholder guards against cycles (which valid metadata does not have)
                        self.derived_levels.insert(key, None);
                        self.walk(&x)
                    }
                    None => None,
                };
                self.derived_levels.insert(key, level);
                level
            }
            ExprKind::JsonMapper { src, .. } => self.walk(src),
            ExprKind::FunctionCall(fc) => match &fc.func {
                FuncRef::Agg(_) => None,
                FuncRef::Builtin(_) => self.walk_all(e.kind.children()),
                FuncRef::Udf(call) => {
                    if let (UdfImpl::Template { .. }, Some(inst)) = (&call.def.imp, &call.instantiated) {
                        return self.walk(inst);
                    }
                    let inner = self.walk_all(e.kind.children());
                    if dispatchable(e).is_none() || reads_scope(e) {
                        return inner;
                    }
                    if let Some(i) = self.index.get(&node_key(e)) {
                        return Some(self.nodes[*i].1);
                    }
                    let level = inner.map_or(0, |l| l + 1);
                    self.index.insert(node_key(e), self.nodes.len());
                    self.nodes.push((e.clone(), level));
                    Some(level)
                }
            },
            _ => self.walk_all(e.kind.children()),
        }
    }

    fn walk_all(&mut self, children: Vec<&ExprRef>) -> Option<usize> {
        children.into_iter().filter_map(|c| self.walk(c)).max()
    }
}

/// True if `exprs` contain a call that `prefetch()` would make.
pub fn has_dispatchable_calls(exprs: &[&ExprRef], derived: &dyn Fn(&ColumnRef) -> Option<ExprRef>) -> bool {
    let mut c = Collector { derived, nodes: Vec::new(), index: HashMap::new(), derived_levels: HashMap::new() };
    for e in exprs {
        c.walk(e);
    }
    !c.nodes.is_empty()
}

/// True if a NULL is bound to a non-nullable parameter (the call yields NULL without running the function).
fn short_circuits(call: &UdfCall, args: &[Value], kwargs: &[Value]) -> bool {
    let Some(sig) = call.signature() else { return false };
    call.binding.bound.iter().any(|(pi, b)| {
        if sig.params[*pi].col_type.as_ref().is_none_or(|t| t.nullable) {
            return false;
        }
        match b {
            BoundArg::Arg(i) => args[*i].is_null(),
            BoundArg::Kwarg(i) => kwargs[*i].is_null(),
            _ => false,
        }
    })
}

/// Phase A for `n_rows` rows: makes the calls in `exprs` and returns their outcomes (None if there are no calls to
/// make ahead). `derived(col)`: the value expression of a column whose value evaluation computes (unstored computed
/// columns). `with_row(i, slot, f)` calls `f` with the evaluation context of row `i`, which must carry `slot`.
/// Returns an error only for an interrupt.
pub fn prefetch(
    exprs: &[&ExprRef],
    derived: &dyn Fn(&ColumnRef) -> Option<ExprRef>,
    n_rows: usize,
    abort_on_error: bool,
    with_row: &dyn Fn(usize, CallSlot<'_>, &mut dyn FnMut(&dyn RowContext)),
) -> Result<Option<CallResults>> {
    let mut c = Collector { derived, nodes: Vec::new(), index: HashMap::new(), derived_levels: HashMap::new() };
    for e in exprs {
        c.walk(e);
    }
    let nodes = c.nodes;
    if nodes.is_empty() || n_rows == 0 || (n_rows == 1 && nodes.len() == 1) {
        return Ok(None);
    }
    let max_level = nodes.iter().map(|(_, l)| *l).max().unwrap_or(0);
    let mut results = CallResults::default();
    let mut abort_row = usize::MAX;
    for level in 0..=max_level {
        let mut work: Vec<NodeCalls> = Vec::new();
        let mut keys: Vec<usize> = Vec::new();
        let mut level_slots: Vec<Vec<Option<Result<Value>>>> = Vec::new();
        for (e, _) in nodes.iter().filter(|(_, l)| *l == level) {
            let Some((fc, call)) = dispatchable(e) else { continue };
            let mut calls = NodeCalls {
                call: call.clone(),
                arg_types: fc.args.iter().map(|a| a.col_type.clone()).collect(),
                kwarg_names: fc.kwargs.iter().map(|(k, _)| k.clone()).collect(),
                kwarg_types: fc.kwargs.iter().map(|(_, a)| a.col_type.clone()).collect(),
                result_type: e.col_type.clone(),
                resource_pool: udf::resource_pool_of(call, fc)?,
                rows: Vec::new(),
                args: Vec::new(),
                kwargs: Vec::new(),
            };
            let mut slots: Vec<Option<Result<Value>>> = (0..n_rows).map(|_| None).collect();
            for (i, slot) in slots.iter_mut().enumerate() {
                if abort_on_error && i > abort_row {
                    break;
                }
                // the arguments go straight to the end of `calls.args`/`calls.kwargs` (removed again if not called)
                let (na, nk) = (calls.args.len(), calls.kwargs.len());
                let mut evaluated: Option<Result<()>> = None;
                with_row(i, CallSlot { results: &results, row: i }, &mut |ctx| {
                    evaluated = Some((|| -> Result<()> {
                        for a in &fc.args {
                            calls.args.push(eval(a, ctx)?);
                        }
                        for (_, a) in &fc.kwargs {
                            calls.kwargs.push(eval(a, ctx)?);
                        }
                        Ok(())
                    })());
                });
                match evaluated {
                    None => {
                        calls.args.truncate(na);
                        calls.kwargs.truncate(nk);
                    }
                    Some(Err(err)) => {
                        calls.args.truncate(na);
                        calls.kwargs.truncate(nk);
                        if udf::is_interrupt(&err) {
                            return Err(err);
                        }
                        abort_row = abort_row.min(i);
                        *slot = Some(Err(err));
                    }
                    Some(Ok(())) => {
                        if short_circuits(call, &calls.args[na..], &calls.kwargs[nk..]) {
                            calls.args.truncate(na);
                            calls.kwargs.truncate(nk);
                            *slot = Some(Ok(Value::Null));
                        } else {
                            calls.rows.push(i);
                        }
                    }
                }
            }
            keys.push(node_key(e));
            level_slots.push(slots);
            work.push(calls);
        }
        dispatch::run(work, &mut level_slots, abort_on_error, if abort_on_error { abort_row } else { usize::MAX })?;
        for (key, slots) in keys.into_iter().zip(level_slots) {
            if abort_on_error {
                if let Some(row) = slots.iter().position(|o| matches!(o, Some(Err(_)))) {
                    abort_row = abort_row.min(row);
                }
            }
            results.nodes.push((key, slots));
        }
    }
    Ok(Some(results))
}
