//! Data changes to existing rows: `update()`, `batch_update()`, `recompute_columns()` and `delete()`, with their
//! propagation to live views.
//!
//! An update rewrites the selected rows of a table: each is retired and replaced by a new row with the same key at
//! the new version, its updated cells set and (with `cascade`) the stored computed columns that depend on them
//! recomputed. Each live view then revisits the base rows that changed: a view whose filter reads a changed column
//! (or whose base gained or lost rows) re-decides membership, adding and removing rows; a view with stored columns
//! that depend on changed columns recomputes them. A view whose rows change gets a new version; its views follow
//! recursively. An update whose expressions may run user code is planned under the catalog lock, computed without it
//! and committed with it if what it read is unchanged (else planned again, see `exec::txn`); one that runs no user
//! code is computed and committed in the plan's lock section.
//!
//! A delete runs no user code (its filter must be expressible without Python), so it runs in one step under the
//! lock (`delete()`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use super::compute::{ComputedCol, MediaTarget, discard_media};
use super::insert::UpdateStatus;
use super::propagate::{ViewCompute, record_version_status};
use super::scan::{ElemInfo, PathCtx, PathInput, eval_rows_into};
use super::txn::{ReadSet, runs_user_code};
use crate::catalog::Catalog;
use crate::catalog::table::{ColId, TableMd};
use crate::error::{Result, py_repr};
use crate::expr::eval::{eval, eval_predicate};
use crate::expr::{Expr, ExprKind, ExprRef};
use crate::functions::FuncRef;
use crate::store::rows::{RowKey, StoredRow, TableData};
use crate::value::Value;

/// A qualified column id.
pub type QCol = (Uuid, ColId);

/// Which rows an update changes, and how.
#[derive(Debug, Clone)]
pub enum RowSelection {
    /// the rows that pass the filter (all rows without one); every selected row gets the same targets
    Where { pred: Option<ExprRef>, targets: Vec<(ColId, ExprRef)> },
    /// `batch_update()`: the rows with the given keys, each with its own values
    Keys(HashMap<RowKey, Vec<(ColId, Value)>>),
}

/// A live view as an update sees it.
pub struct ViewUpdateNode {
    pub compute: ViewCompute,
    pub schema_version: i64,
    pub data: Arc<TableData>,
    /// the view's stored computed columns to recompute (they depend on changed columns), in dependency order
    pub recompute: Vec<ComputedCol>,
    /// true if the update can change which base rows the view contains
    pub membership_change: bool,
    /// component views: true if the rows of a changed base row are generated again (the iterator reads a changed
    /// column, or the base row may enter or leave the view)
    pub reload: bool,
    /// the number of iterator positions in the keys of the base's rows
    pub base_num_pos: usize,
    pub children: Vec<ViewUpdateNode>,
}

impl ViewUpdateNode {
    fn tbl_id(&self) -> Uuid {
        self.compute.info.tbl_id
    }
}

pub struct UpdatePlan {
    pub tbl_id: Uuid,
    pub tbl_name: String,
    pub schema_version: i64,
    pub is_data_versioned: bool,
    /// the table's path at its current version
    pub path: PathInput,
    pub selection: RowSelection,
    /// the table's stored computed columns to recompute, in dependency order
    pub recompute: Vec<ComputedCol>,
    /// qualified names of the updated and recomputed columns (`UpdateStatus.updated_cols`)
    pub updated_cols: Vec<String>,
    /// false: the views are left alone (`cascade=False`), and `views` is empty
    pub cascade: bool,
    pub views: Vec<ViewUpdateNode>,
    /// `recompute_columns(errors_only=True)`: only rows with an error in this column
    pub errors_only: Option<ColId>,
    /// `return_rows`: the columns to return for each updated row (read from `UpdateOutput::own.new_rows`)
    pub return_cols: Option<Vec<super::insert::ReturnCol>>,
    /// `alter_computed_column()`: the column whose value expression changes in the same (schema) version
    pub schema_change: Option<(ColId, ExprRef)>,
    /// false: only the schema change (`alter_computed_column(recompute=False)`), no rows are written
    pub writes_rows: bool,
    /// where in-memory media values and results are written
    pub media: MediaTarget,
    /// what the plan read (the table's path and its views), validated by the commit
    pub read: ReadSet,
}

/// The stored computed columns of `md` (current schema) that depend, transitively, on a column in `changed`, in
/// dependency order; their ids are added to `changed`.
pub fn dependent_columns(md: &TableMd, changed: &mut HashSet<QCol>) -> Result<Vec<ComputedCol>> {
    let mut candidates: Vec<(ComputedCol, Vec<QCol>)> = Vec::new();
    let visible = md.columns_at(None)?;
    let stored: HashSet<ColId> = visible.iter().filter(|c| c.md.stored).map(|c| c.md.id).collect();
    for c in &visible {
        if let Some(e) = &c.md.value_expr {
            let expr = crate::expr::serde::from_json(e)?;
            let refs: Vec<QCol> = expr.column_refs().iter().map(|r| (r.tbl_id, r.col_id)).collect();
            let is_media = c.md.col_type.is_media();
            candidates.push((ComputedCol { col_id: c.md.id, name: c.name.clone(), expr, is_media }, refs));
        }
    }
    // the dependents (unstored columns propagate the dependency but are not recomputed)
    let mut selected: HashSet<ColId> = HashSet::new();
    loop {
        let mut grew = false;
        for (c, refs) in &candidates {
            if !selected.contains(&c.col_id) && refs.iter().any(|r| changed.contains(r)) {
                selected.insert(c.col_id);
                changed.insert((md.id, c.col_id));
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    // dependency order: a column comes after the selected columns it references
    let mut ordered: Vec<ComputedCol> = Vec::new();
    let mut done: HashSet<ColId> = HashSet::new();
    while done.len() < selected.len() {
        let before = done.len();
        for (c, refs) in &candidates {
            if selected.contains(&c.col_id)
                && !done.contains(&c.col_id)
                && refs.iter().all(|r| r.0 != md.id || !selected.contains(&r.1) || done.contains(&r.1))
            {
                done.insert(c.col_id);
                if stored.contains(&c.col_id) {
                    ordered.push(c.clone());
                }
            }
        }
        if done.len() == before {
            // a cycle (cannot happen for a valid schema): keep schema order for the rest
            for (c, _) in &candidates {
                if selected.contains(&c.col_id) && done.insert(c.col_id) && stored.contains(&c.col_id) {
                    ordered.push(c.clone());
                }
            }
        }
    }
    Ok(ordered)
}

/// The update nodes of the live views of `tbl_id` (recursively), for an update that changes the columns in
/// `changed` (which grows by the views' recomputed columns, branch by branch).
pub fn view_update_tree(
    cat: &mut Catalog,
    tbl_id: Uuid,
    changed: &HashSet<QCol>,
    base_membership_change: bool,
    updated_cols: &mut Vec<String>,
) -> Result<Vec<ViewUpdateNode>> {
    let mut out = Vec::new();
    for v in cat.mutable_view_ids(tbl_id) {
        let md = cat.table(v)?.clone();
        super::propagate::check_view_valid(cat, &md, "updated in")?;
        let compute = ViewCompute::current(cat, v)?;
        let mut changed = changed.clone();
        let reads_changed = |e: &ExprRef| e.column_refs().iter().any(|r| changed.contains(&(r.tbl_id, r.col_id)));
        let membership_change = base_membership_change || compute.predicate.as_ref().is_some_and(reads_changed);
        let reload = match &compute.info.iterator {
            Some(it) => membership_change || it.call.arg_exprs().any(reads_changed),
            None => false,
        };
        if reload {
            // regenerated rows: every iterator output may change
            for c in 0..md.num_iterator_cols() {
                changed.insert((v, c as ColId));
            }
        }
        let mut recompute = dependent_columns(&md, &mut changed)?;
        for c in &recompute {
            updated_cols.push(format!("{}.{}", md.name, c.name));
        }
        recompute.extend(super::compute::index_value_columns(cat, v, Some(&changed))?);
        let tbl_display = crate::functions::udf::tbl_version_display(&md);
        for c in &recompute {
            crate::functions::udf::check_valid_column(cat, &tbl_display, &c.name, &c.expr, "updated in")?;
        }
        let data = cat.data(v)?;
        let children = view_update_tree(cat, v, &changed, membership_change || reload, updated_cols)?;
        out.push(ViewUpdateNode {
            compute,
            schema_version: md.current_schema_version,
            data,
            recompute,
            membership_change,
            reload,
            base_num_pos: cat.num_pos(tbl_id),
            children,
        });
    }
    Ok(out)
}

/// True if the expression can be evaluated without running Python code or reading media (upstream requires the
/// rows an update or delete changes to be identifiable in SQL).
pub fn is_sql_expressible(e: &Expr) -> bool {
    let mut ok = true;
    e.visit(&mut |n| {
        if let ExprKind::FunctionCall(fc) = &n.kind {
            match &fc.func {
                FuncRef::Udf(_) | FuncRef::Agg(_) => ok = false,
                FuncRef::Builtin(b) => {
                    if ["image", "video", "audio", "document"].iter().any(|m| b.module.ends_with(&format!(".{m}"))) {
                        ok = false;
                    }
                }
            }
        }
    });
    ok
}

/// Fails unless `pred` can select the rows of an update or delete.
pub fn validate_where(pred: &ExprRef) -> Result<()> {
    if !pred.col_type.is_bool() {
        crate::bail!(
            UnsupportedOperation,
            "`where` argument needs to return `Bool`, but instead returns `{}`",
            pred.col_type.repr()
        );
    }
    if !is_sql_expressible(pred) {
        crate::bail!(UnsupportedOperation, "Filter not expressible in SQL: {}", crate::expr::display::display(pred));
    }
    Ok(())
}

/// The rows an update writes, per table.
#[derive(Debug, Default)]
pub struct TableChanges {
    /// keys of the live rows to retire
    pub retire: HashSet<RowKey>,
    /// the replacement and new rows
    pub new_rows: Vec<StoredRow>,
    pub status: UpdateStatus,
}

pub struct ViewChanges {
    pub changes: TableChanges,
    /// true if the view gets a new version
    pub bump: bool,
    pub children: Vec<ViewChanges>,
}

pub struct UpdateOutput {
    pub own: TableChanges,
    pub views: Vec<ViewChanges>,
    /// `batch_update()`: keys that matched no row
    pub unmatched: Vec<RowKey>,
    /// media files written for the new rows
    pub saved_media: Vec<std::path::PathBuf>,
}

/// A view row's base as an update leaves it: the new base path row, or None if the base row is gone.
type BaseChange<'a> = (RowKey, Option<Vec<&'a StoredRow>>);

/// For an update with a where clause: the candidates (joined rows of the path) that pass `pred`, with their new values
/// (`targets` evaluated against them), in order; the user-function calls of the predicate, then those of the values,
/// are made for all rows together (`exec::prefetch`). Evaluation stops at the first row that fails (in the predicate or
/// the values), whose error is returned alongside the rows before it; an interrupt is returned as the error.
fn where_values(
    infos: &[&ElemInfo],
    candidates: &[Vec<&StoredRow>],
    pred: Option<&ExprRef>,
    targets: &[(ColId, ExprRef)],
) -> Result<(Vec<(usize, Vec<(ColId, Value)>)>, Option<crate::error::Error>)> {
    use super::prefetch::{CallSlot, prefetch};
    let derived = |c: &crate::expr::ColumnRef| infos.iter().find(|e| e.tbl_id == c.tbl_id).and_then(|e| e.unstored.get(&c.col_id).cloned());
    fn ctx<'a>(infos: &'a [&'a ElemInfo], rows: &'a [&'a StoredRow], calls: Option<CallSlot<'a>>) -> PathCtx<'a> {
        PathCtx { elems: infos, rows, python: false, calls }
    }
    let fail = |err: crate::error::Error| -> Result<crate::error::Error> {
        if crate::functions::udf::is_interrupt(&err) { Err(err) } else { Ok(err) }
    };
    let mut error = None;
    let mut keep: Vec<usize> = Vec::new();
    match pred {
        None => keep.extend(0..candidates.len()),
        Some(p) => {
            let calls = prefetch(&[p], &derived, candidates.len(), true, &|i, slot, f| f(&ctx(infos, &candidates[i], Some(slot))))?;
            for i in 0..candidates.len() {
                let slot = calls.as_ref().map(|results| CallSlot { results, row: i });
                match eval_predicate(p, &ctx(infos, &candidates[i], slot)) {
                    Ok(true) => keep.push(i),
                    Ok(false) => {}
                    Err(err) => {
                        error = Some(fail(err)?);
                        break;
                    }
                }
            }
        }
    }
    let exprs: Vec<&ExprRef> = targets.iter().map(|(_, e)| e).collect();
    let calls = prefetch(&exprs, &derived, keep.len(), true, &|k, slot, f| f(&ctx(infos, &candidates[keep[k]], Some(slot))))?;
    let mut out = Vec::with_capacity(keep.len());
    for (k, &i) in keep.iter().enumerate() {
        let ctx = ctx(infos, &candidates[i], calls.as_ref().map(|results| CallSlot { results, row: k }));
        match targets.iter().map(|(c, e)| Ok((*c, eval(e, &ctx)?))).collect::<Result<Vec<_>>>() {
            Ok(values) => out.push((i, values)),
            Err(err) => {
                // this row comes before the predicate's failure (if any)
                error = Some(fail(err)?);
                break;
            }
        }
    }
    Ok((out, error))
}

impl UpdatePlan {
    /// Computes the new rows (no catalog lock held). Media files written along the way are deleted again if this
    /// fails; otherwise they are in `UpdateOutput::saved_media` (adopted by the commit).
    pub fn compute(&self, fail_on_exception: bool) -> Result<UpdateOutput> {
        let mut saved = Vec::new();
        match self.compute_inner(fail_on_exception, &mut saved) {
            Ok(mut out) => {
                out.saved_media = saved;
                Ok(out)
            }
            Err(e) => {
                discard_media(&saved);
                Err(e)
            }
        }
    }

    fn compute_inner(&self, fail_on_exception: bool, saved: &mut Vec<std::path::PathBuf>) -> Result<UpdateOutput> {
        if !self.writes_rows {
            return Ok(UpdateOutput {
                own: TableChanges::default(),
                views: Vec::new(),
                unmatched: Vec::new(),
                saved_media: Vec::new(),
            });
        }
        let infos = self.path.infos();
        let mut own = TableChanges::default();
        let mut olds: Vec<Vec<&StoredRow>> = Vec::new();
        let mut matched: HashSet<RowKey> = HashSet::new();
        // with fail_on_exception: the failure of the first row whose new values could not be computed (rows after it
        // are not updated; a computed column's failure in an earlier row comes first)
        let mut value_error: Option<crate::error::Error> = None;
        let candidates: Vec<Vec<&StoredRow>> = match &self.selection {
            RowSelection::Where { pred, .. } => self.path.rows_where(pred.as_ref())?,
            RowSelection::Keys(rows) => self.path.rows_for_keys(rows.keys().map(|(r, p)| (*r, p.as_slice()))),
        }
        .into_iter()
        .filter(|r| self.errors_only.is_none_or(|c| r[0].error(c).is_some()))
        .collect();
        // the rows to update and their new values
        let updates: Vec<(usize, Vec<(ColId, Value)>)> = match &self.selection {
            RowSelection::Where { pred, targets } => {
                let (updates, err) = where_values(&infos, &candidates, pred.as_ref(), targets)?;
                match err {
                    Some(err) if fail_on_exception => value_error = Some(err),
                    Some(err) => return Err(err),
                    None => {}
                }
                updates
            }
            RowSelection::Keys(rows) => candidates
                .iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let key: RowKey = (r[0].rowid, r[0].pos.clone());
                    let v = rows.get(&key)?.clone();
                    matched.insert(key);
                    Some((i, v))
                })
                .collect(),
        };
        for (i, values) in updates {
            let r = &candidates[i];
            let mut row = r[0].clone();
            for (c, v) in values {
                row.set_cell(c, v);
                row.set_error(c, None);
                self.media.materialize(&mut row, c, saved)?;
            }
            own.status.row_count_stats.upd_rows += 1;
            own.retire.insert((r[0].rowid, r[0].pos.clone()));
            own.new_rows.push(row);
            olds.push(r.clone());
        }
        // the computed columns of all updated rows together (columns with window functions in window order)
        let bases: Vec<&[&StoredRow]> = olds.iter().map(|o| &o[1..]).collect();
        eval_rows_into(
            &mut own.new_rows,
            &bases,
            &self.recompute,
            &infos,
            &self.tbl_name,
            fail_on_exception,
            &mut own.status.row_count_stats,
            &mut own.status.cols_with_excs,
        )?;
        if let Some(err) = value_error {
            return Err(err);
        }
        for row in own.new_rows.iter_mut() {
            for c in self.recompute.iter().filter(|c| c.is_media) {
                self.media.materialize(row, c.col_id, saved)?;
            }
        }
        let unmatched = match &self.selection {
            RowSelection::Keys(rows) => {
                let mut keys: Vec<RowKey> = rows.keys().filter(|k| !matched.contains(*k)).cloned().collect();
                keys.sort();
                keys
            }
            RowSelection::Where { .. } => Vec::new(),
        };
        let changes: Vec<BaseChange<'_>> = own
            .new_rows
            .iter()
            .zip(&olds)
            .map(|(new, old)| {
                let mut p: Vec<&StoredRow> = vec![new];
                p.extend_from_slice(&old[1..]);
                ((new.rowid, new.pos.clone()), Some(p))
            })
            .collect();
        let views = compute_view_changes(&self.views, &infos, &changes, saved)?;
        Ok(UpdateOutput { own, views, unmatched, saved_media: Vec::new() })
    }

    /// True if computing the update may run user code (so it must run without the catalog lock).
    pub fn runs_user_code(&self) -> bool {
        if !self.writes_rows {
            return false;
        }
        let infos = self.path.infos();
        let mut exprs: Vec<&ExprRef> = self.recompute.iter().map(|c| &c.expr).collect();
        if let RowSelection::Where { pred, targets } = &self.selection {
            exprs.extend(pred.iter());
            exprs.extend(targets.iter().map(|(_, e)| e));
        }
        exprs.iter().any(|e| runs_user_code(e, &infos)) || views_run_user_code(&self.views, &infos)
    }

    /// Fails with SERIALIZATION_FAILURE (the update is then planned again) if the rows, schemas or views it read
    /// changed since it was planned.
    fn check_unchanged(&self, cat: &Catalog) -> Result<()> {
        let what = format!("the update of {}", py_repr(&self.tbl_name));
        self.read.check(cat, &what)?;
        if self.cascade && !views_unchanged(cat, self.tbl_id, &self.views)? {
            return Err(super::txn::conflict(&what));
        }
        Ok(())
    }

    /// Writes the changes at a new version of the table (and of each view whose rows change).
    pub fn commit(&self, cat: &mut Catalog, out: UpdateOutput) -> Result<UpdateStatus> {
        cat.adopt_media(out.saved_media.clone());
        self.check_unchanged(cat)?;
        let version = if let Some((col_id, expr)) = &self.schema_change {
            let version = cat.bump_version(self.tbl_id, Some("Altered value expression".to_string()))?;
            let json = crate::expr::serde::to_json(expr)?;
            let md = cat.table_mut(self.tbl_id)?;
            if let Some(sv) = md.last_schema_version_mut() {
                for c in sv.columns.iter_mut().filter(|c| c.col_id == *col_id) {
                    c.value_expr = Some(json.clone());
                }
            }
            if !self.writes_rows {
                cat.commit()?;
                return Ok(UpdateStatus::default());
            }
            version
        } else if self.is_data_versioned {
            cat.bump_version(self.tbl_id, None)?
        } else {
            cat.table(self.tbl_id)?.current_version
        };
        let mut data = cat.data(self.tbl_id)?.as_ref().clone();
        data.retire(&out.own.retire, version, self.is_data_versioned);
        data.append(out.own.new_rows, version);
        let cascade = commit_view_changes(cat, &self.views, out.views, (self.tbl_id, version))?;
        let status = out.own.status + cascade;
        record_version_status(cat, self.tbl_id, &status)?;
        cat.stage_data(self.tbl_id, data)?;
        cat.commit()?;
        Ok(status + UpdateStatus { updated_cols: self.updated_cols.clone(), ..Default::default() })
    }
}

/// True if the live views of `tbl_id` (recursively) are still those of `nodes`, at the same schema versions.
fn views_unchanged(cat: &Catalog, tbl_id: Uuid, nodes: &[ViewUpdateNode]) -> Result<bool> {
    let current = cat.mutable_view_ids(tbl_id);
    let planned: Vec<Uuid> = nodes.iter().map(|n| n.tbl_id()).collect();
    if current != planned {
        return Ok(false);
    }
    for n in nodes {
        if cat.table(n.tbl_id())?.current_schema_version != n.schema_version || !views_unchanged(cat, n.tbl_id(), &n.children)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The tables of a view update tree (recursively).
pub fn view_tree_ids(nodes: &[ViewUpdateNode], out: &mut Vec<Uuid>) {
    for n in nodes {
        out.push(n.tbl_id());
        view_tree_ids(&n.children, out);
    }
}

/// True if updating the views of `nodes` may run user code: their filters, computed columns and iterators (all of
/// them: which ones an update evaluates depends on its rows).
fn views_run_user_code(nodes: &[ViewUpdateNode], base_infos: &[&ElemInfo]) -> bool {
    nodes.iter().any(|n| {
        let mut infos: Vec<&ElemInfo> = vec![&n.compute.info];
        infos.extend_from_slice(base_infos);
        let c = &n.compute;
        c.info.iterator.as_ref().is_some_and(|it| it.call.def.builtin.is_none())
            || c.predicate.iter().chain(c.computed.iter().map(|x| &x.expr)).chain(n.recompute.iter().map(|x| &x.expr)).any(|e| runs_user_code(e, &infos))
            || views_run_user_code(&n.children, &infos)
    })
}

/// A row of a view after the update, as its views see it.
enum ChildBase<'a> {
    Old(&'a StoredRow),
    New(usize),
    Gone,
}

/// Computes how the base changes `changes` (one per changed base row) change each view of `nodes`, recursively.
fn compute_view_changes(
    nodes: &[ViewUpdateNode],
    base_infos: &[&ElemInfo],
    changes: &[BaseChange<'_>],
    saved: &mut Vec<std::path::PathBuf>,
) -> Result<Vec<ViewChanges>> {
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        // the view's live rows of the changed base rows, grouped by the key of their base row
        let base_keys: HashSet<RowKey> = changes.iter().map(|(k, _)| k.clone()).collect();
        let by_base = node.data.live_rows_by_prefix(&base_keys, node.base_num_pos);
        let rewrites = node.membership_change || node.reload || !node.recompute.is_empty();
        let mut ch = TableChanges::default();
        // the dependent columns are recomputed over all of the view's updated rows together (user-function calls
        // dispatched together, window functions in window order): (index in `ch.new_rows`, index of its base change)
        // of each row that awaits that
        let mut pending: Vec<(usize, usize)> = Vec::new();
        // (view row key, the row as the view's views see it, index of its base change)
        let mut child_bases: Vec<(RowKey, ChildBase<'_>, usize)> = Vec::new();
        for (ci, (key, base)) in changes.iter().enumerate() {
            let olds: Vec<&StoredRow> = by_base.get(key).cloned().unwrap_or_default();
            let Some(bp) = base else {
                for o in olds {
                    ch.retire.insert((o.rowid, o.pos.clone()));
                    child_bases.push(((o.rowid, o.pos.clone()), ChildBase::Gone, ci));
                }
                continue;
            };
            if !rewrites {
                for o in olds {
                    child_bases.push(((o.rowid, o.pos.clone()), ChildBase::Old(o), ci));
                }
                continue;
            }
            if node.reload {
                // a component view generates the base row's rows again
                for o in &olds {
                    ch.retire.insert((o.rowid, o.pos.clone()));
                }
                let mut new_status = UpdateStatus::default();
                let new_rows = node.compute.rows_for(base_infos, bp, &mut new_status, saved)?;
                ch.status = std::mem::take(&mut ch.status) + new_status.insert_to_update();
                let new_keys: HashSet<RowKey> = new_rows.iter().map(|r| (r.rowid, r.pos.clone())).collect();
                for o in &olds {
                    if !new_keys.contains(&(o.rowid, o.pos.clone())) {
                        child_bases.push(((o.rowid, o.pos.clone()), ChildBase::Gone, ci));
                    }
                }
                for row in new_rows {
                    child_bases.push(((row.rowid, row.pos.clone()), ChildBase::New(ch.new_rows.len()), ci));
                    ch.new_rows.push(row);
                }
                continue;
            }
            if node.compute.info.iterator.is_some() {
                // a component view whose rows stay: recompute the dependent columns of each
                for o in olds {
                    let row = o.clone();
                    pending.push((ch.new_rows.len(), ci));
                    ch.status.row_count_stats.upd_rows += 1;
                    ch.retire.insert((o.rowid, o.pos.clone()));
                    child_bases.push(((o.rowid, o.pos.clone()), ChildBase::New(ch.new_rows.len()), ci));
                    ch.new_rows.push(row);
                }
                continue;
            }
            let old = olds.first().copied();
            let present = if node.membership_change { node.compute.accepts(base_infos, bp)? } else { old.is_some() };
            match (old, present) {
                (Some(o), true) => {
                    let row = o.clone();
                    pending.push((ch.new_rows.len(), ci));
                    ch.status.row_count_stats.upd_rows += 1;
                    ch.retire.insert(key.clone());
                    child_bases.push((key.clone(), ChildBase::New(ch.new_rows.len()), ci));
                    ch.new_rows.push(row);
                }
                (Some(_), false) => {
                    ch.retire.insert(key.clone());
                    child_bases.push((key.clone(), ChildBase::Gone, ci));
                }
                (None, true) => {
                    let mut new_status = UpdateStatus::default();
                    for row in node.compute.rows_for(base_infos, bp, &mut new_status, saved)? {
                        child_bases.push((key.clone(), ChildBase::New(ch.new_rows.len()), ci));
                        ch.new_rows.push(row);
                    }
                    ch.status = std::mem::take(&mut ch.status) + new_status.insert_to_update();
                }
                (None, false) => {}
            }
        }
        if !pending.is_empty() {
            let mut rows: Vec<StoredRow> = pending.iter().map(|(i, _)| std::mem::replace(&mut ch.new_rows[*i], StoredRow::new(0, Vec::new()))).collect();
            let bases: Vec<&[&StoredRow]> =
                pending.iter().map(|(_, ci)| changes[*ci].1.as_deref().unwrap_or(&[])).collect();
            node.compute.eval_rows(&mut rows, &bases, &node.recompute, base_infos, &mut ch.status, saved, false)?;
            for ((i, _), row) in pending.iter().zip(rows) {
                ch.new_rows[*i] = row;
            }
        }
        let children = {
            let mut infos: Vec<&ElemInfo> = vec![&node.compute.info];
            infos.extend_from_slice(base_infos);
            let child_changes: Vec<BaseChange<'_>> = child_bases
                .iter()
                .map(|(key, cb, ci)| {
                    let base = changes[*ci].1.as_ref();
                    let row = match cb {
                        ChildBase::Old(o) => Some(*o),
                        ChildBase::New(i) => Some(&ch.new_rows[*i]),
                        ChildBase::Gone => None,
                    };
                    let path = match (row, base) {
                        (Some(r), Some(bp)) => {
                            let mut p: Vec<&StoredRow> = vec![r];
                            p.extend_from_slice(bp);
                            Some(p)
                        }
                        _ => None,
                    };
                    (key.clone(), path)
                })
                .collect();
            compute_view_changes(&node.children, &infos, &child_changes, saved)?
        };
        let bump = rewrites || !ch.retire.is_empty() || !ch.new_rows.is_empty();
        out.push(ViewChanges { changes: ch, bump, children });
    }
    Ok(out)
}

/// Writes view changes, each changed view at a new version caused by `root`; returns the views' combined status.
fn commit_view_changes(cat: &mut Catalog, nodes: &[ViewUpdateNode], changes: Vec<ViewChanges>, root: (Uuid, i64)) -> Result<UpdateStatus> {
    let mut total = UpdateStatus::default();
    for (node, vc) in nodes.iter().zip(changes) {
        let view_id = node.tbl_id();
        let mut status = vc.changes.status;
        if vc.bump {
            let version = cat.bump_version(view_id, None)?;
            cat.current_version_md_mut(view_id)?.cascade_from = Some(root);
            let mut data = cat.data(view_id)?.as_ref().clone();
            data.retire(&vc.changes.retire, version, true);
            data.append(vc.changes.new_rows, version);
            let cascade = commit_view_changes(cat, &node.children, vc.children, root)?;
            status = status + cascade;
            record_version_status(cat, view_id, &status)?;
            cat.stage_data(view_id, data)?;
        } else {
            status = status + commit_view_changes(cat, &node.children, vc.children, root)?;
        }
        total = total + status.to_cascade();
    }
    Ok(total)
}

/// `delete()`: retires the rows of `tbl_id` that pass `pred` and the view rows that belong to them; a table or view
/// gets a new version only if it loses rows. Commits.
pub fn delete(cat: &mut Catalog, tbl_id: Uuid, pred: Option<&ExprRef>) -> Result<UpdateStatus> {
    let md = cat.table(tbl_id)?.clone();
    let path = cat.tbl_path(tbl_id, None)?;
    let input = PathInput::snapshot(cat, &path)?;
    let infos = input.infos();
    let mut keys: HashSet<RowKey> = HashSet::new();
    for r in input.rows_where(pred)? {
        if let Some(p) = pred {
            if !eval_predicate(p, &PathCtx::new(&infos, &r))? {
                continue;
            }
        }
        keys.insert((r[0].rowid, r[0].pos.clone()));
    }
    let mut status = UpdateStatus::default();
    status.row_count_stats.del_rows = keys.len() as i64;
    let root_version = md.current_version + 1;
    if !keys.is_empty() {
        let version = if md.is_data_versioned { cat.bump_version(tbl_id, None)? } else { md.current_version };
        let mut data = cat.data(tbl_id)?.as_ref().clone();
        data.retire(&keys, version, md.is_data_versioned);
        cat.stage_data(tbl_id, data)?;
    }
    let cascade = delete_from_views(cat, tbl_id, &keys, cat.num_pos(tbl_id), (tbl_id, root_version))?;
    status = status + cascade;
    if !keys.is_empty() {
        record_version_status(cat, tbl_id, &status)?;
    }
    cat.commit()?;
    Ok(status)
}

/// Retires the rows of the live views of `tbl_id` that belong to the base rows `keys` (recursively).
fn delete_from_views(cat: &mut Catalog, tbl_id: Uuid, keys: &HashSet<RowKey>, num_pos: usize, root: (Uuid, i64)) -> Result<UpdateStatus> {
    let mut total = UpdateStatus::default();
    if keys.is_empty() {
        return Ok(total);
    }
    for v in cat.mutable_view_ids(tbl_id) {
        let mut data = cat.data(v)?.as_ref().clone();
        let next_version = cat.table(v)?.current_version + 1;
        let retired = data.retire_by_prefix(keys, num_pos, next_version);
        let mut status = UpdateStatus::default();
        status.row_count_stats.del_rows = retired.len() as i64;
        let view_keys: HashSet<RowKey> = retired.into_iter().collect();
        if !view_keys.is_empty() {
            cat.bump_version(v, None)?;
            cat.current_version_md_mut(v)?.cascade_from = Some(root);
            cat.stage_data(v, data)?;
        }
        let cascade = delete_from_views(cat, v, &view_keys, cat.num_pos(v), root)?;
        status = status + cascade;
        if !view_keys.is_empty() {
            record_version_status(cat, v, &status)?;
        }
        total = total + status.to_cascade();
    }
    Ok(total)
}
