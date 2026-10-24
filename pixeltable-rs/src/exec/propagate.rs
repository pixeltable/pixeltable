//! Computing view rows from base rows, and propagating base inserts to live views.
//!
//! A view row belongs to one base path row (a row of the base and of each of the base's bases): the view's filter
//! decides whether it exists, and the view's stored computed columns are evaluated in the context of the base path
//! row. `ViewCompute` holds what that needs; `ViewNode` is the tree of live views of a table (each with its own
//! views), captured under the catalog lock when an operation is planned; the rows are computed without the lock
//! and committed with it (`commit_inserted_rows()`).

use uuid::Uuid;

use super::compute::{ComputedCol, MediaTarget};
use super::insert::UpdateStatus;
use super::scan::{ElemInfo, PathCtx, PathInput, eval_into};
use crate::catalog::Catalog;
use crate::catalog::table::{RowCountStats, TableMd};
use crate::error::{Result, py_repr};
use crate::expr::ExprRef;
use crate::expr::eval::eval_predicate;
use crate::store::rows::{StoredRow, TableData};

/// How the rows of a view are computed from base path rows.
#[derive(Debug, Clone)]
pub struct ViewCompute {
    pub info: ElemInfo,
    pub name: String,
    pub predicate: Option<ExprRef>,
    /// the stored computed columns, in schema order (and, see `current()`, the hidden index value columns)
    pub computed: Vec<ComputedCol>,
    /// where in-memory media results are written (None: they stay in memory, `Table.compute()`)
    pub media: Option<MediaTarget>,
    /// `info` with the iterator's unstored outputs read from the row's cells (for rows being generated)
    pub compute_info: ElemInfo,
    /// true: an evaluation error is an error (`compute(on_error='abort')`), else a cell error
    pub fail_on_exception: bool,
    /// the view's sample clause (a live view's is fractional and seeded)
    pub sample: Option<super::SampleClause>,
}

impl ViewCompute {
    /// The current definition of view `md`; its media files are written to `media_dir`.
    pub fn of(md: &TableMd, media_dir: std::path::PathBuf) -> Result<ViewCompute> {
        let predicate = match md.view.as_ref().and_then(|v| v.predicate.as_ref()) {
            Some(p) => Some(crate::expr::serde::from_json(p)?),
            None => None,
        };
        let mut computed = Vec::new();
        let own = md.columns_at(None)?;
        for c in &own {
            if c.md.stored {
                if let Some(e) = &c.md.value_expr {
                    computed.push(ComputedCol {
                        col_id: c.md.id,
                        name: c.name.clone(),
                        expr: crate::expr::serde::from_json(e)?,
                        is_media: c.md.col_type.is_media(),
                    });
                }
            }
        }
        // the value columns of the indexes on the view's own columns (see `current()` for its bases' columns)
        computed.extend(super::compute::index_value_columns_md(md, &own, None)?);
        let media = MediaTarget { tbl_id: md.id, dir: media_dir, version: md.current_version + 1 };
        let info = ElemInfo::of(md, None)?;
        let compute_info = info.with_materialized_outputs();
        let sample = match md.view.as_ref().and_then(|v| v.sample.as_ref()) {
            Some(j) => Some(super::SampleClause::from_json(j)?),
            None => None,
        };
        Ok(ViewCompute {
            info,
            name: md.name.clone(),
            predicate,
            computed,
            media: Some(media),
            compute_info,
            fail_on_exception: false,
            sample,
        })
    }

    /// The current definition of view `tbl_id` in the catalog, including the value columns of the indexes on its
    /// bases' columns.
    pub fn current(cat: &Catalog, tbl_id: Uuid) -> Result<ViewCompute> {
        let mut vc = ViewCompute::of(cat.table(tbl_id)?, cat.store().media_dir(tbl_id))?;
        for c in super::compute::index_value_columns(cat, tbl_id, None)? {
            if !vc.computed.iter().any(|x| x.col_id == c.col_id) {
                vc.computed.push(c);
            }
        }
        Ok(vc)
    }

    /// True if the base path row passes the view's filter (and is in its fractional sample).
    pub fn accepts(&self, base_infos: &[&ElemInfo], base: &[&StoredRow]) -> Result<bool> {
        if let Some(p) = &self.predicate {
            if !eval_predicate(p, &PathCtx::new(base_infos, base))? {
                return Ok(false);
            }
        }
        if let Some(sc) = self.sample.as_ref().filter(|s| s.is_repeatable()) {
            let seed = sc.seed.unwrap_or(0) as u64;
            let h = super::sample::key_hash(seed, base[0].rowid, &base[0].pos);
            return Ok(super::sample::fraction_keeps(h, sc.fraction.unwrap_or(1.0)));
        }
        Ok(true)
    }

    /// Evaluates `cols` into `row`, a row of this view belonging to the base path row `base`. Evaluation errors
    /// become cell errors; in-memory media results are written to the view's media directory (recorded in `saved`).
    /// `materialized`: the row holds all iterator outputs in its cells (it is being generated).
    #[allow(clippy::too_many_arguments)]
    pub fn eval_columns(
        &self,
        row: &mut StoredRow,
        cols: &[ComputedCol],
        base_infos: &[&ElemInfo],
        base: &[&StoredRow],
        status: &mut UpdateStatus,
        saved: &mut Vec<std::path::PathBuf>,
        materialized: bool,
    ) -> Result<()> {
        let mut infos: Vec<&ElemInfo> = Vec::with_capacity(base_infos.len() + 1);
        infos.push(if materialized { &self.compute_info } else { &self.info });
        infos.extend_from_slice(base_infos);
        eval_into(row, cols, &infos, base, &self.name, self.fail_on_exception, &mut status.row_count_stats, &mut status.cols_with_excs)?;
        if let Some(media) = &self.media {
            for c in cols.iter().filter(|c| c.is_media) {
                media.materialize(row, c.col_id, saved)?;
            }
        }
        Ok(())
    }

    /// The new view rows for one base path row (`base[0]` is the base's row): none if the filter rejects it.
    pub fn rows_for(
        &self,
        base_infos: &[&ElemInfo],
        base: &[&StoredRow],
        status: &mut UpdateStatus,
        saved: &mut Vec<std::path::PathBuf>,
    ) -> Result<Vec<StoredRow>> {
        Ok(self.rows_for_all(base_infos, &[base], status, saved)?.into_iter().map(|(_, r)| r).collect())
    }

    /// The new view rows for base path rows `bases`, each with the index of its base path row. Columns computed
    /// with window functions are evaluated over all the new rows, in window order (as for a base table insert).
    pub fn rows_for_all(
        &self,
        base_infos: &[&ElemInfo],
        bases: &[&[&StoredRow]],
        status: &mut UpdateStatus,
        saved: &mut Vec<std::path::PathBuf>,
    ) -> Result<Vec<(usize, StoredRow)>> {
        let materialized = self.info.iterator.is_some();
        let mut out: Vec<(usize, StoredRow)> = Vec::new();
        if !self.computed.iter().any(|c| super::query::has_window_calls(&c.expr)) {
            // row by row, so that only one base row's unstored iterator outputs are held at a time
            for (i, base) in bases.iter().enumerate() {
                for mut row in self.new_rows(base_infos, base)? {
                    self.eval_columns(&mut row, &self.computed, base_infos, base, status, saved, materialized)?;
                    self.finish_row(&mut row, saved)?;
                    out.push((i, row));
                }
            }
        } else {
            let mut parents = Vec::new();
            let mut rows = Vec::new();
            for (i, base) in bases.iter().enumerate() {
                for row in self.new_rows(base_infos, base)? {
                    parents.push(i);
                    rows.push(row);
                }
            }
            let row_bases: Vec<&[&StoredRow]> = parents.iter().map(|i| bases[*i]).collect();
            self.eval_rows(&mut rows, &row_bases, &self.computed, base_infos, status, saved, materialized)?;
            for row in rows.iter_mut() {
                self.finish_row(row, saved)?;
            }
            out = parents.into_iter().zip(rows).collect();
        }
        status.row_count_stats.ins_rows += out.len() as i64;
        Ok(out)
    }

    /// `eval_columns()` for a batch of rows of this view (`rows[i]` belongs to the base path row `bases[i]`); columns
    /// computed with window functions are evaluated over the whole batch.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_rows(
        &self,
        rows: &mut [StoredRow],
        bases: &[&[&StoredRow]],
        cols: &[ComputedCol],
        base_infos: &[&ElemInfo],
        status: &mut UpdateStatus,
        saved: &mut Vec<std::path::PathBuf>,
        materialized: bool,
    ) -> Result<()> {
        let mut infos: Vec<&ElemInfo> = Vec::with_capacity(base_infos.len() + 1);
        infos.push(if materialized { &self.compute_info } else { &self.info });
        infos.extend_from_slice(base_infos);
        super::scan::eval_rows_into(
            rows,
            bases,
            cols,
            &infos,
            &self.name,
            self.fail_on_exception,
            &mut status.row_count_stats,
            &mut status.cols_with_excs,
        )?;
        if let Some(media) = &self.media {
            for row in rows.iter_mut() {
                for c in cols.iter().filter(|c| c.is_media) {
                    media.materialize(row, c.col_id, saved)?;
                }
            }
        }
        Ok(())
    }

    /// The rows of this view for one base path row before their computed columns are evaluated: one row keyed by the
    /// base row's key, or a component view's rows (keyed by the base row's key and the position) holding all the
    /// iterator's outputs.
    fn new_rows(&self, base_infos: &[&ElemInfo], base: &[&StoredRow]) -> Result<Vec<StoredRow>> {
        if !self.accepts(base_infos, base)? {
            return Ok(Vec::new());
        }
        let Some(it) = &self.info.iterator else {
            return Ok(vec![StoredRow::new(base[0].rowid, base[0].pos.clone())]);
        };
        let outputs = it.call.eval_rows(&PathCtx::new(base_infos, base))?;
        let mut rows = Vec::with_capacity(outputs.len());
        for (p, values) in outputs.into_iter().enumerate() {
            let mut pos = base[0].pos.clone();
            pos.push(p as i64);
            let mut row = StoredRow::new(base[0].rowid, pos);
            for ((col_id, _, _), v) in it.outputs.iter().zip(values) {
                row.set_cell(*col_id, v);
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Drops the unstored iterator outputs of a new row and writes its in-memory media outputs to files.
    fn finish_row(&self, row: &mut StoredRow, saved: &mut Vec<std::path::PathBuf>) -> Result<()> {
        let Some(it) = &self.info.iterator else { return Ok(()) };
        for (col_id, _, stored) in &it.outputs {
            if !*stored {
                row.set_cell(*col_id, crate::value::Value::Null);
            } else if let Some(media) = &self.media {
                media.materialize(row, *col_id, saved)?;
            }
        }
        Ok(())
    }

    /// The rows of a new view over the base path `base` (its initial population). A snapshot's sample that is not
    /// fractional and unstratified selects among all the base rows that pass the filter, here.
    pub fn populate(&self, base: &PathInput, saved: &mut Vec<std::path::PathBuf>) -> Result<(TableData, UpdateStatus)> {
        let infos = base.infos();
        let mut status = UpdateStatus::default();
        let mut data = TableData::default();
        let mut rows = base.rows()?;
        if let Some(sc) = self.sample.as_ref().filter(|s| !s.is_repeatable()) {
            let mut candidates = Vec::new();
            for r in rows {
                let passes = match &self.predicate {
                    Some(p) => eval_predicate(p, &PathCtx::new(&infos, &r))?,
                    None => true,
                };
                if passes {
                    candidates.push(r);
                }
            }
            rows = select_sample(candidates, &infos, sc)?;
            let unsampled = ViewCompute { predicate: None, sample: None, ..self.clone() };
            let refs: Vec<&[&StoredRow]> = rows.iter().map(|r| r.as_slice()).collect();
            data.rows.extend(unsampled.rows_for_all(&infos, &refs, &mut status, saved)?.into_iter().map(|(_, r)| r));
            return Ok((data, status));
        }
        let refs: Vec<&[&StoredRow]> = rows.iter().map(|r| r.as_slice()).collect();
        data.rows.extend(self.rows_for_all(&infos, &refs, &mut status, saved)?.into_iter().map(|(_, r)| r));
        Ok((data, status))
    }
}

/// The rows a sample clause keeps (upstream `SqlSampleNode`): rows are ranked by the hash of their key with the seed
/// (a random one if none); `n` keeps the lowest-ranked rows, a fraction the rows below its share of the hash space;
/// stratified samples rank within the strata (the values of the `stratify_by` expressions).
pub fn select_sample<'a>(
    rows: Vec<Vec<&'a StoredRow>>,
    infos: &[&ElemInfo],
    sc: &super::SampleClause,
) -> Result<Vec<Vec<&'a StoredRow>>> {
    use super::sample::{fraction_keeps, key_hash, random_u64};
    let seed = sc.seed.map(|s| s as u64).unwrap_or_else(random_u64);
    let mut keyed: Vec<(u64, Vec<&StoredRow>)> = rows.into_iter().map(|r| (key_hash(seed, r[0].rowid, &r[0].pos), r)).collect();
    keyed.sort_by_key(|(k, _)| *k);
    if !sc.is_stratified() {
        return Ok(match (sc.fraction, sc.n) {
            (Some(f), _) => keyed.into_iter().filter(|(k, _)| fraction_keeps(*k, f)).map(|(_, r)| r).collect(),
            (None, Some(n)) => keyed.into_iter().take(n.max(0) as usize).map(|(_, r)| r).collect(),
            _ => keyed.into_iter().map(|(_, r)| r).collect(),
        });
    }
    let mut strata: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut counts: Vec<usize> = Vec::new();
    let mut ranked: Vec<(usize, usize, u64, Vec<&StoredRow>)> = Vec::new();
    for (k, r) in keyed {
        let ctx = PathCtx::new(infos, &r);
        let vals = sc.stratify_by.iter().map(|e| crate::expr::eval::eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
        let skey = format!("{vals:?}");
        let next = strata.len();
        let sid = *strata.entry(skey).or_insert(next);
        if sid == counts.len() {
            counts.push(0);
        }
        let rank = counts[sid];
        counts[sid] += 1;
        ranked.push((sid, rank, k, r));
    }
    let mut out: Vec<(usize, u64, Vec<&StoredRow>)> = match (sc.n_per_stratum, sc.n, sc.fraction) {
        (Some(per), _, _) => ranked.into_iter().filter(|(_, rank, _, _)| (*rank as i64) < per).map(|(_, rk, k, r)| (rk, k, r)).collect(),
        (None, Some(n), _) => {
            let mut v: Vec<(usize, u64, Vec<&StoredRow>)> = ranked.into_iter().map(|(_, rk, k, r)| (rk, k, r)).collect();
            v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            v.truncate(n.max(0) as usize);
            v
        }
        (None, None, Some(f)) => ranked
            .into_iter()
            .filter(|(sid, rank, _, _)| (*rank as f64) < (f * counts[*sid] as f64).ceil())
            .map(|(_, rk, k, r)| (rk, k, r))
            .collect(),
        _ => Vec::new(),
    };
    out.sort_by_key(|(_, k, _)| *k);
    Ok(out.into_iter().map(|(_, _, r)| r).collect())
}

/// A live view of a table, with its own live views.
#[derive(Debug, Clone)]
pub struct ViewNode {
    pub compute: ViewCompute,
    pub schema_version: i64,
    pub children: Vec<ViewNode>,
}

impl ViewNode {
    /// The live views of `tbl_id`, recursively.
    pub fn tree(cat: &Catalog, tbl_id: Uuid) -> Result<Vec<ViewNode>> {
        cat.mutable_view_ids(tbl_id)
            .into_iter()
            .map(|v| {
                let md = cat.table(v)?;
                check_view_valid(md, "updated in")?;
                Ok(ViewNode {
                    compute: ViewCompute::current(cat, v)?,
                    schema_version: md.current_schema_version,
                    children: ViewNode::tree(cat, v)?,
                })
            })
            .collect()
    }

    pub fn tbl_id(&self) -> Uuid {
        self.compute.info.tbl_id
    }

    /// True if this view or one of its views samples its rows.
    pub fn samples(&self) -> bool {
        self.compute.sample.is_some() || self.children.iter().any(ViewNode::samples)
    }

    /// Fails if the view tree of `tbl_id` changed since it was captured (views added or dropped, schemas changed).
    pub fn check_unchanged(cat: &Catalog, tbl_id: Uuid, nodes: &[ViewNode]) -> Result<()> {
        let current = cat.mutable_view_ids(tbl_id);
        let planned: Vec<Uuid> = nodes.iter().map(|n| n.tbl_id()).collect();
        if current != planned {
            crate::bail!(
                ConcurrentModification,
                "The views of {} changed while the operation was being prepared; retry",
                py_repr(&cat.table(tbl_id)?.name)
            );
        }
        for n in nodes {
            if cat.table(n.tbl_id())?.current_schema_version != n.schema_version {
                crate::bail!(
                    ConcurrentModification,
                    "The schema of view {} changed while the operation was being prepared; retry",
                    py_repr(&n.compute.name)
                );
            }
            ViewNode::check_unchanged(cat, n.tbl_id(), &n.children)?;
        }
        Ok(())
    }
}

/// Fails with upstream's INVALID_STATE error if the view's iterator or computed columns no longer match the code.
pub fn check_view_valid(md: &TableMd, op: &str) -> Result<()> {
    let display = crate::functions::udf::tbl_version_display(md);
    if let Some(j) = md.view.as_ref().and_then(|v| v.iterator.as_ref()) {
        crate::functions::iterator::IteratorCall::from_json(j)?.check_valid(&display, &md.name, op)?;
    }
    for c in md.columns_at(None)?.iter().filter(|c| c.md.stored) {
        if let Some(e) = &c.md.value_expr {
            crate::functions::udf::check_valid_column(&display, &c.name, &*crate::expr::serde::from_json(e)?, op)?;
        }
    }
    Ok(())
}

/// The rows a base insert adds to a view (and to the view's views).
#[derive(Debug, Default)]
pub struct InsertedViewRows {
    pub rows: Vec<StoredRow>,
    pub status: UpdateStatus,
    pub children: Vec<InsertedViewRows>,
}

/// Computes the rows that new base path rows add to each view of `nodes` (recursively). `base_paths[i]` is a new
/// row of the base followed by its rows of the base's bases.
pub fn compute_inserted_rows(
    nodes: &[ViewNode],
    base_infos: &[&ElemInfo],
    base_paths: &[Vec<&StoredRow>],
    saved: &mut Vec<std::path::PathBuf>,
) -> Result<Vec<InsertedViewRows>> {
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        let mut status = UpdateStatus::default();
        let refs: Vec<&[&StoredRow]> = base_paths.iter().map(|p| p.as_slice()).collect();
        let (parents, rows): (Vec<usize>, Vec<StoredRow>) =
            node.compute.rows_for_all(base_infos, &refs, &mut status, saved)?.into_iter().unzip();
        let children = {
            let mut infos: Vec<&ElemInfo> = vec![&node.compute.info];
            infos.extend_from_slice(base_infos);
            let paths: Vec<Vec<&StoredRow>> = rows
                .iter()
                .zip(&parents)
                .map(|(r, &i)| {
                    let mut p = Vec::with_capacity(base_paths[i].len() + 1);
                    p.push(r);
                    p.extend_from_slice(&base_paths[i]);
                    p
                })
                .collect();
            compute_inserted_rows(&node.children, &infos, &paths, saved)?
        };
        out.push(InsertedViewRows { rows, status, children });
    }
    Ok(out)
}

/// Appends computed view rows to their views, each at a new version caused by `root` (the table version of the
/// insert); `rowid_offset` is added to every row's rowid (rows are computed with the rowids the insert expected). Returns the
/// combined status of all views, as a cascade.
pub fn commit_inserted_rows(
    cat: &mut Catalog,
    nodes: &[ViewNode],
    inserted: Vec<InsertedViewRows>,
    root: (Uuid, i64),
    rowid_offset: i64,
) -> Result<UpdateStatus> {
    let mut total = UpdateStatus::default();
    for (node, ins) in nodes.iter().zip(inserted) {
        let view_id = node.tbl_id();
        let version = cat.bump_version(view_id, None)?;
        cat.current_version_md_mut(view_id)?.cascade_from = Some(root);
        let mut data = cat.data(view_id)?.as_ref().clone();
        let rows: Vec<StoredRow> = ins
            .rows
            .into_iter()
            .map(|mut r| {
                r.rowid += rowid_offset;
                r
            })
            .collect();
        data.append(rows, version);
        let cascade = commit_inserted_rows(cat, &node.children, ins.children, root, rowid_offset)?;
        let status = ins.status + cascade;
        record_version_status(cat, view_id, &status)?;
        cat.stage_data(view_id, data)?;
        total = total + status.to_cascade();
    }
    Ok(total)
}

/// Records the status of the operation that created the current version of a table in its version metadata.
pub fn record_version_status(cat: &mut Catalog, tbl_id: Uuid, status: &UpdateStatus) -> Result<()> {
    let vmd = cat.current_version_md_mut(tbl_id)?;
    vmd.row_count_stats = add_stats(&vmd.row_count_stats, &status.row_count_stats);
    vmd.cascade_row_count_stats = add_stats(&vmd.cascade_row_count_stats, &status.cascade_row_count_stats);
    Ok(())
}

pub fn add_stats(a: &RowCountStats, b: &RowCountStats) -> RowCountStats {
    RowCountStats {
        ins_rows: a.ins_rows + b.ins_rows,
        del_rows: a.del_rows + b.del_rows,
        upd_rows: a.upd_rows + b.upd_rows,
        num_excs: a.num_excs + b.num_excs,
        computed_values: a.computed_values + b.computed_values,
    }
}
