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
    /// the stored computed columns, in schema order
    pub computed: Vec<ComputedCol>,
    /// where in-memory media results are written
    pub media: MediaTarget,
    /// `info` with the iterator's unstored outputs read from the row's cells (for rows being generated)
    pub compute_info: ElemInfo,
}

impl ViewCompute {
    /// The current definition of view `md`; its media files are written to `media_dir`.
    pub fn of(md: &TableMd, media_dir: std::path::PathBuf) -> Result<ViewCompute> {
        let predicate = match md.view.as_ref().and_then(|v| v.predicate.as_ref()) {
            Some(p) => Some(crate::expr::serde::from_json(p)?),
            None => None,
        };
        let mut computed = Vec::new();
        for c in md.columns_at(None)? {
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
        let media = MediaTarget { tbl_id: md.id, dir: media_dir, version: md.current_version + 1 };
        let info = ElemInfo::of(md, None)?;
        let compute_info = info.with_materialized_outputs();
        Ok(ViewCompute { info, name: md.name.clone(), predicate, computed, media, compute_info })
    }

    /// True if the base path row passes the view's filter.
    pub fn accepts(&self, base_infos: &[&ElemInfo], base: &[&StoredRow]) -> Result<bool> {
        match &self.predicate {
            Some(p) => eval_predicate(p, &PathCtx::new(base_infos, base)),
            None => Ok(true),
        }
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
        eval_into(row, cols, &infos, base, &self.name, false, &mut status.row_count_stats, &mut status.cols_with_excs)?;
        for c in cols.iter().filter(|c| c.is_media) {
            self.media.materialize(row, c.col_id, saved)?;
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
        if !self.accepts(base_infos, base)? {
            return Ok(Vec::new());
        }
        let Some(it) = &self.info.iterator else {
            let mut row = StoredRow::new(base[0].rowid, base[0].pos.clone());
            self.eval_columns(&mut row, &self.computed, base_infos, base, status, saved, false)?;
            status.row_count_stats.ins_rows += 1;
            return Ok(vec![row]);
        };
        // a component view: one row per iterator output row, keyed by the base row's key and the position
        let outputs = it.call.eval_rows(&PathCtx::new(base_infos, base))?;
        let mut rows = Vec::with_capacity(outputs.len());
        for (p, values) in outputs.into_iter().enumerate() {
            let mut pos = base[0].pos.clone();
            pos.push(p as i64);
            let mut row = StoredRow::new(base[0].rowid, pos);
            for ((col_id, _, _), v) in it.outputs.iter().zip(values) {
                row.set_cell(*col_id, v);
            }
            self.eval_columns(&mut row, &self.computed, base_infos, base, status, saved, true)?;
            for (col_id, _, stored) in &it.outputs {
                if *stored {
                    self.media.materialize(&mut row, *col_id, saved)?;
                } else {
                    row.set_cell(*col_id, crate::value::Value::Null);
                }
            }
            status.row_count_stats.ins_rows += 1;
            rows.push(row);
        }
        Ok(rows)
    }

    /// The rows of a new view over the base path `base` (its initial population).
    pub fn populate(&self, base: &PathInput, saved: &mut Vec<std::path::PathBuf>) -> Result<(TableData, UpdateStatus)> {
        let infos = base.infos();
        let mut status = UpdateStatus::default();
        let mut data = TableData::default();
        for r in base.rows()? {
            data.rows.extend(self.rows_for(&infos, &r, &mut status, saved)?);
        }
        Ok((data, status))
    }
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
                    compute: ViewCompute::of(md, cat.store().media_dir(v))?,
                    schema_version: md.current_schema_version,
                    children: ViewNode::tree(cat, v)?,
                })
            })
            .collect()
    }

    pub fn tbl_id(&self) -> Uuid {
        self.compute.info.tbl_id
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
        let mut rows = Vec::new();
        let mut parents = Vec::new();
        for (i, bp) in base_paths.iter().enumerate() {
            for r in node.compute.rows_for(base_infos, bp, &mut status, saved)? {
                rows.push(r);
                parents.push(i);
            }
        }
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
/// insert); `rowid_offset` is added to every row's rowid (rows are computed with provisional rowids). Returns the
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
