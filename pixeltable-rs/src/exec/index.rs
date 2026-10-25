//! Adding an embedding index: evaluates the embeddings of the existing rows (plan under the lock, compute without
//! it, commit under it if what the plan read is unchanged, as for computed columns, see `exec::txn`), then records
//! the index and its hidden value column.

use uuid::Uuid;

use super::compute::{ComputedCol, KeyedValue};
use super::insert::UpdateStatus;
use super::propagate::record_version_status;
use super::scan::{PathInput, eval_rows_batch};
use super::txn::ReadSet;
use crate::catalog::Catalog;
use crate::catalog::indexes::IndexSpec;
use crate::catalog::table::{ColId, RowCountStats};
use crate::error::{Result, py_repr};
use crate::expr::ExprRef;
use crate::store::rows::StoredRow;
use crate::types::ColumnType;

pub struct AddIndexPlan {
    pub tbl_id: Uuid,
    pub tbl_name: String,
    pub schema_version: i64,
    pub spec: IndexSpec,
    /// the embedding of the indexed column
    pub expr: ExprRef,
    pub val_type: ColumnType,
    /// `<column>:<index>`, as errors are reported
    pub val_col_name: String,
    /// the table's path at its current version (the rows to compute the embeddings for)
    pub path: PathInput,
    pub eval_col_id: ColId,
    /// an existing index to drop at commit (`if_exists='replace'`)
    pub replace: Vec<u32>,
    /// what the plan read (the table's path), validated by the commit
    pub read: ReadSet,
}

/// The result of `AddIndexPlan::compute()`.
pub struct ComputedIndex {
    pub values: Vec<KeyedValue>,
    pub stats: RowCountStats,
    pub cols_with_excs: Vec<String>,
}

impl AddIndexPlan {
    pub fn new(
        cat: &mut Catalog,
        tbl_id: Uuid,
        spec: IndexSpec,
        expr: ExprRef,
        col_name: &str,
        replace: Vec<u32>,
    ) -> Result<AddIndexPlan> {
        let md = cat.table(tbl_id)?.clone();
        let idx_name = spec.name.clone().unwrap_or_else(|| crate::catalog::indexes::generate_idx_name(&md));
        let path = cat.tbl_path(tbl_id, None)?;
        let read = ReadSet::of(cat, path.iter().filter(|e| e.pinned.is_none()).map(|e| e.tbl_id))?;
        Ok(AddIndexPlan {
            tbl_id,
            tbl_name: md.name.clone(),
            schema_version: md.current_schema_version,
            val_type: expr.col_type.with_nullable(true),
            val_col_name: format!("{col_name}:{idx_name}"),
            spec: IndexSpec { name: Some(idx_name), ..spec },
            expr,
            path: PathInput::snapshot(cat, &path)?,
            eval_col_id: md.next_col_id,
            replace,
            read,
        })
    }

    /// Evaluates the embedding of every live row (errors are recorded, as upstream never aborts on index errors).
    pub fn compute(&self) -> Result<ComputedIndex> {
        let mut out = ComputedIndex { values: Vec::new(), stats: RowCountStats::default(), cols_with_excs: Vec::new() };
        out.values = self.compute_values(&mut out.stats, &mut out.cols_with_excs)?;
        Ok(out)
    }

    fn compute_values(&self, stats: &mut RowCountStats, cols_with_excs: &mut Vec<String>) -> Result<Vec<KeyedValue>> {
        let col = ComputedCol { col_id: self.eval_col_id, name: self.val_col_name.clone(), expr: self.expr.clone(), is_media: false };
        let infos = self.path.infos();
        let path_rows = self.path.rows()?;
        let cols = std::slice::from_ref(&col);
        let mut rows = crate::exec::scan::rows_for_eval(&path_rows, &infos, cols);
        let bases: Vec<&[&StoredRow]> = path_rows.iter().map(|r| &r[1..]).collect();
        stats.upd_rows += rows.len() as i64;
        eval_rows_batch(&mut rows, &bases, cols, &infos, &self.tbl_name, false, stats, cols_with_excs, false)?;
        Ok(rows
            .into_iter()
            .map(|mut row| {
                let (v, err) = crate::exec::scan::take_cell(&mut row, col.col_id);
                ((row.rowid, std::mem::take(&mut row.pos)), v, err)
            })
            .collect())
    }

    /// Records the index and its values, if the rows and schemas the plan read are unchanged (else fails with
    /// SERIALIZATION_FAILURE, and the operation is planned again, see `exec::txn`).
    pub fn commit(&self, cat: &mut Catalog, computed: ComputedIndex) -> Result<UpdateStatus> {
        let ComputedIndex { values, stats, cols_with_excs } = computed;
        self.read.check(cat, &format!("the creation of index {}", py_repr(self.spec.name.as_deref().unwrap_or_default())))?;
        if !self.replace.is_empty() {
            let v = cat.table(self.tbl_id)?.current_version;
            for id in &self.replace {
                cat.remove_index(self.tbl_id, *id, v + 1)?;
            }
        }
        let value_json = crate::expr::serde::to_json(&self.expr)?;
        let col_id = cat.push_embedding_index(self.tbl_id, &self.spec, value_json, self.val_type.clone())?;
        let mut data = cat.data(self.tbl_id)?.as_ref().clone();
        data.set_live_cells(col_id, values);
        let status = UpdateStatus { cols_with_excs, row_count_stats: stats, ..Default::default() };
        record_version_status(cat, self.tbl_id, &status)?;
        cat.write_data(self.tbl_id, data)?;
        Ok(status)
    }
}
