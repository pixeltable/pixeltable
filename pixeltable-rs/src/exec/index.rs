//! Adding an embedding index: evaluates the embeddings of the existing rows (plan under the lock, compute without
//! it, commit under it, as for computed columns), then records the index and its hidden value column.

use uuid::Uuid;

use super::compute::{ComputedCol, KeyedValue};
use super::insert::UpdateStatus;
use super::propagate::record_version_status;
use super::scan::{PathInput, eval_into};
use crate::catalog::Catalog;
use crate::catalog::indexes::IndexSpec;
use crate::catalog::table::{ColId, RowCountStats};
use crate::error::{Result, py_repr};
use crate::expr::ExprRef;
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
        })
    }

    /// Evaluates the embedding of every live row (errors are recorded, as upstream never aborts on index errors).
    pub fn compute(&self, stats: &mut RowCountStats, cols_with_excs: &mut Vec<String>) -> Result<Vec<KeyedValue>> {
        let col = ComputedCol { col_id: self.eval_col_id, name: self.val_col_name.clone(), expr: self.expr.clone(), is_media: false };
        let infos = self.path.infos();
        let mut out = Vec::new();
        for r in self.path.rows()? {
            stats.upd_rows += 1;
            let mut row = r[0].clone();
            eval_into(&mut row, std::slice::from_ref(&col), &infos, &r[1..], &self.tbl_name, false, stats, cols_with_excs)?;
            out.push(((row.rowid, row.pos.clone()), row.cell(col.col_id).clone(), row.error(col.col_id).cloned()));
        }
        Ok(out)
    }

    pub fn commit(
        &self,
        cat: &mut Catalog,
        values: Vec<KeyedValue>,
        stats: RowCountStats,
        cols_with_excs: Vec<String>,
    ) -> Result<UpdateStatus> {
        if cat.table(self.tbl_id)?.current_schema_version != self.schema_version {
            crate::bail!(
                ConcurrentModification,
                "The schema of the table changed while index {} was being built; retry",
                py_repr(self.spec.name.as_deref().unwrap_or_default())
            );
        }
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
