//! Adding an embedding index: evaluates the embeddings of the existing rows (plan under the lock, compute without
//! it, commit under it, as for computed columns), then records the index and its hidden value column.

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use super::compute::{ComputedCol, eval_computed, unstored_exprs};
use super::insert::UpdateStatus;
use crate::catalog::Catalog;
use crate::catalog::indexes::IndexSpec;
use crate::catalog::table::{ColId, RowCountStats};
use crate::error::{Result, py_repr};
use crate::expr::ExprRef;
use crate::store::rows::{CellError, LIVE, TableData};
use crate::types::ColumnType;
use crate::value::Value;

pub struct AddIndexPlan {
    pub tbl_id: Uuid,
    pub schema_version: i64,
    pub spec: IndexSpec,
    /// the embedding of the indexed column
    pub expr: ExprRef,
    pub val_type: ColumnType,
    /// `<table>.<column>:<index>`, as errors are reported
    pub qualified_name: String,
    pub data: Arc<TableData>,
    pub eval_col_id: ColId,
    pub unstored: HashMap<ColId, ExprRef>,
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
        let data = cat.data(tbl_id)?;
        Ok(AddIndexPlan {
            tbl_id,
            schema_version: md.current_schema_version,
            val_type: expr.col_type.with_nullable(true),
            qualified_name: format!("{}.{col_name}:{idx_name}", md.name),
            spec: IndexSpec { name: Some(idx_name), ..spec },
            expr,
            data,
            eval_col_id: md.next_col_id,
            unstored: unstored_exprs(&md)?,
            replace,
        })
    }

    /// Evaluates the embedding of every live row (errors are recorded, as upstream never aborts on index errors).
    pub fn compute(&self, stats: &mut RowCountStats, cols_with_excs: &mut Vec<String>) -> Result<Vec<(usize, Value, Option<CellError>)>> {
        let col = ComputedCol { col_id: self.eval_col_id, name: self.qualified_name.clone(), expr: self.expr.clone(), is_media: false };
        let mut out = Vec::new();
        for (i, r) in self.data.rows.iter().enumerate() {
            if r.v_max != LIVE {
                continue;
            }
            let mut row = r.clone();
            stats.upd_rows += 1;
            eval_computed(&mut row, std::slice::from_ref(&col), Some(&self.unstored), false, stats, cols_with_excs)?;
            out.push((i, row.cell(col.col_id).clone(), row.error(col.col_id).cloned()));
        }
        Ok(out)
    }

    pub fn commit(
        &self,
        cat: &mut Catalog,
        values: Vec<(usize, Value, Option<CellError>)>,
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
        for (i, v, err) in values {
            if let Some(r) = data.rows.get_mut(i) {
                r.set_cell(col_id, v);
                r.set_error(col_id, err);
            }
        }
        {
            let md = cat.table_mut(self.tbl_id)?;
            let version = md.current_version;
            if let Some(vmd) = md.versions.iter_mut().find(|v| v.version == version) {
                vmd.row_count_stats.upd_rows += stats.upd_rows;
                vmd.row_count_stats.num_excs += stats.num_excs;
                vmd.row_count_stats.computed_values += stats.computed_values;
            }
        }
        cat.write_data(self.tbl_id, data)?;
        Ok(UpdateStatus {
            updated_cols: Vec::new(),
            cols_with_excs,
            row_count_stats: stats,
            cascade_row_count_stats: RowCountStats::default(),
        })
    }
}
