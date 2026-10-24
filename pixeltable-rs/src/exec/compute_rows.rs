//! `Table.compute()`: the rows a table (or view) would hold for some input rows, without storing anything.
//!
//! Input rows are rows of the root base table. Only the requested output columns and the columns they depend on
//! are evaluated (stored and unstored computed columns alike); a view's filters drop the rows they reject. In-memory
//! media results stay in memory.

use std::collections::HashSet;
use uuid::Uuid;

use super::compute::{ComputedCol, RowCells, eval_computed};
use super::insert::{InputRow, InsertPlan, prepare_media_inputs};
use crate::catalog::Catalog;
use crate::catalog::path::fold_identifier;
use crate::catalog::table::{ColId, RowCountStats};
use crate::error::Result;
use crate::expr::ExprRef;
use crate::expr::eval::eval;
use crate::store::rows::{CellError, StoredRow};
use crate::types::ColumnType;
use crate::value::Value;

pub struct ComputeRowsPlan {
    /// the root base table, whose rows the inputs are
    pub insert: InsertPlan,
    /// filters of the views between the root base table and the target
    pub predicates: Vec<ExprRef>,
    /// (name, column id in the base table, type)
    pub outputs: Vec<(String, ColId, ColumnType)>,
    /// base table columns to evaluate
    pub needed: HashSet<ColId>,
}

/// One computed output row: values (in output order) and the errors of failed cells, by column name.
pub struct ComputedRow {
    pub values: Vec<Value>,
    pub errors: Vec<(String, CellError)>,
}

impl ComputeRowsPlan {
    /// `outputs`: names of the columns to return (None: all columns of the table).
    pub fn new(cat: &mut Catalog, tbl_id: Uuid, version: Option<i64>, outputs: Option<Vec<String>>) -> Result<ComputeRowsPlan> {
        let chain = cat.base_chain(tbl_id);
        if version.is_some() || chain.iter().any(|id| cat.table(*id).is_ok_and(|t| t.is_snapshot())) {
            crate::bail!(UnsupportedOperation, "{}: compute() is not supported for snapshots.", cat.display_str(tbl_id)?);
        }
        let source = cat.scan_source(tbl_id, None)?;
        let base_id = source.base_tbl_id;
        let insert = InsertPlan::new(cat, base_id, None)?;
        let predicates = source.predicates.iter().map(crate::expr::serde::from_json).collect::<Result<Vec<_>>>()?;
        let visible = cat.resolve_columns(tbl_id, None)?;
        if visible.iter().any(|c| c.tbl_id != base_id) {
            return Err(crate::error::Error::not_implemented("compute() on a view with columns of its own"));
        }
        let selected: Vec<_> = match outputs {
            None => visible.clone(),
            Some(names) => {
                if names.is_empty() {
                    crate::bail!(MissingRequired, "At least one output column must be specified");
                }
                names
                    .iter()
                    .map(|n| {
                        let folded = fold_identifier(n);
                        visible
                            .iter()
                            .find(|c| c.name == folded)
                            .cloned()
                            .ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column: {n}"))
                    })
                    .collect::<Result<_>>()?
            }
        };
        let outputs: Vec<(String, ColId, ColumnType)> =
            selected.into_iter().map(|c| (c.name, c.md.id, c.md.col_type)).collect();
        let mut plan = ComputeRowsPlan { insert, predicates, outputs, needed: HashSet::new() };
        let mut todo: Vec<ColId> = plan.outputs.iter().map(|(_, id, _)| *id).collect();
        for p in &plan.predicates {
            todo.extend(p.column_refs().iter().map(|r| r.col_id));
        }
        while let Some(id) = todo.pop() {
            if !plan.needed.insert(id) {
                continue;
            }
            if let Some(e) = plan.insert.columns.iter().find(|c| c.col_id == id).and_then(|c| c.value_expr.as_ref()) {
                todo.extend(e.column_refs().iter().map(|r| r.col_id));
            }
        }
        Ok(plan)
    }

    /// Names of the required (non-nullable, non-computed) input columns that the outputs and filters read.
    pub fn required_input_names(&self) -> Vec<String> {
        self.insert
            .columns
            .iter()
            .filter(|c| c.is_required() && self.needed.contains(&c.col_id))
            .map(|c| c.name.clone())
            .collect()
    }

    pub fn run(&self, inputs: Vec<InputRow>, fail_on_exception: bool) -> Result<Vec<ComputedRow>> {
        let computed: Vec<ComputedCol> = self
            .insert
            .columns
            .iter()
            .filter(|c| self.needed.contains(&c.col_id))
            .filter_map(|c| {
                c.value_expr.as_ref().map(|e| ComputedCol {
                    col_id: c.col_id,
                    name: c.name.clone(),
                    expr: e.clone(),
                    is_media: c.col_type.is_media(),
                })
            })
            .collect();
        let mut stats = RowCountStats::default();
        let mut cols_with_excs = Vec::new();
        let mut out = Vec::new();
        'rows: for input in inputs {
            let mut row = StoredRow::new(0, Vec::new());
            for (col_id, v) in input {
                if self.needed.contains(&col_id) {
                    row.set_cell(col_id, v);
                }
            }
            prepare_media_inputs(&self.insert, &mut row, fail_on_exception, &mut stats, &mut cols_with_excs, None)?;
            // every needed computed column is materialized into the row, so none is evaluated from its expression
            eval_computed(&mut row, &computed, None, fail_on_exception, &mut stats, &mut cols_with_excs)?;
            for p in &self.predicates {
                match eval(p, &RowCells::new(&row)) {
                    Ok(Value::Bool(true)) => {}
                    Ok(_) => continue 'rows,
                    Err(e) if fail_on_exception => return Err(super::compute::eval_error(p, e)),
                    Err(_) => continue 'rows,
                }
            }
            let values = self.outputs.iter().map(|(_, id, _)| row.cell(*id).clone()).collect();
            let errors =
                self.outputs.iter().filter_map(|(name, id, _)| row.error(*id).map(|e| (name.clone(), e.clone()))).collect();
            out.push(ComputedRow { values, errors });
        }
        Ok(out)
    }
}
