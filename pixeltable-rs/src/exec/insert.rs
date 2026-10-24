//! Inserting rows.
//!
//! An insert runs in three steps (see `catalog::runtime` for why): `InsertPlan::new()` captures the target schema
//! under the lock; the bindings convert the Python rows into values and `build_rows()` evaluates computed columns,
//! both without the lock; `commit()` re-checks the schema version, enforces the primary key, appends the rows at a
//! new table version and persists them.

use std::collections::HashSet;
use uuid::Uuid;

use super::compute::{ComputedCol, eval_computed};
use crate::catalog::Catalog;
use crate::catalog::table::{ColId, RowCountStats};
use crate::error::{Result, py_repr};
use crate::expr::ExprRef;
use crate::store::rows::{LIVE, StoredRow};
use crate::types::ColumnType;
use crate::value::Value;

#[derive(Debug, Clone)]
pub struct InsertColumn {
    pub col_id: ColId,
    pub name: String,
    pub col_type: ColumnType,
    pub value_expr: Option<ExprRef>,
    pub stored: bool,
    pub is_pk: bool,
}

impl InsertColumn {
    pub fn is_computed(&self) -> bool {
        self.value_expr.is_some()
    }
    /// Required on insert: non-nullable and not computed.
    pub fn is_required(&self) -> bool {
        !self.col_type.nullable && !self.is_computed()
    }
}

#[derive(Debug, Clone)]
pub struct InsertPlan {
    pub tbl_id: Uuid,
    pub tbl_name: String,
    pub schema_version: i64,
    pub columns: Vec<InsertColumn>,
}

/// Statistics of a completed operation (becomes a Python `UpdateStatus`).
#[derive(Debug, Clone, Default)]
pub struct UpdateStatus {
    pub updated_cols: Vec<String>,
    pub cols_with_excs: Vec<String>,
    pub row_count_stats: RowCountStats,
    pub cascade_row_count_stats: RowCountStats,
}

impl InsertPlan {
    pub fn new(cat: &Catalog, tbl_id: Uuid, version: Option<i64>) -> Result<InsertPlan> {
        let md = cat.table(tbl_id)?;
        if md.is_view() || version.is_some() {
            let kind = if version.is_some() { "snapshot" } else { md.kind.display_name() };
            crate::bail!(UnsupportedOperation, "{}: Cannot insert into a {kind}.", cat.display_str(tbl_id)?);
        }
        let columns = md
            .columns_at(None)?
            .into_iter()
            .map(|c| {
                Ok(InsertColumn {
                    col_id: c.md.id,
                    name: c.name,
                    col_type: c.md.col_type.clone(),
                    value_expr: c.md.value_expr.as_ref().map(crate::expr::serde::from_json).transpose()?,
                    stored: c.md.stored,
                    is_pk: c.md.is_pk,
                })
            })
            .collect::<Result<_>>()?;
        Ok(InsertPlan { tbl_id, tbl_name: md.name.clone(), schema_version: md.current_schema_version, columns })
    }

    pub fn column(&self, name: &str) -> Option<&InsertColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The stored computed columns, in schema order (a computed column only references columns before it).
    pub fn computed_columns(&self) -> Vec<ComputedCol> {
        self.columns
            .iter()
            .filter(|c| c.stored)
            .filter_map(|c| c.value_expr.as_ref().map(|e| ComputedCol { col_id: c.col_id, name: c.name.clone(), expr: e.clone() }))
            .collect()
    }
}

/// A converted input row: a value for every column of the plan that was given (others are NULL or computed).
pub type InputRow = Vec<(ColId, Value)>;

/// Rows ready to be stored (rowids and versions are assigned at commit).
pub struct BuiltRows {
    pub rows: Vec<StoredRow>,
    pub stats: RowCountStats,
    pub cols_with_excs: Vec<String>,
}

/// Evaluates computed columns for the input rows (no catalog lock held).
pub fn build_rows(plan: &InsertPlan, inputs: Vec<InputRow>, fail_on_exception: bool) -> Result<BuiltRows> {
    let computed = plan.computed_columns();
    let mut stats = RowCountStats::default();
    let mut cols_with_excs = Vec::new();
    let mut rows = Vec::with_capacity(inputs.len());
    let has_windows = computed.iter().any(|c| super::query::has_window_calls(&c.expr));
    for input in inputs {
        let mut row = StoredRow { rowid: 0, v_min: 0, v_max: LIVE, cells: Vec::new(), errors: Vec::new() };
        for (col_id, v) in input {
            row.set_cell(col_id, v);
        }
        if !has_windows {
            eval_computed(&mut row, &computed, fail_on_exception, &mut stats, &mut cols_with_excs)?;
        }
        stats.ins_rows += 1;
        rows.push(row);
    }
    if has_windows {
        // column by column: a column with window functions is evaluated over the whole batch in window order
        for col in &computed {
            if super::query::has_window_calls(&col.expr) {
                let refs: Vec<&StoredRow> = rows.iter().collect();
                let results = super::query::eval_rows_with_windows(&refs, &col.expr)?;
                for (row, res) in rows.iter_mut().zip(results) {
                    stats.computed_values += 1;
                    match res {
                        Ok(v) => row.set_cell(col.col_id, v),
                        Err(err) => {
                            if fail_on_exception {
                                return Err(super::compute::eval_error(&col.expr, err));
                            }
                            stats.num_excs += 1;
                            if !cols_with_excs.contains(&col.name) {
                                cols_with_excs.push(col.name.clone());
                            }
                            row.set_cell(col.col_id, Value::Null);
                            row.set_error(col.col_id, Some(super::compute::cell_error(&err)));
                        }
                    }
                }
            } else {
                for row in rows.iter_mut() {
                    eval_computed(row, std::slice::from_ref(col), fail_on_exception, &mut stats, &mut cols_with_excs)?;
                }
            }
        }
    }
    Ok(BuiltRows { rows, stats, cols_with_excs })
}

fn pk_key(row: &StoredRow, pk_cols: &[ColId]) -> String {
    pk_cols.iter().map(|c| format!("{:?}", row.cell(*c))).collect::<Vec<_>>().join("\u{1f}")
}

fn pk_display(row: &StoredRow, plan: &InsertPlan) -> String {
    let pk: Vec<&InsertColumn> = plan.columns.iter().filter(|c| c.is_pk).collect();
    let names: Vec<&str> = pk.iter().map(|c| c.name.as_str()).collect();
    let vals: Vec<String> = pk
        .iter()
        .map(|c| match row.cell(c.col_id) {
            Value::String(s) => s.clone(),
            v => crate::expr::display::literal_str(v, &c.col_type),
        })
        .collect();
    format!("Key ({})=({}) already exists.", names.join(", "), vals.join(", "))
}

/// Appends the rows at a new table version and persists them.
pub fn commit(cat: &mut Catalog, plan: &InsertPlan, built: BuiltRows) -> Result<UpdateStatus> {
    let md = cat.table(plan.tbl_id)?.clone();
    if md.current_schema_version != plan.schema_version {
        crate::bail!(
            ConcurrentModification,
            "The schema of table {} changed while the insert was being prepared; retry the insert",
            py_repr(&plan.tbl_name)
        );
    }
    let mut data = cat.data(plan.tbl_id)?.as_ref().clone();
    let pk_cols: Vec<ColId> = plan.columns.iter().filter(|c| c.is_pk).map(|c| c.col_id).collect();
    if !pk_cols.is_empty() {
        let mut seen: HashSet<String> = data.visible_at(md.current_version).map(|r| pk_key(r, &pk_cols)).collect();
        for r in &built.rows {
            if !seen.insert(pk_key(r, &pk_cols)) {
                crate::bail!(ConstraintViolation, "Duplicate primary key value: {}", pk_display(r, plan));
            }
        }
    }
    let version = if md.is_data_versioned { md.current_version + 1 } else { md.current_version };
    let mut next_rowid = md.next_rowid;
    for mut r in built.rows {
        r.rowid = next_rowid;
        r.v_min = version;
        next_rowid += 1;
        data.rows.push(r);
    }
    if md.is_data_versioned {
        cat.bump_version(plan.tbl_id, None)?;
    }
    {
        let md = cat.table_mut(plan.tbl_id)?;
        md.next_rowid = next_rowid;
        if let Some(vmd) = md.versions.iter_mut().find(|v| v.version == version) {
            vmd.row_count_stats.ins_rows += built.stats.ins_rows;
            vmd.row_count_stats.num_excs += built.stats.num_excs;
            vmd.row_count_stats.computed_values += built.stats.computed_values;
        }
    }
    cat.write_data(plan.tbl_id, data)?;
    Ok(UpdateStatus {
        updated_cols: Vec::new(),
        cols_with_excs: built.cols_with_excs,
        row_count_stats: built.stats,
        cascade_row_count_stats: RowCountStats::default(),
    })
}
