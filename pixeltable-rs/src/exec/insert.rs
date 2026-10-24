//! Inserting rows.
//!
//! An insert runs in three steps (see `catalog::runtime` for why): `InsertPlan::new()` captures the target schema
//! under the lock; the bindings convert the Python rows into values without the lock (`convert` in
//! `bindings::table`); `commit()` re-checks the schema version, evaluates computed columns, enforces the primary
//! key, appends the rows at a new table version and persists them.

use std::collections::HashSet;
use uuid::Uuid;

use crate::catalog::Catalog;
use crate::catalog::table::{ColId, RowCountStats};
use crate::error::{Error, Result, py_repr};
use crate::expr::ExprRef;
use crate::expr::eval::{RowContext, eval};
use crate::store::rows::{CellError, LIVE, StoredRow};
use crate::types::ColumnType;
use crate::value::Value;

#[derive(Debug, Clone)]
pub struct InsertColumn {
    pub col_id: ColId,
    pub name: String,
    pub col_type: ColumnType,
    pub value_expr: Option<ExprRef>,
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
    pub fn new(cat: &Catalog, tbl_id: Uuid) -> Result<InsertPlan> {
        let md = cat.table(tbl_id)?;
        if md.is_view() {
            crate::bail!(
                UnsupportedOperation,
                "{} {}: Cannot insert into a {}.",
                md.kind.display_name(),
                py_repr(&cat.table_path(tbl_id)?.to_string()),
                md.kind.display_name()
            );
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
                    is_pk: c.md.is_pk,
                })
            })
            .collect::<Result<_>>()?;
        Ok(InsertPlan { tbl_id, tbl_name: md.name.clone(), schema_version: md.current_schema_version, columns })
    }

    pub fn column(&self, name: &str) -> Option<&InsertColumn> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// A converted input row: a value for every column of the plan that was given (others are NULL or computed).
pub type InputRow = Vec<(ColId, Value)>;

struct NewRowCtx<'a> {
    row: &'a StoredRow,
}

impl RowContext for NewRowCtx<'_> {
    fn column_value(&self, col: &crate::expr::ColumnRef) -> Result<Value> {
        Ok(self.row.cell(col.col_id).clone())
    }
}

fn pk_key(row: &StoredRow, pk_cols: &[ColId]) -> String {
    pk_cols
        .iter()
        .map(|c| match row.cell(*c) {
            Value::String(s) => s.clone(),
            v => format!("{v:?}"),
        })
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

fn pk_display(row: &StoredRow, plan: &InsertPlan) -> String {
    let pk: Vec<&InsertColumn> = plan.columns.iter().filter(|c| c.is_pk).collect();
    let names: Vec<&str> = pk.iter().map(|c| c.name.as_str()).collect();
    let vals: Vec<String> = pk
        .iter()
        .map(|c| match row.cell(c.col_id) {
            Value::String(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            v => crate::expr::display::literal_str(v, &c.col_type),
        })
        .collect();
    format!("Key ({})=({}) already exists.", names.join(", "), vals.join(", "))
}

/// Appends the rows at a new table version and persists them.
pub fn commit(cat: &mut Catalog, plan: &InsertPlan, rows: Vec<InputRow>, fail_on_exception: bool) -> Result<UpdateStatus> {
    let md = cat.table(plan.tbl_id)?.clone();
    if md.current_schema_version != plan.schema_version {
        crate::bail!(
            ConcurrentModification,
            "The schema of table {} changed while the insert was being prepared; retry the insert",
            py_repr(&plan.tbl_name)
        );
    }
    let data = cat.data(plan.tbl_id)?;
    let mut new_data = data.as_ref().clone();
    let pk_cols: Vec<ColId> = plan.columns.iter().filter(|c| c.is_pk).map(|c| c.col_id).collect();
    let mut pk_seen: HashSet<String> = HashSet::new();
    if !pk_cols.is_empty() {
        for r in new_data.visible_at(md.current_version) {
            pk_seen.insert(pk_key(r, &pk_cols));
        }
    }

    let version = if md.is_data_versioned { md.current_version + 1 } else { md.current_version };
    let mut next_rowid = md.next_rowid;
    let mut stats = RowCountStats::default();
    let mut cols_with_excs: Vec<String> = Vec::new();
    let mut new_rows = Vec::with_capacity(rows.len());
    for input in rows {
        let mut row = StoredRow { rowid: next_rowid, v_min: version, v_max: LIVE, cells: Vec::new(), errors: Vec::new() };
        next_rowid += 1;
        for (col_id, v) in input {
            row.set_cell(col_id, v);
        }
        for c in plan.columns.iter().filter(|c| c.is_computed()) {
            let Some(e) = &c.value_expr else { continue };
            stats.computed_values += 1;
            match eval(e, &NewRowCtx { row: &row }) {
                Ok(v) => row.set_cell(c.col_id, v),
                Err(err) => {
                    if fail_on_exception {
                        return Err(err);
                    }
                    stats.num_excs += 1;
                    if !cols_with_excs.contains(&c.name) {
                        cols_with_excs.push(c.name.clone());
                    }
                    row.set_cell(c.col_id, Value::Null);
                    let errortype = match &err {
                        Error::Pxt { code, .. } => code.name().to_string(),
                        _ => "Error".to_string(),
                    };
                    row.set_error(c.col_id, Some(CellError { errortype, errormsg: err.message() }));
                }
            }
        }
        if !pk_cols.is_empty() && !pk_seen.insert(pk_key(&row, &pk_cols)) {
            crate::bail!(ConstraintViolation, "Duplicate primary key value: {}", pk_display(&row, plan));
        }
        stats.ins_rows += 1;
        new_rows.push(row);
    }

    new_data.rows.extend(new_rows);
    if md.is_data_versioned {
        cat.bump_version(plan.tbl_id, None)?;
    }
    {
        let md = cat.table_mut(plan.tbl_id)?;
        md.next_rowid = next_rowid;
        if let Some(vmd) = md.versions.iter_mut().find(|v| v.version == version) {
            vmd.row_count_stats.ins_rows += stats.ins_rows;
            vmd.row_count_stats.num_excs += stats.num_excs;
            vmd.row_count_stats.computed_values += stats.computed_values;
        }
    }
    cat.write_data(plan.tbl_id, new_data)?;
    Ok(UpdateStatus {
        updated_cols: Vec::new(),
        cols_with_excs,
        row_count_stats: stats,
        cascade_row_count_stats: RowCountStats::default(),
    })
}
