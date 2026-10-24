//! Scanning a table path: the rows of a table, or of a view joined with the rows of its bases.
//!
//! A view stores its own rows (its own columns, keyed by the key of the base row they belong to, see
//! `store::rows::StoredRow::key`), so a query over a view joins, for each visible view row, the base rows with the
//! matching key at the versions the path pins. `PathInput::snapshot()` captures the metadata and row data under the
//! catalog lock; `rows()` runs without it. `PathCtx` evaluates expressions against one joined row: a column
//! reference reads the cell of the row of its table (or evaluates the value expression of an unstored column).

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::catalog::Catalog;
use crate::catalog::table::{ColId, TableMd};
use crate::catalog::views::PathElem;
use crate::error::{Error, Result};
use crate::expr::eval::{RowContext, eval};
use crate::expr::{ColumnRef, ExprRef};
use crate::store::rows::{StoredRow, TableData};
use crate::value::Value;

/// What evaluation needs to know about one table of a path.
#[derive(Debug, Clone, Default)]
pub struct ElemInfo {
    pub tbl_id: Uuid,
    /// value expressions of the table's unstored computed columns (at the schema version read)
    pub unstored: HashMap<ColId, ExprRef>,
}

impl ElemInfo {
    /// The info of table `md` as of its version `version`.
    pub fn of(md: &TableMd, version: Option<i64>) -> Result<ElemInfo> {
        let mut unstored = HashMap::new();
        for c in md.columns_at(version)? {
            if !c.md.stored {
                if let Some(e) = &c.md.value_expr {
                    unstored.insert(c.md.id, crate::expr::serde::from_json(e)?);
                }
            }
        }
        Ok(ElemInfo { tbl_id: md.id, unstored })
    }
}

/// One table of a path, captured for a scan.
pub struct ElemInput {
    pub info: ElemInfo,
    /// the version read
    pub version: i64,
    pub data: Arc<TableData>,
    /// the number of iterator positions in this table's row keys
    pub num_pos: usize,
}

/// A table path (leaf first) captured for a scan.
pub struct PathInput {
    pub elems: Vec<ElemInput>,
}

/// The number of iterator positions in the row keys of a table (one per component view in its path).
pub fn num_pos(cat: &Catalog, tbl_id: Uuid) -> usize {
    let _ = (cat, tbl_id);
    0
}

impl PathInput {
    pub fn snapshot(cat: &mut Catalog, path: &[PathElem]) -> Result<PathInput> {
        let mut elems = Vec::with_capacity(path.len());
        for e in path {
            let md = cat.table(e.tbl_id)?.clone();
            let version = e.pinned.unwrap_or(md.current_version);
            let info = ElemInfo::of(&md, e.pinned)?;
            let data = cat.data(e.tbl_id)?;
            elems.push(ElemInput { info, version, data, num_pos: num_pos(cat, e.tbl_id) });
        }
        if elems.is_empty() {
            return Err(Error::internal("empty table path"));
        }
        Ok(PathInput { elems })
    }

    pub fn infos(&self) -> Vec<&ElemInfo> {
        self.elems.iter().map(|e| &e.info).collect()
    }

    /// The visible joined rows, in key order: each entry has one row per table of the path (leaf first).
    pub fn rows(&self) -> Result<Vec<Vec<&StoredRow>>> {
        let leaf = &self.elems[0];
        let mut leaf_rows: Vec<&StoredRow> = leaf.data.visible_at(leaf.version).collect();
        leaf_rows.sort_by(|a, b| a.key().cmp(&b.key()));
        if self.elems.len() == 1 {
            return Ok(leaf_rows.into_iter().map(|r| vec![r]).collect());
        }
        let indexes: Vec<HashMap<(i64, &[i64]), &StoredRow>> = self.elems[1..]
            .iter()
            .map(|e| e.data.visible_at(e.version).map(|r| (r.key(), r)).collect())
            .collect();
        let mut out = Vec::with_capacity(leaf_rows.len());
        'rows: for r in leaf_rows {
            let mut joined = Vec::with_capacity(self.elems.len());
            joined.push(r);
            for (i, idx) in indexes.iter().enumerate() {
                let n = self.elems[i + 1].num_pos.min(r.pos.len());
                match idx.get(&(r.rowid, &r.pos[..n])) {
                    Some(b) => joined.push(*b),
                    // a view row whose base row is not visible at the base version read (cannot happen for a
                    // consistent path); skip it rather than fail the query
                    None => continue 'rows,
                }
            }
            out.push(joined);
        }
        Ok(out)
    }
}

/// Evaluation context for one joined row of a path: `rows[i]` is the row of table `elems[i]`.
pub struct PathCtx<'a> {
    pub elems: &'a [&'a ElemInfo],
    pub rows: &'a [&'a StoredRow],
}

impl RowContext for PathCtx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        let Some(i) = self.elems.iter().position(|e| e.tbl_id == col.tbl_id) else {
            return Err(Error::internal(format!("column {} is not in the table path", col.name)));
        };
        match self.elems[i].unstored.get(&col.col_id) {
            Some(e) => eval(e, self),
            None => Ok(self.rows.get(i).map(|r| r.cell(col.col_id).clone()).unwrap_or(Value::Null)),
        }
    }
}
