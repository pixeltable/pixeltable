//! Stored rows.
//!
//! A row is visible at table version `v` iff `v_min <= v < v_max`. An insert at version `v` adds rows with
//! `v_min = v, v_max = LIVE`; a delete at `v` sets `v_max = v`; an update is a delete plus an insert of a new row
//! with the same key. Cells are indexed by column id, so a row written before a column was added simply has no
//! cell for it (read as NULL), and dropped columns keep their cells for older versions. The version range of a row
//! is held by the table's `TableData` (see `store::table_data`), not by the row.

use serde::{Deserialize, Serialize};

use crate::catalog::table::ColId;
use crate::value::Value;

pub use super::table_data::TableData;

/// `v_max` of a row that is visible at the current version.
pub const LIVE: i64 = i64::MAX;
/// `v_max` of a row that is visible at no version (an insert undone by `revert()`, a row deleted from an
/// operational table); compaction drops such rows.
pub const DEAD: i64 = i64::MIN;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellError {
    pub errortype: String,
    pub errormsg: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRow {
    pub rowid: i64,
    /// cells by column id (`cells[col_id]`); missing trailing cells are NULL
    pub cells: Vec<Value>,
    /// per-cell errors of computed/media columns, by column id; empty if the row has none
    #[serde(default)]
    pub errors: Vec<Option<CellError>>,
    /// view rows: positions produced by iterators (component views), one per iterator level; the row's key is
    /// `(rowid, pos)`, and a view row joins the base row with the key `(rowid, pos[..n])` (n = the base's levels)
    #[serde(default)]
    pub pos: Vec<i64>,
}

impl StoredRow {
    pub fn new(rowid: i64, pos: Vec<i64>) -> StoredRow {
        StoredRow { rowid, cells: Vec::new(), errors: Vec::new(), pos }
    }

    pub fn key(&self) -> (i64, &[i64]) {
        (self.rowid, &self.pos)
    }

    pub fn cell(&self, col_id: ColId) -> &Value {
        static NULL: Value = Value::Null;
        self.cells.get(col_id as usize).unwrap_or(&NULL)
    }

    pub fn set_cell(&mut self, col_id: ColId, v: Value) {
        let idx = col_id as usize;
        if self.cells.len() <= idx {
            self.cells.resize(idx + 1, Value::Null);
        }
        self.cells[idx] = v;
    }

    pub fn error(&self, col_id: ColId) -> Option<&CellError> {
        self.errors.get(col_id as usize).and_then(Option::as_ref)
    }

    pub fn set_error(&mut self, col_id: ColId, e: Option<CellError>) {
        let idx = col_id as usize;
        if e.is_none() && self.errors.len() <= idx {
            return;
        }
        if self.errors.len() <= idx {
            self.errors.resize(idx + 1, None);
        }
        self.errors[idx] = e;
    }
}

/// The key of a row: its rowid and iterator positions (see `StoredRow::pos`).
pub type RowKey = (i64, Vec<i64>);
