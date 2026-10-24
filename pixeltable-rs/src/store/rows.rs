//! Row storage with version ranges.
//!
//! A row is visible at table version `v` iff `v_min <= v < v_max`. An insert at version `v` adds rows with
//! `v_min = v, v_max = LIVE`; a delete at `v` sets `v_max = v`; an update is a delete plus an insert of a new row
//! with the same rowid. Cells are indexed by column id, so a row written before a column was added simply has no
//! cell for it (read as NULL), and dropped columns keep their cells for older versions.

use serde::{Deserialize, Serialize};

use crate::catalog::table::ColId;
use crate::value::Value;

pub const LIVE: i64 = i64::MAX;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellError {
    pub errortype: String,
    pub errormsg: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRow {
    pub rowid: i64,
    pub v_min: i64,
    pub v_max: i64,
    /// cells by column id (`cells[col_id]`); missing trailing cells are NULL
    pub cells: Vec<Value>,
    /// per-cell errors of computed/media columns, by column id; empty if the row has none
    #[serde(default)]
    pub errors: Vec<Option<CellError>>,
}

impl StoredRow {
    pub fn is_visible_at(&self, version: i64) -> bool {
        self.v_min <= version && version < self.v_max
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

/// All stored rows of a table (every version), in rowid order of insertion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TableData {
    pub rows: Vec<StoredRow>,
}

impl TableData {
    pub fn visible_at(&self, version: i64) -> impl Iterator<Item = &StoredRow> {
        self.rows.iter().filter(move |r| r.is_visible_at(version))
    }
}
