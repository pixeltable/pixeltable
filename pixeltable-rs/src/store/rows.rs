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
    /// view rows: positions produced by iterators (component views), one per iterator level; the row's key is
    /// `(rowid, pos)`, and a view row joins the base row with the key `(rowid, pos[..n])` (n = the base's levels)
    #[serde(default)]
    pub pos: Vec<i64>,
}

impl StoredRow {
    pub fn new(rowid: i64, pos: Vec<i64>) -> StoredRow {
        StoredRow { rowid, v_min: 0, v_max: LIVE, cells: Vec::new(), errors: Vec::new(), pos }
    }

    pub fn key(&self) -> (i64, &[i64]) {
        (self.rowid, &self.pos)
    }

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

/// The key of a row: its rowid and iterator positions (see `StoredRow::pos`).
pub type RowKey = (i64, Vec<i64>);

/// All stored rows of a table (every version), in rowid order of insertion.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TableData {
    pub rows: Vec<StoredRow>,
}

impl TableData {
    pub fn visible_at(&self, version: i64) -> impl Iterator<Item = &StoredRow> {
        self.rows.iter().filter(move |r| r.is_visible_at(version))
    }

    /// The live rows (visible at the current version) by key.
    pub fn live_by_key(&self) -> std::collections::HashMap<(i64, &[i64]), &StoredRow> {
        self.rows.iter().filter(|r| r.v_max == LIVE).map(|r| (r.key(), r)).collect()
    }

    /// Retires the live rows with the given keys at `version` (a data-versioned table keeps them for older
    /// versions; an operational table removes them). Returns the number of rows retired.
    pub fn retire(&mut self, keys: &std::collections::HashSet<RowKey>, version: i64, versioned: bool) -> usize {
        if keys.is_empty() {
            return 0;
        }
        let matches = |r: &StoredRow| r.v_max == LIVE && keys.contains(&(r.rowid, r.pos.clone()));
        if versioned {
            let mut n = 0;
            for r in self.rows.iter_mut().filter(|r| matches(r)) {
                r.v_max = version;
                n += 1;
            }
            n
        } else {
            let before = self.rows.len();
            self.rows.retain(|r| !matches(r));
            before - self.rows.len()
        }
    }

    /// Retires the live rows whose key extends one of `prefixes` (a base row and the view rows that belong to it).
    pub fn retire_by_prefix(&mut self, prefixes: &std::collections::HashSet<RowKey>, prefix_len: usize, version: i64) -> Vec<RowKey> {
        let mut out = Vec::new();
        for r in self.rows.iter_mut().filter(|r| r.v_max == LIVE) {
            let n = prefix_len.min(r.pos.len());
            if prefixes.contains(&(r.rowid, r.pos[..n].to_vec())) {
                r.v_max = version;
                out.push((r.rowid, r.pos.clone()));
            }
        }
        out
    }

    /// Adds rows that become visible at `version`.
    pub fn append(&mut self, rows: Vec<StoredRow>, version: i64) {
        for mut r in rows {
            r.v_min = version;
            r.v_max = LIVE;
            self.rows.push(r);
        }
    }

    /// Undoes the changes made at `version`: removes the rows added at it and makes the rows retired at it live
    /// again.
    pub fn revert(&mut self, version: i64) {
        self.rows.retain(|r| r.v_min != version);
        for r in self.rows.iter_mut().filter(|r| r.v_max == version) {
            r.v_max = LIVE;
        }
    }

    /// Sets the cell of column `col_id` (and its error) in the live rows with the given keys (a column added to
    /// the current schema version: older versions don't see it).
    pub fn set_live_cells(&mut self, col_id: ColId, values: Vec<((i64, Vec<i64>), Value, Option<CellError>)>) {
        let mut by_key: std::collections::HashMap<(i64, Vec<i64>), (Value, Option<CellError>)> =
            values.into_iter().map(|(k, v, e)| (k, (v, e))).collect();
        for r in self.rows.iter_mut().filter(|r| r.v_max == LIVE) {
            if let Some((v, e)) = by_key.remove(&(r.rowid, r.pos.clone())) {
                r.set_cell(col_id, v);
                r.set_error(col_id, e);
            }
        }
    }
}
