//! `Catalog::validate_store()`: consistency checks of the store and of the in-memory catalog, for tests (the hook
//! `pixeltable._testing.validate_store()`, which the compat harness calls after each test as upstream's conftest calls
//! its `Catalog.validate_store()`).

use std::collections::{HashMap, HashSet};

use super::Catalog;
use super::table::TableMd;
use crate::error::Result;
use crate::store::rows::{DEAD, LIVE, RowKey, StoredRow};
use crate::store::table_data::values_identical;
use crate::store::without_gil;
use crate::value::Value;

impl Catalog {
    /// The problems found (empty if none): files of the store against the manifests (a file missing, or one no table
    /// references), the catalog reloaded from disk against the one in memory (directories, table metadata and history,
    /// every stored row with its version range), the version ranges of each table's rows (well formed, within the
    /// table's versions, at most one row of a key visible at any version), the primary-key index (it finds every live
    /// row by its key), and the value columns of B-tree indexes (the truncated values of their columns). The caller
    /// holds the catalog lock.
    pub fn validate_store(&mut self) -> Result<Vec<String>> {
        let mut problems = Vec::new();
        self.validate_files(&mut problems)?;
        if !problems.is_empty() {
            // reading the tables would fail on the missing files
            return Ok(problems);
        }
        let mut disk = without_gil(|| Catalog::load(self.store.clone(), false))?;
        self.validate_metadata(&disk, &mut problems)?;
        let mut ids: Vec<uuid::Uuid> = self.tables.keys().copied().collect();
        ids.sort();
        for id in ids {
            let md = self.table(id)?.clone();
            let data = self.data(id)?;
            let name = md.name.clone();
            let rows: Vec<(&StoredRow, i64, i64)> = data.rows_with_versions().collect();
            validate_versions(&md, &rows, &mut problems);
            self.validate_indexes(&md, &rows, &data, &mut problems);
            if disk.tables.contains_key(&id) {
                let disk_data = disk.data(id)?;
                let disk_rows: Vec<(&StoredRow, i64, i64)> = disk_data.rows_with_versions().collect();
                if let Some(p) = compare_rows(&rows, &disk_rows) {
                    problems.push(format!("table {name:?}: the rows in memory differ from those in the store: {p}"));
                }
            }
        }
        Ok(problems)
    }

    /// Every file a manifest names exists, and every file in `tables/` belongs to a table.
    fn validate_files(&self, problems: &mut Vec<String>) -> Result<()> {
        let mut expected: HashMap<String, &str> = HashMap::new();
        for md in self.tables.values() {
            for f in super::table_files(md) {
                expected.insert(f, &md.name);
            }
        }
        let on_disk: HashSet<String> = self.store.table_file_names()?.into_iter().collect();
        let mut missing: Vec<String> = expected
            .iter()
            .filter(|(f, _)| !on_disk.contains(*f))
            .map(|(f, name)| format!("table {name:?}: file tables/{f} is missing"))
            .collect();
        let mut orphans: Vec<String> =
            on_disk.iter().filter(|f| !expected.contains_key(*f)).map(|f| format!("file tables/{f} belongs to no table")).collect();
        missing.sort();
        orphans.sort();
        problems.extend(missing);
        problems.extend(orphans);
        for f in self.store.leftover_temp_files()? {
            problems.push(format!("file {f} is left over from an unfinished commit"));
        }
        Ok(())
    }

    /// The catalog in the store equals the one in memory.
    fn validate_metadata(&self, disk: &Catalog, problems: &mut Vec<String>) -> Result<()> {
        if disk.stamp() != self.stamp() {
            problems.push(format!("the store's catalog is at {:?}, the one in memory at {:?}", disk.stamp(), self.stamp()));
        }
        let dirs = |c: &Catalog| -> Result<Vec<serde_json::Value>> {
            let mut v: Vec<serde_json::Value> = c.dirs.values().map(serde_json::to_value).collect::<std::result::Result<_, _>>()?;
            v.sort_by_key(|d| d.to_string());
            Ok(v)
        };
        if dirs(self)? != dirs(disk)? {
            problems.push("the directories in memory differ from those in the store".to_string());
        }
        for (id, md) in &self.tables {
            let Some(d) = disk.tables.get(id) else {
                problems.push(format!("table {:?} is not in the store", md.name));
                continue;
            };
            if serde_json::to_value(md.as_ref())? != serde_json::to_value(d.as_ref())? {
                problems.push(format!("table {:?}: its metadata in memory differs from that in the store", md.name));
            }
            let versions = |m: &TableMd| -> Result<(Vec<serde_json::Value>, Vec<serde_json::Value>)> {
                Ok((
                    m.versions.iter().map(|v| serde_json::to_value(v.as_ref())).collect::<std::result::Result<_, _>>()?,
                    m.schema_versions.iter().map(|v| serde_json::to_value(v.as_ref())).collect::<std::result::Result<_, _>>()?,
                ))
            };
            if versions(md)? != versions(d)? {
                problems.push(format!("table {:?}: its version history in memory differs from that in the store", md.name));
            }
        }
        for (id, d) in &disk.tables {
            if !self.tables.contains_key(id) {
                problems.push(format!("table {:?} is in the store but not in memory", d.name));
            }
        }
        Ok(())
    }

    /// The primary-key index finds every live row of a base table, and B-tree value columns hold their columns'
    /// truncated values.
    fn validate_indexes(&self, md: &TableMd, rows: &[(&StoredRow, i64, i64)], data: &crate::store::rows::TableData, problems: &mut Vec<String>) {
        let live = || rows.iter().filter(|(_, _, vmax)| *vmax == LIVE).map(|(r, _, _)| *r);
        let pk: Vec<_> = if md.is_view() { Vec::new() } else { md.columns.iter().filter(|c| c.is_pk).map(|c| c.id).collect() };
        if !pk.is_empty() {
            for r in live() {
                let values: Vec<Value> = pk.iter().map(|c| r.cell(*c).clone()).collect();
                let found = data.find_live_by_pk(&pk, md.is_data_versioned, &values, true).map(|f| f.key());
                if found != Some(r.key()) {
                    problems.push(format!("table {:?}: the primary-key index does not find row {:?} (found {found:?})", md.name, r.key()));
                    break;
                }
            }
        }
        for idx in md.indexes.list.iter().filter(|i| i.is_btree() && i.col_tbl_id == md.id) {
            let Some(val_col) = idx.val_col_id() else { continue };
            for r in live() {
                let expected = crate::functions::btree::truncate(r.cell(idx.col_id));
                if !values_identical(&expected, r.cell(val_col)) {
                    problems.push(format!(
                        "table {:?}: the value of index {:?} of row {:?} does not match its column",
                        md.name,
                        idx.name,
                        r.key()
                    ));
                    break;
                }
            }
        }
    }
}

/// The version ranges of a table's rows: `v_min <= current version`, `v_min < v_max` for a row that is not dead, a
/// retired row retired at a version that exists, and the rows of a key visible at disjoint versions.
fn validate_versions(md: &TableMd, rows: &[(&StoredRow, i64, i64)], problems: &mut Vec<String>) {
    let name = &md.name;
    let mut by_key: HashMap<RowKey, Vec<(i64, i64)>> = HashMap::new();
    for (r, vmin, vmax) in rows {
        let key = (r.rowid, r.pos.clone());
        if *vmax == DEAD {
            continue;
        }
        if *vmin < 0 || *vmin > md.current_version {
            problems.push(format!("table {name:?}: row {key:?} has v_min {vmin} outside the versions 0..={}", md.current_version));
        } else if *vmax != LIVE && (*vmax <= *vmin || *vmax > md.current_version) {
            problems.push(format!("table {name:?}: row {key:?} has the version range [{vmin}, {vmax})"));
        }
        by_key.entry(key).or_default().push((*vmin, *vmax));
    }
    let mut overlapping: Vec<&RowKey> = by_key
        .iter_mut()
        .filter_map(|(k, ranges)| {
            ranges.sort();
            ranges.windows(2).any(|w| w[0].1 > w[1].0).then_some(k)
        })
        .collect();
    overlapping.sort();
    if let Some(k) = overlapping.first() {
        problems.push(format!("table {name:?}: {} key(s) have rows visible at the same version, e.g. {k:?}", overlapping.len()));
    }
}

/// The first difference between the stored rows of two `TableData`s of a table (compared by key and `v_min`), if any.
fn compare_rows(a: &[(&StoredRow, i64, i64)], b: &[(&StoredRow, i64, i64)]) -> Option<String> {
    fn sorted<'r>(rows: &[(&'r StoredRow, i64, i64)]) -> Vec<(RowKey, i64, i64, &'r StoredRow)> {
        let mut v: Vec<(RowKey, i64, i64, &StoredRow)> = rows.iter().map(|(r, vmin, vmax)| ((r.rowid, r.pos.clone()), *vmin, *vmax, *r)).collect();
        v.sort_by(|x, y| (&x.0, x.1).cmp(&(&y.0, y.1)));
        v
    }
    let (a, b) = (sorted(a), sorted(b));
    if a.len() != b.len() {
        return Some(format!("{} rows in memory, {} in the store", a.len(), b.len()));
    }
    for (x, y) in a.iter().zip(&b) {
        if (&x.0, x.1, x.2) != (&y.0, y.1, y.2) {
            return Some(format!("row {:?} [{}, {}) in memory, {:?} [{}, {}) in the store", x.0, x.1, x.2, y.0, y.1, y.2));
        }
        let n = x.3.cells.len().max(y.3.cells.len());
        let cells_equal = (0..n).all(|c| values_identical(x.3.cell(c as _), y.3.cell(c as _)));
        let errors_equal = (0..x.3.errors.len().max(y.3.errors.len())).all(|c| x.3.error(c as _) == y.3.error(c as _));
        if !cells_equal || !errors_equal {
            return Some(format!("the cells of row {:?} [{}, {})", x.0, x.1, x.2));
        }
    }
    None
}
