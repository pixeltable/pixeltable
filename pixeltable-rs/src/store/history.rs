//! A table's version history: its `VersionMd` and `SchemaVersionMd` records, appended to the table's history file
//! (one JSON object per line) by each commit that adds or changes some.
//!
//! The history of a table only changes at its end (a new version, the status of the current version, `revert()`
//! dropping the current version), so a commit appends the records from the first one that differs from what is
//! committed (`changed_records()`); replay (`replay()`) lets a record replace every record of the same or a later
//! version (a version record also replaces the schema versions it rules out: those after its own schema version and
//! not after it), and drops the versions after the table's current version (what `revert()` removed).

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::MAX_CATALOG_DEPTH;
use crate::catalog::table::{SchemaVersionMd, VersionMd};
use crate::error::{Error, Result};
use crate::value::parse_json_bounded;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Record {
    Version(VersionMd),
    Schema(SchemaVersionMd),
}

/// The history of a table: data versions and schema versions, each sorted by version.
#[derive(Debug, Clone, Default)]
pub struct History {
    pub versions: Vec<Arc<VersionMd>>,
    pub schema_versions: Vec<Arc<SchemaVersionMd>>,
}

/// The number of leading entries that `a` and `b` share (by identity: entries are shared between the committed and
/// the current metadata until changed).
fn shared_prefix<T>(a: &[Arc<T>], b: &[Arc<T>]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| Arc::ptr_eq(x, y)).count()
}

/// The records to append so that the committed history (`old_versions`, `old_schema_versions`) replays as the new
/// one (pass empty slices for all records of the new one).
pub fn changed_records(
    old_versions: &[Arc<VersionMd>],
    old_schema_versions: &[Arc<SchemaVersionMd>],
    versions: &[Arc<VersionMd>],
    schema_versions: &[Arc<SchemaVersionMd>],
) -> Vec<Record> {
    let mut out = Vec::new();
    let s = shared_prefix(old_schema_versions, schema_versions);
    out.extend(schema_versions[s..].iter().map(|x| Record::Schema(x.as_ref().clone())));
    let v = shared_prefix(old_versions, versions);
    out.extend(versions[v..].iter().map(|x| Record::Version(x.as_ref().clone())));
    out
}

pub fn serialize(records: &[Record]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for r in records {
        serde_json::to_writer(&mut out, r)?;
        out.push(b'\n');
    }
    if crate::value::json_nesting_depth(&out) > MAX_CATALOG_DEPTH {
        return Err(Error::internal("cannot save the version history: its metadata is nested too deeply"));
    }
    Ok(out)
}

pub fn parse(bytes: &[u8]) -> Result<Vec<Record>> {
    bytes
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| parse_json_bounded(l, MAX_CATALOG_DEPTH).map_err(|e| Error::internal(format!("corrupt version history: {e}"))))
        .collect()
}

/// Applies `records` (in order) to `h`, then drops the versions after `current_version`.
pub fn replay(h: &mut History, records: Vec<Record>, current_version: i64) {
    for r in records {
        match r {
            Record::Version(v) => {
                let k = h.versions.partition_point(|x| x.version < v.version);
                h.versions.truncate(k);
                // a schema version is numbered by the version that created it: one after this version's schema version
                // and not after this version belongs to a reverted version that this one replaces
                h.schema_versions.retain(|x| x.schema_version <= v.schema_version || x.schema_version > v.version);
                h.versions.push(Arc::new(v));
            }
            Record::Schema(s) => {
                let k = h.schema_versions.partition_point(|x| x.schema_version < s.schema_version);
                h.schema_versions.truncate(k);
                h.schema_versions.push(Arc::new(s));
            }
        }
    }
    let k = h.versions.partition_point(|x| x.version <= current_version);
    h.versions.truncate(k);
    let k = h.schema_versions.partition_point(|x| x.schema_version <= current_version);
    h.schema_versions.truncate(k);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(v: i64, ins: i64) -> Arc<VersionMd> {
        let mut m = VersionMd::new(v, 0.0, 0, "data", None);
        m.row_count_stats.ins_rows = ins;
        Arc::new(m)
    }

    fn replayed(records: &[Record], current: i64) -> Vec<(i64, i64)> {
        let mut h = History::default();
        replay(&mut h, parse(&serialize(records).unwrap_or_default()).unwrap_or_default(), current);
        h.versions.iter().map(|v| (v.version, v.row_count_stats.ins_rows)).collect()
    }

    fn changed(old: &History, new: &History) -> Vec<Record> {
        changed_records(&old.versions, &old.schema_versions, &new.versions, &new.schema_versions)
    }

    #[test]
    fn appends_replay_to_the_current_history() {
        let committed = History { versions: vec![version(0, 0), version(1, 5)], schema_versions: Vec::new() };
        let mut log = changed(&History::default(), &committed);
        // a new version whose status changes in the same operation, and a change of the previous one
        let mut current = committed.clone();
        current.versions[1] = version(1, 6);
        current.versions.push(version(2, 1));
        log.extend(changed(&committed, &current));
        assert_eq!(replayed(&log, 2), vec![(0, 0), (1, 6), (2, 1)]);
        // revert() of version 2 appends nothing: the current version drops it
        assert_eq!(replayed(&log, 1), vec![(0, 0), (1, 6)]);
        // a new version 2 after the revert replaces the old one
        let mut after = current.clone();
        after.versions.pop();
        after.versions.push(version(2, 9));
        log.extend(changed(&current, &after));
        assert_eq!(replayed(&log, 2), vec![(0, 0), (1, 6), (2, 9)]);
    }

    fn schema(s: i64) -> Arc<SchemaVersionMd> {
        Arc::new(SchemaVersionMd {
            schema_version: s,
            preceding_schema_version: None,
            columns: Vec::new(),
            comment: None,
            custom_metadata: serde_json::Value::Null,
            media_validation: crate::catalog::table::MediaValidation::OnWrite,
        })
    }

    #[test]
    fn data_version_replaces_reverted_schema_version() {
        // a schema change at version 2 is reverted, and a data change takes version 2 again (with schema version 1)
        let v1 = Arc::new(VersionMd::new(1, 0.0, 1, "schema", None));
        let committed = History { versions: vec![version(0, 0), v1], schema_versions: vec![schema(0), schema(1)] };
        let mut log = changed(&History::default(), &committed);
        let mut altered = committed.clone();
        altered.schema_versions.push(schema(2));
        let mut v2 = VersionMd::new(2, 0.0, 2, "schema", None);
        v2.row_count_stats.ins_rows = 0;
        altered.versions.push(Arc::new(v2));
        log.extend(changed(&committed, &altered));
        let mut after = committed.clone();
        let mut v2 = VersionMd::new(2, 0.0, 1, "data", None);
        v2.row_count_stats.ins_rows = 3;
        after.versions.push(Arc::new(v2));
        log.extend(changed(&altered, &after));
        let mut h = History::default();
        replay(&mut h, parse(&serialize(&log).unwrap_or_default()).unwrap_or_default(), 2);
        assert_eq!(h.schema_versions.iter().map(|s| s.schema_version).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(h.versions.iter().map(|v| (v.version, v.schema_version)).collect::<Vec<_>>(), vec![(0, 0), (1, 1), (2, 1)]);
    }
}
