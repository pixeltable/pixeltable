//! The rows of one table, as a list of immutable segments plus the current version range of every row.
//!
//! A segment (`Segment`) is a run of rows sorted by key (then `v_min`), written once as one file. Each commit that
//! adds rows adds a segment; the `v_max` changes a commit makes to rows of earlier segments (deletes, the old
//! version of updated rows, `revert()`) are recorded as `Patch`es carried by the segment it writes (a segment
//! without rows if it adds none). `TableData` holds the segments (shared, via `Arc`, between the catalog, its
//! checkpoint and query snapshots) and the current `v_max` of every row (copy-on-write per segment), so cloning it
//! costs O(segments) and a mutation costs O(rows touched) plus a copy of the `v_max` array of each segment it
//! touches.
//!
//! Writing (`Catalog::flush()`): `prepare_write()` merges segments where the compaction policy says so and attaches
//! the pending patches to the last segment; the store writes every segment that is not in a file yet; `mark_written()`
//! records that. Reading (`from_segments()`) applies the patches of every segment in order.
//!
//! Lookups by key binary-search each segment (a key has at most one visible row per version); the primary-key
//! index (`PkIndex`) maps primary-key values to rowids.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::rows::{CellError, DEAD, LIVE, RowKey, StoredRow};
use crate::catalog::table::ColId;
use crate::value::Value;

/// A version at which exactly the live rows (`v_max == LIVE`) are visible.
const NOW: i64 = LIVE - 1;

/// Segments with at least this many rows (or patches) are not merged with others; they are only rewritten on their
/// own (see `TableData::plan_merge()`), which bounds the cost of a single commit.
const MAX_MERGE_WEIGHT: usize = 1 << 16;
/// A segment of at least this many rows is rewritten when half of it is dead or retired, or a quarter of it is
/// patched by later segments.
const MIN_REWRITE_ROWS: usize = 1024;

/// A change of `v_max` of row `idx` of segment `seg`, carried by a later segment's file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    pub seg: u64,
    pub idx: u32,
    pub vmax: i64,
}

/// An immutable run of rows, stored as one file.
#[derive(Debug, Clone)]
pub struct Segment {
    /// unique within the table (names the file)
    pub id: u64,
    /// sorted by key, then `v_min`
    pub rows: Vec<StoredRow>,
    pub vmin: Vec<i64>,
    /// `v_max` changes of rows of earlier segments, carried by this segment's file
    pub patches: Vec<Patch>,
}

/// A new segment id (random, so that files of a failed operation never collide with committed ones).
pub fn new_segment_id() -> u64 {
    loop {
        let id = uuid::Uuid::new_v4().as_u64_pair().0;
        if id != 0 {
            return id;
        }
    }
}

/// A segment of a `TableData`, with the current `v_max` of its rows.
#[derive(Clone)]
struct Seg {
    seg: Arc<Segment>,
    vmax: Arc<Vec<i64>>,
    /// true once the segment is in a file referenced by the committed catalog (or being written by this commit)
    written: bool,
    /// rows with `v_max == LIVE`
    live: usize,
    /// rows with `v_max == DEAD`
    dead: usize,
    /// an upper bound of the `v_max` of the rows that are neither live nor dead
    hist_max: i64,
    min_vmin: i64,
    max_vmin: i64,
}

impl Seg {
    fn new(seg: Arc<Segment>, vmax: Arc<Vec<i64>>, written: bool) -> Seg {
        let mut s = Seg { seg, vmax, written, live: 0, dead: 0, hist_max: DEAD, min_vmin: LIVE, max_vmin: DEAD };
        s.recount();
        s
    }

    fn recount(&mut self) {
        let (mut live, mut dead, mut hist_max) = (0, 0, DEAD);
        for &v in self.vmax.iter() {
            match v {
                LIVE => live += 1,
                DEAD => dead += 1,
                v => hist_max = hist_max.max(v),
            }
        }
        self.live = live;
        self.dead = dead;
        self.hist_max = hist_max;
        self.min_vmin = self.seg.vmin.iter().copied().min().unwrap_or(LIVE);
        self.max_vmin = self.seg.vmin.iter().copied().max().unwrap_or(DEAD);
    }

    fn len(&self) -> usize {
        self.seg.rows.len()
    }

    /// True if no row of the segment can be visible at `version`.
    fn skip_at(&self, version: i64) -> bool {
        self.seg.rows.is_empty() || self.min_vmin > version || (self.live == 0 && self.hist_max <= version)
    }

    fn visible(&self, i: usize, version: i64) -> bool {
        self.seg.vmin[i] <= version && version < self.vmax[i]
    }

    /// The index range of the rows with key `key`.
    fn key_range(&self, key: (i64, &[i64])) -> std::ops::Range<usize> {
        let rows = &self.seg.rows;
        match (rows.first(), rows.last()) {
            (Some(f), Some(l)) if f.key() <= key && key <= l.key() => {
                let start = rows.partition_point(|r| r.key() < key);
                let end = start + rows[start..].iter().take_while(|r| r.key() == key).count();
                start..end
            }
            _ => 0..0,
        }
    }

    /// The index range of the rows whose key extends `(rowid, prefix)`.
    fn prefix_range(&self, rowid: i64, prefix: &[i64]) -> std::ops::Range<usize> {
        let rows = &self.seg.rows;
        let start = rows.partition_point(|r| r.key() < (rowid, prefix));
        let end = start + rows[start..].iter().take_while(|r| r.rowid == rowid && r.pos.starts_with(prefix)).count();
        start..end
    }

    /// The weight of the segment for the merge policy.
    fn weight(&self) -> usize {
        self.len() + self.seg.patches.len()
    }
}

/// The rows of a table (every version).
#[derive(Clone, Default)]
pub struct TableData {
    segs: Vec<Seg>,
    /// `v_max` changes of rows of written segments made since the data was last written
    patches: Vec<Patch>,
    /// the primary-key index, shared by all clones (see `PkIndex`)
    pk: Arc<Mutex<Option<PkIndex>>>,
}

impl std::fmt::Debug for TableData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TableData({} segments, {} rows)", self.segs.len(), self.num_rows())
    }
}

/// Where a row is: (segment position, row index).
type Addr = (usize, usize);

impl TableData {
    // ------------------------------------------------------------------------------------------- loading, writing

    /// The data stored in `segments` (in manifest order), each with the `v_max` its file records; applies the
    /// patches the segments carry, in order.
    pub fn from_segments(segments: Vec<(Segment, Vec<i64>)>) -> TableData {
        let mut vmaxes: Vec<Vec<i64>> = Vec::with_capacity(segments.len());
        let mut segs: Vec<Arc<Segment>> = Vec::with_capacity(segments.len());
        for (s, v) in segments {
            segs.push(Arc::new(s));
            vmaxes.push(v);
        }
        let pos: HashMap<u64, usize> = segs.iter().enumerate().map(|(i, s)| (s.id, i)).collect();
        for s in &segs {
            for p in &s.patches {
                if let Some(&t) = pos.get(&p.seg) {
                    if let Some(v) = vmaxes[t].get_mut(p.idx as usize) {
                        *v = p.vmax;
                    }
                }
            }
        }
        let segs = segs.into_iter().zip(vmaxes).map(|(s, v)| Seg::new(s, Arc::new(v), true)).collect();
        TableData { segs, patches: Vec::new(), pk: Default::default() }
    }

    /// This data followed by the segments `more` (that another process wrote after it), with their patches applied.
    pub fn extended(&self, more: Vec<(Segment, Vec<i64>)>) -> TableData {
        let mut out = TableData { segs: self.segs.clone(), patches: Vec::new(), pk: self.pk.clone() };
        let first_new = out.segs.len();
        for (s, v) in more {
            out.segs.push(Seg::new(Arc::new(s), Arc::new(v), true));
        }
        let pos: HashMap<u64, usize> = out.segs.iter().enumerate().map(|(i, s)| (s.seg.id, i)).collect();
        for k in first_new..out.segs.len() {
            let patches = out.segs[k].seg.patches.clone();
            for p in patches {
                if let Some(&t) = pos.get(&p.seg) {
                    if (p.idx as usize) < out.segs[t].len() {
                        out.set_vmax((t, p.idx as usize), p.vmax);
                    }
                }
            }
        }
        out.patches.clear();
        out
    }

    /// The ids of the segments, in order (the table's manifest).
    pub fn segment_ids(&self) -> Vec<u64> {
        self.segs.iter().map(|s| s.seg.id).collect()
    }

    /// True if some change is not written yet.
    pub fn has_unwritten(&self) -> bool {
        !self.patches.is_empty() || self.segs.iter().any(|s| !s.written)
    }

    /// Prepares writing: merges segments as the compaction policy says and attaches the pending patches to the last
    /// segment. Returns the segments to write, each with the `v_max` to record (the current one); call
    /// `mark_written()` once they are written.
    pub fn prepare_write(&mut self) -> Vec<(Arc<Segment>, Arc<Vec<i64>>)> {
        while let Some(range) = self.plan_merge() {
            self.merge(range);
        }
        let pending = std::mem::take(&mut self.patches);
        if !pending.is_empty() {
            let pos: HashMap<u64, usize> =
                self.segs.iter().enumerate().filter(|(_, s)| s.written).map(|(i, s)| (s.seg.id, i)).collect();
            let mut targets: Vec<(u64, u32)> = Vec::new();
            let mut seen = HashSet::new();
            for p in pending {
                if pos.contains_key(&p.seg) && seen.insert((p.seg, p.idx)) {
                    targets.push((p.seg, p.idx));
                }
            }
            let mut patches: Vec<Patch> = targets
                .into_iter()
                .map(|(seg, idx)| Patch { seg, idx, vmax: self.segs[pos[&seg]].vmax[idx as usize] })
                .collect();
            if !patches.is_empty() {
                match self.segs.last_mut() {
                    Some(last) if !last.written => {
                        let s = Arc::make_mut(&mut last.seg);
                        let carried: HashSet<(u64, u32)> = patches.iter().map(|p| (p.seg, p.idx)).collect();
                        s.patches.retain(|p| !carried.contains(&(p.seg, p.idx)));
                        s.patches.append(&mut patches);
                    }
                    _ => {
                        let seg = Segment { id: new_segment_id(), rows: Vec::new(), vmin: Vec::new(), patches };
                        self.segs.push(Seg::new(Arc::new(seg), Arc::new(Vec::new()), false));
                    }
                }
            }
        }
        self.segs.iter().filter(|s| !s.written).map(|s| (s.seg.clone(), s.vmax.clone())).collect()
    }

    pub fn mark_written(&mut self) {
        for s in self.segs.iter_mut() {
            s.written = true;
        }
    }

    /// The next merge the compaction policy asks for, as a range of segment positions. Segments of at least
    /// `MAX_MERGE_WEIGHT` rows and patches are not merged with others; among the segments after the last of them, the
    /// policy keeps every segment heavier than all later ones together (so there are O(log rows) of them, and a row
    /// is rewritten O(log rows) times), merging the trailing segments from the first one that is not. Besides, a
    /// large segment that is mostly dead or retired rows, or mostly patched by later segments, is rewritten alone.
    fn plan_merge(&self) -> Option<std::ops::RangeInclusive<usize>> {
        let n = self.segs.len();
        if n == 0 {
            return None;
        }
        let start = self.segs.iter().rposition(|s| s.weight() >= MAX_MERGE_WEIGHT).map_or(0, |p| p + 1);
        let mut later = 0;
        let mut first_violation = None;
        for j in (start..n).rev() {
            let w = self.segs[j].weight();
            if j + 1 < n && w <= later {
                first_violation = Some(j);
            }
            later += w;
        }
        if let Some(i) = first_violation {
            return Some(i..=n - 1);
        }
        let mut patched: HashMap<u64, usize> = HashMap::new();
        for s in &self.segs {
            for p in &s.seg.patches {
                *patched.entry(p.seg).or_default() += 1;
            }
        }
        for (j, s) in self.segs.iter().enumerate() {
            let rows = s.len();
            if rows < MIN_REWRITE_ROWS {
                continue;
            }
            let retired = rows - s.live;
            let mostly_dead = s.dead * 2 > rows;
            let mostly_history = s.live > 0 && retired * 2 > rows;
            let patched = patched.get(&s.seg.id).copied().unwrap_or(0) * 4 > rows;
            if mostly_dead || mostly_history || patched {
                return Some(j..=j);
            }
        }
        None
    }

    /// Replaces the segments in `range` by new ones holding their rows that are not dead, with their current
    /// `v_max`: one segment, or (when the retired rows are many) one of retired rows followed by one of live rows, so
    /// that reads of the current version skip the former. The new segments carry the patches of the old ones that
    /// target earlier segments, with the targets' current `v_max`.
    fn merge(&mut self, range: std::ops::RangeInclusive<usize>) {
        let (start, end) = (*range.start(), *range.end());
        let members: Vec<Seg> = self.segs[range.clone()].to_vec();
        let earlier: HashMap<u64, usize> = self.segs[..start].iter().enumerate().map(|(i, s)| (s.seg.id, i)).collect();
        let mut carried: Vec<Patch> = Vec::new();
        let mut seen = HashSet::new();
        for s in &members {
            for p in &s.seg.patches {
                if let Some(&t) = earlier.get(&p.seg) {
                    if seen.insert((p.seg, p.idx)) {
                        if let Some(&vmax) = self.segs[t].vmax.get(p.idx as usize) {
                            carried.push(Patch { seg: p.seg, idx: p.idx, vmax });
                        }
                    }
                }
            }
        }
        // (key, vmin, vmax, row) of the rows that stay
        let mut rows: Vec<(i64, i64, StoredRow)> = Vec::with_capacity(members.iter().map(|s| s.len() - s.dead).sum());
        for s in &members {
            for (i, r) in s.seg.rows.iter().enumerate() {
                if s.vmax[i] != DEAD {
                    rows.push((s.seg.vmin[i], s.vmax[i], r.clone()));
                }
            }
        }
        rows.sort_by(|a, b| a.2.key().cmp(&b.2.key()).then(a.0.cmp(&b.0)));
        // (a rewrite for `mostly_history` in `plan_merge()` always splits, so its output does not qualify again)
        let num_retired = rows.iter().filter(|r| r.1 != LIVE).count();
        let split = num_retired * 2 >= MIN_REWRITE_ROWS && num_retired < rows.len();
        let mut parts: Vec<Vec<(i64, i64, StoredRow)>> = Vec::new();
        if split {
            let (retired, live): (Vec<_>, Vec<_>) = rows.into_iter().partition(|r| r.1 != LIVE);
            parts.push(retired);
            parts.push(live);
        } else {
            parts.push(rows);
        }
        let mut new_segs = Vec::with_capacity(parts.len());
        let num_parts = parts.len();
        for (k, part) in parts.into_iter().enumerate() {
            let mut vmin = Vec::with_capacity(part.len());
            let mut vmax = Vec::with_capacity(part.len());
            let mut out = Vec::with_capacity(part.len());
            for (a, b, r) in part {
                vmin.push(a);
                vmax.push(b);
                out.push(r);
            }
            let patches = if k + 1 == num_parts { std::mem::take(&mut carried) } else { Vec::new() };
            if out.is_empty() && patches.is_empty() {
                continue;
            }
            let seg = Segment { id: new_segment_id(), rows: out, vmin, patches };
            new_segs.push(Seg::new(Arc::new(seg), Arc::new(vmax), false));
        }
        self.segs.splice(start..=end, new_segs);
    }

    // ------------------------------------------------------------------------------------------- reading

    /// The number of stored rows (all versions).
    pub fn num_rows(&self) -> usize {
        self.segs.iter().map(Seg::len).sum()
    }

    /// The number of live rows (visible at the current version).
    pub fn num_live(&self) -> usize {
        self.segs.iter().map(|s| s.live).sum()
    }

    /// The rows visible at `version`, in storage order (each segment in key order).
    pub fn visible_at(&self, version: i64) -> impl Iterator<Item = &StoredRow> {
        self.segs
            .iter()
            .filter(move |s| !s.skip_at(version))
            .flat_map(move |s| s.seg.rows.iter().enumerate().filter(move |(i, _)| s.visible(*i, version)).map(|(_, r)| r))
    }

    /// The number of rows visible at `version`.
    pub fn count_at(&self, version: i64) -> usize {
        self.segs
            .iter()
            .filter(|s| !s.skip_at(version))
            .map(|s| if version >= s.max_vmin && s.dead == 0 && s.live == s.len() { s.len() } else { (0..s.len()).filter(|i| s.visible(*i, version)).count() })
            .sum()
    }

    /// The live rows (visible at the current version), in storage order.
    pub fn live_rows(&self) -> impl Iterator<Item = &StoredRow> {
        self.visible_at(NOW)
    }

    /// The live rows by key.
    pub fn live_by_key(&self) -> HashMap<(i64, &[i64]), &StoredRow> {
        self.live_rows().map(|r| (r.key(), r)).collect()
    }

    fn find(&self, key: (i64, &[i64]), version: i64) -> Option<Addr> {
        for (p, s) in self.segs.iter().enumerate().rev() {
            if s.skip_at(version) {
                continue;
            }
            for i in s.key_range(key) {
                if s.visible(i, version) {
                    return Some((p, i));
                }
            }
        }
        None
    }

    fn row(&self, (p, i): Addr) -> &StoredRow {
        &self.segs[p].seg.rows[i]
    }

    /// The row with key `key` visible at `version`, if any.
    pub fn row_at(&self, key: (i64, &[i64]), version: i64) -> Option<&StoredRow> {
        self.find(key, version).map(|a| self.row(a))
    }

    /// The live row with key `key`, if any.
    pub fn live_row(&self, key: (i64, &[i64])) -> Option<&StoredRow> {
        self.row_at(key, NOW)
    }

    /// True if looking up `n` keys one by one costs more than a scan of the live rows.
    fn prefer_scan(&self, n: usize) -> bool {
        n.saturating_mul(8) > self.num_live().max(64)
    }

    /// Addresses of the live rows whose key extends one of `prefixes` (`(rowid, pos[..n])` for keys of rows of a
    /// base with n iterator levels).
    fn live_with_prefixes(&self, prefixes: &HashSet<RowKey>, prefix_len: usize) -> Vec<Addr> {
        let mut out = Vec::new();
        if self.prefer_scan(prefixes.len()) {
            for (p, s) in self.segs.iter().enumerate() {
                if s.live == 0 {
                    continue;
                }
                for (i, r) in s.seg.rows.iter().enumerate() {
                    if s.vmax[i] == LIVE {
                        let n = prefix_len.min(r.pos.len());
                        if prefixes.contains(&(r.rowid, r.pos[..n].to_vec())) {
                            out.push((p, i));
                        }
                    }
                }
            }
        } else {
            for (rowid, prefix) in prefixes {
                for (p, s) in self.segs.iter().enumerate() {
                    if s.live == 0 {
                        continue;
                    }
                    for i in s.prefix_range(*rowid, prefix) {
                        if s.vmax[i] == LIVE {
                            out.push((p, i));
                        }
                    }
                }
            }
        }
        out
    }

    /// The live rows whose key extends one of `prefixes` (see `retire_by_prefix()`), grouped by prefix, each group
    /// in key order.
    pub fn live_rows_by_prefix(&self, prefixes: &HashSet<RowKey>, prefix_len: usize) -> HashMap<RowKey, Vec<&StoredRow>> {
        let mut out: HashMap<RowKey, Vec<&StoredRow>> = HashMap::new();
        for a in self.live_with_prefixes(prefixes, prefix_len) {
            let r = self.row(a);
            let n = prefix_len.min(r.pos.len());
            out.entry((r.rowid, r.pos[..n].to_vec())).or_default().push(r);
        }
        for rows in out.values_mut() {
            rows.sort_by(|a, b| a.key().cmp(&b.key()));
        }
        out
    }

    // ------------------------------------------------------------------------------------------- mutations

    fn set_vmax(&mut self, (p, i): Addr, vmax: i64) {
        let s = &mut self.segs[p];
        let old = s.vmax[i];
        if old == vmax {
            return;
        }
        Arc::make_mut(&mut s.vmax)[i] = vmax;
        match old {
            LIVE => s.live -= 1,
            DEAD => s.dead -= 1,
            _ => {}
        }
        match vmax {
            LIVE => s.live += 1,
            DEAD => s.dead += 1,
            v => s.hist_max = s.hist_max.max(v),
        }
        if s.written {
            let seg = s.seg.id;
            self.patches.push(Patch { seg, idx: i as u32, vmax });
        }
    }

    /// Retires the live rows with the given keys at `version` (a data-versioned table keeps them for older
    /// versions; an operational table removes them). Returns the number of rows retired.
    pub fn retire(&mut self, keys: &HashSet<RowKey>, version: i64, versioned: bool) -> usize {
        if keys.is_empty() {
            return 0;
        }
        let vmax = if versioned { version } else { DEAD };
        let addrs: Vec<Addr> = if self.prefer_scan(keys.len()) {
            let mut out = Vec::new();
            for (p, s) in self.segs.iter().enumerate() {
                if s.live == 0 {
                    continue;
                }
                for (i, r) in s.seg.rows.iter().enumerate() {
                    if s.vmax[i] == LIVE && keys.contains(&(r.rowid, r.pos.clone())) {
                        out.push((p, i));
                    }
                }
            }
            out
        } else {
            keys.iter().filter_map(|(rowid, pos)| self.find((*rowid, pos), NOW)).collect()
        };
        for a in &addrs {
            self.set_vmax(*a, vmax);
        }
        addrs.len()
    }

    /// Retires the live rows whose key extends one of `prefixes` (a base row and the view rows that belong to it);
    /// returns their keys.
    pub fn retire_by_prefix(&mut self, prefixes: &HashSet<RowKey>, prefix_len: usize, version: i64) -> Vec<RowKey> {
        if prefixes.is_empty() {
            return Vec::new();
        }
        let addrs = self.live_with_prefixes(prefixes, prefix_len);
        let keys = addrs.iter().map(|a| {
            let r = self.row(*a);
            (r.rowid, r.pos.clone())
        });
        let keys: Vec<RowKey> = keys.collect();
        for a in addrs {
            self.set_vmax(a, version);
        }
        keys
    }

    /// Adds rows that become visible at `version`.
    pub fn append(&mut self, mut rows: Vec<StoredRow>, version: i64) {
        if rows.is_empty() {
            return;
        }
        rows.sort_by(|a, b| a.key().cmp(&b.key()));
        let n = rows.len();
        let seg = Segment { id: new_segment_id(), rows, vmin: vec![version; n], patches: Vec::new() };
        self.segs.push(Seg::new(Arc::new(seg), Arc::new(vec![LIVE; n]), false));
    }

    /// Undoes the changes made at `version`: removes the rows added at it and makes the rows retired at it live
    /// again.
    pub fn revert(&mut self, version: i64) {
        let mut changes: Vec<(Addr, i64)> = Vec::new();
        for (p, s) in self.segs.iter().enumerate() {
            if s.max_vmin < version && s.hist_max < version {
                continue;
            }
            for i in 0..s.len() {
                if s.seg.vmin[i] == version {
                    if s.vmax[i] != DEAD {
                        changes.push(((p, i), DEAD));
                    }
                } else if s.vmax[i] == version {
                    changes.push(((p, i), LIVE));
                }
            }
        }
        for (a, v) in changes {
            self.set_vmax(a, v);
        }
    }

    /// A segment that this data may change in place: a written segment is replaced by an unwritten copy (with a new
    /// id) that carries its patches.
    fn rewrite_seg(&mut self, p: usize) -> &mut Segment {
        let s = &mut self.segs[p];
        if s.written || Arc::strong_count(&s.seg) > 1 {
            let mut copy = s.seg.as_ref().clone();
            copy.id = new_segment_id();
            s.seg = Arc::new(copy);
            s.written = false;
        }
        match Arc::get_mut(&mut s.seg) {
            Some(seg) => seg,
            None => unreachable!("a segment copy is not shared"),
        }
    }

    /// Sets the cell of column `col_id` (and its error) in the live rows with the given keys (a column added to
    /// the current schema version: older versions don't see it).
    pub fn set_live_cells(&mut self, col_id: ColId, values: Vec<((i64, Vec<i64>), Value, Option<CellError>)>) {
        if values.is_empty() {
            return;
        }
        let mut by_seg: HashMap<usize, Vec<(usize, Value, Option<CellError>)>> = HashMap::new();
        if self.prefer_scan(values.len()) {
            let mut by_key: HashMap<RowKey, (Value, Option<CellError>)> = values.into_iter().map(|(k, v, e)| (k, (v, e))).collect();
            for (p, s) in self.segs.iter().enumerate() {
                if s.live == 0 {
                    continue;
                }
                for (i, r) in s.seg.rows.iter().enumerate() {
                    if s.vmax[i] == LIVE {
                        if let Some((v, e)) = by_key.remove(&(r.rowid, r.pos.clone())) {
                            by_seg.entry(p).or_default().push((i, v, e));
                        }
                    }
                }
            }
        } else {
            for ((rowid, pos), v, e) in values {
                if let Some((p, i)) = self.find((rowid, &pos), NOW) {
                    by_seg.entry(p).or_default().push((i, v, e));
                }
            }
        }
        for (p, changes) in by_seg {
            let seg = self.rewrite_seg(p);
            for (i, v, e) in changes {
                let r = &mut seg.rows[i];
                r.set_cell(col_id, v);
                r.set_error(col_id, e);
            }
        }
    }

    /// Applies `f` to every stored row (all versions).
    pub fn update_all_rows(&mut self, mut f: impl FnMut(&mut StoredRow)) {
        for p in 0..self.segs.len() {
            if self.segs[p].len() == 0 {
                continue;
            }
            for r in self.rewrite_seg(p).rows.iter_mut() {
                f(r);
            }
        }
    }

    // ------------------------------------------------------------------------------------------- primary key

    /// The live row whose primary key (columns `pk_cols`) is `values`: compared after truncating strings as the
    /// primary-key index of a data-versioned table does (`truncate`), or exactly (`exact`).
    pub fn find_live_by_pk(&self, pk_cols: &[ColId], truncate: bool, values: &[Value], exact: bool) -> Option<&StoredRow> {
        self.find_live_by_pks(pk_cols, truncate, std::slice::from_ref(&values), exact).pop().flatten()
    }

    /// `find_live_by_pk()` for several primary-key values.
    pub fn find_live_by_pks<V: AsRef<[Value]>>(&self, pk_cols: &[ColId], truncate: bool, keys: &[V], exact: bool) -> Vec<Option<&StoredRow>> {
        let candidates = self.pk_candidates_many(pk_cols, truncate, keys);
        keys.iter()
            .zip(candidates)
            .map(|(values, rowids)| {
                let values = values.as_ref();
                rowids.into_iter().filter_map(|rowid| self.live_row((rowid, &[]))).find(|r| {
                    pk_cols.iter().zip(values).all(|(c, v)| {
                        if exact {
                            values_identical(r.cell(*c), v)
                        } else {
                            values_identical(&pk_part(r.cell(*c), truncate), &pk_part(v, truncate))
                        }
                    })
                })
            })
            .collect()
    }

    /// The rowids of the rows (of any version) whose primary key may be `values` (a superset).
    pub fn pk_candidates(&self, pk_cols: &[ColId], truncate: bool, values: &[Value]) -> Vec<i64> {
        self.pk_candidates_many(pk_cols, truncate, std::slice::from_ref(&values)).pop().unwrap_or_default()
    }

    fn pk_candidates_many<V: AsRef<[Value]>>(&self, pk_cols: &[ColId], truncate: bool, keys: &[V]) -> Vec<Vec<i64>> {
        let mut guard = self.pk.lock();
        let idx = match guard.as_mut() {
            Some(idx) if idx.cols == pk_cols && idx.truncate == truncate => idx,
            _ => guard.insert(PkIndex { cols: pk_cols.to_vec(), truncate, ..Default::default() }),
        };
        for s in &self.segs {
            if !idx.covered.contains(&s.seg.id) {
                for r in &s.seg.rows {
                    let h = pk_hash(idx.cols.iter().map(|c| r.cell(*c)), truncate);
                    idx.add(h, r.rowid);
                }
                idx.covered.insert(s.seg.id);
            }
        }
        keys.iter().map(|values| idx.get(pk_hash(values.as_ref().iter(), truncate))).collect()
    }
}

/// Maps the primary-key values of the rows of a table (of every version) to their rowids, by a hash of the values
/// (truncated as the index of a data-versioned table truncates strings). It only grows: the rows of a segment are
/// added when a lookup first sees the segment, and entries of rows that are gone are harmless (lookups check the
/// rows they find), so all clones of a `TableData` share one index.
#[derive(Default)]
struct PkIndex {
    cols: Vec<ColId>,
    truncate: bool,
    /// ids of the segments whose rows are in the index
    covered: HashSet<u64>,
    first: HashMap<u64, i64>,
    more: HashMap<u64, Vec<i64>>,
}

impl PkIndex {
    fn add(&mut self, h: u64, rowid: i64) {
        match self.first.get(&h) {
            None => {
                self.first.insert(h, rowid);
            }
            Some(&r) if r == rowid => {}
            Some(_) => {
                let more = self.more.entry(h).or_default();
                if !more.contains(&rowid) {
                    more.push(rowid);
                }
            }
        }
    }

    fn get(&self, h: u64) -> Vec<i64> {
        let mut out: Vec<i64> = self.first.get(&h).copied().into_iter().collect();
        if let Some(more) = self.more.get(&h) {
            out.extend(more.iter().copied());
        }
        out
    }
}

/// A primary-key value as the index holds it.
pub fn pk_part(v: &Value, truncate: bool) -> Value {
    if truncate { crate::functions::btree::truncate(v) } else { v.clone() }
}

/// The hash of a primary-key value (`values`, one per key column) by which the index finds it.
pub fn pk_hash<'a>(values: impl Iterator<Item = &'a Value>, truncate: bool) -> u64 {
    let mut h = std::hash::DefaultHasher::new();
    for v in values {
        hash_value(&pk_part(v, truncate), &mut h);
    }
    h.finish()
}

fn hash_value(v: &Value, h: &mut impl Hasher) {
    std::mem::discriminant(v).hash(h);
    match v {
        Value::Null | Value::PyObj(_) => {}
        Value::Bool(b) => b.hash(h),
        Value::Int(i) | Value::Timestamp(i) => i.hash(h),
        Value::Float(f) => (if f.is_nan() { f64::NAN.to_bits() } else { f.to_bits() }).hash(h),
        Value::String(s) | Value::Media(s) => s.hash(h),
        Value::Date(d) => d.hash(h),
        Value::Uuid(u) => u.hash(h),
        Value::Binary(b) => b.hash(h),
        Value::Json(j) => j.to_string().hash(h),
        Value::Array(a) => format!("{a:?}").hash(h),
    }
}

/// True if `a` and `b` are the same value (same kind and payload; floats by their bits, any NaN equal to any other):
/// the equality of primary-key values.
pub fn values_identical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) | (Value::Timestamp(x), Value::Timestamp(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
        (Value::String(x), Value::String(y)) | (Value::Media(x), Value::Media(y)) => x == y,
        (Value::Date(x), Value::Date(y)) => x == y,
        (Value::Uuid(x), Value::Uuid(y)) => x == y,
        (Value::Binary(x), Value::Binary(y)) => x == y,
        (Value::Json(x), Value::Json(y)) => x.to_string() == y.to_string(),
        (Value::Array(x), Value::Array(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(rowid: i64, v: i64) -> StoredRow {
        let mut r = StoredRow::new(rowid, Vec::new());
        r.set_cell(0, Value::Int(v));
        r
    }

    fn keys(rowids: &[i64]) -> HashSet<RowKey> {
        rowids.iter().map(|r| (*r, Vec::new())).collect()
    }

    fn visible(d: &TableData, version: i64) -> Vec<(i64, i64)> {
        let mut out: Vec<(i64, i64)> = d
            .visible_at(version)
            .map(|r| (r.rowid, match r.cell(0) {
                Value::Int(i) => *i,
                _ => -1,
            }))
            .collect();
        out.sort();
        out
    }

    /// Writes and reloads `d` the way the catalog does (without files): the segments' rows with their current
    /// `v_max`, then `from_segments()`.
    fn roundtrip(d: &mut TableData) -> TableData {
        d.prepare_write();
        d.mark_written();
        let segs = d.segs.iter().map(|s| (s.seg.as_ref().clone(), file_vmax(d, s))).collect();
        TableData::from_segments(segs)
    }

    /// The v_max a segment's file records: its current v_max with the patches later segments carry undone (the
    /// file keeps what was current when it was written; here, the patched rows get a placeholder that the patches
    /// must overwrite).
    fn file_vmax(d: &TableData, s: &Seg) -> Vec<i64> {
        let mut v = s.vmax.as_ref().clone();
        for t in &d.segs {
            for p in &t.seg.patches {
                if p.seg == s.seg.id {
                    v[p.idx as usize] = -12345;
                }
            }
        }
        v
    }

    #[test]
    fn versions_lookups_and_reload() {
        let mut d = TableData::default();
        d.append((0..10).map(|i| row(i, i)).collect(), 1);
        d.prepare_write();
        d.mark_written();
        // update rows 2 and 3 at version 2
        let n = d.retire(&keys(&[2, 3]), 2, true);
        assert_eq!(n, 2);
        d.append(vec![row(2, 20), row(3, 30)], 2);
        // delete row 5 at version 3
        d.retire(&keys(&[5]), 3, true);
        assert_eq!(visible(&d, 1).len(), 10);
        assert_eq!(visible(&d, 2).iter().filter(|r| r.0 == 2).collect::<Vec<_>>(), vec![&(2, 20)]);
        assert_eq!(visible(&d, 3).len(), 9);
        assert_eq!(d.count_at(3), 9);
        assert_eq!(d.count_at(1), 10);
        assert!(d.row_at((5, &[]), 2).is_some());
        assert!(d.row_at((5, &[]), 3).is_none());
        assert!(matches!(d.live_row((3, &[])).map(|r| r.cell(0)), Some(Value::Int(30))));
        let reloaded = roundtrip(&mut d);
        for v in 0..4 {
            assert_eq!(visible(&reloaded, v), visible(&d, v));
        }
        // revert version 3, then version 2
        let mut r = reloaded.clone();
        r.revert(3);
        assert_eq!(visible(&r, NOW).len(), 10);
        r.revert(2);
        assert_eq!(visible(&r, NOW), (0..10).map(|i| (i, i)).collect::<Vec<_>>());
        let again = roundtrip(&mut r);
        assert_eq!(visible(&again, NOW), visible(&r, NOW));
        // the original is unaffected by changes to its clone
        assert_eq!(visible(&reloaded, NOW).len(), 9);
    }

    #[test]
    fn merges_keep_every_version() {
        let mut d = TableData::default();
        let mut expected: Vec<Vec<(i64, i64)>> = vec![Vec::new()];
        for v in 1..=200i64 {
            // insert 3 rows, update one older row, delete another
            let base = (v - 1) * 3;
            d.append((base..base + 3).map(|i| row(i, i)).collect(), v);
            if v > 2 {
                let k = (v * 7) % base;
                if d.live_row((k, &[])).is_some() {
                    d.retire(&keys(&[k]), v, true);
                    d.append(vec![row(k, k + 1000 * v)], v);
                }
                let del = (v * 13) % base;
                d.retire(&keys(&[del]), v, true);
            }
            d.prepare_write();
            d.mark_written();
            expected.push(visible(&d, v));
            assert!(d.segs.len() <= 40, "{} segments after {v} commits", d.segs.len());
        }
        let reloaded = roundtrip(&mut d);
        for (v, rows) in expected.iter().enumerate() {
            assert_eq!(&visible(&d, v as i64), rows, "version {v}");
            assert_eq!(&visible(&reloaded, v as i64), rows, "version {v} after reload");
            assert_eq!(reloaded.count_at(v as i64), rows.len());
        }
    }

    #[test]
    fn operational_deletes_and_rewrites() {
        let mut d = TableData::default();
        d.append((0..2000).map(|i| row(i, i)).collect(), 0);
        d.prepare_write();
        d.mark_written();
        let half: Vec<i64> = (0..1500).collect();
        assert_eq!(d.retire(&keys(&half), 0, false), 1500);
        assert_eq!(d.count_at(0), 500);
        let reloaded = roundtrip(&mut d);
        // the mostly dead segment was rewritten without its dead rows
        assert_eq!(d.num_rows(), 500);
        assert_eq!(visible(&reloaded, 0), (1500..2000).map(|i| (i, i)).collect::<Vec<_>>());
        let mut d = reloaded;
        d.set_live_cells(0, vec![((1600, Vec::new()), Value::Int(-1), None)]);
        d.update_all_rows(|r| r.set_cell(1, Value::Int(7)));
        let r = roundtrip(&mut d);
        assert!(matches!(r.live_row((1600, &[])).map(|r| r.cell(0)), Some(Value::Int(-1))));
        assert!(r.live_rows().all(|r| matches!(r.cell(1), Value::Int(7))));
    }

    #[test]
    fn primary_key_index() {
        let mut d = TableData::default();
        d.append((0..100).map(|i| row(i, i * 10)).collect(), 1);
        let pk = [0 as ColId];
        assert_eq!(d.find_live_by_pk(&pk, true, &[Value::Int(420)], true).map(|r| r.rowid), Some(42));
        assert!(d.find_live_by_pk(&pk, true, &[Value::Int(421)], true).is_none());
        d.retire(&keys(&[42]), 2, true);
        assert!(d.find_live_by_pk(&pk, true, &[Value::Int(420)], true).is_none());
        d.append(vec![row(100, 420)], 3);
        assert_eq!(d.find_live_by_pk(&pk, true, &[Value::Int(420)], true).map(|r| r.rowid), Some(100));
        let mut r = d.clone();
        r.revert(3);
        r.revert(2);
        assert_eq!(r.find_live_by_pk(&pk, true, &[Value::Int(420)], true).map(|r| r.rowid), Some(42));
    }

    #[test]
    fn view_rows_by_prefix() {
        let mut d = TableData::default();
        let mut rows = Vec::new();
        for rowid in 0..50 {
            for p in 0..3 {
                rows.push(StoredRow::new(rowid, vec![p]));
            }
        }
        d.append(rows, 1);
        let prefixes: HashSet<RowKey> = [(7, Vec::new()), (8, Vec::new())].into_iter().collect();
        let groups = d.live_rows_by_prefix(&prefixes, 0);
        assert_eq!(groups[&(7, Vec::new())].len(), 3);
        let retired = d.retire_by_prefix(&prefixes, 0, 2);
        assert_eq!(retired.len(), 6);
        assert_eq!(d.count_at(2), 144);
        assert_eq!(d.count_at(1), 150);
    }
}
