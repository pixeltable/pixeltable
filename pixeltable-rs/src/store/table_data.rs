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
//! Lookups by key binary-search each segment's keys (a key has at most one visible row per version), which a segment
//! read from a file has without decoding its cells; key indexes (`KeyIndex`, and the sorted key hashes stored with
//! segments) map the values of the primary key, or of a column with a B-tree index, to rowids.

use std::collections::{HashMap, HashSet};
use std::hash::Hasher;
use std::sync::Arc;

use parking_lot::Mutex;

use super::rows::{CellError, DEAD, LIVE, RowKey, StoredRow};
pub use super::segment::{Patch, Segment, new_segment_id};
use crate::catalog::table::ColId;
use crate::value::Value;

/// A version at which exactly the live rows (`v_max == LIVE`) are visible.
const NOW: i64 = LIVE - 1;

/// The most rows a segment holds (larger appends and merges produce several), so that the copy of a segment's `v_max`
/// array that a change to it makes stays small. Segments of at least this many rows (or rows and patches) are not
/// merged with others; they are only rewritten on their own (see `TableData::plan_merge()`), which bounds the cost of
/// a single commit.
const MAX_SEGMENT_ROWS: usize = 1 << 16;
/// A segment of at least this many rows is rewritten when half of it is dead, or a quarter of it is patched by later
/// segments.
const MIN_REWRITE_ROWS: usize = 1024;

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
        self.seg.len()
    }

    /// True if no row of the segment can be visible at `version`.
    fn skip_at(&self, version: i64) -> bool {
        self.seg.is_empty() || self.min_vmin > version || (self.live == 0 && self.hist_max <= version)
    }

    fn visible(&self, i: usize, version: i64) -> bool {
        self.seg.vmin[i] <= version && version < self.vmax[i]
    }

    /// The position of the first row whose key is not less than `key`.
    fn lower_bound(&self, key: (i64, &[i64])) -> usize {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.seg.key(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The index range of the rows with key `key`.
    fn key_range(&self, key: (i64, &[i64])) -> std::ops::Range<usize> {
        let n = self.len();
        if n == 0 || key < self.seg.key(0) || self.seg.key(n - 1) < key {
            return 0..0;
        }
        let start = self.lower_bound(key);
        let end = start + (start..n).take_while(|i| self.seg.key(*i) == key).count();
        start..end
    }

    /// The index range of the rows whose key extends `(rowid, prefix)`.
    fn prefix_range(&self, rowid: i64, prefix: &[i64]) -> std::ops::Range<usize> {
        let start = self.lower_bound((rowid, prefix));
        let end = start
            + (start..self.len())
                .take_while(|i| {
                    let (r, p) = self.seg.key(*i);
                    r == rowid && p.starts_with(prefix)
                })
                .count();
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
    /// the key indexes (primary key, indexed columns), built on first use and shared by all clones (see `KeyIndex`)
    indexes: Arc<Mutex<Vec<KeyIndex>>>,
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
        TableData { segs, patches: Vec::new(), indexes: Default::default() }
    }

    /// This data followed by the segments `more` (that another process wrote after it), with their patches applied.
    pub fn extended(&self, more: Vec<(Segment, Vec<i64>)>) -> TableData {
        let mut out = TableData { segs: self.segs.clone(), patches: Vec::new(), indexes: self.indexes.clone() };
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

    /// The number of decoded chunks of rows (see `segment::CHUNK_ROWS`), and of all chunks.
    pub fn decoded_chunks(&self) -> (usize, usize) {
        let all = self.segs.iter().map(|s| s.len().div_ceil(super::segment::CHUNK_ROWS)).sum();
        (self.segs.iter().map(|s| s.seg.decoded_chunks()).sum(), all)
    }

    /// The ids of the segments, in order (the table's manifest).
    pub fn segment_ids(&self) -> Vec<u64> {
        self.segs.iter().map(|s| s.seg.id).collect()
    }

    /// True if some change is not written yet.
    pub fn has_unwritten(&self) -> bool {
        !self.patches.is_empty() || self.segs.iter().any(|s| !s.written)
    }

    /// Prepares writing: attaches the pending patches to the last segment and merges segments as the compaction policy
    /// says. Returns the segments to write, each with the `v_max` to record (the current one); call `mark_written()`
    /// once they are written.
    pub fn prepare_write(&mut self) -> Vec<(Arc<Segment>, Arc<Vec<i64>>)> {
        // the pending patches first, so that the merge policy sees them
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
                        let seg = Segment::new(new_segment_id(), Vec::new(), Vec::new(), patches);
                        self.segs.push(Seg::new(Arc::new(seg), Arc::new(Vec::new()), false));
                    }
                }
            }
        }
        while let Some(range) = self.plan_merge() {
            self.merge(range);
        }
        self.segs.iter().filter(|s| !s.written).map(|s| (s.seg.clone(), s.vmax.clone())).collect()
    }

    pub fn mark_written(&mut self) {
        for s in self.segs.iter_mut() {
            s.written = true;
        }
    }

    /// The next merge the compaction policy asks for, as a range of segment positions. Segments of at least
    /// `MAX_SEGMENT_ROWS` rows and patches ("full" ones) are not merged with others; among the segments after the last
    /// of them, the policy keeps every segment heavier than all later ones together (so there are O(log rows) of them,
    /// and a row is rewritten O(log rows) times), merging the trailing segments from the first one that is not.
    /// Besides, a large segment that is mostly dead rows, or mostly patched by later segments, is rewritten alone.
    /// (Retired rows stay: older versions read them; scans of other versions skip them by their version ranges without
    /// reading them.) Every merge removes a segment, adds a full one, or removes the condition that called for it, so
    /// `prepare_write()` terminates.
    fn plan_merge(&self) -> Option<std::ops::RangeInclusive<usize>> {
        let n = self.segs.len();
        if n == 0 {
            return None;
        }
        let start = self.segs.iter().rposition(|s| s.weight() >= MAX_SEGMENT_ROWS).map_or(0, |p| p + 1);
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
            let mostly_dead = s.dead * 2 > rows;
            let patched = patched.get(&s.seg.id).copied().unwrap_or(0) * 4 > rows;
            if mostly_dead || patched {
                return Some(j..=j);
            }
        }
        None
    }

    /// Replaces the segments in `range` by new ones holding their rows that are not dead, in key order, with their
    /// current `v_max` (at most `MAX_SEGMENT_ROWS` rows each; none if nothing is left). The last one carries the patches
    /// of the old ones that target earlier segments, with the targets' current `v_max` (which is what a later patch of
    /// the same row, if any, also records).
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
        // (vmin, vmax, row) of the rows that stay
        let mut rows: Vec<(i64, i64, StoredRow)> = Vec::with_capacity(members.iter().map(|s| s.len() - s.dead).sum());
        for s in &members {
            for (i, r) in s.seg.rows().enumerate() {
                if s.vmax[i] != DEAD {
                    rows.push((s.seg.vmin[i], s.vmax[i], r.clone()));
                }
            }
        }
        rows.sort_by(|a, b| a.2.key().cmp(&b.2.key()).then(a.0.cmp(&b.0)));
        let mut new_segs = Vec::new();
        let mut rows = rows.into_iter().peekable();
        while rows.peek().is_some() || (new_segs.is_empty() && !carried.is_empty()) {
            let mut vmin = Vec::new();
            let mut vmax = Vec::new();
            let mut out = Vec::new();
            for (a, b, r) in rows.by_ref().take(MAX_SEGMENT_ROWS) {
                vmin.push(a);
                vmax.push(b);
                out.push(r);
            }
            let patches = if rows.peek().is_none() { std::mem::take(&mut carried) } else { Vec::new() };
            let seg = Segment::new(new_segment_id(), out, vmin, patches);
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
            .flat_map(move |s| {
                // decode the chunks in parallel first when many rows are read; else each chunk with a visible row is
                // decoded when reached
                if !s.seg.is_decoded() && (0..s.len()).filter(|i| s.visible(*i, version)).count() * 4 >= s.len() {
                    s.seg.decode_all();
                }
                (0..s.len()).filter(move |i| s.visible(*i, version)).map(move |i| s.seg.row(i))
            })
    }

    /// The number of rows visible at `version`.
    pub fn count_at(&self, version: i64) -> usize {
        self.segs
            .iter()
            .filter(|s| !s.skip_at(version))
            .map(|s| if version >= s.max_vmin && s.dead == 0 && s.live == s.len() { s.len() } else { (0..s.len()).filter(|i| s.visible(*i, version)).count() })
            .sum()
    }

    /// Every stored row with its version range `[v_min, v_max)`, in storage order (`Catalog::validate_store()`).
    pub fn rows_with_versions(&self) -> impl Iterator<Item = (&StoredRow, i64, i64)> {
        self.segs.iter().flat_map(|s| (0..s.len()).map(move |i| (s.seg.row(i), s.seg.vmin[i], s.vmax[i])))
    }

    /// The live rows (visible at the current version), in storage order.
    pub fn live_rows(&self) -> impl Iterator<Item = &StoredRow> {
        self.visible_at(NOW)
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
        self.segs[p].seg.row(i)
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
                for i in 0..s.len() {
                    if s.vmax[i] == LIVE {
                        let (rowid, pos) = s.seg.key(i);
                        let n = prefix_len.min(pos.len());
                        if prefixes.contains(&(rowid, pos[..n].to_vec())) {
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
                for i in 0..s.len() {
                    let (rowid, pos) = s.seg.key(i);
                    if s.vmax[i] == LIVE && keys.contains(&(rowid, pos.to_vec())) {
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
        let keys: Vec<RowKey> = addrs
            .iter()
            .map(|(p, i)| {
                let (rowid, pos) = self.segs[*p].seg.key(*i);
                (rowid, pos.to_vec())
            })
            .collect();
        for a in addrs {
            self.set_vmax(a, version);
        }
        keys
    }

    /// Adds rows that become visible at `version` (as segments of at most `MAX_SEGMENT_ROWS` rows).
    pub fn append(&mut self, mut rows: Vec<StoredRow>, version: i64) {
        rows.sort_by(|a, b| a.key().cmp(&b.key()));
        let mut rows = rows.into_iter();
        loop {
            let chunk: Vec<StoredRow> = rows.by_ref().take(MAX_SEGMENT_ROWS).collect();
            if chunk.is_empty() {
                break;
            }
            let n = chunk.len();
            let seg = Segment::new(new_segment_id(), chunk, vec![version; n], Vec::new());
            self.segs.push(Seg::new(Arc::new(seg), Arc::new(vec![LIVE; n]), false));
        }
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
                for i in 0..s.len() {
                    if s.vmax[i] == LIVE {
                        let (rowid, pos) = s.seg.key(i);
                        if let Some((v, e)) = by_key.remove(&(rowid, pos.to_vec())) {
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
                let r = seg.row_mut(i);
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
            for r in self.rewrite_seg(p).rows_mut() {
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

    /// The rowids of the rows (of any version) in which column `col_id` may hold one of `values` (a superset), by the
    /// column's key index; sorted.
    pub fn eq_candidates(&self, col_id: ColId, values: &[Value]) -> Vec<i64> {
        let keys: Vec<&[Value]> = values.iter().map(std::slice::from_ref).collect();
        let mut out: Vec<i64> = self.pk_candidates_many(&[col_id], false, &keys).into_iter().flatten().collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn pk_candidates_many<V: AsRef<[Value]>>(&self, cols: &[ColId], truncate: bool, keys: &[V]) -> Vec<Vec<i64>> {
        let mut guard = self.indexes.lock();
        let i = match guard.iter().position(|x| x.cols == cols && x.truncate == truncate) {
            Some(i) => i,
            None => {
                guard.push(KeyIndex { cols: cols.to_vec(), truncate, ..Default::default() });
                guard.len() - 1
            }
        };
        let idx = &mut guard[i];
        idx.cover(&self.segs);
        keys.iter()
            .map(|values| {
                let h = pk_hash(values.as_ref().iter(), truncate);
                let mut out = idx.get(h);
                // segments with stored hashes for this key are searched directly
                for s in &self.segs {
                    if let Some(kh) = s.seg.hashes(cols, truncate) {
                        out.extend(kh.rows_with(h).map(|i| s.seg.rowids()[i]));
                    }
                }
                out.sort_unstable();
                out.dedup();
                out
            })
            .collect()
    }
}

/// Maps the values of some columns (the primary key, or an indexed column) in the rows of a table, of every version,
/// to their rowids, by a hash of the values (truncated as the primary-key index of a data-versioned table truncates
/// strings). The rows of a segment are added when a lookup first sees the segment, unless the segment was written
/// with sorted hashes of the same key (`segment::KeyHashes`), which lookups search directly. Lookups check the rows
/// they find, so entries that lead nowhere are harmless, and rowids stay valid across merges (which keep every row
/// version that is visible at some version): all clones of a `TableData` share the index, and it is rebuilt from the
/// segments of the clone at hand once most of its entries come from segments that clone no longer has.
#[derive(Default)]
struct KeyIndex {
    cols: Vec<ColId>,
    truncate: bool,
    /// the segments whose rows are in the index, with their number of rows
    covered: HashMap<u64, usize>,
    /// keyed by `pk_hash()`, which is already well mixed
    first: HashMap<u64, i64, std::hash::BuildHasherDefault<IdentityHasher>>,
    more: HashMap<u64, Vec<i64>, std::hash::BuildHasherDefault<IdentityHasher>>,
}

impl KeyIndex {
    /// Adds the rows of the segments in `segs` that the index does not cover yet (rebuilding it if most of its
    /// entries come from segments not in `segs`).
    fn cover(&mut self, segs: &[Seg]) {
        use rayon::prelude::*;
        let current: HashSet<u64> = segs.iter().map(|s| s.seg.id).collect();
        let stale: usize = self.covered.iter().filter(|(id, _)| !current.contains(id)).map(|(_, n)| n).sum();
        let total: usize = self.covered.values().sum();
        if stale > 1024 && stale * 2 > total {
            self.covered.clear();
            self.first.clear();
            self.more.clear();
        }
        for s in segs {
            let (cols, truncate) = (&self.cols, self.truncate);
            if self.covered.contains_key(&s.seg.id) || s.seg.hashes(cols, truncate).is_some() {
                continue;
            }
            s.seg.decode_all();
            let hashes: Vec<u64> = if s.len() >= 4096 {
                (0..s.len()).into_par_iter().map(|i| pk_hash(cols.iter().map(|c| s.seg.row(i).cell(*c)), truncate)).collect()
            } else {
                s.seg.rows().map(|r| pk_hash(cols.iter().map(|c| r.cell(*c)), truncate)).collect()
            };
            self.first.reserve(hashes.len());
            for (h, rowid) in hashes.into_iter().zip(s.seg.rowids()) {
                self.add(h, *rowid);
            }
            self.covered.insert(s.seg.id, s.len());
        }
    }

    fn add(&mut self, h: u64, rowid: i64) {
        match self.first.get(&h) {
            None => {
                self.first.insert(h, rowid);
            }
            Some(&r) if r == rowid => {}
            // (a rowid may repeat here; `get()` removes duplicates)
            Some(_) => self.more.entry(h).or_default().push(rowid),
        }
    }

    /// The rowids with hash `h`, without duplicates.
    fn get(&self, h: u64) -> Vec<i64> {
        let mut out: Vec<i64> = self.first.get(&h).copied().into_iter().collect();
        if let Some(more) = self.more.get(&h) {
            out.extend(more.iter().copied());
            out.sort_unstable();
            out.dedup();
        }
        out
    }
}

/// The hasher of maps keyed by a hash.
#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 << 8) | u64::from(*b);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

/// A primary-key value as the index holds it.
pub fn pk_part(v: &Value, truncate: bool) -> Value {
    if truncate { crate::functions::btree::truncate(v) } else { v.clone() }
}

/// The hash of a primary-key value (`values`, one per key column) by which the index finds it: of the values as
/// `pk_part()` makes them (without copying them).
pub fn pk_hash<'a>(values: impl Iterator<Item = &'a Value>, truncate: bool) -> u64 {
    let mut h = KeyHasher(0);
    for v in values {
        hash_value(v, truncate, &mut h);
    }
    // murmur3's finalizer, so that every bit of the result depends on every input bit
    let mut x = h.0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// A fast, non-cryptographic hasher (rustc's FxHasher) for primary-key values; collisions only cost lookups a check.
struct KeyHasher(u64);

impl KeyHasher {
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut b = [0u8; 8];
            b[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(b));
        }
        self.add(bytes.len() as u64);
    }
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }
}

fn hash_value(v: &Value, truncate: bool, h: &mut KeyHasher) {
    use crate::functions::btree::MAX_STRING_LEN;
    let tag: u64 = match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Int(_) => 2,
        Value::Float(_) => 3,
        Value::String(_) => 4,
        Value::Timestamp(_) => 5,
        Value::Date(_) => 6,
        Value::Uuid(_) => 7,
        Value::Binary(_) => 8,
        Value::Json(_) => 9,
        Value::Array(_) => 10,
        Value::Media(_) => 11,
        Value::PyObj(_) => 12,
    };
    h.add(tag);
    match v {
        Value::Null | Value::PyObj(_) => {}
        Value::Bool(b) => h.add(u64::from(*b)),
        Value::Int(i) | Value::Timestamp(i) => h.add(*i as u64),
        Value::Float(f) => h.add(if f.is_nan() { f64::NAN.to_bits() } else { f.to_bits() }),
        Value::String(s) if truncate => {
            // the first MAX_STRING_LEN characters, as `btree::truncate()` keeps them
            let end = s.char_indices().nth(MAX_STRING_LEN).map_or(s.len(), |(i, _)| i);
            h.write(&s.as_bytes()[..end]);
        }
        Value::String(s) | Value::Media(s) => h.write(s.as_bytes()),
        Value::Date(d) => h.add(*d as u64),
        Value::Uuid(u) => h.write(u.as_bytes()),
        Value::Binary(b) => h.write(b),
        Value::Json(j) => h.write(j.to_string().as_bytes()),
        Value::Array(a) => h.write(format!("{a:?}").as_bytes()),
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
    fn large_updates_keep_every_version() {
        // updates of most rows, alternating with inserts: merges see segments of mostly retired rows and heavily
        // patched ones
        let mut d = TableData::default();
        let mut expected: Vec<Vec<(i64, i64)>> = vec![Vec::new()];
        d.append((0..3000).map(|i| row(i, i)).collect(), 1);
        d.prepare_write();
        d.mark_written();
        expected.push(visible(&d, 1));
        let mut next = 3000;
        for v in 2..30i64 {
            let lo = (v * 997) % 2500;
            let keys: Vec<i64> = (lo..lo + 2000).filter(|k| d.live_row((*k, &[])).is_some()).collect();
            d.retire(&keys.iter().map(|k| (*k, Vec::new())).collect(), v, true);
            d.append(keys.iter().map(|k| row(*k, k + v * 10_000)).collect(), v);
            d.append((next..next + 1500).map(|i| row(i, i)).collect(), v);
            next += 1500;
            d.prepare_write();
            d.mark_written();
            expected.push(visible(&d, v));
        }
        let reloaded = roundtrip(&mut d);
        for (v, rows) in expected.iter().enumerate() {
            assert_eq!(&visible(&reloaded, v as i64), rows, "version {v} after reload");
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
    fn segments_are_capped() {
        let mut d = TableData::default();
        let n = 2 * MAX_SEGMENT_ROWS as i64 + 100;
        d.append((0..n).map(|i| row(i, i)).collect(), 1);
        assert_eq!(d.segs.len(), 3);
        d.prepare_write();
        d.mark_written();
        // small commits merge among themselves, never into the full segments
        for v in 2..20 {
            d.retire(&keys(&[v * 1000]), v, true);
            d.append((n + v * 10..n + v * 10 + 10).map(|i| row(i, i)).collect(), v);
            d.prepare_write();
            d.mark_written();
        }
        assert!(d.segs.iter().all(|s| s.len() <= MAX_SEGMENT_ROWS));
        assert_eq!(d.count_at(19), n as usize + 18 * 10 - 18);
        let r = roundtrip(&mut d);
        assert_eq!(visible(&r, 19), visible(&d, 19));
        assert_eq!(visible(&r, 5), visible(&d, 5));
    }

    #[test]
    fn column_index() {
        let mut d = TableData::default();
        d.append((0..5000).map(|i| row(i, i % 7)).collect(), 1);
        let c = d.eq_candidates(0, &[Value::Int(3)]);
        assert_eq!(c, (0..5000).filter(|i| i % 7 == 3).collect::<Vec<_>>());
        // a new version of a row with another value: both rowid lists name it (lookups check the rows)
        d.retire(&keys(&[3]), 2, true);
        d.append(vec![row(3, 5)], 2);
        assert!(d.eq_candidates(0, &[Value::Int(5)]).contains(&3));
        assert!(d.eq_candidates(0, &[Value::Int(3), Value::Int(5)]).windows(2).all(|w| w[0] < w[1]));
        // merges rewrite segments; the index is rebuilt once most of it refers to segments that are gone
        for v in 3..40 {
            d.append((5000 + v * 200..5200 + v * 200).map(|i| row(i, i % 7)).collect(), v);
            d.prepare_write();
            d.mark_written();
            let c = d.eq_candidates(0, &[Value::Int(1)]);
            let expected: Vec<i64> = d.live_rows().filter(|r| matches!(r.cell(0), Value::Int(1))).map(|r| r.rowid).collect();
            assert!(expected.iter().all(|r| c.binary_search(r).is_ok()), "version {v}");
        }
        let idx = d.indexes.lock();
        let total: usize = idx.iter().map(|i| i.covered.values().sum::<usize>()).sum();
        assert!(total < 3 * d.num_rows(), "{total} indexed rows for {} rows", d.num_rows());
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
