//! A segment: an immutable run of a table's rows, sorted by key (then `v_min`), stored as one file.
//!
//! A segment read from a file keeps its keys, version ranges, patches and stored key hashes decoded, and its cells
//! encoded, in chunks of `CHUNK_ROWS` rows, until something reads them: loading a table costs little more than
//! reading its files, counting rows needs no decoding, a point lookup decodes one chunk, and a scan decodes a segment's
//! chunks in parallel (`decode_all()`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::codec;
use super::rows::StoredRow;
use crate::catalog::table::ColId;

/// The rows of a segment are encoded, and decoded, in chunks of this many rows.
pub const CHUNK_ROWS: usize = 8192;

/// A change of `v_max` of row `idx` of segment `seg`, carried by a later segment's file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    pub seg: u64,
    pub idx: u32,
    pub vmax: i64,
}

/// The hashes of a key (`table_data::pk_hash()` of columns `cols`, truncated or not) of every row of a segment, as
/// (hash, row index) sorted by hash: written with the segment, so that key lookups binary-search it instead of
/// decoding the rows into a map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyHashes {
    pub cols: Vec<ColId>,
    pub truncate: bool,
    pub sorted: Vec<(u64, u32)>,
}

impl KeyHashes {
    /// The indexes of the rows whose key has hash `h`.
    pub fn rows_with(&self, h: u64) -> impl Iterator<Item = usize> + '_ {
        let start = self.sorted.partition_point(|(x, _)| *x < h);
        self.sorted[start..].iter().take_while(move |(x, _)| *x == h).map(|(_, i)| *i as usize)
    }
}

/// The encoded cells of a segment read from a file: `bytes[start..]`, as chunks of `CHUNK_ROWS` rows, each with its
/// offset (from `start`) and checksum.
#[derive(Debug)]
pub struct Body {
    pub bytes: super::file_bytes::FileBytes,
    pub start: usize,
    pub chunks: Vec<(u64, u64)>,
}

/// One column of a chunk's rows in contiguous typed form, for scans that compare or aggregate many values (built on
/// first use, `Segment::column()`): the cells of the chunk's rows for one column, when they are all of one numeric kind
/// (or NULL).
#[derive(Debug)]
pub enum ColChunk {
    Int(TypedCol<i64>),
    Float(TypedCol<f64>),
    Timestamp(TypedCol<i64>),
    Str(StrCol),
    /// cells of other kinds (or of several kinds), or only NULLs: no typed form
    Mixed,
}

/// Values and NULL flags (None: no NULLs); a NULL's value is 0.
#[derive(Debug)]
pub struct TypedCol<T> {
    pub vals: Vec<T>,
    pub nulls: Option<Vec<bool>>,
}

impl<T> TypedCol<T> {
    pub fn is_null(&self, i: usize) -> bool {
        self.nulls.as_ref().is_some_and(|n| n[i])
    }
}

/// Strings, one after the other in one buffer (`ends[i]`: the end of string `i`), and NULL flags; a NULL is empty.
#[derive(Debug)]
pub struct StrCol {
    pub bytes: String,
    pub ends: Vec<u32>,
    pub nulls: Option<Vec<bool>>,
}

impl StrCol {
    pub fn is_null(&self, i: usize) -> bool {
        self.nulls.as_ref().is_some_and(|n| n[i])
    }

    pub fn get(&self, i: usize) -> &str {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.bytes[start..self.ends[i] as usize]
    }
}

/// The kind of a typed column (the codec's value tags).
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum CellKind {
    Int = 3,
    Float = 4,
    Str = 5,
    Timestamp = 6,
}

/// Builds a `ColChunk` cell by cell: the first non-NULL cell decides the kind; a cell of another kind makes the column
/// `Mixed`.
pub(crate) struct ColBuilder {
    kind: Option<CellKind>,
    mixed: bool,
    n: usize,
    /// ints, timestamps and float bits
    vals: Vec<i64>,
    bytes: Vec<u8>,
    ends: Vec<u32>,
    nulls: Option<Vec<bool>>,
}

impl ColBuilder {
    pub(crate) fn new(capacity: usize) -> ColBuilder {
        ColBuilder { kind: None, mixed: false, n: 0, vals: Vec::with_capacity(capacity), bytes: Vec::new(), ends: Vec::with_capacity(capacity), nulls: None }
    }

    /// True once the column has no typed form (later cells need not be given).
    pub(crate) fn is_mixed(&self) -> bool {
        self.mixed
    }

    pub(crate) fn null(&mut self) {
        let (n, cap) = (self.n, self.vals.capacity());
        self.nulls.get_or_insert_with(|| {
            let mut v = Vec::with_capacity(cap.max(n + 1));
            v.resize(n, false);
            v
        });
        self.push(0, None);
        if let Some(nulls) = self.nulls.as_mut() {
            nulls.push(true);
        }
    }

    /// A non-NULL cell: an int, a timestamp or a float's bits (`v`), or a string (`s`).
    pub(crate) fn cell(&mut self, kind: CellKind, v: i64, s: Option<&[u8]>) {
        match self.kind {
            None => self.kind = Some(kind),
            Some(k) if k != kind => {
                self.mixed = true;
                return;
            }
            _ => {}
        }
        self.push(v, s);
        if let Some(nulls) = self.nulls.as_mut() {
            nulls.push(false);
        }
    }

    fn push(&mut self, v: i64, s: Option<&[u8]>) {
        self.vals.push(v);
        if let Some(s) = s {
            self.bytes.extend_from_slice(s);
        }
        self.ends.push(self.bytes.len() as u32);
        self.n += 1;
    }

    /// The column (None: its strings are not UTF-8).
    pub(crate) fn finish(self) -> Option<ColChunk> {
        if self.mixed || self.bytes.len() > u32::MAX as usize {
            return Some(ColChunk::Mixed);
        }
        let nulls = self.nulls;
        Some(match self.kind {
            Some(CellKind::Int) => ColChunk::Int(TypedCol { vals: self.vals, nulls }),
            Some(CellKind::Timestamp) => ColChunk::Timestamp(TypedCol { vals: self.vals, nulls }),
            Some(CellKind::Float) => ColChunk::Float(TypedCol {
                vals: self
                    .vals
                    .iter()
                    .enumerate()
                    .map(|(i, v)| if nulls.as_ref().is_some_and(|n| n[i]) { 0.0 } else { f64::from_bits(*v as u64) })
                    .collect(),
                nulls,
            }),
            Some(CellKind::Str) => ColChunk::Str(StrCol { bytes: String::from_utf8(self.bytes).ok()?, ends: self.ends, nulls }),
            None => ColChunk::Mixed,
        })
    }
}

impl ColChunk {
    pub(crate) fn of(rows: &[StoredRow], col: ColId) -> ColChunk {
        use crate::value::Value;
        let mut b = ColBuilder::new(rows.len());
        for r in rows {
            match r.cell(col) {
                Value::Null => b.null(),
                Value::Int(v) => b.cell(CellKind::Int, *v, None),
                Value::Timestamp(v) => b.cell(CellKind::Timestamp, *v, None),
                Value::Float(v) => b.cell(CellKind::Float, v.to_bits() as i64, None),
                Value::String(s) => b.cell(CellKind::Str, 0, Some(s.as_bytes())),
                _ => return ColChunk::Mixed,
            }
            if b.is_mixed() {
                return ColChunk::Mixed;
            }
        }
        b.finish().unwrap_or(ColChunk::Mixed)
    }
}

#[derive(Debug)]
pub struct Segment {
    /// unique within the table (names the file)
    pub id: u64,
    pub vmin: Vec<i64>,
    /// `v_max` changes of rows of earlier segments, carried by this segment's file
    pub patches: Vec<Patch>,
    rowids: Vec<i64>,
    /// iterator positions of each row; empty if no row has any
    pos: Vec<Vec<i64>>,
    /// the rows, `CHUNK_ROWS` per chunk
    chunks: Vec<OnceLock<Vec<StoredRow>>>,
    /// the encoded rows, until every chunk is decoded
    body: Mutex<Option<Arc<Body>>>,
    undecoded: AtomicUsize,
    /// one per key index known when the segment was written
    hashes: Vec<KeyHashes>,
    /// per chunk: the typed columns built so far (`column()`)
    columns: Vec<parking_lot::RwLock<Vec<(ColId, Arc<ColChunk>)>>>,
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

impl Segment {
    /// A segment of the given rows (sorted by key, then `vmin`).
    pub fn new(id: u64, rows: Vec<StoredRow>, vmin: Vec<i64>, patches: Vec<Patch>) -> Segment {
        let rowids = rows.iter().map(|r| r.rowid).collect();
        let pos = if rows.iter().any(|r| !r.pos.is_empty()) { rows.iter().map(|r| r.pos.clone()).collect() } else { Vec::new() };
        let mut chunks = Vec::with_capacity(rows.len().div_ceil(CHUNK_ROWS));
        let mut rows = rows.into_iter();
        loop {
            let chunk: Vec<StoredRow> = rows.by_ref().take(CHUNK_ROWS).collect();
            if chunk.is_empty() {
                break;
            }
            chunks.push(OnceLock::from(chunk));
        }
        let columns = (0..chunks.len()).map(|_| Default::default()).collect();
        Segment { id, vmin, patches, rowids, pos, chunks, body: Mutex::new(None), undecoded: AtomicUsize::new(0), hashes: Vec::new(), columns }
    }

    /// A segment read from a file whose rows are still encoded (see `codec`).
    pub(super) fn encoded(
        id: u64,
        vmin: Vec<i64>,
        patches: Vec<Patch>,
        rowids: Vec<i64>,
        pos: Vec<Vec<i64>>,
        body: Body,
        hashes: Vec<KeyHashes>,
    ) -> Segment {
        let n = body.chunks.len();
        Segment {
            id,
            vmin,
            patches,
            rowids,
            pos,
            chunks: (0..n).map(|_| OnceLock::new()).collect(),
            body: Mutex::new((n > 0).then(|| Arc::new(body))),
            undecoded: AtomicUsize::new(n),
            hashes,
            columns: (0..n).map(|_| Default::default()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.rowids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rowids.is_empty()
    }

    /// The key of row `i`.
    pub fn key(&self, i: usize) -> (i64, &[i64]) {
        (self.rowids[i], self.pos.get(i).map_or(&[][..], |p| p.as_slice()))
    }

    pub fn rowids(&self) -> &[i64] {
        &self.rowids
    }

    /// The hashes of key `cols` (truncated or not) written with the segment, if any.
    pub fn hashes(&self, cols: &[ColId], truncate: bool) -> Option<&KeyHashes> {
        self.hashes.iter().find(|h| h.cols == cols && h.truncate == truncate)
    }

    /// True once all rows are decoded.
    pub fn is_decoded(&self) -> bool {
        self.undecoded.load(Ordering::Acquire) == 0
    }

    /// The number of decoded chunks.
    pub fn decoded_chunks(&self) -> usize {
        self.chunks.iter().filter(|c| c.get().is_some()).count()
    }

    /// Chunk `k`, decoded on first use. Decoding cannot fail for a segment the store wrote (each chunk has a
    /// checksum against corruption), so a failure is reported as a panic, which operations turn into an internal
    /// error. Decoding never touches the Python interpreter (a thread waiting here may hold it).
    fn chunk(&self, k: usize) -> &[StoredRow] {
        self.chunks[k].get_or_init(|| {
            let body = self.body.lock().clone();
            let Some(body) = body else { unreachable!("an undecoded chunk has an encoded body") };
            let first = k * CHUNK_ROWS;
            let last = self.len().min(first + CHUNK_ROWS);
            let pos = self.pos.get(first..last).unwrap_or(&[]);
            let rows = match codec::decode_chunk(&body, k, &self.rowids[first..last], pos) {
                Ok(rows) => rows,
                Err(e) => panic!("table data segment {:016x} cannot be decoded: {}", self.id, e.message()),
            };
            if self.undecoded.fetch_sub(1, Ordering::AcqRel) == 1 {
                // the last chunk: the encoded rows are no longer needed
                *self.body.lock() = None;
            }
            rows
        })
    }

    /// True if chunk `k` is decoded.
    pub fn chunk_decoded(&self, k: usize) -> bool {
        self.chunks[k].get().is_some()
    }

    /// Decodes chunk `k` if it is not decoded yet.
    pub fn decode_chunk(&self, k: usize) {
        self.chunk(k);
    }

    /// Row `i` (decoding its chunk if needed).
    pub fn row(&self, i: usize) -> &StoredRow {
        &self.chunk(i / CHUNK_ROWS)[i % CHUNK_ROWS]
    }

    /// Decodes all chunks not decoded yet, in parallel.
    pub fn decode_all(&self) {
        if self.is_decoded() {
            return;
        }
        use rayon::prelude::*;
        (0..self.chunks.len()).into_par_iter().for_each(|k| {
            self.chunk(k);
        });
    }

    /// The rows, in order.
    pub fn rows(&self) -> impl Iterator<Item = &StoredRow> {
        self.decode_all();
        (0..self.chunks.len()).flat_map(move |k| self.chunk(k).iter())
    }

    /// Column `col` of chunk `k` in typed form (built on first use; `ColChunk::Mixed` if its cells are not all of one
    /// numeric kind).
    pub fn column(&self, k: usize, col: ColId) -> Arc<ColChunk> {
        if let Some((_, c)) = self.columns[k].read().iter().find(|(id, _)| *id == col) {
            return c.clone();
        }
        // from the decoded rows, or else from the encoded chunk (without decoding the rows)
        let body = if self.chunks[k].get().is_some() { None } else { self.body.lock().clone() };
        let built = Arc::new(match body {
            Some(body) => {
                let n = self.len().min((k + 1) * CHUNK_ROWS) - k * CHUNK_ROWS;
                match codec::decode_column(&body, k, n, col) {
                    Ok(c) => c,
                    Err(e) => panic!("table data segment {:016x} cannot be decoded: {}", self.id, e.message()),
                }
            }
            None => ColChunk::of(self.chunk(k), col),
        });
        let mut cache = self.columns[k].write();
        match cache.iter().find(|(id, _)| *id == col) {
            Some((_, c)) => c.clone(),
            None => {
                cache.push((col, built.clone()));
                built
            }
        }
    }

    /// The rows, for changing their cells (keys stay); drops the stored key hashes.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut StoredRow> {
        self.decode_all();
        self.hashes.clear();
        for c in &mut self.columns {
            c.get_mut().clear();
        }
        self.chunks.iter_mut().flat_map(|c| match c.get_mut() {
            Some(rows) => rows.iter_mut(),
            None => unreachable!("decode_all() decoded every chunk"),
        })
    }

    /// Row `i`, for changing its cells (its key stays); drops the stored key hashes.
    pub fn row_mut(&mut self, i: usize) -> &mut StoredRow {
        self.chunk(i / CHUNK_ROWS);
        self.hashes.clear();
        self.columns[i / CHUNK_ROWS].get_mut().clear();
        match self.chunks[i / CHUNK_ROWS].get_mut() {
            Some(rows) => &mut rows[i % CHUNK_ROWS],
            None => unreachable!("chunk() decoded the chunk"),
        }
    }
}

impl Clone for Segment {
    fn clone(&self) -> Segment {
        let mut copy = Segment::new(self.id, self.rows().cloned().collect(), self.vmin.clone(), self.patches.clone());
        copy.hashes = self.hashes.clone();
        copy
    }
}
