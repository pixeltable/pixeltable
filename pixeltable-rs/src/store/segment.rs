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
        Segment { id, vmin, patches, rowids, pos, chunks, body: Mutex::new(None), undecoded: AtomicUsize::new(0), hashes: Vec::new() }
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

    /// The rows, for changing their cells (keys stay); drops the stored key hashes.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut StoredRow> {
        self.decode_all();
        self.hashes.clear();
        self.chunks.iter_mut().flat_map(|c| match c.get_mut() {
            Some(rows) => rows.iter_mut(),
            None => unreachable!("decode_all() decoded every chunk"),
        })
    }

    /// Row `i`, for changing its cells (its key stays); drops the stored key hashes.
    pub fn row_mut(&mut self, i: usize) -> &mut StoredRow {
        self.chunk(i / CHUNK_ROWS);
        self.hashes.clear();
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
