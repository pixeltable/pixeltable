//! A segment: an immutable run of a table's rows, sorted by key (then `v_min`), stored as one file.
//!
//! A segment read from a file keeps its keys, version ranges, patches and stored key hashes decoded, and its cells
//! encoded until something reads them (`rows()`), so that loading a table costs little more than reading its files,
//! counting rows needs no decoding, and a point lookup decodes one segment.

use std::sync::OnceLock;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::codec;
use super::rows::StoredRow;
use crate::catalog::table::ColId;

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

/// The encoded cells of a segment read from a file: `bytes[start..]`, with its checksum.
#[derive(Debug)]
pub struct Body {
    pub bytes: Vec<u8>,
    pub start: usize,
    pub checksum: u64,
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
    rows: OnceLock<Vec<StoredRow>>,
    /// the encoded rows, until decoded
    body: Mutex<Option<Body>>,
    hashes: Option<KeyHashes>,
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
        Segment { id, vmin, patches, rowids, pos, rows: OnceLock::from(rows), body: Mutex::new(None), hashes: None }
    }

    /// A segment read from a file whose rows are still encoded (see `codec`).
    pub(super) fn encoded(
        id: u64,
        vmin: Vec<i64>,
        patches: Vec<Patch>,
        rowids: Vec<i64>,
        pos: Vec<Vec<i64>>,
        body: Body,
        hashes: Option<KeyHashes>,
    ) -> Segment {
        Segment { id, vmin, patches, rowids, pos, rows: OnceLock::new(), body: Mutex::new(Some(body)), hashes }
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

    /// The key hashes written with the segment, if any.
    pub fn hashes(&self) -> Option<&KeyHashes> {
        self.hashes.as_ref()
    }

    /// The rows, decoded on first use. Decoding cannot fail for a segment the store wrote (the file's checksum
    /// guards against corruption), so a failure is reported as a panic, which operations turn into an internal error.
    pub fn rows(&self) -> &[StoredRow] {
        self.rows.get_or_init(|| {
            let body = self.body.lock().take();
            let Some(body) = body else { unreachable!("a segment has either rows or an encoded body") };
            match codec::decode_body(&body, &self.rowids, &self.pos) {
                Ok(rows) => rows,
                Err(e) => panic!("table data segment {:016x} cannot be decoded: {}", self.id, e.message()),
            }
        })
    }

    /// True once the rows are decoded.
    pub fn is_decoded(&self) -> bool {
        self.rows.get().is_some()
    }

    /// The rows, for changing their cells (keys stay); drops the stored key hashes.
    pub fn rows_mut(&mut self) -> &mut Vec<StoredRow> {
        self.rows();
        self.hashes = None;
        match self.rows.get_mut() {
            Some(rows) => rows,
            None => unreachable!("rows() initialized them"),
        }
    }
}

impl Clone for Segment {
    fn clone(&self) -> Segment {
        Segment {
            id: self.id,
            vmin: self.vmin.clone(),
            patches: self.patches.clone(),
            rowids: self.rowids.clone(),
            pos: self.pos.clone(),
            rows: OnceLock::from(self.rows().to_vec()),
            body: Mutex::new(None),
            hashes: self.hashes.clone(),
        }
    }
}
