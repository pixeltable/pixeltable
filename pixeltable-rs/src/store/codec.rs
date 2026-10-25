//! The binary encoding of segment files.
//!
//! ```text
//! header: magic "PXTSEG" 0x04 0x00 | table id (16) | segment id (u64) | rows (u64) | patches (u64) | flags (u8) |
//!         body length (u64) | rows per chunk (u32) | v_min (i64 per row) | v_max (i64 per row) | rowid (i64 per row) |
//!         [flag 1: positions per row (count, int each)] |
//!         [flag 2: key hashes: count, then per key: columns (count, int each), truncated (u8),
//!                  (hash u64, row u32) per row, by hash] |
//!         patches (segment u64, row u32, v_max i64 each) | chunks: offset in the body (u64), checksum (u64) each
//! body:   chunks of `segment::CHUNK_ROWS` rows; per row: cells (count, value each) | errors (count, each 0 or 1 +
//!         type + message)
//! value:  tag (u8) + payload: 0 NULL, 1 false, 2 true, 3 Int (int), 4 Float (f64), 5 String, 6 Timestamp (int),
//!         7 Date (int), 8 UUID (16), 9 Binary, 10 JSON (text), 11 Array (MessagePack of NdArray), 12 Media (path/URL)
//! counts and lengths: LEB128; int: zigzag LEB128; strings and byte strings: length + bytes; fixed-size numbers are
//! little-endian
//! ```
//!
//! `decode()` reads the header and keeps the body encoded (`Segment` decodes a chunk on first use, checking its
//! checksum). JSON values are written as text, refusing values nested more deeply than `value::MAX_JSON_DEPTH` (so
//! every stored value can be read back) and read with the same limit. Chunks are encoded in parallel (rayon).

use uuid::Uuid;

use super::rows::{CellError, StoredRow};
use super::segment::{Body, CHUNK_ROWS, CellKind, ColBuilder, ColChunk, KeyHashes, Patch, Segment};
use crate::catalog::table::ColId;
use crate::error::{Error, ErrorCode, Result};
use crate::value::{MAX_JSON_DEPTH, Value, json_nesting_depth, json_too_deep_msg, parse_json_bounded};

const MAGIC: &[u8; 8] = b"PXTSEG\x04\x00";
const FLAG_POS: u8 = 1;
const FLAG_HASHES: u8 = 2;

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }
    fn int(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn len(&mut self, n: usize) -> Result<()> {
        self.varint(n as u64);
        Ok(())
    }
    fn bytes(&mut self, b: &[u8]) -> Result<()> {
        self.len(b.len())?;
        self.0.extend_from_slice(b);
        Ok(())
    }

    fn value(&mut self, v: &Value) -> Result<()> {
        match v {
            Value::Null => self.u8(0),
            Value::Bool(false) => self.u8(1),
            Value::Bool(true) => self.u8(2),
            Value::Int(i) => {
                self.u8(3);
                self.int(*i);
            }
            Value::Float(f) => {
                self.u8(4);
                self.u64(f.to_bits());
            }
            Value::String(s) => {
                self.u8(5);
                self.bytes(s.as_bytes())?;
            }
            Value::Timestamp(t) => {
                self.u8(6);
                self.int(*t);
            }
            Value::Date(d) => {
                self.u8(7);
                self.int(i64::from(*d));
            }
            Value::Uuid(u) => {
                self.u8(8);
                self.0.extend_from_slice(u.as_bytes());
            }
            Value::Binary(b) => {
                self.u8(9);
                self.bytes(b)?;
            }
            Value::Json(j) => {
                let text = j.to_string();
                if json_nesting_depth(text.as_bytes()) > MAX_JSON_DEPTH {
                    // a computed JSON value nested more deeply than the store accepts
                    return Err(Error::new(ErrorCode::UnsupportedOperation, json_too_deep_msg()));
                }
                self.u8(10);
                self.bytes(text.as_bytes())?;
            }
            Value::Array(a) => {
                let b = rmp_serde::to_vec(a.as_ref()).map_err(|e| Error::internal(format!("cannot serialize table data: {e}")))?;
                self.u8(11);
                self.bytes(&b)?;
            }
            Value::Media(s) => {
                self.u8(12);
                self.bytes(s.as_bytes())?;
            }
            Value::PyObj(_) => return Err(Error::internal("cannot serialize table data: a cell holds a Python object")),
        }
        Ok(())
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn corrupt() -> Error {
    Error::internal("corrupt table data file")
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|e| *e <= self.buf.len()).ok_or_else(corrupt)?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.arr()?))
    }
    /// `n` little-endian i64s (one bounds check for all of them).
    fn i64s(&mut self, n: usize) -> Result<Vec<i64>> {
        let bytes = self.take(n.checked_mul(8).ok_or_else(corrupt)?)?;
        Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
    }
    fn varint(&mut self) -> Result<u64> {
        let mut v: u64 = 0;
        for shift in (0..64).step_by(7) {
            let b = self.u8()?;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(corrupt())
    }
    fn int(&mut self) -> Result<i64> {
        let v = self.varint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }
    /// A count of items of at least one byte each (bounded by what is left, so that a corrupt count cannot make a
    /// reader allocate without limit).
    fn count(&mut self) -> Result<usize> {
        let n = usize::try_from(self.varint()?).map_err(|_| corrupt())?;
        if n > self.buf.len() - self.pos {
            return Err(corrupt());
        }
        Ok(n)
    }
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.count()?;
        self.take(n)
    }
    fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| corrupt())
    }

    /// Skips an encoded value (see `value()`).
    fn skip_value(&mut self) -> Result<()> {
        match self.u8()? {
            0..=2 => {}
            3 | 6 | 7 => {
                self.varint()?;
            }
            4 => {
                self.take(8)?;
            }
            8 => {
                self.take(16)?;
            }
            5 | 9..=12 => {
                self.bytes()?;
            }
            _ => return Err(corrupt()),
        }
        Ok(())
    }

    fn value(&mut self) -> Result<Value> {
        Ok(match self.u8()? {
            0 => Value::Null,
            1 => Value::Bool(false),
            2 => Value::Bool(true),
            3 => Value::Int(self.int()?),
            4 => Value::Float(f64::from_bits(self.u64()?)),
            5 => Value::String(self.string()?),
            6 => Value::Timestamp(self.int()?),
            7 => Value::Date(i32::try_from(self.int()?).map_err(|_| corrupt())?),
            8 => Value::Uuid(Uuid::from_bytes(self.arr()?)),
            9 => Value::Binary(self.bytes()?.to_vec()),
            10 => Value::Json(parse_json_bounded(self.bytes()?, MAX_JSON_DEPTH).map_err(|_| corrupt())?),
            11 => Value::Array(std::sync::Arc::new(rmp_serde::from_slice(self.bytes()?).map_err(|_| corrupt())?)),
            12 => Value::Media(self.string()?),
            _ => return Err(corrupt()),
        })
    }
}

/// Encodes segment `seg` of table `tbl_id`, with `vmax` as the `v_max` of its rows, and with the hashes of the keys
/// `keys` (columns, truncated or not).
pub fn encode(tbl_id: Uuid, seg: &Segment, vmax: &[i64], keys: &[(Vec<ColId>, bool)]) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    let rows: Vec<&StoredRow> = seg.rows().collect();
    let n = rows.len();
    let chunks = rows
        .par_chunks(CHUNK_ROWS)
        .map(|rows| {
            let mut w = Writer(Vec::with_capacity(rows.len() * 48));
            rows.iter().try_for_each(|r| w.row(r))?;
            Ok(w.0)
        })
        .collect::<Result<Vec<Vec<u8>>>>()?;
    let hashes: Vec<Vec<(u64, u32)>> = keys
        .iter()
        .map(|(cols, truncate)| {
            let hash = |(i, r): (usize, &&StoredRow)| (super::table_data::pk_hash(cols.iter().map(|c| r.cell(*c)), *truncate), i as u32);
            let mut sorted: Vec<(u64, u32)> =
                if n >= 4096 { rows.par_iter().enumerate().map(hash).collect() } else { rows.iter().enumerate().map(hash).collect() };
            sorted.par_sort_unstable();
            sorted
        })
        .collect();
    let has_pos = rows.iter().any(|r| !r.pos.is_empty());
    let body_len: usize = chunks.iter().map(Vec::len).sum();
    let mut w = Writer(Vec::with_capacity(96 + n * 32 + body_len));
    w.0.extend_from_slice(MAGIC);
    w.0.extend_from_slice(tbl_id.as_bytes());
    w.u64(seg.id);
    w.u64(n as u64);
    w.u64(seg.patches.len() as u64);
    w.u8(if has_pos { FLAG_POS } else { 0 } | if hashes.is_empty() { 0 } else { FLAG_HASHES });
    w.u64(body_len as u64);
    w.u32(CHUNK_ROWS as u32);
    for v in &seg.vmin {
        w.i64(*v);
    }
    for v in vmax {
        w.i64(*v);
    }
    for r in &rows {
        w.i64(r.rowid);
    }
    if has_pos {
        for r in &rows {
            w.len(r.pos.len())?;
            for p in &r.pos {
                w.int(*p);
            }
        }
    }
    if !hashes.is_empty() {
        w.len(hashes.len())?;
        for ((cols, truncate), sorted) in keys.iter().zip(&hashes) {
            w.len(cols.len())?;
            for c in cols {
                w.varint(u64::from(*c));
            }
            w.u8(u8::from(*truncate));
            for (h, i) in sorted {
                w.u64(*h);
                w.u32(*i);
            }
        }
    }
    for p in &seg.patches {
        w.u64(p.seg);
        w.u32(p.idx);
        w.i64(p.vmax);
    }
    let mut offset = 0u64;
    for c in &chunks {
        w.u64(offset);
        w.u64(checksum(c));
        offset += c.len() as u64;
    }
    for c in &chunks {
        w.0.extend_from_slice(c);
    }
    Ok(w.0)
}

impl Writer {
    /// A row's cells and errors (its key is in the header).
    fn row(&mut self, r: &StoredRow) -> Result<()> {
        self.len(r.cells.len())?;
        for c in &r.cells {
            self.value(c)?;
        }
        self.len(r.errors.len())?;
        for e in &r.errors {
            match e {
                None => self.u8(0),
                Some(e) => {
                    self.u8(1);
                    self.bytes(e.errortype.as_bytes())?;
                    self.bytes(e.errormsg.as_bytes())?;
                }
            }
        }
        Ok(())
    }
}

/// Decodes the header of segment file `bytes` (segment `id` of table `tbl_id`): the segment, with its rows still
/// encoded, and the `v_max` its file records.
pub fn decode(bytes: impl Into<super::file_bytes::FileBytes>, tbl_id: Uuid, id: u64) -> Result<(Segment, Vec<i64>)> {
    let bytes: super::file_bytes::FileBytes = bytes.into();
    let mut r = Reader { buf: &bytes, pos: 0 };
    if r.take(8)? != MAGIC || r.take(16)? != tbl_id.as_bytes() || r.u64()? != id {
        return Err(corrupt());
    }
    let n = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    let m = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    let flags = r.u8()?;
    let body_len = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    if r.u32()? as usize != CHUNK_ROWS {
        return Err(corrupt());
    }
    if n.saturating_mul(24).saturating_add(m.saturating_mul(20)).saturating_add(body_len) > bytes.len() {
        return Err(corrupt());
    }
    let vmin = r.i64s(n)?;
    let vmax = r.i64s(n)?;
    let rowids = r.i64s(n)?;
    let pos = if flags & FLAG_POS != 0 {
        (0..n)
            .map(|_| {
                let k = r.count()?;
                (0..k).map(|_| r.int()).collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut hashes = Vec::new();
    if flags & FLAG_HASHES != 0 {
        for _ in 0..r.count()? {
            let k = r.count()?;
            let cols = (0..k).map(|_| u32::try_from(r.varint()?).map_err(|_| corrupt())).collect::<Result<Vec<_>>>()?;
            let truncate = r.u8()? != 0;
            let bytes = r.take(n.checked_mul(12).ok_or_else(corrupt)?)?;
            let sorted: Vec<(u64, u32)> = bytes
                .chunks_exact(12)
                .map(|c| {
                    let h = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                    (h, u32::from_le_bytes([c[8], c[9], c[10], c[11]]))
                })
                .collect();
            if sorted.iter().any(|(_, i)| *i as usize >= n) {
                return Err(corrupt());
            }
            hashes.push(KeyHashes { cols, truncate, sorted });
        }
    }
    let patches = (0..m).map(|_| Ok(Patch { seg: r.u64()?, idx: r.u32()?, vmax: r.i64()? })).collect::<Result<Vec<_>>>()?;
    let chunks = (0..n.div_ceil(CHUNK_ROWS))
        .map(|_| {
            let (offset, sum) = (r.u64()?, r.u64()?);
            if offset > body_len as u64 { Err(corrupt()) } else { Ok((offset, sum)) }
        })
        .collect::<Result<Vec<_>>>()?;
    let start = r.pos;
    if bytes.len() - start != body_len || chunks.windows(2).any(|w| w[0].0 > w[1].0) {
        return Err(corrupt());
    }
    Ok((Segment::encoded(id, vmin, patches, rowids, pos, Body { bytes, start, chunks }, hashes), vmax))
}

/// The rows of chunk `k` of a segment from its encoded body and the keys of the chunk's rows.
pub fn decode_chunk(body: &Body, k: usize, rowids: &[i64], pos: &[Vec<i64>]) -> Result<Vec<StoredRow>> {
    let (offset, sum) = body.chunks[k];
    let end = body.chunks.get(k + 1).map_or(body.bytes.len() - body.start, |c| c.0 as usize);
    let bytes = &body.bytes[body.start + offset as usize..body.start + end];
    if checksum(bytes) != sum {
        return Err(Error::internal("table data file is corrupt (checksum mismatch)"));
    }
    let mut r = Reader { buf: bytes, pos: 0 };
    let rows = (0..rowids.len())
        .map(|i| r.row((rowids[i], pos.get(i).cloned().unwrap_or_default())))
        .collect::<Result<Vec<_>>>()?;
    if r.pos == bytes.len() { Ok(rows) } else { Err(corrupt()) }
}

/// Column `col` of chunk `k` of a segment (of `n` rows) in typed form, read from the encoded body without decoding the
/// rows: the other cells are skipped. The same column as `ColChunk::of()` makes of the decoded rows.
pub fn decode_column(body: &Body, k: usize, n: usize, col: ColId) -> Result<ColChunk> {
    let (offset, sum) = body.chunks[k];
    let end = body.chunks.get(k + 1).map_or(body.bytes.len() - body.start, |c| c.0 as usize);
    let bytes = &body.bytes[body.start + offset as usize..body.start + end];
    if checksum(bytes) != sum {
        return Err(Error::internal("table data file is corrupt (checksum mismatch)"));
    }
    let mut r = Reader { buf: bytes, pos: 0 };
    let mut b = ColBuilder::new(n);
    for _ in 0..n {
        let ncells = r.count()?;
        if ncells <= col as usize {
            b.null();
        }
        for c in 0..ncells {
            if c == col as usize {
                match r.u8()? {
                    0 => b.null(),
                    3 => b.cell(CellKind::Int, r.int()?, None),
                    6 => b.cell(CellKind::Timestamp, r.int()?, None),
                    4 => b.cell(CellKind::Float, r.u64()? as i64, None),
                    5 => b.cell(CellKind::Str, 0, Some(r.bytes()?)),
                    // a cell of another kind: the column has no typed form
                    _ => return Ok(ColChunk::Mixed),
                }
                if b.is_mixed() {
                    return Ok(ColChunk::Mixed);
                }
            } else {
                r.skip_value()?;
            }
        }
        for _ in 0..r.count()? {
            match r.u8()? {
                0 => {}
                1 => {
                    r.bytes()?;
                    r.bytes()?;
                }
                _ => return Err(corrupt()),
            }
        }
    }
    b.finish().ok_or_else(corrupt)
}

/// A fast checksum of a segment body (four independent multiply-rotate lanes over 8-byte words).
fn checksum(b: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [K, K.rotate_left(16), K.rotate_left(32), K.rotate_left(48)];
    let mut chunks = b.chunks_exact(32);
    for c in &mut chunks {
        for (k, lane) in lanes.iter_mut().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&c[k * 8..k * 8 + 8]);
            *lane = (*lane ^ u64::from_le_bytes(w)).wrapping_mul(K).rotate_left(29);
        }
    }
    let mut tail = [0u8; 32];
    let rest = chunks.remainder();
    tail[..rest.len()].copy_from_slice(rest);
    for (k, lane) in lanes.iter_mut().enumerate() {
        let mut w = [0u8; 8];
        w.copy_from_slice(&tail[k * 8..k * 8 + 8]);
        *lane = (*lane ^ u64::from_le_bytes(w)).wrapping_mul(K).rotate_left(29);
    }
    let mut x = lanes.iter().fold(b.len() as u64, |acc, l| (acc ^ l).wrapping_mul(K).rotate_left(31));
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^ (x >> 33)
}

impl Reader<'_> {
    fn row(&mut self, (rowid, pos): (i64, Vec<i64>)) -> Result<StoredRow> {
        let ncells = self.count()?;
        let cells = (0..ncells).map(|_| self.value()).collect::<Result<Vec<_>>>()?;
        let nerr = self.count()?;
        let mut errors = Vec::with_capacity(nerr);
        for _ in 0..nerr {
            errors.push(match self.u8()? {
                0 => None,
                1 => Some(CellError { errortype: self.string()?, errormsg: self.string()? }),
                _ => return Err(corrupt()),
            });
        }
        Ok(StoredRow { rowid, cells, errors, pos })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{ArrayData, NdArray};

    #[test]
    fn columns_from_encoded_chunks() -> Result<()> {
        let tbl = Uuid::new_v4();
        let rows: Vec<StoredRow> = (0..300)
            .map(|i| {
                let mut r = StoredRow::new(i, Vec::new());
                r.set_cell(0, if i % 9 == 0 { Value::Null } else { Value::String(format!("s{i}\u{e9}")) });
                r.set_cell(5, Value::Media(format!("/m/{i}")));
                r.set_cell(1, if i % 7 == 0 { Value::Null } else { Value::Int(i * 3 - 100) });
                r.set_cell(2, if i % 5 == 0 { Value::Null } else { Value::Float(i as f64 / 3.0) });
                r.set_cell(3, Value::Timestamp(i * 1000));
                if i % 2 == 0 {
                    r.set_cell(4, Value::Int(i));
                }
                if i % 11 == 0 {
                    r.set_error(2, Some(CellError { errortype: "E".into(), errormsg: "m".into() }));
                }
                r
            })
            .collect();
        let n = rows.len();
        let seg = Segment::new(9, rows.clone(), vec![1; n], Vec::new());
        let bytes = encode(tbl, &seg, &vec![crate::store::rows::LIVE; n], &[])?;
        let (back, _) = decode(bytes, tbl, 9)?;
        for col in 0..7 {
            let from_rows = format!("{:?}", ColChunk::of(&rows, col));
            assert_eq!(format!("{:?}", back.column(0, col)), from_rows, "column {col}");
        }
        Ok(())
    }

    #[test]
    fn roundtrip() -> Result<()> {
        let tbl = Uuid::new_v4();
        let values = vec![
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(-5),
            Value::Int(i64::MIN),
            Value::Int(i64::MAX),
            Value::Float(f64::NAN),
            Value::Float(-0.0),
            Value::String("héllo".into()),
            Value::Timestamp(1_700_000_000_000_000),
            Value::Date(-3),
            Value::Uuid(Uuid::new_v4()),
            Value::Binary(vec![0, 255, 7]),
            Value::Json(serde_json::json!({"a": [1, 2.5, null, "x"]})),
            Value::Array(std::sync::Arc::new(NdArray { shape: vec![2], data: ArrayData::I32(vec![1, 2]) })),
            Value::Media("/tmp/x.jpg".into()),
        ];
        let mut row = StoredRow::new(42, vec![1, 2]);
        for (i, v) in values.iter().enumerate() {
            row.set_cell(i as u32, v.clone());
        }
        row.set_error(3, Some(CellError { errortype: "ValueError".into(), errormsg: "bad".into() }));
        let seg = Segment::new(9, vec![row, StoredRow::new(43, Vec::new())], vec![1, 2], vec![Patch { seg: 3, idx: 4, vmax: 5 }]);
        let bytes = encode(tbl, &seg, &[7, i64::MAX], &[(vec![3], false), (vec![0, 1], true)])?;
        let (back, vmax) = decode(bytes.clone(), tbl, 9)?;
        assert_eq!(vmax, vec![7, i64::MAX]);
        assert_eq!(back.vmin, seg.vmin);
        assert_eq!(back.patches, seg.patches);
        assert_eq!(back.key(0), (42, &[1, 2][..]));
        assert_eq!(back.key(1), (43, &[][..]));
        assert_eq!(back.hashes(&[3], false).map(|h| h.sorted.len()), Some(2));
        assert_eq!(back.hashes(&[0, 1], true).map(|h| h.sorted.len()), Some(2));
        assert!(back.hashes(&[3], true).is_none());
        let rows: Vec<&StoredRow> = back.rows().collect();
        assert_eq!(rows[0].pos, vec![1, 2]);
        assert_eq!(rows[1].rowid, 43);
        for (i, v) in values.iter().enumerate() {
            let got = rows[0].cell(i as u32);
            assert!(crate::store::table_data::values_identical(got, v), "{v:?} came back as {got:?}");
        }
        assert_eq!(rows[0].error(3).map(|e| e.errormsg.as_str()), Some("bad"));
        // truncated or altered files are refused, never panic
        for cut in [0, 10, bytes.len() / 2, bytes.len() - 1] {
            assert!(decode(bytes[..cut].to_vec(), tbl, 9).is_err());
        }
        assert!(decode(bytes.clone(), tbl, 10).is_err());
        // a chunk that does not match its checksum is refused (when it is decoded)
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        let (seg, _) = decode(flipped, tbl, 9)?;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| seg.row(0).rowid)).is_err());
        Ok(())
    }

    #[test]
    fn large_segments_decode_in_chunks() -> Result<()> {
        let tbl = Uuid::new_v4();
        let n = 3 * CHUNK_ROWS + 17;
        let rows: Vec<StoredRow> = (0..n as i64)
            .map(|i| {
                let mut r = StoredRow::new(i, if i % 3 == 0 { vec![i, -i] } else { Vec::new() });
                r.set_cell(0, Value::Int(i));
                r.set_cell(2, Value::String(format!("s{i}")));
                if i % 5 == 0 {
                    r.set_error(1, Some(CellError { errortype: "E".into(), errormsg: format!("m{i}") }));
                }
                r
            })
            .collect();
        let seg = Segment::new(1, rows, vec![1; n], Vec::new());
        let bytes = encode(tbl, &seg, &vec![i64::MAX; n], &[])?;
        let (back, _) = decode(bytes.clone(), tbl, 1)?;
        // one row decodes its chunk only
        assert_eq!(back.row(CHUNK_ROWS + 5).rowid, (CHUNK_ROWS + 5) as i64);
        assert_eq!(back.decoded_chunks(), 1);
        assert_eq!(back.rows().count(), n);
        assert!(back.is_decoded());
        for (a, b) in seg.rows().zip(back.rows()) {
            assert_eq!(a.rowid, b.rowid);
            assert_eq!(a.pos, b.pos);
            assert!(a.cells.iter().zip(&b.cells).all(|(x, y)| crate::store::table_data::values_identical(x, y)));
            assert_eq!(a.errors, b.errors);
        }
        assert!(decode(bytes[..bytes.len() - 3].to_vec(), tbl, 1).is_err());
        Ok(())
    }

    #[test]
    fn too_deep_json_is_refused() {
        let mut j = serde_json::json!(1);
        for _ in 0..=MAX_JSON_DEPTH {
            j = serde_json::json!([j]);
        }
        let mut row = StoredRow::new(0, Vec::new());
        row.set_cell(0, Value::Json(j));
        let seg = Segment::new(1, vec![row], vec![0], Vec::new());
        match encode(Uuid::nil(), &seg, &[i64::MAX], &[]) {
            Err(Error::Pxt { code: ErrorCode::UnsupportedOperation, message }) => assert_eq!(message, json_too_deep_msg()),
            other => panic!("expected UNSUPPORTED_OPERATION, got {other:?}"),
        }
    }
}
