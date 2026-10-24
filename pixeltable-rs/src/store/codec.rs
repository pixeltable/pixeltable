//! The binary encoding of segment files.
//!
//! ```text
//! header: magic "PXTSEG" 0x03 0x00 | table id (16) | segment id (u64) | rows (u64) | patches (u64) | flags (u8) |
//!         body length (u64) | body checksum (u64) | v_min (i64 per row) | v_max (i64 per row) | rowid (i64 per row) |
//!         [flag 1: positions per row (count, int each)] |
//!         [flag 2: key hashes: columns (count, int each), truncated (u8), (hash u64, row u32) per row by hash] |
//!         patches (segment u64, row u32, v_max i64 each)
//! body:   per row: cells (count, value each) | errors (count, each 0 or 1 + type + message)
//! value:  tag (u8) + payload: 0 NULL, 1 false, 2 true, 3 Int (int), 4 Float (f64), 5 String, 6 Timestamp (int),
//!         7 Date (int), 8 UUID (16), 9 Binary, 10 JSON (text), 11 Array (MessagePack of NdArray), 12 Media (path/URL)
//! counts and lengths: LEB128; int: zigzag LEB128; strings and byte strings: length + bytes; fixed-size numbers are
//! little-endian
//! ```
//!
//! `decode()` reads the header and keeps the body encoded (`Segment::rows()` decodes it on first use, checking its
//! checksum). JSON values are written as text, refusing values nested more deeply than `value::MAX_JSON_DEPTH` (so
//! every stored value can be read back) and read with the same limit. Large segments are encoded and decoded in
//! chunks of `DECODE_CHUNK` rows in parallel (rayon); decoding first skips through the rows to find where the chunks
//! start.

use uuid::Uuid;

use super::rows::{CellError, StoredRow};
use super::segment::{Body, KeyHashes, Patch, Segment};
use crate::catalog::table::ColId;
use crate::error::{Error, ErrorCode, Result};
use crate::value::{MAX_JSON_DEPTH, Value, json_nesting_depth, json_too_deep_msg, parse_json_bounded};

const MAGIC: &[u8; 8] = b"PXTSEG\x03\x00";
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

/// Encodes segment `seg` of table `tbl_id`, with `vmax` as the `v_max` of its rows, and with the hashes of key `key`
/// (columns, truncated or not) if given.
pub fn encode(tbl_id: Uuid, seg: &Segment, vmax: &[i64], key: Option<(&[ColId], bool)>) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    let rows = seg.rows();
    let n = rows.len();
    let body = if n < 2 * DECODE_CHUNK {
        let mut w = Writer(Vec::with_capacity(n * 48));
        rows.iter().try_for_each(|r| w.row(r))?;
        w.0
    } else {
        let parts = rows
            .par_chunks(DECODE_CHUNK)
            .map(|rows| {
                let mut pw = Writer(Vec::with_capacity(rows.len() * 48));
                rows.iter().try_for_each(|r| pw.row(r))?;
                Ok(pw.0)
            })
            .collect::<Result<Vec<Vec<u8>>>>()?;
        parts.concat()
    };
    let hashes: Option<Vec<(u64, u32)>> = key.map(|(cols, truncate)| {
        let hash = |(i, r): (usize, &StoredRow)| (super::table_data::pk_hash(cols.iter().map(|c| r.cell(*c)), truncate), i as u32);
        let mut sorted: Vec<(u64, u32)> =
            if n >= 4096 { rows.par_iter().enumerate().map(hash).collect() } else { rows.iter().enumerate().map(hash).collect() };
        sorted.par_sort_unstable();
        sorted
    });
    let has_pos = rows.iter().any(|r| !r.pos.is_empty());
    let mut w = Writer(Vec::with_capacity(96 + n * 32 + body.len()));
    w.0.extend_from_slice(MAGIC);
    w.0.extend_from_slice(tbl_id.as_bytes());
    w.u64(seg.id);
    w.u64(n as u64);
    w.u64(seg.patches.len() as u64);
    w.u8(if has_pos { FLAG_POS } else { 0 } | if hashes.is_some() { FLAG_HASHES } else { 0 });
    w.u64(body.len() as u64);
    w.u64(checksum(&body));
    for v in &seg.vmin {
        w.i64(*v);
    }
    for v in vmax {
        w.i64(*v);
    }
    for r in rows {
        w.i64(r.rowid);
    }
    if has_pos {
        for r in rows {
            w.len(r.pos.len())?;
            for p in &r.pos {
                w.int(*p);
            }
        }
    }
    if let (Some(hashes), Some((cols, truncate))) = (&hashes, key) {
        w.len(cols.len())?;
        for c in cols {
            w.varint(u64::from(*c));
        }
        w.u8(u8::from(truncate));
        for (h, i) in hashes {
            w.u64(*h);
            w.u32(*i);
        }
    }
    for p in &seg.patches {
        w.u64(p.seg);
        w.u32(p.idx);
        w.i64(p.vmax);
    }
    w.0.extend_from_slice(&body);
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
pub fn decode(bytes: Vec<u8>, tbl_id: Uuid, id: u64) -> Result<(Segment, Vec<i64>)> {
    let mut r = Reader { buf: &bytes, pos: 0 };
    if r.take(8)? != MAGIC || r.take(16)? != tbl_id.as_bytes() || r.u64()? != id {
        return Err(corrupt());
    }
    let n = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    let m = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    let flags = r.u8()?;
    let body_len = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    let checksum = r.u64()?;
    if n.saturating_mul(24).saturating_add(m.saturating_mul(20)).saturating_add(body_len) > bytes.len() {
        return Err(corrupt());
    }
    let vmin = (0..n).map(|_| r.i64()).collect::<Result<Vec<_>>>()?;
    let vmax = (0..n).map(|_| r.i64()).collect::<Result<Vec<_>>>()?;
    let rowids = (0..n).map(|_| r.i64()).collect::<Result<Vec<_>>>()?;
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
    let hashes = if flags & FLAG_HASHES != 0 {
        let k = r.count()?;
        let cols = (0..k).map(|_| u32::try_from(r.varint()?).map_err(|_| corrupt())).collect::<Result<Vec<_>>>()?;
        let truncate = r.u8()? != 0;
        let sorted = (0..n)
            .map(|_| {
                let (h, i) = (r.u64()?, r.u32()?);
                if i as usize >= n { Err(corrupt()) } else { Ok((h, i)) }
            })
            .collect::<Result<Vec<_>>>()?;
        Some(KeyHashes { cols, truncate, sorted })
    } else {
        None
    };
    let patches = (0..m).map(|_| Ok(Patch { seg: r.u64()?, idx: r.u32()?, vmax: r.i64()? })).collect::<Result<Vec<_>>>()?;
    let start = r.pos;
    if bytes.len() - start != body_len {
        return Err(corrupt());
    }
    Ok((Segment::encoded(id, vmin, patches, rowids, pos, Body { bytes, start, checksum }, hashes), vmax))
}

/// The rows of a segment from its encoded body and the keys of its rows.
pub fn decode_body(body: &Body, rowids: &[i64], pos: &[Vec<i64>]) -> Result<Vec<StoredRow>> {
    use rayon::prelude::*;
    let bytes = &body.bytes[body.start..];
    if checksum(bytes) != body.checksum {
        return Err(Error::internal("table data file is corrupt (checksum mismatch)"));
    }
    let n = rowids.len();
    let key = |i: usize| (rowids[i], pos.get(i).cloned().unwrap_or_default());
    let mut r = Reader { buf: bytes, pos: 0 };
    if n < 2 * DECODE_CHUNK {
        let rows = (0..n).map(|i| r.row(key(i))).collect::<Result<Vec<_>>>()?;
        return if r.pos == bytes.len() { Ok(rows) } else { Err(corrupt()) };
    }
    // find where every chunk of rows starts (skipping is much cheaper than decoding), then decode the chunks in parallel
    let mut starts = Vec::with_capacity(n / DECODE_CHUNK + 1);
    for i in 0..n {
        if i % DECODE_CHUNK == 0 {
            starts.push(r.pos);
        }
        r.skip_row()?;
    }
    if r.pos != bytes.len() {
        return Err(corrupt());
    }
    let chunks = starts
        .par_iter()
        .enumerate()
        .map(|(k, &start)| {
            let mut cr = Reader { buf: bytes, pos: start };
            let first = k * DECODE_CHUNK;
            (first..n.min(first + DECODE_CHUNK)).map(|i| cr.row(key(i))).collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<Vec<StoredRow>>>>()?;
    Ok(chunks.into_iter().flatten().collect())
}

/// The number of rows a thread encodes or decodes at a time.
const DECODE_CHUNK: usize = 8192;

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

    /// Moves past a row without decoding it (checking only its structure; `row()` checks the rest).
    fn skip_row(&mut self) -> Result<()> {
        for _ in 0..self.count()? {
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
        }
        for _ in 0..self.count()? {
            if self.u8()? == 1 {
                self.bytes()?;
                self.bytes()?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{ArrayData, NdArray};

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
        let bytes = encode(tbl, &seg, &[7, i64::MAX], Some((&[3], false)))?;
        let (back, vmax) = decode(bytes.clone(), tbl, 9)?;
        assert_eq!(vmax, vec![7, i64::MAX]);
        assert_eq!(back.vmin, seg.vmin);
        assert_eq!(back.patches, seg.patches);
        assert_eq!(back.key(0), (42, &[1, 2][..]));
        assert_eq!(back.key(1), (43, &[][..]));
        let hashes = back.hashes().map(|h| (h.cols.clone(), h.truncate, h.sorted.len()));
        assert_eq!(hashes, Some((vec![3], false, 2)));
        let rows = back.rows();
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
        // a body that does not match its checksum is refused
        let le64 = |b: &[u8]| b.iter().rev().fold(0u64, |acc, x| (acc << 8) | u64::from(*x));
        let (body_len, sum) = (le64(&bytes[49..57]) as usize, le64(&bytes[57..65]));
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        let body = Body { start: flipped.len() - body_len, bytes: flipped, checksum: sum };
        assert!(decode_body(&body, &[42, 43], &[vec![1, 2], Vec::new()]).is_err());
        Ok(())
    }

    #[test]
    fn large_segments_decode_in_chunks() -> Result<()> {
        let tbl = Uuid::new_v4();
        let n = 3 * DECODE_CHUNK + 17;
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
        let bytes = encode(tbl, &seg, &vec![i64::MAX; n], None)?;
        let (back, _) = decode(bytes.clone(), tbl, 1)?;
        assert_eq!(back.rows().len(), n);
        for (a, b) in seg.rows().iter().zip(back.rows()) {
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
        match encode(Uuid::nil(), &seg, &[i64::MAX], None) {
            Err(Error::Pxt { code: ErrorCode::UnsupportedOperation, message }) => assert_eq!(message, json_too_deep_msg()),
            other => panic!("expected UNSUPPORTED_OPERATION, got {other:?}"),
        }
    }
}
