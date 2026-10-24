//! The binary encoding of segment files.
//!
//! ```text
//! magic "PXTSEG" 0x02 0x00 | table id (16) | segment id (u64) | rows (u64) | patches (u64)
//! v_min (i64 per row) | v_max (i64 per row) | patches (segment u64, row u32, v_max i64 each)
//! rows: rowid (int) | positions (count, int each) | cells (count, value each) | errors (count, each 0 or
//!       1 + type + message)
//! value: tag (u8) + payload: 0 NULL, 1 false, 2 true, 3 Int (int), 4 Float (f64), 5 String, 6 Timestamp (int),
//!        7 Date (int), 8 UUID (16), 9 Binary, 10 JSON (text), 11 Array (MessagePack of NdArray), 12 Media (path/URL)
//! counts and lengths: LEB128; int: zigzag LEB128; strings and byte strings: length + bytes; fixed-size numbers are
//! little-endian
//! ```
//!
//! JSON values are written as text, refusing values nested more deeply than `value::MAX_JSON_DEPTH` (so every stored
//! value can be read back) and read with the same limit. Large segments are encoded and decoded in chunks of
//! `DECODE_CHUNK` rows in parallel (rayon); decoding first skips through the rows to find where the chunks start.

use uuid::Uuid;

use super::rows::{CellError, StoredRow};
use super::table_data::{Patch, Segment};
use crate::error::{Error, ErrorCode, Result};
use crate::value::{MAX_JSON_DEPTH, Value, json_nesting_depth, json_too_deep_msg, parse_json_bounded};

const MAGIC: &[u8; 8] = b"PXTSEG\x02\x00";

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

/// Encodes segment `seg` of table `tbl_id`, with `vmax` as the `v_max` of its rows.
pub fn encode(tbl_id: Uuid, seg: &Segment, vmax: &[i64]) -> Result<Vec<u8>> {
    let mut w = Writer(Vec::with_capacity(64 + seg.rows.len() * 64));
    w.0.extend_from_slice(MAGIC);
    w.0.extend_from_slice(tbl_id.as_bytes());
    w.u64(seg.id);
    w.u64(seg.rows.len() as u64);
    w.u64(seg.patches.len() as u64);
    for v in &seg.vmin {
        w.i64(*v);
    }
    for v in vmax {
        w.i64(*v);
    }
    for p in &seg.patches {
        w.u64(p.seg);
        w.u32(p.idx);
        w.i64(p.vmax);
    }
    if seg.rows.len() < 2 * DECODE_CHUNK {
        for r in &seg.rows {
            w.row(r)?;
        }
    } else {
        use rayon::prelude::*;
        let parts = seg
            .rows
            .par_chunks(DECODE_CHUNK)
            .map(|rows| {
                let mut pw = Writer(Vec::with_capacity(rows.len() * 64));
                rows.iter().try_for_each(|r| pw.row(r))?;
                Ok(pw.0)
            })
            .collect::<Result<Vec<Vec<u8>>>>()?;
        for p in parts {
            w.0.extend_from_slice(&p);
        }
    }
    Ok(w.0)
}

impl Writer {
    fn row(&mut self, r: &StoredRow) -> Result<()> {
        self.int(r.rowid);
        self.len(r.pos.len())?;
        for p in &r.pos {
            self.int(*p);
        }
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

/// Decodes segment `id` of table `tbl_id`: the segment and the `v_max` its file records.
pub fn decode(bytes: &[u8], tbl_id: Uuid, id: u64) -> Result<(Segment, Vec<i64>)> {
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.take(8)? != MAGIC || r.take(16)? != tbl_id.as_bytes() || r.u64()? != id {
        return Err(corrupt());
    }
    let n = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    let m = usize::try_from(r.u64()?).map_err(|_| corrupt())?;
    if n.saturating_mul(16).saturating_add(m.saturating_mul(20)) > bytes.len() {
        return Err(corrupt());
    }
    let vmin = (0..n).map(|_| r.i64()).collect::<Result<Vec<_>>>()?;
    let vmax = (0..n).map(|_| r.i64()).collect::<Result<Vec<_>>>()?;
    let patches = (0..m).map(|_| Ok(Patch { seg: r.u64()?, idx: r.u32()?, vmax: r.i64()? })).collect::<Result<Vec<_>>>()?;
    let rows = if n < 2 * DECODE_CHUNK {
        let rows = (0..n).map(|_| r.row()).collect::<Result<Vec<_>>>()?;
        if r.pos != bytes.len() {
            return Err(corrupt());
        }
        rows
    } else {
        // find where every chunk of rows starts (skipping is much cheaper than decoding), then decode the chunks in
        // parallel
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
        use rayon::prelude::*;
        let chunks = starts
            .par_iter()
            .enumerate()
            .map(|(k, &start)| {
                let mut cr = Reader { buf: bytes, pos: start };
                let count = DECODE_CHUNK.min(n - k * DECODE_CHUNK);
                (0..count).map(|_| cr.row()).collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<Vec<StoredRow>>>>()?;
        chunks.into_iter().flatten().collect()
    };
    Ok((Segment { id, rows, vmin, patches }, vmax))
}

/// The number of rows a thread encodes or decodes at a time.
const DECODE_CHUNK: usize = 8192;

impl Reader<'_> {
    fn row(&mut self) -> Result<StoredRow> {
        let rowid = self.int()?;
        let npos = self.count()?;
        let pos = (0..npos).map(|_| self.int()).collect::<Result<Vec<_>>>()?;
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
        self.varint()?;
        for _ in 0..self.count()? {
            self.varint()?;
        }
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
        let seg = Segment { id: 9, rows: vec![row, StoredRow::new(43, Vec::new())], vmin: vec![1, 2], patches: vec![Patch { seg: 3, idx: 4, vmax: 5 }] };
        let bytes = encode(tbl, &seg, &[7, i64::MAX])?;
        let (back, vmax) = decode(&bytes, tbl, 9)?;
        assert_eq!(vmax, vec![7, i64::MAX]);
        assert_eq!(back.vmin, seg.vmin);
        assert_eq!(back.patches, seg.patches);
        assert_eq!(back.rows[0].pos, vec![1, 2]);
        assert_eq!(back.rows[1].rowid, 43);
        for (i, v) in values.iter().enumerate() {
            let got = back.rows[0].cell(i as u32);
            assert!(crate::store::table_data::values_identical(got, v), "{v:?} came back as {got:?}");
        }
        assert_eq!(back.rows[0].error(3).map(|e| e.errormsg.as_str()), Some("bad"));
        // truncated or altered files are refused, never panic
        for cut in [0, 10, bytes.len() / 2, bytes.len() - 1] {
            assert!(decode(&bytes[..cut], tbl, 9).is_err());
        }
        assert!(decode(&bytes, tbl, 10).is_err());
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
        let seg = Segment { id: 1, rows, vmin: vec![1; n], patches: Vec::new() };
        let bytes = encode(tbl, &seg, &vec![i64::MAX; n])?;
        let (back, _) = decode(&bytes, tbl, 1)?;
        assert_eq!(back.rows.len(), n);
        for (a, b) in seg.rows.iter().zip(&back.rows) {
            assert_eq!(a.rowid, b.rowid);
            assert_eq!(a.pos, b.pos);
            assert!(a.cells.iter().zip(&b.cells).all(|(x, y)| crate::store::table_data::values_identical(x, y)));
            assert_eq!(a.errors, b.errors);
        }
        assert!(decode(&bytes[..bytes.len() - 3], tbl, 1).is_err());
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
        let seg = Segment { id: 1, rows: vec![row], vmin: vec![0], patches: Vec::new() };
        match encode(Uuid::nil(), &seg, &[i64::MAX]) {
            Err(Error::Pxt { code: ErrorCode::UnsupportedOperation, message }) => assert_eq!(message, json_too_deep_msg()),
            other => panic!("expected UNSUPPORTED_OPERATION, got {other:?}"),
        }
    }
}
