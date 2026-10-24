//! Persistence under `$PIXELTABLE_HOME` (default `~/.pixeltable`).
//!
//! Layout (format version 2):
//!
//! ```text
//! $PIXELTABLE_HOME/
//!   pxtrs/<db>/catalog.json                 directories, table metadata and table manifests (the commit point)
//!   pxtrs/<db>/catalog.lock                 the store lock
//!   pxtrs/<db>/tables/<tbl id>.<id>.seg     a segment of a table's rows (binary, see `codec`)
//!   pxtrs/<db>/tables/<tbl id>.<id>.hist    a table's version history (JSON lines, see `history`)
//!   media/<db>/<tbl id>/...                 media files written by the store (in-memory images etc.)
//! ```
//!
//! A table's rows are a list of immutable segments (`table_data::Segment`): every commit that changes rows writes
//! one new segment with the rows it adds and the `v_max` changes it makes to older rows; compaction replaces
//! segments by merged ones. A table's version history (`VersionMd`/`SchemaVersionMd` records) is appended to its
//! history file. `catalog.json` holds the current metadata of every table without its history, plus its manifest
//! (`TableStorage`: the segment list and the valid length of the history file), so its size does not grow with
//! rows or versions.
//!
//! A commit writes the new segment files (fsync), appends to the history files (fsync), fsyncs `tables/`, and then
//! atomically replaces `catalog.json` (temp file, fsync, rename, fsync of the directory): the rename is the commit
//! point. A crash or error before it leaves the previous state intact: new segment files are not referenced yet
//! (`gc()` removes them), and history bytes past the recorded length are ignored and overwritten. Superseded files
//! are deleted after the rename.
//!
//! Several processes may share a store: each catalog operation holds an exclusive lock on `catalog.lock` (`lock()`),
//! and `catalog.json` starts with a stamp (`catalog_id`, `generation`; `read_stamp()`) that every commit advances,
//! so a process notices that another one committed and reloads (see `catalog::runtime`).

pub mod codec;
pub mod history;
pub mod rows;
pub mod table_data;

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path as FsPath, PathBuf};
use uuid::Uuid;

use crate::catalog::table::TableMd;
use crate::error::{Error, Result};
use crate::expr::MAX_EXPR_DEPTH;
use crate::value::{MAX_JSON_DEPTH, json_nesting_depth, parse_json_bounded};
use table_data::Segment;

pub const FORMAT_NAME: &str = "pixeltable-rs";
pub const FORMAT_VERSION: u32 = 2;

/// Maximum nesting of `catalog.json` and of serialized expressions and queries: room for an expression of
/// `MAX_EXPR_DEPTH` levels (two JSON levels each) holding a JSON literal of `MAX_JSON_DEPTH` levels. A commit that
/// would exceed it is refused and reads use the same limit, so whatever is committed can be loaded again.
pub const MAX_CATALOG_DEPTH: usize = 2 * MAX_EXPR_DEPTH + MAX_JSON_DEPTH + 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirMd {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub name: String,
    pub created_at: f64,
}

/// Where a table's data is: its manifest, recorded in `catalog.json` with the table's metadata.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TableStorage {
    /// the table's segments, in order
    #[serde(default)]
    pub segments: Vec<u64>,
    /// the table's history file (None until the first commit writes it)
    #[serde(default)]
    pub history: Option<HistoryRef>,
}

/// A history file and how much of it is valid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HistoryRef {
    pub file: u64,
    /// the committed length in bytes (later bytes are left over from an operation that did not commit)
    pub len: u64,
    /// the number of records in the committed part
    pub records: u64,
}

/// The content of `catalog.json`. `catalog_id` and `generation` are serialized before the bulk of the metadata, so
/// that `Store::read_stamp()` finds them in the first bytes of the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    pub format: String,
    pub format_version: u32,
    /// identifies one catalog across its generations (a new one after `reset_db()`)
    #[serde(default)]
    pub catalog_id: Uuid,
    /// advanced by every commit
    #[serde(default)]
    pub generation: u64,
    pub dirs: Vec<DirMd>,
    /// the tables' metadata (without their history, which is in their history files)
    pub tables: Vec<TableMd>,
}

/// What identifies a committed state of the catalog: `(catalog_id, generation)`.
pub type CatalogStamp = (Uuid, u64);

#[derive(Deserialize)]
struct CatalogHeader {
    #[serde(default)]
    catalog_id: Uuid,
    #[serde(default)]
    generation: u64,
}

/// Only the format of `catalog.json`, read before the rest so that a store of another format gets a clear error.
#[derive(Deserialize)]
struct FormatHeader {
    format: String,
    format_version: u32,
}

#[cfg(test)]
thread_local! {
    /// Test hook: makes `save_catalog_bytes()` fail on this thread, as if the process failed after writing a commit's
    /// segment and history files but before the commit point.
    pub static FAIL_BEFORE_COMMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

const LOCK_FILE: &str = "catalog.lock";
const CATALOG_FILE: &str = "catalog.json";

/// The exclusive lock on a store (an OS file lock on `catalog.lock`, so it also excludes other processes); released
/// when dropped.
pub struct StoreLock {
    _file: fs::File,
}

#[derive(Debug, Clone)]
pub struct Store {
    home: PathBuf,
    root: PathBuf,
    media_root: PathBuf,
}

/// `$PIXELTABLE_HOME`, defaulting to `~/.pixeltable`.
pub fn pixeltable_home() -> PathBuf {
    if let Ok(h) = std::env::var("PIXELTABLE_HOME") {
        if !h.is_empty() {
            return PathBuf::from(h);
        }
    }
    let user_home = std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
    user_home.join(".pixeltable")
}

fn db_name() -> String {
    match std::env::var("PIXELTABLE_DB") {
        Ok(db) if !db.is_empty() => db,
        _ => "pixeltable".to_string(),
    }
}

/// Runs `f` (file I/O, (de)serialization) detached from the Python interpreter if this thread is attached to it,
/// so that other Python threads run meanwhile.
pub fn without_gil<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    // SAFETY: plain queries of the interpreter state; PyGILState_Check() is only called on an initialized interpreter
    let attached = unsafe { pyo3::ffi::Py_IsInitialized() != 0 && pyo3::ffi::PyGILState_Check() == 1 };
    let mut f = Some(f);
    if attached {
        if let Some(r) = pyo3::Python::try_attach(|py| py.detach(|| f.take().map(|f| f()))) {
            if let Some(r) = r {
                return r;
            }
        }
    }
    match f.take() {
        Some(f) => f(),
        None => unreachable!("the closure ran but returned no result"),
    }
}

impl Store {
    /// Opens (creating if needed) the store of the configured database.
    pub fn open_default() -> Result<Store> {
        Store::open(&pixeltable_home(), &db_name())
    }

    pub fn open(home: &FsPath, db: &str) -> Result<Store> {
        let root = home.join("pxtrs").join(db);
        let media_root = home.join("media").join(db);
        fs::create_dir_all(root.join("tables"))?;
        fs::create_dir_all(&media_root)?;
        Ok(Store { home: home.to_path_buf(), root, media_root })
    }

    pub fn home(&self) -> &FsPath {
        &self.home
    }

    pub fn media_dir(&self, tbl_id: Uuid) -> PathBuf {
        self.media_root.join(tbl_id.to_string())
    }

    fn catalog_path(&self) -> PathBuf {
        self.root.join(CATALOG_FILE)
    }

    fn tables_dir(&self) -> PathBuf {
        self.root.join("tables")
    }

    pub fn segment_file_name(tbl_id: Uuid, id: u64) -> String {
        format!("{tbl_id}.{id:016x}.seg")
    }

    pub fn history_file_name(tbl_id: Uuid, id: u64) -> String {
        format!("{tbl_id}.{id:016x}.hist")
    }

    /// Waits for the exclusive lock on this store.
    pub fn lock(&self) -> Result<StoreLock> {
        let path = self.root.join(LOCK_FILE);
        let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
        file.lock()?;
        Ok(StoreLock { _file: file })
    }

    /// The stamp of the committed catalog (None if there is none yet), read from the first bytes of `catalog.json`.
    pub fn read_stamp(&self) -> Result<Option<CatalogStamp>> {
        use std::io::Read;
        let f = match fs::File::open(self.catalog_path()) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let mut head = Vec::with_capacity(512);
        f.take(512).read_to_end(&mut head)?;
        const KEY: &[u8] = b"\"generation\":";
        if let Some(pos) = head.windows(KEY.len()).position(|w| w == KEY) {
            let start = pos + KEY.len();
            let end = start + head[start..].iter().take_while(|b| b.is_ascii_digit()).count();
            if end > start && end < head.len() {
                let mut text = head[..end].to_vec();
                text.push(b'}');
                if let Ok(h) = serde_json::from_slice::<CatalogHeader>(&text) {
                    return Ok(Some((h.catalog_id, h.generation)));
                }
            }
        }
        // not a catalog this version wrote; `load_catalog()` says why
        Ok(Some((Uuid::nil(), 0)))
    }

    /// Reads `catalog.json` (None if the store is new). The tables' metadata comes without history (see
    /// `history::read()`). A store written in another format is refused.
    pub fn load_catalog(&self) -> Result<Option<CatalogFile>> {
        let path = self.catalog_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        let header: FormatHeader = parse_json_bounded(&bytes, MAX_CATALOG_DEPTH)
            .map_err(|e| Error::internal(format!("corrupt or unserializable metadata: {e}")))?;
        if header.format != FORMAT_NAME || header.format_version != FORMAT_VERSION {
            return Err(crate::pxt_err!(
                StoreUnavailable,
                "The store at {} has format {} v{}, which this version of pixeltable-rs (format v{FORMAT_VERSION}) cannot read; \
                 remove that directory (and {}) to start over",
                self.root.display(),
                header.format,
                header.format_version,
                self.media_root.display()
            ));
        }
        let cf: CatalogFile = parse_json_bounded(&bytes, MAX_CATALOG_DEPTH)
            .map_err(|e| Error::internal(format!("corrupt or unserializable metadata: {e}")))?;
        Ok(Some(cf))
    }

    /// Serializes `catalog.json` (refusing metadata that would be too deeply nested to load again).
    pub fn serialize_catalog(cf: &CatalogFile) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(cf)?;
        let depth = json_nesting_depth(&bytes);
        if depth > MAX_CATALOG_DEPTH {
            return Err(Error::internal(format!(
                "cannot save the catalog: its metadata is nested {depth} levels deep (the limit is {MAX_CATALOG_DEPTH})"
            )));
        }
        Ok(bytes)
    }

    /// Atomically replaces `catalog.json` (the commit point). Returns once the new file has replaced the old one;
    /// the directory is synced after that, as far as possible.
    pub fn save_catalog_bytes(&self, bytes: &[u8]) -> Result<()> {
        #[cfg(test)]
        if FAIL_BEFORE_COMMIT.with(|f| f.get()) {
            return Err(Error::internal("injected failure before the commit point"));
        }
        write_atomic(&self.catalog_path(), bytes)
    }

    pub fn save_catalog(&self, cf: &CatalogFile) -> Result<()> {
        self.save_catalog_bytes(&Store::serialize_catalog(cf)?)
    }

    /// Writes a segment of table `tbl_id` with the given `v_max` of its rows (fsynced; the directory is synced by
    /// `sync_tables()`). Returns the file name.
    pub fn write_segment(&self, tbl_id: Uuid, seg: &Segment, vmax: &[i64]) -> Result<String> {
        let bytes = codec::encode(tbl_id, seg, vmax)?;
        let name = Store::segment_file_name(tbl_id, seg.id);
        let mut f = fs::File::create(self.tables_dir().join(&name))?;
        f.write_all(&bytes)?;
        f.sync_data()?;
        Ok(name)
    }

    /// Reads segment `id` of table `tbl_id`: the segment and the `v_max` its file records.
    pub fn read_segment(&self, tbl_id: Uuid, id: u64) -> Result<(Segment, Vec<i64>)> {
        let name = Store::segment_file_name(tbl_id, id);
        let bytes = fs::read(self.tables_dir().join(&name))
            .map_err(|e| Error::internal(format!("cannot read table data file {name}: {e}")))?;
        codec::decode(&bytes, tbl_id, id).map_err(|e| Error::internal(format!("{} {name}", e.message())))
    }

    /// Appends `bytes` to history file `file` of table `tbl_id` at offset `len` (dropping whatever follows the
    /// committed part), fsynced. Returns the new length.
    pub fn append_history(&self, tbl_id: Uuid, file: u64, len: u64, bytes: &[u8]) -> Result<u64> {
        let path = self.tables_dir().join(Store::history_file_name(tbl_id, file));
        let mut f = fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
        f.set_len(len)?;
        f.seek(SeekFrom::Start(len))?;
        f.write_all(bytes)?;
        f.sync_data()?;
        Ok(len + bytes.len() as u64)
    }

    /// The committed part of a history file.
    pub fn read_history(&self, tbl_id: Uuid, h: &HistoryRef) -> Result<Vec<u8>> {
        use std::io::Read;
        let name = Store::history_file_name(tbl_id, h.file);
        let f = fs::File::open(self.tables_dir().join(&name))
            .map_err(|e| Error::internal(format!("cannot read history file {name}: {e}")))?;
        let mut out = Vec::with_capacity(h.len as usize);
        f.take(h.len).read_to_end(&mut out)?;
        if out.len() as u64 != h.len {
            return Err(Error::internal(format!("history file {name} is shorter than its committed length")));
        }
        Ok(out)
    }

    /// Bytes `from..to` of a history file (records a commit appended after `from`).
    pub fn read_history_range(&self, tbl_id: Uuid, file: u64, from: u64, to: u64) -> Result<Vec<u8>> {
        use std::io::Read;
        let name = Store::history_file_name(tbl_id, file);
        let mut f = fs::File::open(self.tables_dir().join(&name))
            .map_err(|e| Error::internal(format!("cannot read history file {name}: {e}")))?;
        f.seek(SeekFrom::Start(from))?;
        let mut out = Vec::with_capacity((to - from) as usize);
        f.take(to - from).read_to_end(&mut out)?;
        if out.len() as u64 != to - from {
            return Err(Error::internal(format!("history file {name} is shorter than its committed length")));
        }
        Ok(out)
    }

    /// Makes the creation of the files written in `tables/` durable.
    pub fn sync_tables(&self) -> Result<()> {
        sync_dir(&self.tables_dir())
    }

    /// Deletes a file in `tables/` (best effort: a leftover file is garbage, not corruption).
    pub fn delete_table_file(&self, name: &str) {
        let _ = fs::remove_file(self.tables_dir().join(name));
    }

    /// Deletes the files in `tables/` that `keep` does not name (left over from operations that failed or crashed
    /// before committing, or superseded files a crash kept from being deleted), and leftover temp files. The caller
    /// holds the store lock, so no other process is writing files that it has yet to commit.
    pub fn gc(&self, keep: &HashSet<String>) -> Result<()> {
        for entry in fs::read_dir(self.tables_dir())? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !keep.contains(&name) {
                let _ = fs::remove_file(entry.path());
            }
        }
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("catalog.tmp") {
                let _ = fs::remove_file(entry.path());
            }
        }
        Ok(())
    }

    pub fn delete_media(&self, tbl_id: Uuid) {
        let _ = fs::remove_dir_all(self.media_dir(tbl_id));
    }

    /// Deletes the media files written for table version `version` (named `<tbl id>_<col id>_<version>_<id>...`,
    /// see `media::save_object()`).
    pub fn delete_media_version(&self, tbl_id: Uuid, version: i64) {
        let Ok(entries) = fs::read_dir(self.media_dir(tbl_id)) else { return };
        let prefix = format!("{}_", tbl_id.simple());
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(rest) = name.strip_prefix(&prefix) else { continue };
            let mut parts = rest.splitn(3, '_');
            let _col = parts.next();
            if parts.next() == Some(version.to_string().as_str()) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    /// Deletes everything in this database (test hook `reset_db()`); the caller holds the lock, whose file stays.
    pub fn wipe(&self) -> Result<()> {
        if self.root.exists() {
            for entry in fs::read_dir(&self.root)? {
                let entry = entry?;
                if entry.file_name() == LOCK_FILE {
                    continue;
                }
                if entry.file_type()?.is_dir() {
                    fs::remove_dir_all(entry.path())?;
                } else {
                    fs::remove_file(entry.path())?;
                }
            }
        }
        if self.media_root.exists() {
            fs::remove_dir_all(&self.media_root)?;
        }
        fs::create_dir_all(self.root.join("tables"))?;
        fs::create_dir_all(&self.media_root)?;
        Ok(())
    }
}

fn write_atomic(path: &FsPath, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_data()?;
    }
    fs::rename(&tmp, path)?;
    // make the rename itself durable; it has happened, so a failure here does not undo the commit
    if let Some(dir) = path.parent() {
        let _ = sync_dir(dir);
    }
    Ok(())
}

#[cfg(unix)]
fn sync_dir(dir: &FsPath) -> Result<()> {
    fs::File::open(dir)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_dir(_dir: &FsPath) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rows::StoredRow;
    use table_data::Patch;

    fn temp_store() -> Store {
        let tmp = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test-tmp");
        let dir = tmp.join(Uuid::new_v4().to_string());
        Store::open(&dir, "test").unwrap_or_else(|e| panic!("cannot create test store: {e}"))
    }

    #[test]
    fn stamp_is_read_from_the_header() -> Result<()> {
        let store = temp_store();
        assert_eq!(store.read_stamp()?, None);
        let id = Uuid::new_v4();
        let cf = CatalogFile {
            format: FORMAT_NAME.into(),
            format_version: FORMAT_VERSION,
            catalog_id: id,
            generation: 42,
            dirs: Vec::new(),
            tables: Vec::new(),
        };
        store.save_catalog(&cf)?;
        assert_eq!(store.read_stamp()?, Some((id, 42)));
        assert_eq!(store.load_catalog()?.map(|c| c.generation), Some(42));
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }

    #[test]
    fn older_formats_are_refused() -> Result<()> {
        let store = temp_store();
        let v1 = br#"{"format":"pixeltable-rs","format_version":1,"catalog_id":"00000000-0000-0000-0000-000000000000","generation":3,"dirs":[],"tables":[]}"#;
        std::fs::write(store.catalog_path(), v1)?;
        match store.load_catalog() {
            Err(Error::Pxt { code: crate::error::ErrorCode::StoreUnavailable, message }) => {
                assert!(message.contains("format pixeltable-rs v1"), "{message}");
            }
            other => panic!("expected StoreUnavailable, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }

    #[test]
    fn wipe_keeps_the_lock_file() -> Result<()> {
        let store = temp_store();
        let _lock = store.lock()?;
        std::fs::write(store.catalog_path(), b"{}")?;
        store.wipe()?;
        assert!(!store.catalog_path().exists());
        assert!(store.root.join(LOCK_FILE).exists());
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }

    #[test]
    fn segments_and_history_roundtrip() -> Result<()> {
        let store = temp_store();
        let tbl = Uuid::new_v4();
        let mut r = StoredRow::new(3, Vec::new());
        r.set_cell(1, crate::value::Value::String("x".into()));
        let seg = Segment { id: 77, rows: vec![r], vmin: vec![2], patches: vec![Patch { seg: 5, idx: 1, vmax: 2 }] };
        store.write_segment(tbl, &seg, &[rows::LIVE])?;
        let (back, vmax) = store.read_segment(tbl, 77)?;
        assert_eq!(back.rows.len(), 1);
        assert_eq!(back.vmin, vec![2]);
        assert_eq!(vmax, vec![rows::LIVE]);
        assert_eq!(back.patches, seg.patches);
        let len = store.append_history(tbl, 9, 0, b"abc\n")?;
        // bytes past the committed length are dropped by the next append
        store.append_history(tbl, 9, len, b"garbage\n")?;
        let len2 = store.append_history(tbl, 9, len, b"def\n")?;
        assert_eq!(store.read_history(tbl, &HistoryRef { file: 9, len: len2, records: 2 })?, b"abc\ndef\n");
        let keep: HashSet<String> = [Store::segment_file_name(tbl, 77)].into_iter().collect();
        store.gc(&keep)?;
        assert!(store.read_segment(tbl, 77).is_ok());
        assert!(store.read_history(tbl, &HistoryRef { file: 9, len: 4, records: 1 }).is_err());
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }
}
