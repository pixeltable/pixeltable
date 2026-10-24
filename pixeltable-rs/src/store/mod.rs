//! Persistence under `$PIXELTABLE_HOME` (default `~/.pixeltable`).
//!
//! Layout (format version 1):
//!
//! ```text
//! $PIXELTABLE_HOME/
//!   pxtrs/<db>/catalog.json          directories and all table metadata (the commit point)
//!   pxtrs/<db>/tables/<id>.v<N>.bin  rows of table <id> as of version N (MessagePack of `TableData`)
//!   media/<db>/<tbl id>/...          media files written by the store (in-memory images etc.)
//! ```
//!
//! Every mutation writes the new data file (if rows changed) and then atomically replaces `catalog.json`
//! (write to a temp file, fsync, rename, fsync the directory), which names the current data file of every table. A
//! crash before the rename leaves the previous state intact; superseded data files are deleted after the rename.
//!
//! Several processes may share a store: each catalog operation holds an exclusive lock on `catalog.lock` (`lock()`),
//! and `catalog.json` starts with a stamp (`catalog_id`, `generation`; `read_stamp()`) that every commit advances,
//! so a process notices that another one committed and reloads (see `catalog::runtime`).

pub mod rows;

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path as FsPath, PathBuf};
use uuid::Uuid;

use crate::catalog::table::TableMd;
use crate::error::{Error, Result};
use crate::expr::MAX_EXPR_DEPTH;
use crate::value::{MAX_JSON_DEPTH, json_nesting_depth, parse_json_bounded};
use rows::TableData;

pub const FORMAT_NAME: &str = "pixeltable-rs";
pub const FORMAT_VERSION: u32 = 1;

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

/// The content of `catalog.json`. `catalog_id` and `generation` are serialized before the bulk of the metadata, so
/// that `Store::read_stamp()` finds them in the first bytes of the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    pub format: String,
    pub format_version: u32,
    /// identifies one catalog across its generations (a new one after `reset_db()`); nil in files written before
    /// stamps were recorded
    #[serde(default)]
    pub catalog_id: Uuid,
    /// advanced by every commit
    #[serde(default)]
    pub generation: u64,
    pub dirs: Vec<DirMd>,
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

const LOCK_FILE: &str = "catalog.lock";

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
        self.root.join("catalog.json")
    }

    fn table_path(&self, file: &str) -> PathBuf {
        self.root.join("tables").join(file)
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
        // a file written before stamps were recorded
        Ok(Some((Uuid::nil(), 0)))
    }

    pub fn load_catalog(&self) -> Result<Option<CatalogFile>> {
        let path = self.catalog_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        let cf: CatalogFile = parse_json_bounded(&bytes, MAX_CATALOG_DEPTH)
            .map_err(|e| Error::internal(format!("corrupt or unserializable metadata: {e}")))?;
        if cf.format != FORMAT_NAME || cf.format_version > FORMAT_VERSION {
            return Err(crate::pxt_err!(
                StoreUnavailable,
                "The store at {} has format {} v{}, which this version of pixeltable-rs cannot read",
                self.root.display(),
                cf.format,
                cf.format_version
            ));
        }
        Ok(Some(cf))
    }

    pub fn save_catalog(&self, cf: &CatalogFile) -> Result<()> {
        let bytes = serde_json::to_vec(cf)?;
        let depth = json_nesting_depth(&bytes);
        if depth > MAX_CATALOG_DEPTH {
            return Err(Error::internal(format!(
                "cannot save the catalog: its metadata is nested {depth} levels deep (the limit is {MAX_CATALOG_DEPTH})"
            )));
        }
        write_atomic(&self.catalog_path(), &bytes)
    }

    pub fn load_table_data(&self, file: &str) -> Result<TableData> {
        let bytes = fs::read(self.table_path(file))?;
        rmp_serde::from_slice(&bytes).map_err(|e| Error::internal(format!("corrupt table data file {file}: {e}")))
    }

    /// Writes the rows of `tbl_id` as of `version`; returns the file name to record in the metadata.
    pub fn write_table_data(&self, tbl_id: Uuid, version: i64, data: &TableData) -> Result<String> {
        let bytes = rmp_serde::to_vec(data).map_err(|e| Error::internal(format!("cannot serialize table data: {e}")))?;
        // the sequence number makes the name unique even when an operational table rewrites rows at one version
        let seq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let file = format!("{tbl_id}.v{version}.{seq}.bin");
        write_atomic(&self.table_path(&file), &bytes)?;
        Ok(file)
    }

    pub fn delete_table_data(&self, file: &str) {
        // best effort: a leftover file is garbage, not corruption
        let _ = fs::remove_file(self.table_path(file));
    }

    pub fn delete_media(&self, tbl_id: Uuid) {
        let _ = fs::remove_dir_all(self.media_dir(tbl_id));
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
    // make the rename itself durable
    if let Some(dir) = path.parent() {
        sync_dir(dir)?;
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
        // a file written before stamps were recorded
        let legacy = br#"{"format":"pixeltable-rs","format_version":1,"dirs":[],"tables":[]}"#;
        std::fs::write(store.catalog_path(), legacy)?;
        assert_eq!(store.read_stamp()?, Some((Uuid::nil(), 0)));
        assert_eq!(store.load_catalog()?.map(|c| c.generation), Some(0));
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
}
