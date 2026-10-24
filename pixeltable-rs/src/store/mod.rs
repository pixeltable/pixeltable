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
//! (write to a temp file, then rename), which names the current data file of every table. A crash before the
//! rename leaves the previous state intact; superseded data files are deleted after the rename.

pub mod rows;

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path as FsPath, PathBuf};
use uuid::Uuid;

use crate::catalog::table::TableMd;
use crate::error::{Error, Result};
use rows::TableData;

pub const FORMAT_NAME: &str = "pixeltable-rs";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirMd {
    pub id: Uuid,
    pub parent_id: Option<Uuid>,
    pub name: String,
    pub created_at: f64,
}

/// The content of `catalog.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    pub format: String,
    pub format_version: u32,
    pub dirs: Vec<DirMd>,
    pub tables: Vec<TableMd>,
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

    pub fn load_catalog(&self) -> Result<Option<CatalogFile>> {
        let path = self.catalog_path();
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        let cf: CatalogFile = serde_json::from_slice(&bytes)?;
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

    /// Deletes everything in this database (test hook `reset_db()`).
    pub fn wipe(&self) -> Result<()> {
        if self.root.exists() {
            fs::remove_dir_all(&self.root)?;
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
    Ok(())
}
