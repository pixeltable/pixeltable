//! The catalog: directories and tables, their metadata, and the row data of every table.
//!
//! `Catalog` is the in-memory state of one database. It is loaded from the store on first use and every mutation
//! is persisted before the operation returns (see `store`). The process-wide instance lives in `runtime`.
//!
//! Operations are split by concern: `dirs.rs` (directory operations and listings), `tables.rs` (create, get, drop,
//! metadata), `views.rs` (views and snapshots); data mutations (insert, ...) are in `exec`.

pub mod dirs;
pub mod indexes;
pub mod path;
pub mod runtime;
pub mod table;
pub mod tables;
pub mod versions;
pub mod views;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::store::history::{self, History};
use crate::store::rows::TableData;
use crate::store::{CatalogFile, CatalogStamp, DirMd, FORMAT_NAME, FORMAT_VERSION, HistoryRef, Store, without_gil};
use path::Path;
use table::{ColId, TableMd};

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// What a name in a directory refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entry {
    Dir(Uuid),
    Table(Uuid),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfExists {
    Error,
    Ignore,
    Replace,
    ReplaceForce,
}

impl IfExists {
    pub fn parse(s: &str, param_name: &str) -> Result<IfExists> {
        match s.to_ascii_uppercase().as_str() {
            "ERROR" => Ok(IfExists::Error),
            "IGNORE" => Ok(IfExists::Ignore),
            "REPLACE" => Ok(IfExists::Replace),
            "REPLACE_FORCE" => Ok(IfExists::ReplaceForce),
            _ => crate::bail!(InvalidArgument, "{param_name} must be one of: ['error', 'ignore', 'replace', 'replace_force']"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfNotExists {
    Error,
    Ignore,
}

impl IfNotExists {
    pub fn parse(s: &str, param_name: &str) -> Result<IfNotExists> {
        match s.to_ascii_uppercase().as_str() {
            "ERROR" => Ok(IfNotExists::Error),
            "IGNORE" => Ok(IfNotExists::Ignore),
            _ => crate::bail!(InvalidArgument, "{param_name} must be one of: ['error', 'ignore']"),
        }
    }
}

/// A file-level effect of an operation, applied only once the operation has succeeded.
#[derive(Debug, Clone)]
enum PendingDelete {
    /// a file in the store's `tables/` directory (a superseded segment or history file, or one of a dropped table)
    TableFile(String),
    Media(Uuid),
    /// the media files a table version wrote (revert)
    MediaVersion(Uuid, i64),
}

pub struct Catalog {
    store: Store,
    /// the stamp of the committed state this catalog reflects (see `store::CatalogFile`)
    catalog_id: Uuid,
    generation: u64,
    root_id: Uuid,
    dirs: HashMap<Uuid, DirMd>,
    tables: HashMap<Uuid, Arc<TableMd>>,
    /// row data, loaded on first access
    data: HashMap<Uuid, Arc<TableData>>,
    /// the metadata of each table as last written or loaded; `flush()` appends the history records that differ
    persisted: HashMap<Uuid, Arc<TableMd>>,
    /// tables whose rows the current operation changed (`stage_data()`), written by `flush()`
    staged: HashSet<Uuid>,
    /// true once the current operation has committed (`commit()`), so `flush()` writes `catalog.json`
    dirty: bool,
    /// files to delete when the current operation succeeds
    pending_deletes: Vec<PendingDelete>,
    /// media files written for the current operation (deleted if it fails)
    written_media: Vec<std::path::PathBuf>,
}

/// The in-memory state at the start of an operation (see `Catalog::checkpoint()`).
pub struct Checkpoint {
    dirs: HashMap<Uuid, DirMd>,
    tables: HashMap<Uuid, Arc<TableMd>>,
    data: HashMap<Uuid, Arc<TableData>>,
}

impl Catalog {
    /// Loads the catalog from the store, initializing an empty one if the store is new. With `gc`, also deletes the
    /// files in the store that the catalog does not reference (see `Store::gc()`). The caller holds the store lock.
    pub fn load(store: Store, gc: bool) -> Result<Catalog> {
        let Some(cf) = store.load_catalog()? else {
            let root = DirMd { id: Uuid::new_v4(), parent_id: None, name: String::new(), created_at: now() };
            let mut cat = Catalog::new(store, Uuid::new_v4(), 0, root.id, HashMap::from([(root.id, root)]), HashMap::new());
            cat.commit()?;
            cat.flush()?;
            return Ok(cat);
        };
        let root_id = cf
            .dirs
            .iter()
            .find(|d| d.parent_id.is_none())
            .ok_or_else(|| Error::internal("catalog has no root directory"))?
            .id;
        let mut tables = HashMap::with_capacity(cf.tables.len());
        for mut md in cf.tables {
            let h = md.storage.history.ok_or_else(|| Error::internal(format!("table {} has no version history", md.id)))?;
            let records = history::parse(&store.read_history(md.id, &h)?)?;
            let mut hist = History::default();
            history::replay(&mut hist, records, md.current_version);
            md.versions = hist.versions;
            md.schema_versions = hist.schema_versions;
            tables.insert(md.id, Arc::new(md));
        }
        let dirs = cf.dirs.into_iter().map(|d| (d.id, d)).collect();
        let cat = Catalog::new(store, cf.catalog_id, cf.generation, root_id, dirs, tables);
        if gc {
            let mut keep = HashSet::new();
            for md in cat.tables.values() {
                keep.extend(table_files(md));
            }
            cat.store.gc(&keep)?;
        }
        Ok(cat)
    }

    /// Loads the catalog that another process committed, reusing what `previous` (this process's catalog, between
    /// operations) holds that is still valid: the history of a table whose history file only grew, and the rows of
    /// a table whose segment list only grew (segments are immutable; the new ones are read and their patches
    /// applied).
    pub fn reload(store: Store, previous: Catalog) -> Result<Catalog> {
        let Some(cf) = store.load_catalog()? else { return Catalog::load(store, false) };
        let root_id = cf
            .dirs
            .iter()
            .find(|d| d.parent_id.is_none())
            .ok_or_else(|| Error::internal("catalog has no root directory"))?
            .id;
        let same_catalog = cf.catalog_id == previous.catalog_id;
        let mut tables = HashMap::with_capacity(cf.tables.len());
        let mut data = HashMap::new();
        for mut md in cf.tables {
            let h = md.storage.history.ok_or_else(|| Error::internal(format!("table {} has no version history", md.id)))?;
            let old = previous.persisted.get(&md.id).filter(|_| same_catalog);
            let mut hist = History::default();
            let records = match old.and_then(|o| o.storage.history.map(|oh| (o, oh))) {
                Some((o, oh)) if oh.file == h.file && oh.len <= h.len => {
                    hist.versions = o.versions.clone();
                    hist.schema_versions = o.schema_versions.clone();
                    history::parse(&store.read_history_range(md.id, h.file, oh.len, h.len)?)?
                }
                _ => history::parse(&store.read_history(md.id, &h)?)?,
            };
            history::replay(&mut hist, records, md.current_version);
            md.versions = hist.versions;
            md.schema_versions = hist.schema_versions;
            if let (Some(o), Some(d)) = (old, previous.data.get(&md.id)) {
                let (old_segs, new_segs) = (&o.storage.segments, &md.storage.segments);
                if new_segs.starts_with(old_segs) {
                    let more = new_segs[old_segs.len()..]
                        .iter()
                        .map(|id| store.read_segment(md.id, *id))
                        .collect::<Result<Vec<_>>>()?;
                    data.insert(md.id, if more.is_empty() { d.clone() } else { Arc::new(d.extended(more)) });
                }
            }
            tables.insert(md.id, Arc::new(md));
        }
        let dirs = cf.dirs.into_iter().map(|d| (d.id, d)).collect();
        let mut cat = Catalog::new(store, cf.catalog_id, cf.generation, root_id, dirs, tables);
        cat.data = data;
        Ok(cat)
    }

    fn new(
        store: Store,
        catalog_id: Uuid,
        generation: u64,
        root_id: Uuid,
        dirs: HashMap<Uuid, DirMd>,
        tables: HashMap<Uuid, Arc<TableMd>>,
    ) -> Catalog {
        Catalog {
            store,
            catalog_id,
            generation,
            root_id,
            dirs,
            persisted: tables.clone(),
            tables,
            data: HashMap::new(),
            staged: HashSet::new(),
            dirty: false,
            pending_deletes: Vec::new(),
            written_media: Vec::new(),
        }
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// The stamp of the committed state this in-memory catalog reflects.
    pub fn stamp(&self) -> CatalogStamp {
        (self.catalog_id, self.generation)
    }

    // ---------------------------------------------------------------------------------------- atomicity
    //
    // Every operation runs between `checkpoint()` and `finish()` (see `runtime::with_catalog`). Its changes stay in
    // memory (`commit()` and `stage_data()` only mark what to write) until `finish()`, which on success writes them
    // (`flush()`) and then applies the deferred file deletions; on failure, or if writing fails, the in-memory state
    // is restored and the files the operation wrote are removed. File deletions are therefore never done directly:
    // use `defer_delete_*`.

    pub fn checkpoint(&mut self) -> Checkpoint {
        self.pending_deletes.clear();
        self.written_media.clear();
        self.staged.clear();
        self.dirty = false;
        Checkpoint { dirs: self.dirs.clone(), tables: self.tables.clone(), data: self.data.clone() }
    }

    pub fn finish(&mut self, checkpoint: Checkpoint, succeeded: bool) -> Result<()> {
        if succeeded {
            match self.flush() {
                Ok(()) => {
                    let deletes = std::mem::take(&mut self.pending_deletes);
                    let store = &self.store;
                    without_gil(|| {
                        for d in deletes {
                            match d {
                                PendingDelete::TableFile(f) => store.delete_table_file(&f),
                                PendingDelete::Media(id) => store.delete_media(id),
                                PendingDelete::MediaVersion(id, version) => store.delete_media_version(id, version),
                            }
                        }
                    });
                    self.written_media.clear();
                    return Ok(());
                }
                Err(e) => {
                    self.rollback(checkpoint);
                    return Err(e);
                }
            }
        }
        self.rollback(checkpoint);
        Ok(())
    }

    /// Restores the state at `checkpoint` and deletes the media files written for the operation. Row data that the
    /// operation only loaded is kept.
    fn rollback(&mut self, checkpoint: Checkpoint) {
        let mut data = checkpoint.data;
        for (id, d) in self.data.drain() {
            if !data.contains_key(&id) && !self.staged.contains(&id) && checkpoint.tables.contains_key(&id) {
                data.insert(id, d);
            }
        }
        self.dirs = checkpoint.dirs;
        self.tables = checkpoint.tables;
        self.data = data;
        self.pending_deletes.clear();
        self.staged.clear();
        self.dirty = false;
        for f in std::mem::take(&mut self.written_media) {
            let _ = std::fs::remove_file(f);
        }
    }

    /// Makes media files written (without the lock) for the current operation part of it: they are deleted if the
    /// operation fails.
    pub fn adopt_media(&mut self, files: Vec<std::path::PathBuf>) {
        self.written_media.extend(files);
    }

    /// Deletes the media files that table version `version` wrote, when the operation succeeds (revert).
    pub fn defer_delete_media_version(&mut self, tbl_id: Uuid, version: i64) {
        self.pending_deletes.push(PendingDelete::MediaVersion(tbl_id, version));
    }

    fn defer_delete_media(&mut self, tbl_id: Uuid) {
        self.pending_deletes.push(PendingDelete::Media(tbl_id));
    }

    /// Marks the operation's changes to be persisted when it finishes (every operation that changes metadata or
    /// rows calls it once its changes are complete).
    pub fn commit(&mut self) -> Result<()> {
        self.dirty = true;
        Ok(())
    }

    /// Writes what the current operation changed, detached from the interpreter: the new segments of the tables
    /// whose rows changed, the new history records, then `catalog.json` (the commit point). On failure the files it
    /// wrote are removed and the store is unchanged.
    fn flush(&mut self) -> Result<()> {
        if !self.dirty && self.staged.is_empty() {
            return Ok(());
        }
        let mut written: Vec<String> = Vec::new();
        let result = without_gil(|| self.flush_inner(&mut written));
        if result.is_err() {
            let store = &self.store;
            without_gil(|| written.iter().for_each(|f| store.delete_table_file(f)));
        }
        result
    }

    fn flush_inner(&mut self, written: &mut Vec<String>) -> Result<()> {
        let mut staged: Vec<Uuid> = self.staged.iter().copied().filter(|id| self.tables.contains_key(id)).collect();
        staged.sort();
        for tbl_id in staged {
            let Some(entry) = self.data.get_mut(&tbl_id) else { continue };
            let data = Arc::make_mut(entry);
            if data.has_unwritten() {
                let md = self.tables.get(&tbl_id).ok_or_else(|| Error::internal("staged table has no metadata"))?;
                let keys = key_indexes(md);
                for (seg, vmax) in data.prepare_write() {
                    written.push(self.store.write_segment(tbl_id, &seg, &vmax, &keys)?);
                }
                data.mark_written();
            }
            let ids = data.segment_ids();
            if self.table(tbl_id)?.storage.segments == ids {
                continue;
            }
            let md = self.table_mut(tbl_id)?;
            let old = std::mem::replace(&mut md.storage.segments, ids);
            let current: HashSet<u64> = md.storage.segments.iter().copied().collect();
            for id in old.into_iter().filter(|id| !current.contains(id)) {
                self.pending_deletes.push(PendingDelete::TableFile(Store::segment_file_name(tbl_id, id)));
            }
        }
        let mut changed: Vec<Uuid> = self
            .tables
            .iter()
            .filter(|(id, md)| self.persisted.get(*id).is_none_or(|p| !Arc::ptr_eq(p, md)))
            .map(|(id, _)| *id)
            .collect();
        changed.sort();
        for tbl_id in changed {
            self.write_history(tbl_id, written)?;
        }
        self.store.sync_tables()?;
        let mut dirs: Vec<DirMd> = self.dirs.values().cloned().collect();
        dirs.sort_by(|a, b| a.created_at.total_cmp(&b.created_at).then(a.id.cmp(&b.id)));
        let mut tables: Vec<TableMd> = self.tables.values().map(|t| t.as_ref().clone()).collect();
        tables.sort_by(|a, b| a.created_at.total_cmp(&b.created_at).then(a.id.cmp(&b.id)));
        let generation = self.generation + 1;
        let bytes = Store::serialize_catalog(&CatalogFile {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            catalog_id: self.catalog_id,
            generation,
            dirs,
            tables,
        })?;
        self.store.save_catalog_bytes(&bytes)?;
        self.generation = generation;
        self.persisted = self.tables.clone();
        self.staged.clear();
        self.dirty = false;
        Ok(())
    }

    /// Appends the history records of table `tbl_id` that differ from what is committed (rewriting the history file
    /// when most of it is superseded records) and records the new length in its manifest.
    fn write_history(&mut self, tbl_id: Uuid, written: &mut Vec<String>) -> Result<()> {
        let md = self.table(tbl_id)?.clone();
        let records = match self.persisted.get(&tbl_id) {
            Some(old) => history::changed_records(&old.versions, &old.schema_versions, &md.versions, &md.schema_versions),
            None => history::changed_records(&[], &[], &md.versions, &md.schema_versions),
        };
        let live = (md.versions.len() + md.schema_versions.len()) as u64;
        let new_ref = match md.storage.history {
            Some(h) if records.is_empty() => h,
            Some(h) if h.records + (records.len() as u64) <= 2 * live + 64 => {
                let bytes = history::serialize(&records)?;
                let len = self.store.append_history(tbl_id, h.file, h.len, &bytes)?;
                HistoryRef { file: h.file, len, records: h.records + records.len() as u64 }
            }
            old => {
                // a new history file with every record
                let all = history::changed_records(&[], &[], &md.versions, &md.schema_versions);
                let bytes = history::serialize(&all)?;
                let file = crate::store::table_data::new_segment_id();
                written.push(Store::history_file_name(tbl_id, file));
                let len = self.store.append_history(tbl_id, file, 0, &bytes)?;
                if let Some(old) = old {
                    self.pending_deletes.push(PendingDelete::TableFile(Store::history_file_name(tbl_id, old.file)));
                }
                HistoryRef { file, len, records: all.len() as u64 }
            }
        };
        if md.storage.history != Some(new_ref) {
            self.table_mut(tbl_id)?.storage.history = Some(new_ref);
        }
        Ok(())
    }

    // ---------------------------------------------------------------------------------------- lookups

    pub fn root_id(&self) -> Uuid {
        self.root_id
    }

    pub fn dir(&self, id: Uuid) -> Option<&DirMd> {
        self.dirs.get(&id)
    }

    /// The table with the given id; TABLE_NOT_FOUND if it was dropped.
    pub fn table(&self, id: Uuid) -> Result<&Arc<TableMd>> {
        self.tables
            .get(&id)
            .ok_or_else(|| crate::pxt_err!(TableNotFound, "Table was dropped (no record found for {id})"))
    }

    pub fn table_mut(&mut self, id: Uuid) -> Result<&mut TableMd> {
        self.tables
            .get_mut(&id)
            .map(Arc::make_mut)
            .ok_or_else(|| crate::pxt_err!(TableNotFound, "Table was dropped (no record found for {id})"))
    }

    pub fn all_tables(&self) -> impl Iterator<Item = &Arc<TableMd>> {
        self.tables.values()
    }

    /// Subdirectories of a directory, sorted by name.
    pub fn child_dirs(&self, dir_id: Uuid) -> Vec<&DirMd> {
        let mut v: Vec<&DirMd> = self.dirs.values().filter(|d| d.parent_id == Some(dir_id)).collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// Tables in a directory, sorted by name.
    pub fn child_tables(&self, dir_id: Uuid) -> Vec<&Arc<TableMd>> {
        let mut v: Vec<&Arc<TableMd>> = self.tables.values().filter(|t| t.dir_id == dir_id).collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    pub fn entry_in(&self, dir_id: Uuid, name: &str) -> Option<Entry> {
        if let Some(d) = self.dirs.values().find(|d| d.parent_id == Some(dir_id) && d.name == name) {
            return Some(Entry::Dir(d.id));
        }
        self.tables.values().find(|t| t.dir_id == dir_id && t.name == name).map(|t| Entry::Table(t.id))
    }

    /// The directory at `path` (None if it does not exist or is not a directory).
    pub fn dir_at(&self, path: &Path) -> Option<Uuid> {
        let mut cur = self.root_id;
        for c in &path.components {
            match self.entry_in(cur, c) {
                Some(Entry::Dir(id)) => cur = id,
                _ => return None,
            }
        }
        Some(cur)
    }

    pub fn dir_path(&self, mut dir_id: Uuid) -> Result<Path> {
        let mut names = Vec::new();
        loop {
            let d = self.dirs.get(&dir_id).ok_or_else(|| crate::pxt_err!(DirectoryNotFound, "Directory not found: {dir_id}"))?;
            match d.parent_id {
                None => break,
                Some(p) => {
                    names.push(d.name.clone());
                    dir_id = p;
                }
            }
        }
        names.reverse();
        Ok(Path { components: names, version: None })
    }

    pub fn table_path(&self, tbl_id: Uuid) -> Result<Path> {
        let t = self.table(tbl_id)?;
        Ok(self.dir_path(t.dir_id)?.append(&t.name))
    }

    pub fn entry_display_name(&self, e: Entry) -> &'static str {
        match e {
            Entry::Dir(_) => "directory",
            Entry::Table(id) => self.tables.get(&id).map(|t| t.kind.display_name()).unwrap_or("table"),
        }
    }

    // ---------------------------------------------------------------------------------------- data

    /// The row data of a table (loading it from the store if needed).
    pub fn data(&mut self, tbl_id: Uuid) -> Result<Arc<TableData>> {
        if let Some(d) = self.data.get(&tbl_id) {
            return Ok(d.clone());
        }
        let ids = self.table(tbl_id)?.storage.segments.clone();
        let store = &self.store;
        let segments = without_gil(|| {
            use rayon::prelude::*;
            ids.par_iter().map(|id| store.read_segment(tbl_id, *id)).collect::<Result<Vec<_>>>()
        })?;
        let data = Arc::new(TableData::from_segments(segments));
        self.data.insert(tbl_id, data.clone());
        Ok(data)
    }

    /// Replaces the row data of a table and commits; it is written when the operation finishes.
    pub fn write_data(&mut self, tbl_id: Uuid, data: TableData) -> Result<()> {
        self.stage_data(tbl_id, data)?;
        self.commit()
    }

    /// Replaces the row data of a table, without committing (the operation commits once all its tables are staged).
    /// Only the changes (`TableData` tracks them) are written, when the operation finishes.
    pub fn stage_data(&mut self, tbl_id: Uuid, data: TableData) -> Result<()> {
        self.table(tbl_id)?;
        self.data.insert(tbl_id, Arc::new(data));
        self.staged.insert(tbl_id);
        Ok(())
    }

    /// Removes a table's record; its files are deleted when the operation succeeds. No dependency checks (see
    /// `tables::drop_table_by_id`).
    fn remove_table_record(&mut self, tbl_id: Uuid) -> Option<Arc<TableMd>> {
        self.data.remove(&tbl_id);
        self.staged.remove(&tbl_id);
        let md = self.tables.remove(&tbl_id);
        if let Some(md) = &md {
            for f in table_files(md) {
                self.pending_deletes.push(PendingDelete::TableFile(f));
            }
            self.defer_delete_media(tbl_id);
        }
        md
    }

    fn insert_dir(&mut self, parent_id: Uuid, name: &str) -> Uuid {
        let d = DirMd { id: Uuid::new_v4(), parent_id: Some(parent_id), name: name.to_string(), created_at: now() };
        let id = d.id;
        self.dirs.insert(id, d);
        id
    }
}

/// The keys whose hashes the segments of table `md` carry (see `store::table_data::KeyIndex`): its primary key
/// (truncated as the index of a data-versioned table truncates strings) and each of its columns with a B-tree index.
fn key_indexes(md: &TableMd) -> Vec<(Vec<ColId>, bool)> {
    let mut out = Vec::new();
    let pk: Vec<ColId> = if md.is_view() { Vec::new() } else { md.columns.iter().filter(|c| c.is_pk).map(|c| c.id).collect() };
    if !pk.is_empty() {
        out.push((pk, md.is_data_versioned));
    }
    for i in md.indexes.list.iter().filter(|i| i.is_btree() && i.col_tbl_id == md.id) {
        if !out.iter().any(|(cols, t)| *cols == [i.col_id] && !*t) {
            out.push((vec![i.col_id], false));
        }
    }
    out
}

/// The files in the store's `tables/` directory that table `md` references.
fn table_files(md: &TableMd) -> Vec<String> {
    let mut out: Vec<String> = md.storage.segments.iter().map(|id| Store::segment_file_name(md.id, *id)).collect();
    if let Some(h) = &md.storage.history {
        out.push(Store::history_file_name(md.id, h.file));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> Store {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("test-tmp").join(Uuid::new_v4().to_string());
        match Store::open(&dir, "test") {
            Ok(s) => s,
            Err(e) => panic!("cannot create test store: {e}"),
        }
    }

    fn create_table(cat: &mut Catalog, name: &str) -> Result<Uuid> {
        create_table_pk(cat, name, false)
    }

    fn create_table_pk(cat: &mut Catalog, name: &str, is_pk: bool) -> Result<Uuid> {
        let cp = cat.checkpoint();
        let spec = tables::ColumnSpec {
            name: "c".into(),
            col_type: crate::types::ColumnType::int(!is_pk),
            value_expr: None,
            is_pk,
            stored: None,
            media_validation: None,
            comment: None,
            custom_metadata: serde_json::Value::Null,
            destination: None,
        };
        let path = Path::parse(name, false, false)?;
        let (id, _) = cat.create_table(&path, vec![spec], IfExists::Error, None, serde_json::Value::Null, table::MediaValidation::OnWrite, true, false)?;
        cat.finish(cp, true)?;
        Ok(id)
    }

    /// Inserts rows with the given rowids (cell `c` = rowid) at a new version, as one operation.
    fn insert(cat: &mut Catalog, id: Uuid, rowids: std::ops::Range<i64>) -> Result<()> {
        let cp = cat.checkpoint();
        let mut data = cat.data(id)?.as_ref().clone();
        let v = cat.bump_version(id, None)?;
        let rows = rowids
            .map(|i| {
                let mut r = crate::store::rows::StoredRow::new(i, Vec::new());
                r.set_cell(0, crate::value::Value::Int(i));
                r
            })
            .collect();
        data.append(rows, v);
        cat.stage_data(id, data)?;
        cat.commit()?;
        cat.finish(cp, true)
    }

    fn delete(cat: &mut Catalog, id: Uuid, rowids: std::ops::Range<i64>) -> Result<()> {
        let cp = cat.checkpoint();
        let mut data = cat.data(id)?.as_ref().clone();
        let v = cat.bump_version(id, None)?;
        data.retire(&rowids.map(|r| (r, Vec::new())).collect(), v, true);
        cat.stage_data(id, data)?;
        cat.commit()?;
        cat.finish(cp, true)
    }

    fn rowids_at(cat: &mut Catalog, id: Uuid, version: Option<i64>) -> Result<Vec<i64>> {
        let v = version.unwrap_or(cat.table(id)?.current_version);
        let mut out: Vec<i64> = cat.data(id)?.visible_at(v).map(|r| r.rowid).collect();
        out.sort();
        Ok(out)
    }

    fn table_dir_files(store: &Store) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(store.home().join("pxtrs").join("test").join("tables"))
            .map(|d| d.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect())
            .unwrap_or_default();
        out.sort();
        out
    }

    fn referenced_files(cat: &Catalog) -> Vec<String> {
        let mut out: Vec<String> = cat.tables.values().flat_map(|md| table_files(md)).collect();
        out.sort();
        out
    }

    #[test]
    fn failure_before_the_commit_point_keeps_the_previous_state() -> Result<()> {
        let store = temp_store();
        let mut cat = Catalog::load(store.clone(), false)?;
        let id = create_table(&mut cat, "t")?;
        insert(&mut cat, id, 0..10)?;
        let stamp = cat.stamp();
        crate::store::FAIL_BEFORE_COMMIT.with(|f| f.set(true));
        let failed = insert(&mut cat, id, 10..20);
        crate::store::FAIL_BEFORE_COMMIT.with(|f| f.set(false));
        assert!(failed.is_err());
        // the in-memory state is the previous one, and so is the store; the failed commit's files are gone
        assert_eq!(cat.stamp(), stamp);
        assert_eq!(cat.table(id)?.current_version, 1);
        assert_eq!(rowids_at(&mut cat, id, None)?, (0..10).collect::<Vec<_>>());
        assert_eq!(table_dir_files(&store), referenced_files(&cat));
        let mut reloaded = Catalog::load(store.clone(), false)?;
        assert_eq!(reloaded.stamp(), stamp);
        assert_eq!(reloaded.table(id)?.versions.len(), 2);
        assert_eq!(rowids_at(&mut reloaded, id, None)?, (0..10).collect::<Vec<_>>());
        // the next operation commits normally
        insert(&mut cat, id, 10..15)?;
        let mut reloaded = Catalog::load(store.clone(), false)?;
        assert_eq!(rowids_at(&mut reloaded, id, None)?, (0..15).collect::<Vec<_>>());
        assert_eq!(rowids_at(&mut reloaded, id, Some(1))?, (0..10).collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }

    #[test]
    fn leftovers_of_a_crash_are_ignored_and_collected() -> Result<()> {
        let store = temp_store();
        let mut cat = Catalog::load(store.clone(), false)?;
        let id = create_table(&mut cat, "t")?;
        insert(&mut cat, id, 0..10)?;
        // what a process that crashed before its commit point leaves behind: a segment file nothing references and
        // history records past the committed length
        let h = cat.table(id)?.storage.history.ok_or_else(|| Error::internal("no history"))?;
        let stray = crate::store::segment::Segment::new(99, Vec::new(), Vec::new(), Vec::new());
        store.write_segment(id, &stray, &[], &[])?;
        store.append_history(id, h.file, h.len, b"{\"version\": \"garbage\"}\n")?;
        let mut reloaded = Catalog::load(store.clone(), true)?;
        assert_eq!(rowids_at(&mut reloaded, id, None)?, (0..10).collect::<Vec<_>>());
        assert_eq!(table_dir_files(&store), referenced_files(&reloaded));
        insert(&mut reloaded, id, 10..12)?;
        let mut again = Catalog::load(store.clone(), false)?;
        assert_eq!(again.table(id)?.versions.len(), 3);
        assert_eq!(rowids_at(&mut again, id, None)?, (0..12).collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }

    #[test]
    fn reloaded_rows_are_decoded_when_read() -> Result<()> {
        let store = temp_store();
        let mut cat = Catalog::load(store.clone(), false)?;
        let id = create_table_pk(&mut cat, "t", true)?;
        insert(&mut cat, id, 0..150_000)?;
        let mut cat = Catalog::load(store.clone(), false)?;
        let data = cat.data(id)?;
        let (decoded, total) = data.decoded_chunks();
        assert!(total >= 18 && decoded == 0, "{decoded} of {total} decoded");
        // counting reads only version ranges; a primary-key lookup reads the stored key hashes and decodes the one
        // chunk holding the row
        assert_eq!(data.count_at(1), 150_000);
        let found = data.find_live_by_pk(&[0], true, &[crate::value::Value::Int(70_000)], true).map(|r| r.rowid);
        assert_eq!(found, Some(70_000));
        assert_eq!(data.decoded_chunks().0, 1);
        assert_eq!(rowids_at(&mut cat, id, None)?.len(), 150_000);
        assert_eq!(data.decoded_chunks().0, total);
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }

    #[test]
    fn reload_picks_up_another_process_commits() -> Result<()> {
        let store = temp_store();
        let mut a = Catalog::load(store.clone(), false)?;
        let id = create_table(&mut a, "t")?;
        insert(&mut a, id, 0..100)?;
        let mut b = Catalog::load(store.clone(), false)?;
        assert_eq!(rowids_at(&mut b, id, None)?.len(), 100);
        // a appends (b's cached rows are extended), then deletes (a patch of an older segment)
        insert(&mut a, id, 100..110)?;
        delete(&mut a, id, 0..5)?;
        let mut b = Catalog::reload(store.clone(), b)?;
        assert_eq!(b.stamp(), a.stamp());
        assert_eq!(rowids_at(&mut b, id, None)?, (5..110).collect::<Vec<_>>());
        assert_eq!(rowids_at(&mut b, id, Some(2))?, (0..110).collect::<Vec<_>>());
        assert_eq!(b.table(id)?.versions.len(), 4);
        // many commits by b (merging segments), then a reloads
        for k in 0..20 {
            insert(&mut b, id, 110 + k * 3..113 + k * 3)?;
        }
        delete(&mut b, id, 5..50)?;
        let mut a = Catalog::reload(store.clone(), a)?;
        let expected = rowids_at(&mut Catalog::load(store.clone(), false)?, id, None)?;
        assert_eq!(rowids_at(&mut a, id, None)?, expected);
        assert_eq!(expected.len(), 110 + 60 - 50);
        for v in 0..=a.table(id)?.current_version {
            assert_eq!(rowids_at(&mut a, id, Some(v))?, rowids_at(&mut b, id, Some(v))?, "version {v}");
        }
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }

    #[test]
    fn rollback_restores_memory_and_disk() -> Result<()> {
        let store = temp_store();
        let mut cat = Catalog::load(store.clone(), false)?;
        let cp = cat.checkpoint();
        cat.create_dir(&Path::parse("d1", false, false)?, IfExists::Error, false)?;
        cat.finish(cp, true)?;

        let cp = cat.checkpoint();
        cat.create_dir(&Path::parse("d2", false, false)?, IfExists::Error, false)?;
        cat.finish(cp, false)?;
        assert!(cat.dir_at(&Path::parse("d1", false, false)?).is_some());
        assert!(cat.dir_at(&Path::parse("d2", false, false)?).is_none());

        let reloaded = Catalog::load(store.clone(), false)?;
        assert!(reloaded.dir_at(&Path::parse("d1", false, false)?).is_some());
        assert!(reloaded.dir_at(&Path::parse("d2", false, false)?).is_none());
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }
}
