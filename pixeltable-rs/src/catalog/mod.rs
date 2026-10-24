//! The catalog: directories and tables, their metadata, and the row data of every table.
//!
//! `Catalog` is the in-memory state of one database. It is loaded from the store on first use and every mutation
//! is persisted before the operation returns (see `store`). The process-wide instance lives in `runtime`.
//!
//! Operations are split by concern: `dirs.rs` (directory operations and listings), `tables.rs` (create, get, drop,
//! metadata), `views.rs` (views and snapshots); data mutations (insert, ...) are in `exec`.

pub mod dirs;
pub mod path;
pub mod runtime;
pub mod table;
pub mod tables;
pub mod views;

use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::store::rows::TableData;
use crate::store::{CatalogFile, CatalogStamp, DirMd, FORMAT_NAME, FORMAT_VERSION, Store};
use path::Path;
use table::TableMd;

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
    DataFile(String),
    Media(Uuid),
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
    /// files to delete when the current operation succeeds
    pending_deletes: Vec<PendingDelete>,
    /// data files written by the current operation (deleted if it fails)
    written_files: Vec<String>,
    /// true once the current operation has written catalog.json
    committed: bool,
}

/// The in-memory state at the start of an operation (see `Catalog::checkpoint()`).
pub struct Checkpoint {
    dirs: HashMap<Uuid, DirMd>,
    tables: HashMap<Uuid, Arc<TableMd>>,
    data: HashMap<Uuid, Arc<TableData>>,
}

impl Catalog {
    /// Loads the catalog from the store, initializing an empty one if the store is new.
    pub fn load(store: Store) -> Result<Catalog> {
        match store.load_catalog()? {
            Some(cf) => {
                let root = cf
                    .dirs
                    .iter()
                    .find(|d| d.parent_id.is_none())
                    .ok_or_else(|| Error::internal("catalog has no root directory"))?;
                let root_id = root.id;
                Ok(Catalog {
                    store,
                    catalog_id: cf.catalog_id,
                    generation: cf.generation,
                    root_id,
                    dirs: cf.dirs.into_iter().map(|d| (d.id, d)).collect(),
                    tables: cf.tables.into_iter().map(|t| (t.id, Arc::new(t))).collect(),
                    data: HashMap::new(),
                    pending_deletes: Vec::new(),
                    written_files: Vec::new(),
                    committed: false,
                })
            }
            None => {
                let root = DirMd { id: Uuid::new_v4(), parent_id: None, name: String::new(), created_at: now() };
                let mut cat = Catalog {
                    store,
                    catalog_id: Uuid::new_v4(),
                    generation: 0,
                    root_id: root.id,
                    dirs: HashMap::from([(root.id, root)]),
                    tables: HashMap::new(),
                    data: HashMap::new(),
                    pending_deletes: Vec::new(),
                    written_files: Vec::new(),
                    committed: false,
                };
                cat.commit()?;
                Ok(cat)
            }
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
    // Every operation runs between `checkpoint()` and `finish()` (see `runtime::with_catalog`): on success the
    // deferred file deletions are applied; on failure the in-memory state is restored, catalog.json is rewritten if
    // the operation had already written it, and the data files it wrote are removed. File deletions are therefore
    // never done directly: use `defer_delete_*`.

    pub fn checkpoint(&mut self) -> Checkpoint {
        self.pending_deletes.clear();
        self.written_files.clear();
        self.committed = false;
        Checkpoint { dirs: self.dirs.clone(), tables: self.tables.clone(), data: self.data.clone() }
    }

    pub fn finish(&mut self, checkpoint: Checkpoint, succeeded: bool) -> Result<()> {
        if succeeded {
            for d in std::mem::take(&mut self.pending_deletes) {
                match d {
                    PendingDelete::DataFile(f) => self.store.delete_table_data(&f),
                    PendingDelete::Media(id) => self.store.delete_media(id),
                }
            }
            self.written_files.clear();
            return Ok(());
        }
        self.dirs = checkpoint.dirs;
        self.tables = checkpoint.tables;
        self.data = checkpoint.data;
        self.pending_deletes.clear();
        let written = std::mem::take(&mut self.written_files);
        if self.committed {
            self.commit()?;
        }
        for f in written {
            self.store.delete_table_data(&f);
        }
        Ok(())
    }

    fn defer_delete_data_file(&mut self, file: String) {
        self.pending_deletes.push(PendingDelete::DataFile(file));
    }

    fn defer_delete_media(&mut self, tbl_id: Uuid) {
        self.pending_deletes.push(PendingDelete::Media(tbl_id));
    }

    /// Persists the metadata (the commit point of every operation).
    pub fn commit(&mut self) -> Result<()> {
        let mut dirs: Vec<DirMd> = self.dirs.values().cloned().collect();
        dirs.sort_by(|a, b| a.created_at.total_cmp(&b.created_at).then(a.id.cmp(&b.id)));
        let mut tables: Vec<TableMd> = self.tables.values().map(|t| t.as_ref().clone()).collect();
        tables.sort_by(|a, b| a.created_at.total_cmp(&b.created_at).then(a.id.cmp(&b.id)));
        self.generation += 1;
        self.store.save_catalog(&CatalogFile {
            format: FORMAT_NAME.to_string(),
            format_version: FORMAT_VERSION,
            catalog_id: self.catalog_id,
            generation: self.generation,
            dirs,
            tables,
        })?;
        self.committed = true;
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
        let md = self.table(tbl_id)?;
        let data = match &md.data_file {
            Some(f) => Arc::new(self.store.load_table_data(f)?),
            None => Arc::new(TableData::default()),
        };
        self.data.insert(tbl_id, data.clone());
        Ok(data)
    }

    /// Replaces the row data of a table: writes the data file, records it in the metadata and commits.
    pub fn write_data(&mut self, tbl_id: Uuid, data: TableData) -> Result<()> {
        let version = self.table(tbl_id)?.current_version;
        let file = self.store.write_table_data(tbl_id, version, &data)?;
        self.written_files.push(file.clone());
        let md = self.table_mut(tbl_id)?;
        let old = md.data_file.replace(file);
        self.data.insert(tbl_id, Arc::new(data));
        if let Some(old) = old {
            self.defer_delete_data_file(old);
        }
        self.commit()
    }

    /// Removes a table's record; its files are deleted when the operation succeeds. No dependency checks (see
    /// `tables::drop_table_by_id`).
    fn remove_table_record(&mut self, tbl_id: Uuid) -> Option<Arc<TableMd>> {
        self.data.remove(&tbl_id);
        let md = self.tables.remove(&tbl_id);
        if let Some(md) = &md {
            if let Some(f) = &md.data_file {
                self.defer_delete_data_file(f.clone());
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

    #[test]
    fn rollback_restores_memory_and_disk() -> Result<()> {
        let store = temp_store();
        let mut cat = Catalog::load(store.clone())?;
        let cp = cat.checkpoint();
        cat.create_dir(&Path::parse("d1", false, false)?, IfExists::Error, false)?;
        cat.finish(cp, true)?;

        let cp = cat.checkpoint();
        cat.create_dir(&Path::parse("d2", false, false)?, IfExists::Error, false)?;
        cat.finish(cp, false)?;
        assert!(cat.dir_at(&Path::parse("d1", false, false)?).is_some());
        assert!(cat.dir_at(&Path::parse("d2", false, false)?).is_none());

        let reloaded = Catalog::load(store.clone())?;
        assert!(reloaded.dir_at(&Path::parse("d1", false, false)?).is_some());
        assert!(reloaded.dir_at(&Path::parse("d2", false, false)?).is_none());
        let _ = std::fs::remove_dir_all(store.home());
        Ok(())
    }
}
