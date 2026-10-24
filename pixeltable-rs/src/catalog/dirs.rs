//! Directory operations and listings, following upstream `Catalog` semantics and messages.

use uuid::Uuid;

use super::path::{Path, is_valid_identifier};
use super::table::TableKind;
use super::{Catalog, Entry, IfExists, IfNotExists};
use crate::error::Result;

/// What a path lookup expects to find.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expected {
    Any,
    Dir,
    Table,
}

impl Expected {
    fn name(self) -> &'static str {
        match self {
            Expected::Dir => "directory",
            _ => "table",
        }
    }
}

/// What is being created at a path (for collision handling).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Creating {
    Dir,
    Table,
    View,
    Snapshot,
}

impl Creating {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Creating::Dir => "directory",
            Creating::Table => "table",
            Creating::View => "view",
            Creating::Snapshot => "snapshot",
        }
    }
}

/// A node of a directory listing.
#[derive(Debug, Clone)]
pub enum ContentNode {
    Dir { name: String, entries: Vec<ContentNode> },
    Table { name: String, id: Uuid },
}

impl Catalog {
    /// Upstream `_get_schema_object()`.
    pub fn get_object(
        &self,
        path: &Path,
        expected: Expected,
        raise_if_exists: bool,
        raise_if_not_exists: bool,
    ) -> Result<Option<Entry>> {
        if path.is_root() {
            if expected == Expected::Table {
                crate::bail!(InvalidArgument, "{} needs to be a table but is a dir", path.repr());
            }
            return Ok(Some(Entry::Dir(self.root_id)));
        }
        let parent_path = path.parent();
        let Some(parent) = self.dir_at(&parent_path) else {
            if raise_if_not_exists {
                crate::bail!(DirectoryNotFound, "Directory {} does not exist.", parent_path.repr());
            }
            return Ok(None);
        };
        let obj = self.entry_in(parent, path.name());
        match obj {
            None if raise_if_not_exists => {
                crate::bail!(PathNotFound, "Path {} does not exist.", path.without_version().repr())
            }
            Some(e) if raise_if_exists => crate::bail!(
                PathAlreadyExists,
                "Path {} is an existing {}.",
                path.repr(),
                self.entry_display_name(e)
            ),
            Some(e) => {
                let mismatch = match (expected, e) {
                    (Expected::Dir, Entry::Table(_)) | (Expected::Table, Entry::Dir(_)) => true,
                    _ => false,
                };
                if mismatch {
                    crate::bail!(
                        InvalidArgument,
                        "{} needs to be a {} but is a {}.",
                        path.repr(),
                        expected.name(),
                        self.entry_display_name(e)
                    );
                }
                Ok(Some(e))
            }
            None => Ok(None),
        }
    }

    /// Upstream `_prepare_dir_op()` for adding an entry: the parent directory must exist; returns the existing entry.
    pub fn prepare_add(&self, path: &Path, raise_if_exists: bool) -> Result<(Uuid, Option<Entry>)> {
        let parent_path = path.parent();
        let Some(dir) = self.dir_at(&parent_path) else {
            crate::bail!(
                DirectoryNotFound,
                "Directory {} does not exist. Create it first with:\npxt.create_dir({})",
                parent_path.repr(),
                parent_path.repr()
            );
        };
        let obj = self.entry_in(dir, path.name());
        if obj.is_some() && raise_if_exists {
            crate::bail!(PathAlreadyExists, "Path {} already exists.", path.repr());
        }
        Ok((dir, obj))
    }

    /// Upstream `_prepare_dir_op()` for dropping an entry.
    fn prepare_drop(&self, path: &Path, expected: Expected, raise_if_not_exists: bool) -> Result<Option<Entry>> {
        let parent_path = path.parent();
        let Some(dir) = self.dir_at(&parent_path) else {
            if raise_if_not_exists {
                crate::bail!(DirectoryNotFound, "Directory {} does not exist.", parent_path.repr());
            }
            return Ok(None);
        };
        let obj = self.entry_in(dir, path.name());
        match obj {
            None if raise_if_not_exists => crate::bail!(PathNotFound, "Path {} does not exist.", path.repr()),
            Some(e) => {
                let mismatch = matches!((expected, e), (Expected::Dir, Entry::Table(_)) | (Expected::Table, Entry::Dir(_)));
                if mismatch {
                    crate::bail!(
                        InvalidArgument,
                        "{} needs to be a {} but is a {}",
                        path.repr(),
                        expected.name(),
                        self.entry_display_name(e)
                    );
                }
                Ok(Some(e))
            }
            None => Ok(None),
        }
    }

    /// Upstream `_handle_path_collision()`: returns the existing entry to reuse (if_exists='ignore'), or None after
    /// clearing the path (replace) or if the path is free.
    pub fn handle_path_collision(&mut self, path: &Path, creating: Creating, if_exists: IfExists) -> Result<Option<Entry>> {
        let (_, obj) = self.prepare_add(path, false)?;
        let Some(obj) = obj else { return Ok(None) };
        let creating_dir = creating == Creating::Dir;
        match if_exists {
            IfExists::Error => crate::bail!(
                PathAlreadyExists,
                "Path {} is an existing {}",
                path.repr(),
                self.entry_display_name(obj)
            ),
            IfExists::Ignore => {
                let type_ok = match (obj, creating) {
                    (Entry::Dir(_), Creating::Dir) => true,
                    (Entry::Table(id), Creating::Table) => self.table(id)?.kind == TableKind::Table,
                    (Entry::Table(id), Creating::View) => self.table(id)?.kind != TableKind::Table,
                    (Entry::Table(id), Creating::Snapshot) => self.table(id)?.kind == TableKind::Snapshot,
                    _ => false,
                };
                if !type_ok {
                    crate::bail!(PathAlreadyExists, "Path {} already exists and is not a {}", path.repr(), creating.name());
                }
                Ok(Some(obj))
            }
            IfExists::Replace | IfExists::ReplaceForce => {
                if creating_dir && matches!(obj, Entry::Table(_)) {
                    crate::bail!(
                        PathAlreadyExists,
                        "Path {} already exists as a {}; expected a directory",
                        path.repr(),
                        self.entry_display_name(obj)
                    );
                }
                if !creating_dir && matches!(obj, Entry::Dir(_)) {
                    crate::bail!(
                        PathAlreadyExists,
                        "Path {} already exists as a directory; expected a table, view or snapshot",
                        path.repr()
                    );
                }
                match obj {
                    Entry::Dir(id) => {
                        let non_empty = !self.child_dirs(id).is_empty() || !self.child_tables(id).is_empty();
                        if non_empty && if_exists == IfExists::Replace {
                            crate::bail!(
                                PathAlreadyExists,
                                "Directory {} already exists and is not empty. Use `if_exists=\"replace_force\"` to replace it.",
                                path.repr()
                            );
                        }
                        self.drop_dir_recursive(id, path, true)?;
                    }
                    Entry::Table(id) => {
                        self.drop_table_by_id(id, if_exists == IfExists::ReplaceForce, true)?;
                    }
                }
                Ok(None)
            }
        }
    }

    pub fn create_dir(&mut self, path: &Path, if_exists: IfExists, parents: bool) -> Result<Uuid> {
        let parent = if parents {
            let mut last: Option<Uuid> = None;
            for ancestor in path.ancestors() {
                let obj = self.get_object(&ancestor, Expected::Dir, false, false)?;
                last = Some(match (obj, last) {
                    (Some(Entry::Dir(id)), _) => id,
                    (None, Some(parent)) => self.insert_dir(parent, ancestor.name()),
                    _ => return Err(crate::error::Error::internal("create_dir(): inconsistent ancestors")),
                });
            }
            last
        } else {
            match self.get_object(&path.parent(), Expected::Any, false, false)? {
                Some(Entry::Dir(id)) => Some(id),
                _ => None,
            }
        };
        if let Some(Entry::Dir(existing)) = self.handle_path_collision(path, Creating::Dir, if_exists)? {
            return Ok(existing);
        }
        let parent = match parent {
            Some(p) => p,
            None => self.prepare_add(path, false)?.0,
        };
        let id = self.insert_dir(parent, path.name());
        self.commit()?;
        Ok(id)
    }

    pub fn drop_dir(&mut self, path: &Path, if_not_exists: IfNotExists, force: bool) -> Result<()> {
        let raise = if_not_exists == IfNotExists::Error && !force;
        let Some(Entry::Dir(id)) = self.prepare_drop(path, Expected::Dir, raise)? else { return Ok(()) };
        self.drop_dir_recursive(id, path, force)?;
        self.commit()
    }

    pub(crate) fn drop_dir_recursive(&mut self, dir_id: Uuid, dir_path: &Path, force: bool) -> Result<()> {
        let sub_dirs: Vec<(Uuid, String)> = self.child_dirs(dir_id).iter().map(|d| (d.id, d.name.clone())).collect();
        let tables: Vec<Uuid> = self.child_tables(dir_id).iter().map(|t| t.id).collect();
        if !force && !(sub_dirs.is_empty() && tables.is_empty()) {
            crate::bail!(UnsupportedOperation, "Directory {} is not empty.", dir_path.repr());
        }
        for (id, name) in sub_dirs {
            self.drop_dir_recursive(id, &dir_path.append(&name), true)?;
        }
        for id in tables {
            // a view may already be gone if its base was dropped earlier in this loop
            if self.tables.contains_key(&id) {
                self.drop_table_by_id(id, true, false)?;
            }
        }
        self.dirs.remove(&dir_id);
        Ok(())
    }

    pub fn move_entry(
        &mut self,
        path: &Path,
        new_path: &Path,
        if_exists: IfExists,
        if_not_exists: IfNotExists,
    ) -> Result<()> {
        let mut dir_paths = vec![new_path.parent(), path.parent()];
        dir_paths.sort();
        dir_paths.dedup();
        for p in &dir_paths {
            if self.dir_at(p).is_none() {
                crate::bail!(
                    DirectoryNotFound,
                    "Directory {} does not exist. Create it first with:\npxt.create_dir({})",
                    p.repr(),
                    p.repr()
                );
            }
        }
        let (dest_dir, dest_obj) = self.prepare_add(new_path, if_exists == IfExists::Error)?;
        let src_obj = self.prepare_drop(path, Expected::Any, if_not_exists == IfNotExists::Error)?;
        if dest_obj.is_some() {
            return Ok(());
        }
        match src_obj {
            None => Ok(()),
            Some(Entry::Table(id)) => {
                let md = self.table_mut(id)?;
                md.dir_id = dest_dir;
                md.name = new_path.name().to_string();
                self.commit()
            }
            Some(Entry::Dir(id)) => {
                if let Some(d) = self.dirs.get_mut(&id) {
                    d.parent_id = Some(dest_dir);
                    d.name = new_path.name().to_string();
                }
                self.commit()
            }
        }
    }

    /// The contents of the directory at `path` (upstream `get_dir_contents()`).
    pub fn dir_contents(&self, path: &Path, recursive: bool) -> Result<Vec<ContentNode>> {
        let Some(Entry::Dir(id)) = self.get_object(path, Expected::Dir, false, true)? else {
            return Err(crate::error::Error::internal("get_object() returned no directory"));
        };
        Ok(self.dir_contents_of(id, recursive))
    }

    fn dir_contents_of(&self, dir_id: Uuid, recursive: bool) -> Vec<ContentNode> {
        let mut out = Vec::new();
        for d in self.child_dirs(dir_id) {
            let entries = if recursive { self.dir_contents_of(d.id, true) } else { Vec::new() };
            out.push(ContentNode::Dir { name: d.name.clone(), entries });
        }
        for t in self.child_tables(dir_id) {
            out.push(ContentNode::Table { name: t.name.clone(), id: t.id });
        }
        out
    }
}

/// Flattens a listing into paths (upstream `_extract_paths()`): the direct matches first, then the matches in each
/// subdirectory. `want_dirs` selects directories, otherwise tables.
pub fn extract_paths(nodes: &[ContentNode], parent: &Path, want_dirs: bool) -> Vec<String> {
    let mut out = Vec::new();
    for n in nodes {
        match n {
            ContentNode::Dir { name, .. } if want_dirs && is_valid_identifier(name, true) => {
                out.push(parent.append(name).to_string())
            }
            ContentNode::Table { name, .. } if !want_dirs && is_valid_identifier(name, true) => {
                out.push(parent.append(name).to_string())
            }
            _ => {}
        }
    }
    for n in nodes {
        if let ContentNode::Dir { name, entries } = n {
            if !entries.is_empty() && is_valid_identifier(name, true) {
                out.extend(extract_paths(entries, &parent.append(name), want_dirs));
            }
        }
    }
    out
}

/// `get_dir_contents()`'s lists: every path prefixed with the caller's `dir_path` string as given.
pub fn assemble_contents(dir_path: &str, nodes: &[ContentNode], dirs: &mut Vec<String>, tables: &mut Vec<String>) {
    for n in nodes {
        let name = match n {
            ContentNode::Dir { name, .. } | ContentNode::Table { name, .. } => name,
        };
        if name.starts_with('_') {
            continue;
        }
        let path = if dir_path.is_empty() { name.clone() } else { format!("{dir_path}/{name}") };
        match n {
            ContentNode::Dir { entries, .. } => {
                dirs.push(path.clone());
                assemble_contents(&path, entries, dirs, tables);
            }
            ContentNode::Table { .. } => tables.push(path),
        }
    }
}
