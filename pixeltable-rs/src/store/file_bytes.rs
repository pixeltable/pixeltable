//! The bytes of a segment file, mapped into memory: pages are read when first touched, so loading a table reads its
//! segments' headers, and a scan or lookup reads only the chunks it decodes.
//!
//! Segment files are immutable once written (a commit writes new files, never changes one), and a mapping stays valid
//! after its file is deleted (by compaction, `Store::gc()` or another process), so a segment keeps its encoded rows
//! until it decodes them all, whatever happens to the file. Where mapping fails, the file is read into memory.

use std::fs::File;
use std::path::Path;

/// A segment file's bytes: mapped read-only, or in memory.
pub enum FileBytes {
    Owned(Vec<u8>),
    Mapped { ptr: *const u8, len: usize },
}

// SAFETY: a read-only private mapping of an immutable file is plain shared memory; nothing mutates it
unsafe impl Send for FileBytes {}
unsafe impl Sync for FileBytes {}

impl FileBytes {
    /// The file at `path`, mapped (an empty file, or one that cannot be mapped, is read).
    pub fn open(path: &Path) -> std::io::Result<FileBytes> {
        use std::os::fd::AsRawFd;
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).map_err(std::io::Error::other)?;
        if len == 0 {
            return Ok(FileBytes::Owned(Vec::new()));
        }
        // SAFETY: a fresh read-only private mapping of `len` bytes of an open file; the fd may be closed afterwards
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Ok(FileBytes::Owned(std::fs::read(path)?));
        }
        Ok(FileBytes::Mapped { ptr: ptr as *const u8, len })
    }
}

impl From<Vec<u8>> for FileBytes {
    fn from(v: Vec<u8>) -> FileBytes {
        FileBytes::Owned(v)
    }
}

impl std::ops::Deref for FileBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            FileBytes::Owned(v) => v,
            // SAFETY: the mapping is valid for `len` bytes until drop
            FileBytes::Mapped { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

impl Drop for FileBytes {
    fn drop(&mut self) {
        if let FileBytes::Mapped { ptr, len } = *self {
            // SAFETY: unmaps the mapping made by `open()`, which nothing references any more
            unsafe {
                libc::munmap(ptr as *mut libc::c_void, len);
            }
        }
    }
}

impl std::fmt::Debug for FileBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileBytes::Owned(v) => write!(f, "FileBytes::Owned({} bytes)", v.len()),
            FileBytes::Mapped { len, .. } => write!(f, "FileBytes::Mapped({len} bytes)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_file_outlives_its_deletion() {
        let dir = std::env::temp_dir().join(format!("pxtrs-filebytes-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let bytes = FileBytes::open(&path).unwrap();
        assert!(matches!(bytes, FileBytes::Mapped { .. }));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(&bytes[..], &data[..]);
    }
}
