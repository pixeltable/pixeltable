//! Media files: what a media cell refers to, fetching remote files into the file cache, writing in-memory media to
//! the media store, and validating files.
//!
//! A stored media cell (`Value::Media`) is either an absolute local path or a URL (`http(s)://`, `s3://`, ...).
//! Inputs are normalized by `normalize_input()` (relative paths become absolute, `file://` URLs become paths), as
//! upstream's `DataRow.__setitem__` does. Remote files are downloaded on first use into
//! `$PIXELTABLE_HOME/file_cache/` (`local_path()`); in-memory media (PIL images, bytes) are written to the table's
//! media directory (`save_object()`), with file names `<tbl id>_<col id>_<version>_<unique id>[.<ext>]` so that a
//! table's files can be found by version.
//!
//! Python is used only for I/O: PIL and PyAV to decode and validate files, `urllib` to download.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::store::rows::CellError;
use crate::types::{ColumnType, TypeKind};
use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
    Document,
}

impl MediaKind {
    pub fn of(t: &ColumnType) -> Option<MediaKind> {
        match t.kind {
            TypeKind::Image { .. } => Some(MediaKind::Image),
            TypeKind::Video => Some(MediaKind::Video),
            TypeKind::Audio => Some(MediaKind::Audio),
            TypeKind::Document { .. } => Some(MediaKind::Document),
            _ => None,
        }
    }
}

// ------------------------------------------------------------------------------------------------ paths and URLs

/// The URL scheme of `s` as `urllib.parse.urlparse()` determines it (lowercased), if any.
pub fn url_scheme(s: &str) -> Option<String> {
    let i = s.find(':')?;
    let candidate = &s[..i];
    let mut chars = candidate.chars();
    let first = chars.next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    if candidate.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.') {
        Some(candidate.to_ascii_lowercase())
    } else {
        None
    }
}

/// True if `s` refers to a local file (a path, or a one-letter "scheme" such as a Windows drive).
pub fn is_local(s: &str) -> bool {
    url_scheme(s).is_none_or(|scheme| scheme.len() <= 1)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// `urllib.parse.unquote()`
pub fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                out.push(h * 16 + l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `urllib.parse.quote(s, safe='/')`
pub fn percent_encode_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The path component of a URL (after the scheme and the authority, before query and fragment).
fn url_path(url: &str) -> &str {
    let rest = match url.find(':') {
        Some(i) => &url[i + 1..],
        None => url,
    };
    let rest = match rest.strip_prefix("//") {
        Some(r) => match r.find('/') {
            Some(i) => &r[i..],
            None => "",
        },
        None => rest,
    };
    let end = rest.find(['?', '#']).unwrap_or(rest.len());
    &rest[..end]
}

/// The absolute form of a local path (`Path(p).absolute()`: no symlink resolution, no `..` folding).
pub fn absolute_path(p: &str) -> String {
    std::path::absolute(p).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| p.to_string())
}

/// The stored form of a media input string: an absolute path for local files (including `file://` URLs), the URL
/// itself otherwise.
pub fn normalize_input(s: &str) -> String {
    match url_scheme(s) {
        Some(scheme) if scheme == "file" => percent_decode(&percent_decode(url_path(s))),
        Some(scheme) if scheme.len() > 1 => s.to_string(),
        _ => absolute_path(s),
    }
}

/// The `fileurl` of a stored media value (`Path.as_uri()` for local files).
pub fn file_url(stored: &str) -> String {
    if is_local(stored) { format!("file://{}", percent_encode_path(stored)) } else { stored.to_string() }
}

// ------------------------------------------------------------------------------------------------ file cache

fn file_cache_dir() -> PathBuf {
    crate::store::pixeltable_home().join("file_cache")
}

/// The file name extension (with the dot) of a URL's path, if any.
fn url_extension(url: &str) -> String {
    let path = url_path(url);
    let name = path.rsplit('/').next().unwrap_or("");
    match name.rfind('.') {
        Some(i) if i > 0 && i + 1 < name.len() => name[i..].to_string(),
        _ => String::new(),
    }
}

/// Where the file cache keeps (or will keep) the download of `url`.
pub fn cache_path(url: &str) -> PathBuf {
    let key = Uuid::new_v5(&Uuid::NAMESPACE_URL, url.as_bytes()).simple().to_string();
    file_cache_dir().join(format!("{key}{}", url_extension(url)))
}

/// The URL actually fetched: public `s3://bucket/key` objects are read over HTTPS.
fn fetch_url(url: &str) -> String {
    if let Some(rest) = url.strip_prefix("s3://") {
        if let Some((bucket, key)) = rest.split_once('/') {
            return format!("https://{bucket}.s3.amazonaws.com/{key}");
        }
    }
    url.to_string()
}

/// Downloads `url` into the file cache (if it is not there yet) and returns the local path.
pub fn fetch(py: Python<'_>, url: &str) -> Result<PathBuf> {
    let dest = cache_path(url);
    if dest.is_file() {
        return Ok(dest);
    }
    std::fs::create_dir_all(file_cache_dir())?;
    let tmp = dest.with_extension(format!("part{}", Uuid::new_v4().simple()));
    let result = py
        .import("urllib.request")
        .and_then(|m| m.call_method1("urlretrieve", (fetch_url(url), tmp.to_string_lossy().to_string())));
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        let exc_type = e.get_type(py).name().map(|n| n.to_string()).unwrap_or_else(|_| "Exception".to_string());
        crate::bail!(ProviderError, "Failed to download {url}: {exc_type}");
    }
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

/// The local file of a stored media value: the path itself, or the cached download of a URL.
pub fn local_path(py: Python<'_>, stored: &str) -> Result<PathBuf> {
    if is_local(stored) { Ok(PathBuf::from(stored)) } else { fetch(py, stored) }
}

// ------------------------------------------------------------------------------------------------ media store

static PIL_IMAGE_MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// `PIL.Image`, with the HEIF opener registered when `pillow_heif` is installed (as upstream does at startup).
pub fn pil_image(py: Python<'_>) -> Result<Bound<'_, PyModule>> {
    let m = PIL_IMAGE_MODULE.get_or_try_init(py, || {
        let m = py.import("PIL.Image")?;
        if let Ok(heif) = py.import("pillow_heif") {
            heif.call_method0("register_heif_opener")?;
        }
        Ok::<_, PyErr>(m.unbind())
    })?;
    Ok(m.bind(py).clone())
}

/// Writes an in-memory media object (a PIL image or bytes) of column `col_id` to `media_dir`; returns its path.
pub fn save_object(
    obj: &Bound<'_, PyAny>,
    media_dir: &Path,
    tbl_id: Uuid,
    col_id: u32,
    version: i64,
) -> Result<String> {
    let py = obj.py();
    std::fs::create_dir_all(media_dir)?;
    let stem = format!("{}_{col_id}_{version}_{}", tbl_id.simple(), Uuid::new_v4().simple());
    if obj.is_instance(&pil_image(py)?.getattr("Image")?)? {
        // upstream `default_format()`: JPEG unless the image has transparency, which JPEG cannot represent
        let has_transparency: bool = obj.getattr("has_transparency_data")?.extract()?;
        let format = if has_transparency { "webp" } else { "jpeg" };
        let path = media_dir.join(format!("{stem}.{format}"));
        let path_str = path.to_string_lossy().to_string();
        let kwargs = PyDict::new(py);
        kwargs.set_item("format", format)?;
        if let Err(e) = obj.call_method("save", (&path_str,), Some(&kwargs)) {
            // modes JPEG cannot store (palette, 16-bit, ...): fall back to PNG
            let _ = std::fs::remove_file(&path);
            if format != "jpeg" {
                return Err(e.into());
            }
            let path = media_dir.join(format!("{stem}.png"));
            let path_str = path.to_string_lossy().to_string();
            obj.call_method1("save", (&path_str,))?;
            return Ok(path_str);
        }
        return Ok(path_str);
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        let path = media_dir.join(stem);
        std::fs::write(&path, b.as_bytes())?;
        return Ok(path.to_string_lossy().to_string());
    }
    Err(Error::internal(format!(
        "unsupported in-memory media value of type {}",
        obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
    )))
}

/// The value as a PIL image: opens (and loads) the file of a stored image, or returns an in-memory image.
pub fn open_image<'py>(py: Python<'py>, v: &Value) -> Result<Bound<'py, PyAny>> {
    match v {
        Value::PyObj(o) => Ok(o.bind(py).clone()),
        Value::Media(s) => {
            let path = local_path(py, s)?;
            let img = pil_image(py)?.call_method1("open", (path.to_string_lossy().to_string(),))?;
            img.call_method0("load")?;
            Ok(img)
        }
        other => Err(Error::internal(format!("expected an image, got a value of type {}", other.py_type_name()))),
    }
}

// ------------------------------------------------------------------------------------------------ validation

fn invalid(msg: String) -> Error {
    Error::new(crate::error::ErrorCode::InvalidDataFormat, msg)
}

fn is_instance_of(py: Python<'_>, e: &Error, module: &str, cls: &str) -> bool {
    match e {
        Error::Python(err) => py
            .import(module)
            .and_then(|m| m.getattr(cls))
            .map(|c| err.get_type(py).is_subclass(&c).unwrap_or(false))
            .unwrap_or(false),
        _ => false,
    }
}

fn validate_image(py: Python<'_>, path: &str) -> Result<()> {
    let r = (|| -> Result<()> {
        let img = pil_image(py)?.call_method1("open", (path,))?;
        img.call_method0("close")?;
        Ok(())
    })();
    match r {
        Err(e) if is_instance_of(py, &e, "PIL", "UnidentifiedImageError") => Err(invalid(format!("Not a valid image: {path}"))),
        other => other,
    }
}

/// Runs `f` with an open PyAV container, closing it afterwards.
fn with_av_container<T>(py: Python<'_>, path: &str, f: impl FnOnce(&Bound<'_, PyAny>) -> Result<T>) -> Result<T> {
    let container = py.import("av")?.call_method1("open", (path, "r"))?;
    let result = f(&container);
    let _ = container.call_method0("close");
    result
}

fn ffmpeg_error_text(py: Python<'_>, e: &Error) -> Option<String> {
    if is_instance_of(py, e, "av", "FFmpegError") { Some(e.message()) } else { None }
}

fn validate_video(py: Python<'_>, path: &str) -> Result<()> {
    let not_valid = || invalid(format!("Not a valid video: {path}"));
    let r = with_av_container(py, path, |c| {
        if c.getattr("streams")?.getattr("video")?.len()? == 0 {
            return Err(not_valid());
        }
        // decode a few frames to make sure the file is playable
        let kwargs = PyDict::new(py);
        kwargs.set_item("video", 0)?;
        let frames = c.call_method("decode", (), Some(&kwargs))?;
        let mut num_decoded = 0;
        for frame in frames.try_iter()? {
            frame?.call_method0("to_image")?;
            num_decoded += 1;
            if num_decoded == 10 {
                break;
            }
        }
        if num_decoded < 2 {
            // most likely an image file
            return Err(not_valid());
        }
        Ok(())
    });
    match r {
        Err(e) if ffmpeg_error_text(py, &e).is_some() => Err(not_valid()),
        other => other,
    }
}

fn validate_audio(py: Python<'_>, path: &str) -> Result<()> {
    let r = with_av_container(py, path, |c| {
        let audio = c.getattr("streams")?.getattr("audio")?;
        if audio.len()? == 0 {
            return Err(invalid(format!("No audio stream in file: {path}")));
        }
        let stream = audio.get_item(0)?;
        // decode everything to make sure the file is playable
        for packet in c.call_method1("demux", (stream,))?.try_iter()? {
            for frame in packet?.call_method0("decode")?.try_iter()? {
                frame?;
            }
        }
        Ok(())
    });
    match r {
        Err(e) => match ffmpeg_error_text(py, &e) {
            Some(text) => Err(invalid(format!("Not a valid audio file: {path}\n{text}"))),
            None => Err(e),
        },
        ok => ok,
    }
}

/// The document format of a file extension (upstream `DocumentType.DocumentFormat.from_extension()`).
fn document_format(ext: &str) -> Option<&'static str> {
    Some(match ext {
        ".htm" | ".html" => "HTML",
        ".md" => "MD",
        ".pdf" => "PDF",
        ".xml" => "XML",
        ".txt" => "TXT",
        ".pptx" | ".ppt" => "PPTX",
        ".docx" | ".doc" => "DOCX",
        ".xlsx" | ".xls" => "XLSX",
        _ => return None,
    })
}

/// Checks that a document of the given format can be read (a structural check of the content).
fn check_document(path: &str, format: &str) -> Result<()> {
    let processing_error = || invalid(format!("An error occurred processing a DocumentFormat.{format} document: {path}"));
    let bytes = std::fs::read(path).map_err(|_| processing_error())?;
    match format {
        "HTML" | "XML" => {
            let text = String::from_utf8(bytes).map_err(|_| processing_error())?;
            let has_element = text.match_indices('<').any(|(i, _)| {
                text[i + 1..].chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            });
            if !has_element {
                let kind = if format == "HTML" { "HTML" } else { "XML" };
                return Err(invalid(format!("Not a valid {kind} document: {path}")));
            }
            Ok(())
        }
        "MD" | "TXT" => String::from_utf8(bytes).map(|_| ()).map_err(|_| processing_error()),
        "PDF" => {
            if bytes.starts_with(b"%PDF") { Ok(()) } else { Err(processing_error()) }
        }
        _ => {
            // Office documents: OOXML (zip) or the legacy OLE container
            let is_zip = bytes.starts_with(b"PK\x03\x04");
            let is_ole = bytes.starts_with(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]);
            if is_zip || is_ole { Ok(()) } else { Err(processing_error()) }
        }
    }
}

/// The document format of a file without a known extension, from its content.
fn sniff_document_format(path: &str) -> Option<&'static str> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.starts_with(b"%PDF") {
        return Some("PDF");
    }
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(1024)]).to_ascii_lowercase();
    let head = head.trim_start();
    if head.starts_with("<!doctype html") || head.starts_with("<html") {
        return Some("HTML");
    }
    if head.starts_with("<?xml") {
        return Some("XML");
    }
    None
}

fn validate_document(path: &str) -> Result<()> {
    let ext = Path::new(path).extension().map(|e| format!(".{}", e.to_string_lossy().to_ascii_lowercase()));
    if let Some(format) = ext.as_deref().and_then(document_format) {
        return check_document(path, format);
    }
    if let Some(format) = sniff_document_format(path) {
        return check_document(path, format);
    }
    Err(invalid(format!("Unrecognized document format: {path}")))
}

/// Validates the local file `path` as media of the given kind (upstream `ColumnType.validate_media()`).
pub fn validate_file(py: Python<'_>, path: &str, kind: MediaKind) -> Result<()> {
    match kind {
        MediaKind::Image => validate_image(py, path),
        MediaKind::Video => validate_video(py, path),
        MediaKind::Audio => validate_audio(py, path),
        MediaKind::Document => validate_document(path),
    }
}

/// Validates a stored media value (fetching it first if it is remote).
pub fn validate_stored(py: Python<'_>, stored: &str, kind: MediaKind) -> Result<()> {
    let path = local_path(py, stored)?;
    validate_file(py, &path.to_string_lossy(), kind)
}

/// The cell error recorded for a failed validation or fetch.
pub fn validation_cell_error(e: &Error) -> CellError {
    CellError { errortype: e.py_type_name(), errormsg: e.message() }
}

/// Read-time validation of the media columns of a table whose files are validated on read: which columns, and
/// the outcome per file (memoized for the duration of a query).
#[derive(Debug, Default)]
pub struct ReadValidation {
    cols: std::collections::HashMap<u32, MediaKind>,
    cache: parking_lot::Mutex<std::collections::HashMap<(String, MediaKind), Option<CellError>>>,
}

impl ReadValidation {
    pub fn for_table(md: &crate::catalog::table::TableMd) -> Result<ReadValidation> {
        use crate::catalog::table::MediaValidation;
        let tbl_default = md.current_schema()?.media_validation;
        let cols = md
            .columns
            .iter()
            .filter(|c| !c.is_computed() && c.media_validation.unwrap_or(tbl_default) == MediaValidation::OnRead)
            .filter_map(|c| MediaKind::of(&c.col_type).map(|k| (c.id, k)))
            .collect();
        Ok(ReadValidation { cols, cache: Default::default() })
    }

    /// The media kind of `col_id` if it is validated on read.
    pub fn kind(&self, col_id: u32) -> Option<MediaKind> {
        self.cols.get(&col_id).copied()
    }

    /// The error of validating `stored`, if it is not valid media of `kind`.
    pub fn error(&self, stored: &str, kind: MediaKind) -> Option<CellError> {
        let key = (stored.to_string(), kind);
        if let Some(r) = self.cache.lock().get(&key) {
            return r.clone();
        }
        let outcome = Python::attach(|py| validate_stored(py, stored, kind)).err().map(|e| validation_cell_error(&e));
        self.cache.lock().insert(key, outcome.clone());
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemes_and_paths() {
        assert_eq!(url_scheme("s3://bucket/key").as_deref(), Some("s3"));
        assert_eq!(url_scheme("/a/b:c"), None);
        assert_eq!(url_scheme("C:\\x"), Some("c".to_string()));
        assert!(is_local("tests/data/x.jpg"));
        assert!(!is_local("https://x.org/a.jpg"));
        assert_eq!(normalize_input("file:///tmp/a%20b.jpg"), "/tmp/a b.jpg");
        assert_eq!(file_url("/tmp/#_x name!@$.jpg"), "file:///tmp/%23_x%20name%21%40%24.jpg");
        assert_eq!(url_extension("https://x.org/a/b.mp4?x=1"), ".mp4");
        assert_eq!(url_extension("https://x.org/download?file=a"), "");
        assert_eq!(percent_decode("a%2"), "a%2");
    }
}
