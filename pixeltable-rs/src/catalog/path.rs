//! Catalog paths (`dir1/sub1/tbl`, optionally with a trailing `:version`), with upstream's validation rules and
//! messages. Paths are case-insensitive: components are folded to lower case.

use crate::error::{Result, py_repr};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Path {
    /// folded components; empty for the root directory
    pub components: Vec<String>,
    pub version: Option<i64>,
}

const URL_PREFIXES: [&str; 4] = [
    "https://www.pixeltable.com/t/",
    "http://www.pixeltable.com/t/",
    "https://pixeltable.com/t/",
    "http://pixeltable.com/t/",
];

/// Upstream `is_valid_identifier()`: an ASCII Python identifier that does not start with `_` (or `-`).
pub fn is_valid_identifier(name: &str, allow_hyphens: bool) -> bool {
    if !name.is_ascii() || name.starts_with('_') || name.starts_with('-') {
        return false;
    }
    let mut chars = name.chars().map(|c| if allow_hyphens && c == '-' { '_' } else { c });
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn fold_identifier(name: &str) -> String {
    name.to_ascii_lowercase()
}

impl Path {
    pub fn root() -> Path {
        Path { components: Vec::new(), version: None }
    }

    /// `Path.parse()`: validates the syntax and folds the components.
    pub fn parse(path: &str, allow_empty_path: bool, allow_versioned_path: bool) -> Result<Path> {
        let mut normalized = path.to_string();
        for prefix in URL_PREFIXES {
            if let Some(rest) = path.strip_prefix(prefix) {
                normalized = format!("pxt://{rest}");
            }
        }
        if normalized.starts_with("pxt://") {
            crate::bail!(
                UnsupportedOperation,
                "Hosted catalogs (pxt:// paths) are not supported by pixeltable-rs: {}",
                py_repr(path)
            );
        }
        let mut path_part = normalized.as_str();
        let mut version: Option<i64> = None;
        if path_part.contains(':') {
            let parts: Vec<&str> = path_part.split(':').collect();
            if parts.len() != 2 {
                crate::bail!(InvalidPath, "Invalid path: {path}");
            }
            path_part = parts[0];
            match parts[1].trim().parse::<i64>() {
                Ok(v) => version = Some(v),
                Err(_) => crate::bail!(InvalidPath, "Invalid path: {path}"),
            }
        }
        let components: Vec<&str> = if path_part.contains('/') {
            path_part.split('/').collect()
        } else if path_part.contains('.') {
            path_part.split('.').collect()
        } else if path_part.is_empty() {
            Vec::new()
        } else {
            vec![path_part]
        };
        if components.is_empty() && !allow_empty_path {
            crate::bail!(InvalidPath, "Invalid path: {path}");
        }
        if !components.iter().all(|c| is_valid_identifier(c, true)) {
            crate::bail!(InvalidPath, "Invalid path: {path}");
        }
        if version.is_some() && !allow_versioned_path {
            crate::bail!(InvalidPath, "Versioned path not allowed here: {path}");
        }
        if let Some(v) = version {
            if v < 0 {
                crate::bail!(InvalidPath, "Version must be non-negative: {v}");
            }
        }
        Ok(Path { components: components.iter().map(|c| fold_identifier(c)).collect(), version })
    }

    pub fn is_root(&self) -> bool {
        self.components.is_empty()
    }

    pub fn len(&self) -> usize {
        self.components.len()
    }

    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// The last component (empty for the root).
    pub fn name(&self) -> &str {
        self.components.last().map(String::as_str).unwrap_or("")
    }

    pub fn parent(&self) -> Path {
        let mut components = self.components.clone();
        components.pop();
        Path { components, version: None }
    }

    pub fn append(&self, name: &str) -> Path {
        let mut components = self.components.clone();
        components.push(name.to_string());
        Path { components, version: None }
    }

    pub fn without_version(&self) -> Path {
        Path { components: self.components.clone(), version: None }
    }

    /// True if `self` is a proper ancestor of `other`.
    pub fn is_ancestor(&self, other: &Path) -> bool {
        if self.len() >= other.len() || other.is_root() {
            return false;
        }
        self.components[..] == other.components[..self.len()]
    }

    /// All proper ancestors, top-down, starting with the root.
    pub fn ancestors(&self) -> Vec<Path> {
        (0..self.components.len()).map(|i| Path { components: self.components[..i].to_vec(), version: None }).collect()
    }

    /// `repr(path)` in upstream messages: the quoted path string.
    pub fn repr(&self) -> String {
        py_repr(&self.to_string())
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.components.join("/"))?;
        if let Some(v) = self.version {
            write!(f, ":{v}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers() {
        assert!(is_valid_identifier("dir1", false));
        assert!(!is_valid_identifier("1dir", false));
        assert!(!is_valid_identifier("_dir", false));
        assert!(!is_valid_identifier("dir 1", false));
        assert!(is_valid_identifier("hyphen-ated", true));
        assert!(!is_valid_identifier("hyphen-ated", false));
    }

    #[test]
    fn parse() {
        let p = Path::parse("Dir1/Sub", false, false).ok();
        assert_eq!(p.map(|p| p.components), Some(vec!["dir1".to_string(), "sub".to_string()]));
        assert!(Path::parse("dir1..sub2", false, false).is_err());
        assert!(Path::parse("dir1:120", false, false).is_err());
        assert_eq!(Path::parse("t:3", false, true).ok().and_then(|p| p.version), Some(3));
        assert!(Path::parse("", true, false).map(|p| p.is_root()).unwrap_or(false));
    }
}
