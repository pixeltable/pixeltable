//! Configuration values, looked up as upstream's `Config` does: the environment variable `<SECTION>_<KEY>`
//! (uppercased) first, then `key` in section `[section]` of the config file (`$PIXELTABLE_CONFIG`, else
//! `$PIXELTABLE_HOME/config.toml`). The file is read once per process.

use std::sync::OnceLock;

fn config_file() -> &'static Option<toml::Table> {
    static FILE: OnceLock<Option<toml::Table>> = OnceLock::new();
    FILE.get_or_init(|| {
        let path = match std::env::var("PIXELTABLE_CONFIG") {
            Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
            _ => crate::store::pixeltable_home().join("config.toml"),
        };
        std::fs::read_to_string(path).ok().and_then(|s| s.parse::<toml::Table>().ok())
    })
}

/// The value of `section.key` as a string (a TOML integer, float or boolean is rendered as text).
pub fn get(section: &str, key: &str) -> Option<String> {
    let env_name = format!("{}_{}", section.to_uppercase(), key.to_uppercase());
    if let Ok(v) = std::env::var(&env_name) {
        if !v.trim().is_empty() {
            return Some(v.trim().to_string());
        }
    }
    // a dotted section (`openai.rate_limits`) is a nested table in the file
    let mut table = config_file().as_ref()?;
    for part in section.split('.') {
        table = table.get(part)?.as_table()?;
    }
    match table.get(key)? {
        toml::Value::String(s) => Some(s.clone()),
        toml::Value::Integer(i) => Some(i.to_string()),
        toml::Value::Float(f) => Some(f.to_string()),
        toml::Value::Boolean(b) => Some(b.to_string()),
        _ => None,
    }
}

/// The value of `section.key` as an integer (None if unset or not an integer).
pub fn get_int(section: &str, key: &str) -> Option<i64> {
    get(section, key)?.parse().ok()
}
