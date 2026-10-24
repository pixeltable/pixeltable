//! Errors produced by the backend.
//!
//! User-facing errors are `Error::Pxt { code, message }`; at the Python boundary they are raised as
//! `pixeltable.exceptions._error_class(code)(code, message)`, so the Python class always matches the code.
//! A few Python builtin exceptions are part of the public contract (for example `TypeError` from
//! `ColumnType.validate_literal()` or `AttributeError` for an unknown column attribute); those have their own
//! variants. Errors raised by Python code that Rust calls (UDFs, PIL, numpy) pass through unchanged as `Python`.

use pyo3::exceptions::{PyAttributeError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::fmt;

/// Mirrors `pixeltable.exceptions.ErrorCode`; `name()` must match the Python member name exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    InternalError,
    GenericUserError,
    // NotFoundError
    ColumnNotFound,
    PathNotFound,
    TableNotFound,
    DirectoryNotFound,
    IndexNotFound,
    FunctionNotFound,
    RowNotFound,
    StorageNotFound,
    // AlreadyExistsError
    ColumnAlreadyExists,
    PathAlreadyExists,
    IndexAlreadyExists,
    FunctionAlreadyExists,
    // RequestError
    InvalidColumnName,
    InvalidPath,
    InvalidExpression,
    InvalidType,
    InvalidSchema,
    InvalidArgument,
    InvalidDataFormat,
    MissingRequired,
    TypeMismatch,
    ConstraintViolation,
    UnsupportedOperation,
    InvalidState,
    InvalidConfiguration,
    SchemaMismatch,
    // ExternalServiceError
    ProviderError,
    // ServiceUnavailableError
    StoreUnavailable,
    // ConcurrencyError
    ConcurrentModification,
}

impl ErrorCode {
    pub fn name(self) -> &'static str {
        use ErrorCode::*;
        match self {
            InternalError => "INTERNAL_ERROR",
            GenericUserError => "GENERIC_USER_ERROR",
            ColumnNotFound => "COLUMN_NOT_FOUND",
            PathNotFound => "PATH_NOT_FOUND",
            TableNotFound => "TABLE_NOT_FOUND",
            DirectoryNotFound => "DIRECTORY_NOT_FOUND",
            IndexNotFound => "INDEX_NOT_FOUND",
            FunctionNotFound => "FUNCTION_NOT_FOUND",
            RowNotFound => "ROW_NOT_FOUND",
            StorageNotFound => "STORAGE_NOT_FOUND",
            ColumnAlreadyExists => "COLUMN_ALREADY_EXISTS",
            PathAlreadyExists => "PATH_ALREADY_EXISTS",
            IndexAlreadyExists => "INDEX_ALREADY_EXISTS",
            FunctionAlreadyExists => "FUNCTION_ALREADY_EXISTS",
            InvalidColumnName => "INVALID_COLUMN_NAME",
            InvalidPath => "INVALID_PATH",
            InvalidExpression => "INVALID_EXPRESSION",
            InvalidType => "INVALID_TYPE",
            InvalidSchema => "INVALID_SCHEMA",
            InvalidArgument => "INVALID_ARGUMENT",
            InvalidDataFormat => "INVALID_DATA_FORMAT",
            MissingRequired => "MISSING_REQUIRED",
            TypeMismatch => "TYPE_MISMATCH",
            ConstraintViolation => "CONSTRAINT_VIOLATION",
            UnsupportedOperation => "UNSUPPORTED_OPERATION",
            InvalidState => "INVALID_STATE",
            InvalidConfiguration => "INVALID_CONFIGURATION",
            SchemaMismatch => "SCHEMA_MISMATCH",
            ProviderError => "PROVIDER_ERROR",
            StoreUnavailable => "STORE_UNAVAILABLE",
            ConcurrentModification => "CONCURRENT_MODIFICATION",
        }
    }
}

#[derive(Debug)]
pub enum Error {
    /// A `pixeltable.exceptions.Error` subclass, chosen by the code.
    Pxt { code: ErrorCode, message: String },
    /// Python `TypeError` (part of the public contract of type validation and operator overloading).
    Type(String),
    /// Python `AttributeError`.
    Attribute(String),
    /// Python `ValueError`.
    Value(String),
    /// Python `NotImplementedError`: API that exists but is not implemented yet.
    NotImplemented(String),
    /// An exception raised by Python code that the backend called; re-raised unchanged.
    Python(PyErr),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Error::Pxt { code, message: message.into() }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Error::new(ErrorCode::InternalError, message)
    }

    pub fn not_implemented(what: impl fmt::Display) -> Self {
        Error::NotImplemented(format!("{what} is not implemented yet in pixeltable-rs"))
    }

    /// The error code, for `Pxt` errors.
    pub fn code(&self) -> Option<ErrorCode> {
        match self {
            Error::Pxt { code, .. } => Some(*code),
            _ => None,
        }
    }

    /// The name of the Python exception class this error is raised as.
    pub fn py_type_name(&self) -> String {
        match self {
            Error::Pxt { code, .. } => {
                use ErrorCode::*;
                match code {
                    InternalError => "InternalError",
                    GenericUserError => "UserError",
                    ColumnNotFound | PathNotFound | TableNotFound | DirectoryNotFound | IndexNotFound
                    | FunctionNotFound | RowNotFound | StorageNotFound => "NotFoundError",
                    ColumnAlreadyExists | PathAlreadyExists | IndexAlreadyExists | FunctionAlreadyExists => {
                        "AlreadyExistsError"
                    }
                    ProviderError => "ExternalServiceError",
                    StoreUnavailable => "ServiceUnavailableError",
                    ConcurrentModification => "ConcurrencyError",
                    _ => "RequestError",
                }
                .to_string()
            }
            Error::Type(_) => "TypeError".to_string(),
            Error::Attribute(_) => "AttributeError".to_string(),
            Error::Value(_) => "ValueError".to_string(),
            Error::NotImplemented(_) => "NotImplementedError".to_string(),
            Error::Python(e) => Python::attach(|py| {
                e.get_type(py).name().map(|n| n.to_string()).unwrap_or_else(|_| "Exception".to_string())
            }),
        }
    }

    pub fn message(&self) -> String {
        match self {
            Error::Pxt { message, .. }
            | Error::Type(message)
            | Error::Attribute(message)
            | Error::Value(message)
            | Error::NotImplemented(message) => message.clone(),
            // `str(exc)`, as upstream records and reports exceptions
            Error::Python(e) => Python::attach(|py| e.value(py).str().map(|s| s.to_string()).unwrap_or_else(|_| e.to_string())),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Pxt { code, message } => write!(f, "{}: {}", code.name(), message),
            other => write!(f, "{}", other.message()),
        }
    }
}

impl From<PyErr> for Error {
    fn from(e: PyErr) -> Self {
        Error::Python(e)
    }
}

impl From<std::convert::Infallible> for Error {
    fn from(e: std::convert::Infallible) -> Self {
        match e {}
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::new(ErrorCode::StoreUnavailable, format!("I/O error in the Pixeltable store: {e}"))
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::internal(format!("corrupt or unserializable metadata: {e}"))
    }
}

fn make_pxt_exception(py: Python<'_>, code: ErrorCode, message: &str) -> PyResult<PyErr> {
    let excs = py.import("pixeltable.exceptions")?;
    let code_obj = excs.getattr("ErrorCode")?.get_item(code.name())?;
    let cls = excs.getattr("_error_class")?.call1((&code_obj,))?;
    let args = PyTuple::new(py, [code_obj, message.into_pyobject(py)?.into_any()])?;
    let exc = cls.call1(args)?;
    Ok(PyErr::from_value(exc))
}

impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        match e {
            Error::Pxt { code, message } => Python::attach(|py| match make_pxt_exception(py, code, &message) {
                Ok(err) => err,
                Err(import_err) => PyRuntimeError::new_err(format!(
                    "{}: {} (could not construct pixeltable exception: {})",
                    code.name(),
                    message,
                    import_err
                )),
            }),
            Error::Type(m) => PyTypeError::new_err(m),
            Error::Attribute(m) => PyAttributeError::new_err(m),
            Error::Value(m) => PyValueError::new_err(m),
            Error::NotImplemented(m) => PyNotImplementedError::new_err(m),
            Error::Python(e) => e,
        }
    }
}

/// `pxt_err!(Code, "fmt", args...)` builds an `Error::Pxt`.
#[macro_export]
macro_rules! pxt_err {
    ($code:ident, $($arg:tt)*) => {
        $crate::error::Error::new($crate::error::ErrorCode::$code, format!($($arg)*))
    };
}

/// `bail!(Code, "fmt", args...)` returns an `Error::Pxt` from the enclosing function.
#[macro_export]
macro_rules! bail {
    ($code:ident, $($arg:tt)*) => {
        return Err($crate::pxt_err!($code, $($arg)*))
    };
}

/// Python's `repr()` of a str, used to reproduce upstream messages that format with `{x!r}`.
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}
