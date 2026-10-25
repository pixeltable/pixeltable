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
    SerializationFailure,
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
            SerializationFailure => "SERIALIZATION_FAILURE",
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
    /// Another Python builtin exception (by class name, e.g. `ZeroDivisionError`), for failures of builtin
    /// functions and operators that upstream reports as the exception Python raises.
    PyExc(&'static str, String),
    /// An exception raised by Python code that the backend called; re-raised unchanged.
    Python(PyErr),
    /// An evaluation failure with where it happened (upstream `ExprEvalError`), for the message that
    /// `exec::compute::eval_error()` builds; everything else (code, class, message) is that of the wrapped error.
    Located(Box<Located>),
}

/// Where an evaluation failed: the innermost node whose own evaluation failed (or an aggregate's `init()`/`update()`/
/// `value()`) and the values of its inputs (upstream's `dependencies()` of the node), in input order.
#[derive(Debug)]
pub struct Located {
    pub site: EvalSite,
    pub inputs: Vec<(crate::expr::ExprRef, crate::value::Value)>,
    pub err: Error,
}

#[derive(Debug, Clone)]
pub enum EvalSite {
    /// "the expression {node}"
    Expr(crate::expr::ExprRef),
    /// "the init() function of the aggregate {call}"
    AggInit(std::sync::Arc<crate::expr::FunctionCall>),
    /// "the update() function of the aggregate {call}"
    AggUpdate(std::sync::Arc<crate::expr::FunctionCall>),
    /// value() of an aggregate: "the expression {call}"
    AggValue(std::sync::Arc<crate::expr::FunctionCall>),
    /// not an evaluation failure but an invalid setup that upstream detects before evaluating any row (it raises the
    /// error as it is, whatever `on_error` says)
    Setup,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Error::Pxt { code, message: message.into() }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Error::new(ErrorCode::InternalError, message)
    }

    pub fn py_exc(class_name: &'static str, message: impl Into<String>) -> Self {
        Error::PyExc(class_name, message.into())
    }

    pub fn not_implemented(what: impl fmt::Display) -> Self {
        Error::NotImplemented(format!("{what} is not implemented yet in pixeltable-rs"))
    }

    /// The error code, for `Pxt` errors.
    pub fn code(&self) -> Option<ErrorCode> {
        match self {
            Error::Pxt { code, .. } => Some(*code),
            Error::Located(l) => l.err.code(),
            _ => None,
        }
    }

    /// The error without the location of an evaluation failure.
    pub fn unlocated(self) -> Error {
        match self {
            Error::Located(l) => l.err,
            e => e,
        }
    }

    /// A copy of this error (a Python exception is shared by reference).
    pub fn clone_ref(&self, py: Python<'_>) -> Error {
        match self {
            Error::Python(e) => Error::Python(e.clone_ref(py)),
            other => other.clone_plain(),
        }
    }

    /// `clone_ref()`, attaching to the interpreter only for a Python exception.
    pub fn duplicate(&self) -> Error {
        match self {
            Error::Python(_) => Python::attach(|py| self.clone_ref(py)),
            other => other.clone_plain(),
        }
    }

    fn clone_plain(&self) -> Error {
        match self {
            Error::Pxt { code, message } => Error::Pxt { code: *code, message: message.clone() },
            Error::Type(m) => Error::Type(m.clone()),
            Error::Attribute(m) => Error::Attribute(m.clone()),
            Error::Value(m) => Error::Value(m.clone()),
            Error::NotImplemented(m) => Error::NotImplemented(m.clone()),
            Error::PyExc(name, m) => Error::PyExc(name, m.clone()),
            Error::Python(e) => Python::attach(|py| Error::Python(e.clone_ref(py))),
            Error::Located(l) => Error::Located(Box::new(Located {
                site: l.site.clone(),
                inputs: l.inputs.clone(),
                err: l.err.clone_plain(),
            })),
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
                    SerializationFailure | ConcurrentModification => "ConcurrencyError",
                    _ => "RequestError",
                }
                .to_string()
            }
            Error::Type(_) => "TypeError".to_string(),
            Error::Attribute(_) => "AttributeError".to_string(),
            Error::Value(_) => "ValueError".to_string(),
            Error::NotImplemented(_) => "NotImplementedError".to_string(),
            Error::PyExc(name, _) => name.to_string(),
            Error::Python(e) => Python::attach(|py| {
                e.get_type(py).name().map(|n| n.to_string()).unwrap_or_else(|_| "Exception".to_string())
            }),
            Error::Located(l) => l.err.py_type_name(),
        }
    }

    pub fn message(&self) -> String {
        match self {
            Error::Pxt { message, .. }
            | Error::Type(message)
            | Error::Attribute(message)
            | Error::Value(message)
            | Error::NotImplemented(message)
            | Error::PyExc(_, message) => message.clone(),
            // `str(exc)`, as upstream records it in cell errors and messages
            Error::Python(e) => Python::attach(|py| e.value(py).str().map(|s| s.to_string()).unwrap_or_else(|_| e.to_string())),
            Error::Located(l) => l.err.message(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Pxt { code, message } => write!(f, "{}: {}", code.name(), message),
            Error::Located(l) => l.err.fmt(f),
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
            Error::PyExc(name, m) => Python::attach(|py| {
                match py.import("builtins").and_then(|b| b.getattr(name)).and_then(|cls| cls.call1((m.clone(),))) {
                    Ok(exc) => PyErr::from_value(exc),
                    Err(e) => e,
                }
            }),
            Error::Python(e) => e,
            Error::Located(l) => l.err.into(),
        }
    }
}

/// Builds the `pixeltable.exceptions` error for `code` with `message` and the diagnostic `detail` (shown after the
/// message by `str()`), caused by `cause`.
pub fn pxt_exception_with_detail(py: Python<'_>, code: ErrorCode, message: &str, detail: &str, cause: Option<&PyErr>) -> Error {
    let exc = match make_pxt_exception(py, code, message) {
        Ok(e) => e,
        Err(_) => return Error::new(code, message),
    };
    let value = exc.value(py);
    if value.setattr("detail", detail).is_err() {
        return Error::new(code, message);
    }
    if let Some(c) = cause {
        value.setattr("__cause__", c.value(py)).ok();
    }
    Error::Python(exc)
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
