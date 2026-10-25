//! Builtin iterators over strings: `pixeltable.functions.string.string_splitter` (upstream
//! `pixeltable/functions/string.py`); sentence segmentation goes through spaCy.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

use super::iterator::{BuiltinIterator, IteratorCall, RowSource};
use super::signature::{Param, ParamKind};
use crate::error::Result;
use crate::expr::Expr;
use crate::types::ColumnType;
use crate::value::Value;

pub static STRING_SPLITTER: BuiltinIterator = BuiltinIterator {
    fqn: "pixeltable.functions.string.string_splitter",
    defined_at: "pixeltable.functions.string.string_splitter",
    params: splitter_params,
    unstored_cols: &[],
    output_schema: splitter_schema,
    validate: validate_splitter_args,
    start: start_string_splitter,
};

/// `(text: str, separators: str, *, spacy_model: str = 'en_core_web_sm')`
fn splitter_params() -> Vec<Param> {
    let param = |name: &str, kind: ParamKind, default: Option<&str>| Param {
        name: name.into(),
        kind,
        col_type: Some(ColumnType::string(false)),
        default: default.map(|d| Expr::literal(Value::String(d.into()), ColumnType::string(false))),
        is_batched: false,
    };
    vec![
        param("text", ParamKind::PositionalOrKeyword, None),
        param("separators", ParamKind::PositionalOrKeyword, None),
        param("spacy_model", ParamKind::KeywordOnly, Some("en_core_web_sm")),
    ]
}

fn splitter_schema(_args: &[(String, Value)]) -> Result<Vec<(String, ColumnType)>> {
    Ok(vec![("text".to_string(), ColumnType::string(false))])
}

fn str_arg<'a>(args: &'a [(String, Value)], name: &str) -> Option<&'a str> {
    match args.iter().find(|(n, _)| n == name).map(|(_, v)| v) {
        Some(Value::String(s)) => Some(s),
        _ => None,
    }
}

fn validate_splitter_args(args: &[(String, Value)]) -> Result<()> {
    match str_arg(args, "separators") {
        Some(s) if s != "sentence" => crate::bail!(
            UnsupportedOperation,
            "Only 'sentence' is supported as a separator in `string_splitter` iterator."
        ),
        _ => Ok(()),
    }
}

/// The loaded spaCy pipeline `name` (upstream `get_spacy_model()`), loaded once per process.
fn spacy_model(py: Python<'_>, name: &str) -> Result<Py<PyAny>> {
    static MODELS: Mutex<Option<HashMap<String, Py<PyAny>>>> = Mutex::new(None);
    if let Some(m) = MODELS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(name)) {
        return Ok(m.clone_ref(py));
    }
    let spacy = py.import("spacy").map_err(|_| {
        crate::pxt_err!(
            UnsupportedOperation,
            "This feature requires the `spacy` package. To install it, run: `pip install -U spacy`"
        )
    })?;
    let model = match spacy.call_method1("load", (name,)) {
        Ok(m) => m.unbind(),
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyOSError>(py) => crate::bail!(
            UnsupportedOperation,
            "Failed to locate spaCy model {}. To install it, run:\n    python -m spacy download {name}",
            crate::error::py_repr(name)
        ),
        Err(e) => return Err(e.into()),
    };
    let mut models = MODELS.lock().unwrap_or_else(|e| e.into_inner());
    Ok(models.get_or_insert_with(HashMap::new).entry(name.to_string()).or_insert(model).clone_ref(py))
}

/// The sentences of the text, computed when the iterator starts.
struct Sentences {
    sentences: Vec<String>,
    next: usize,
}

fn start_string_splitter(py: Python<'_>, args: &[(String, Value)]) -> Result<Box<dyn RowSource>> {
    let text = str_arg(args, "text").unwrap_or_default();
    let model = spacy_model(py, str_arg(args, "spacy_model").unwrap_or("en_core_web_sm"))?;
    let doc = model.bind(py).call1((text,))?;
    let sentences = doc
        .getattr("sents")?
        .try_iter()?
        .map(|s| Ok(s?.getattr("text")?.extract::<String>()?))
        .collect::<Result<Vec<_>>>()?;
    Ok(Box::new(Sentences { sentences, next: 0 }))
}

impl RowSource for Sentences {
    fn next_row(&mut self, _py: Python<'_>, call: &IteratorCall) -> Result<Option<Vec<Value>>> {
        let Some(s) = self.sentences.get(self.next) else { return Ok(None) };
        self.next += 1;
        Ok(Some(call.value_outputs().iter().map(|_| Value::String(s.clone())).collect()))
    }

    fn seek(&mut self, _py: Python<'_>, _call: &IteratorCall, pos: i64, _stored: &[(String, Value)]) -> Result<()> {
        self.next = usize::try_from(pos).unwrap_or(usize::MAX);
        Ok(())
    }

    fn close(&mut self, _py: Python<'_>) {}
}
