//! Index operations behind `Table.add_btree_index()`, `add_embedding_index()`, `drop_index()`,
//! `drop_embedding_index()`, `ColumnRef.similarity()`, and the `indexes` section of `get_metadata()`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use std::sync::Arc;

use super::catalog::parse_uuid;
use super::expr::{PyExpr, as_expr};
use crate::catalog::indexes::{ColumnArg, IndexSpec};
use crate::catalog::runtime::with_catalog;
use crate::catalog::table::{IndexKind, ResolvedColumn};
use crate::catalog::{Catalog, IfExists, IfNotExists};
use crate::error::{Error, Result, py_repr};
use crate::exec::column_ref_expr;
use crate::exec::index::AddIndexPlan;
use crate::expr::{Expr, ExprKind, ExprRef, Similarity};
use crate::functions::udf;
use crate::types::{ColumnType, TypeKind};
use crate::value::Json;

const METRICS: [&str; 3] = ["cosine", "ip", "l2"];
const PRECISIONS: [&str; 2] = ["fp16", "fp32"];
/// Query modalities of embedding indexes, as `ColumnType.Type` names.
const MODALITIES: [&str; 5] = ["STRING", "IMAGE", "AUDIO", "VIDEO", "DOCUMENT"];

/// A column parameter (a name or a column reference); upstream `_resolve_column_parameter()` checks.
pub fn column_arg(obj: &Bound<'_, PyAny>) -> Result<ColumnArg> {
    if obj.is_instance_of::<PyString>() {
        return Ok(ColumnArg::Name(obj.extract()?));
    }
    if let Some(e) = as_expr(obj) {
        if let Some(c) = e.as_column_ref() {
            return Ok(ColumnArg::Ref { tbl_id: c.tbl_id, col_id: c.col_id, name: c.name.clone() });
        }
    }
    crate::bail!(TypeMismatch, "Invalid column parameter type: {}", obj.get_type().str()?)
}

fn type_name_of(t: &ColumnType) -> &'static str {
    match t.kind {
        TypeKind::String => "STRING",
        TypeKind::Image { .. } => "IMAGE",
        TypeKind::Audio => "AUDIO",
        TypeKind::Video => "VIDEO",
        TypeKind::Document { .. } => "DOCUMENT",
        TypeKind::Array { .. } => "ARRAY",
        _ => "",
    }
}

fn fn_name(f: &Bound<'_, PyAny>) -> String {
    f.getattr("name").and_then(|n| n.str()).map(|s| s.to_string()).unwrap_or_else(|_| "None".to_string())
}

/// Upstream `EmbeddingIndex._validate_embedding_fn()`.
fn validate_embedding_return_type(name: &str, t: &ColumnType) -> Result<()> {
    let TypeKind::Array { shape, .. } = &t.kind else {
        crate::bail!(
            InvalidConfiguration,
            "The function `{name}` is not a valid embedding: it must return an array, but returns {}",
            t.repr()
        );
    };
    match shape.as_deref() {
        Some([Some(n)]) if *n > 0 => Ok(()),
        Some([Some(n)]) => crate::bail!(
            InvalidConfiguration,
            "The function `{name}` is not a valid embedding: it returns an array of invalid length {n}"
        ),
        Some([None]) => crate::bail!(
            InvalidConfiguration,
            "The function `{name}` is not a valid embedding: it must return a 1-dimensional array of a specific length"
        ),
        _ => crate::bail!(
            InvalidConfiguration,
            "The function `{name}` is not a valid embedding: it must return a 1-dimensional array, but returns {}",
            t.repr()
        ),
    }
}

/// Upstream `EmbeddingIndex.resolve_embeddings()` and its validation: (modality, unary embedding function) pairs.
fn resolve_embeddings<'py>(
    py: Python<'py>,
    embedding: Option<&Bound<'py, PyAny>>,
    per_type: &[(&'static str, Option<&Bound<'py, PyAny>>)],
) -> Result<Vec<(&'static str, Bound<'py, PyAny>)>> {
    let helper = py.import("pixeltable._typehints")?;
    let mut out = Vec::new();
    for (t, f) in per_type {
        let lower = t.to_ascii_lowercase();
        if let Some(f) = f {
            let resolved = helper.call_method1("resolve_embedding_fn", (f, *t))?;
            if resolved.is_none() {
                crate::bail!(
                    InvalidConfiguration,
                    "The function `{}` is not a valid {lower} embedding: it must take a single {lower} parameter",
                    fn_name(f)
                );
            }
            out.push((*t, resolved));
        } else if let Some(e) = embedding {
            let resolved = helper.call_method1("resolve_embedding_fn", (e, *t))?;
            if !resolved.is_none() {
                out.push((*t, resolved));
            }
        }
    }
    if let Some(e) = embedding {
        if out.is_empty() {
            crate::bail!(
                InvalidConfiguration,
                "The function `{}` is not a valid embedding: it must take a single string, image, audio, video, or document parameter",
                fn_name(e)
            );
        }
    }
    Ok(out)
}

/// `Table.add_embedding_index()`
#[pyfunction]
#[pyo3(signature = (
    tbl_id, column, idx_name, embedding, string_embed, image_embed, audio_embed, video_embed, document_embed,
    metric, precision, if_exists
))]
#[allow(clippy::too_many_arguments)]
pub fn tbl_add_embedding_index<'py>(
    py: Python<'py>,
    tbl_id: &str,
    column: &Bound<'py, PyAny>,
    idx_name: Option<&str>,
    embedding: Option<&Bound<'py, PyAny>>,
    string_embed: Option<&Bound<'py, PyAny>>,
    image_embed: Option<&Bound<'py, PyAny>>,
    audio_embed: Option<&Bound<'py, PyAny>>,
    video_embed: Option<&Bound<'py, PyAny>>,
    document_embed: Option<&Bound<'py, PyAny>>,
    metric: &str,
    precision: &str,
    if_exists: &str,
) -> Result<()> {
    let id = parse_uuid(tbl_id)?;
    let some = |f: Option<&Bound<'py, PyAny>>| f.filter(|f| !f.is_none()).cloned();
    let (embedding, string_embed, image_embed) = (some(embedding), some(string_embed), some(image_embed));
    let (audio_embed, video_embed, document_embed) = (some(audio_embed), some(video_embed), some(document_embed));
    let function_cls = py.import("pixeltable.func")?.getattr("Function")?;
    for (name, f) in [("embedding", &embedding), ("string_embed", &string_embed), ("image_embed", &image_embed)] {
        if let Some(f) = f {
            if !f.is_instance(&function_cls)? {
                crate::bail!(InvalidArgument, "`{name}` must be a Pixeltable function; got `{}`", f.get_type().name()?);
            }
        }
    }
    let col_arg = column_arg(column)?;
    let (col, md) = with_catalog(py, |cat| {
        cat.check_mutable(id, None, "add an index to")?;
        Ok((cat.resolve_column_arg(id, &col_arg)?, cat.table(id)?.clone()))
    })?;
    let mut replace = Vec::new();
    let idx_name = idx_name.map(Catalog::validate_index_name).transpose()?;
    if let Some(n) = &idx_name {
        if let Some(existing) = md.index_by_name(n) {
            let ie = IfExists::parse(if_exists, "if_exists")?;
            if ie == IfExists::Error {
                crate::bail!(IndexAlreadyExists, "Duplicate index name: {n}");
            }
            if existing.is_btree() {
                let op = if ie == IfExists::Ignore { "ignore" } else if ie == IfExists::Replace { "replace" } else { "replace_force" };
                crate::bail!(UnsupportedOperation, "Index {} is not an embedding index. Cannot {op} it.", py_repr(n));
            }
            if ie == IfExists::Ignore {
                return Ok(());
            }
            replace.push(existing.id);
        }
    }
    // upstream `EmbeddingIndex.__init__()`
    let col_type = &col.md.col_type;
    let any_fn = [&embedding, &string_embed, &image_embed, &audio_embed, &video_embed, &document_embed].iter().any(|f| f.is_some());
    if !col_type.is_array() && !any_fn {
        crate::bail!(
            MissingRequired,
            "At least one of `embed`, `string_embed`, `image_embed`, `audio_embed`, `video_embed`, or `document_embed` must be specified"
        );
    }
    if !METRICS.contains(&metric.to_ascii_lowercase().as_str()) {
        crate::bail!(InvalidArgument, "Invalid metric {metric}, must be one of ['cosine', 'ip', 'l2']");
    }
    let per_type = [
        ("STRING", string_embed.as_ref()),
        ("IMAGE", image_embed.as_ref()),
        ("AUDIO", audio_embed.as_ref()),
        ("VIDEO", video_embed.as_ref()),
        ("DOCUMENT", document_embed.as_ref()),
    ];
    let resolved = resolve_embeddings(py, embedding.as_ref(), &per_type)?;
    let mut defs = Vec::new();
    for (t, f) in &resolved {
        let def = udf::def_of(f)?;
        if !def.storable {
            let display: String = f.getattr("display_name")?.str()?.to_string();
            crate::bail!(
                UnsupportedOperation,
                "The {} embedding `{display}()` was created with `.apply()` or defined as a local function.\nTo use it in an embedding index, define it as a module-level `@pxt.udf` and use that instead.",
                t.to_ascii_lowercase()
            );
        }
        let ret = def.signatures.first().map(|s| s.return_type.clone()).unwrap_or_else(|| ColumnType::invalid(true));
        validate_embedding_return_type(&fn_name(f), &ret)?;
        defs.push((*t, def, f.clone()));
    }
    if !PRECISIONS.contains(&precision) {
        crate::bail!(InvalidArgument, "Invalid precision '{precision}'. Must be one of: ['fp16', 'fp32']");
    }
    // upstream `create_value_expr()`
    let t_name = type_name_of(col_type);
    if t_name.is_empty() {
        crate::bail!(
            TypeMismatch,
            "Type `{}` of column {} is not a valid type for an embedding index.",
            col_type.repr(),
            py_repr(&col.name)
        );
    }
    let col_expr = column_ref_expr(&col, None);
    let value_expr: ExprRef = if col_type.is_array() {
        col_expr
    } else {
        let Some((_, def, _)) = defs.iter().find(|(t, _, _)| *t == t_name) else {
            crate::bail!(
                TypeMismatch,
                "The specified embedding function does not support the type `{}` of column {}.",
                col_type.repr(),
                py_repr(&col.name)
            );
        };
        udf::make_call(def, vec![col_expr], vec![], false)?
    };
    let mut embeddings: Vec<(String, Json)> = Vec::new();
    for (t, def, _) in &defs {
        embeddings.push((t.to_string(), def.descriptor()?));
    }
    let metric = metric.to_ascii_lowercase();
    if idx_name.is_none() {
        let matches: Vec<(u32, String)> = md
            .indexes
            .list
            .iter()
            .filter(|i| i.col_tbl_id == col.tbl_id && i.col_id == col.md.id)
            .filter(|i| match &i.kind {
                IndexKind::Embedding { metric: m, precision: p, embeddings: e, .. } => {
                    *m == metric && p == precision && *e == embeddings
                }
                IndexKind::Btree { .. } => false,
            })
            .map(|i| (i.id, i.name.clone()))
            .collect();
        if let Some((_, first)) = matches.first() {
            let ie = IfExists::parse(if_exists, "if_exists")?;
            match ie {
                IfExists::Error => crate::bail!(
                    IndexAlreadyExists,
                    "An identical embedding index already exists on column {} (index {}). Pass a distinct idx_name, or if_exists='ignore' / 'replace'.",
                    py_repr(&col.name),
                    py_repr(first)
                ),
                IfExists::Ignore => return Ok(()),
                _ => replace.extend(matches.iter().map(|(id, _)| *id)),
            }
        }
    }
    let spec = IndexSpec {
        name: idx_name,
        col_tbl_id: col.tbl_id,
        col_id: col.md.id,
        metric,
        precision: precision.to_string(),
        embeddings,
    };
    let plan = with_catalog(py, |cat| AddIndexPlan::new(cat, id, spec, value_expr, &col.name, replace))?;
    let mut stats = crate::catalog::table::RowCountStats::default();
    let mut cols_with_excs = Vec::new();
    let values = plan.compute(&mut stats, &mut cols_with_excs)?;
    with_catalog(py, |cat| plan.commit(cat, values, stats, cols_with_excs))?;
    Ok(())
}

/// `Table.add_btree_index()`
#[pyfunction]
#[pyo3(signature = (tbl_id, column, idx_name, if_exists))]
pub fn tbl_add_btree_index(py: Python<'_>, tbl_id: &str, column: &Bound<'_, PyAny>, idx_name: Option<&str>, if_exists: &str) -> Result<()> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| cat.check_mutable(id, None, "add an index to"))?;
    if !matches!(if_exists, "error" | "ignore") {
        crate::bail!(InvalidArgument, "if_exists must be one of: ['error', 'ignore']; got {}", py_repr(if_exists));
    }
    let col = column_arg(column)?;
    with_catalog(py, |cat| cat.add_btree_index(id, &col, idx_name, if_exists == "ignore"))
}

/// `Table.drop_index()` / `Table.drop_embedding_index()`
#[pyfunction]
#[pyo3(signature = (tbl_id, column, idx_name, embedding_only, if_not_exists))]
pub fn tbl_drop_index(
    py: Python<'_>,
    tbl_id: &str,
    column: Option<&Bound<'_, PyAny>>,
    idx_name: Option<&str>,
    embedding_only: bool,
    if_not_exists: &str,
) -> Result<()> {
    let id = parse_uuid(tbl_id)?;
    let column = column.filter(|c| !c.is_none());
    if column.is_none() == idx_name.is_none() {
        crate::bail!(MissingRequired, "Exactly one of 'column' or 'idx_name' must be provided");
    }
    let col = column.map(column_arg).transpose()?;
    let ine = IfNotExists::parse(if_not_exists, "if_not_exists")?;
    with_catalog(py, |cat| cat.drop_index(id, col.as_ref(), idx_name, embedding_only, ine == IfNotExists::Ignore))
}

/// The `indexes` section of `get_metadata()`.
pub fn indexes_metadata<'py>(py: Python<'py>, cat: &Catalog, tbl_id: uuid::Uuid) -> Result<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let md = cat.table(tbl_id)?;
    for (col_name, idx) in cat.visible_indexes(tbl_id)? {
        let d = PyDict::new(py);
        d.set_item("name", &idx.name)?;
        d.set_item("columns", vec![col_name])?;
        match &idx.kind {
            IndexKind::Btree { .. } => {
                d.set_item("index_type", "btree")?;
                d.set_item("parameters", py.None())?;
            }
            IndexKind::Embedding { metric, precision, embeddings, val_col_id } => {
                d.set_item("index_type", "embedding")?;
                let params = PyDict::new(py);
                params.set_item("metric", metric)?;
                params.set_item("precision", precision)?;
                let embedding = match md.column(*val_col_id).and_then(|c| c.value_expr.as_ref()) {
                    Some(e) => crate::expr::display::display(&*crate::expr::serde::from_json(e)?),
                    None => String::new(),
                };
                params.set_item("embedding", embedding)?;
                let fns = PyList::empty(py);
                for (_, descriptor) in embeddings {
                    let f = py
                        .import("pixeltable.func")?
                        .call_method1("_resolve_function", (crate::value::py::json_to_py(py, descriptor)?,))?;
                    fns.append(f.str()?)?;
                }
                params.set_item("embedding_functions", fns)?;
                d.set_item("parameters", params)?;
            }
        }
        out.set_item(&idx.name, d)?;
    }
    Ok(out)
}

/// Upstream `ColumnRef.similarity()`: the query item as a constant expression.
fn similarity_item<'py>(
    py: Python<'py>,
    item: Option<&Bound<'py, PyAny>>,
    named: &[(&'static str, Option<&Bound<'py, PyAny>>)],
) -> Result<ExprRef> {
    let n_named = named.iter().filter(|(_, v)| v.is_some()).count();
    if item.is_some() && n_named != 0 {
        crate::bail!(
            UnsupportedOperation,
            "similarity(): `item` is deprecated and cannot be used together with modality arguments"
        );
    }
    if n_named > 1 {
        crate::bail!(
            UnsupportedOperation,
            "similarity(): expected exactly one of string=..., image=..., audio=..., video=..., document=..., vector=..."
        );
    }
    let pil_cls = crate::media::pil_image(py)?.getattr("Image")?;
    let literal = |obj: &Bound<'py, PyAny>| -> Result<ExprRef> { super::expr::literal(obj) };
    if let Some(item) = item {
        if let Some(e) = as_expr(item) {
            if !(e.col_type.is_string() || e.col_type.is_image()) {
                crate::bail!(TypeMismatch, "similarity(): expected `String` or `Image`; got `{}`", e.col_type.repr());
            }
            return Ok(e);
        }
        if !(item.is_instance_of::<PyString>() || item.is_instance(&pil_cls)?) {
            crate::bail!(TypeMismatch, "similarity(): expected `str` or `PIL.Image.Image`; got `{}`", item.get_type().name()?);
        }
        return literal(item);
    }
    let Some((name, Some(v))) = named.iter().find(|(_, v)| v.is_some()).map(|(n, v)| (*n, *v)) else {
        crate::bail!(
            UnsupportedOperation,
            "similarity(): expected exactly one of string=..., image=..., audio=..., video=..., document=..., vector=..."
        );
    };
    let expected = |t: &str| -> &'static str {
        match t {
            "string" => "String",
            "image" => "Image",
            "audio" => "Audio",
            "video" => "Video",
            "document" => "Document",
            _ => "Array",
        }
    };
    if let Some(e) = as_expr(v) {
        let ok = match name {
            "string" => e.col_type.is_string(),
            "image" => e.col_type.is_image(),
            "audio" => e.col_type.is_audio(),
            "video" => e.col_type.is_video(),
            "document" => e.col_type.is_document(),
            _ => e.col_type.is_array(),
        };
        if !ok {
            crate::bail!(TypeMismatch, "similarity({name}=...): expected `{}`; got `{}`", expected(name), e.col_type.repr());
        }
        return Ok(e);
    }
    let type_name = v.get_type().name()?.to_string();
    match name {
        "string" => {
            if !v.is_instance_of::<PyString>() {
                crate::bail!(TypeMismatch, "similarity(string=...): expected `str`; got `{type_name}`");
            }
            literal(v)
        }
        "image" => {
            if v.is_instance_of::<PyString>() {
                let stored = crate::media::normalize_input(&v.extract::<String>()?);
                let img = crate::media::open_image(py, &crate::value::Value::Media(stored))?;
                return literal(&img);
            }
            if !v.is_instance(&pil_cls)? {
                crate::bail!(TypeMismatch, "similarity(image=...): expected `str` or `PIL.Image.Image`; got `{type_name}`");
            }
            literal(v)
        }
        "vector" => {
            let np = py.import("numpy")?;
            if !v.is_instance(&np.getattr("ndarray")?)? {
                crate::bail!(
                    UnsupportedOperation,
                    "similarity(vector=...): expected `numpy.ndarray`, or array `Expr`; got `{type_name}`"
                );
            }
            let ndim: usize = v.getattr("ndim")?.extract()?;
            if ndim != 1 {
                crate::bail!(
                    UnsupportedOperation,
                    "similarity(vector=...): expected 1-dimensional array; got shape {}",
                    v.getattr("shape")?.str()?
                );
            }
            let is_float: bool = np.call_method1("issubdtype", (v.getattr("dtype")?, np.getattr("floating")?))?.extract()?;
            if !is_float {
                crate::bail!(UnsupportedOperation, "similarity(vector=...): expected float array; got dtype {}", v.getattr("dtype")?.str()?);
            }
            literal(v)
        }
        media => {
            if !v.is_instance_of::<PyString>() {
                let what = match media {
                    "audio" => "audio file",
                    "video" => "video file",
                    _ => "document file",
                };
                crate::bail!(UnsupportedOperation, "similarity({media}=...): expected `str` (path to {what}); got `{type_name}`");
            }
            let stored = crate::media::normalize_input(&v.extract::<String>()?);
            let path = crate::media::local_path(py, &stored)?.to_string_lossy().into_owned();
            let kind = match media {
                "audio" => TypeKind::Audio,
                "video" => TypeKind::Video,
                _ => TypeKind::Document { formats: None },
            };
            Ok(Expr::literal(crate::value::Value::Media(path), ColumnType::new(kind, false)))
        }
    }
}

/// `col.similarity(...)` (a method of column references).
#[allow(clippy::too_many_arguments)]
pub fn similarity<'py>(
    py: Python<'py>,
    col_expr: &ExprRef,
    item: Option<&Bound<'py, PyAny>>,
    named: &[(&'static str, Option<&Bound<'py, PyAny>>)],
    idx: Option<&str>,
) -> Result<PyExpr> {
    let Some(c) = col_expr.as_column_ref() else {
        return Err(Error::Attribute(format!("Unknown method (of type {}): similarity", col_expr.col_type.repr())));
    };
    if item.is_some() {
        let msg = "Use of similarity() without specifying an explicit modality is deprecated -- since version 0.5.7. Please use one of the following instead:\n  .similarity(string=...)\n  .similarity(image=...)\n  .similarity(audio=...)\n  .similarity(video=...)\n  .similarity(document=...)\n  .similarity(vector=...)";
        let builtins = py.import("builtins")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("stacklevel", 2)?;
        py.import("warnings")?.call_method("warn", (msg, builtins.getattr("DeprecationWarning")?), Some(&kwargs))?;
    }
    let item_expr = similarity_item(py, item, named)?;
    let (col, idx_md) = with_catalog(py, |cat| -> Result<(ResolvedColumn, crate::catalog::table::IndexMd)> {
        let md = cat.table(c.tbl_id)?;
        let col = md
            .column_by_name(&c.name, c.tbl_version)?
            .filter(|r| r.md.id == c.col_id)
            .or_else(|| md.columns_at(None).ok()?.into_iter().find(|r| r.md.id == c.col_id))
            .ok_or_else(|| crate::pxt_err!(ColumnNotFound, "Unknown column: {}", c.name))?;
        let idx = cat.embedding_index(c.tbl_id, c.tbl_id, c.col_id, &col.name, idx)?;
        Ok((col, idx))
    })?;
    let IndexKind::Embedding { metric, embeddings, val_col_id, .. } = &idx_md.kind else {
        return Err(Error::internal("embedding_index() returned a B-tree index"));
    };
    let item_embedding = if item_expr.col_type.is_array() {
        item_expr.clone()
    } else {
        let t = type_name_of(&item_expr.col_type);
        let lower = t.to_ascii_lowercase();
        let Some((_, descriptor)) = embeddings.iter().find(|(m, _)| m == t) else {
            let article = if lower.starts_with(['a', 'e', 'i', 'o', 'u']) { "an" } else { "a" };
            crate::bail!(
                UnsupportedOperation,
                "Embedding index {} on column {} does not have {article} {lower} embedding and does not support {lower} queries",
                py_repr(&idx_md.name),
                py_repr(&col.name)
            );
        };
        let def = udf::resolve_descriptor(descriptor)?;
        udf::make_call(&def, vec![item_expr.clone()], vec![], false)?
    };
    let _ = MODALITIES;
    let sim = Similarity {
        col: col_expr.clone(),
        item: item_expr,
        idx_name: idx_md.name.clone(),
        idx_tbl_id: c.tbl_id,
        val_col_id: *val_col_id,
        metric: metric.clone(),
        item_embedding,
        cache: Arc::default(),
    };
    Ok(PyExpr::new(Expr::new(ExprKind::Similarity(sim), ColumnType::float(false))))
}

/// The content of `Table.describe()` / `repr(table)` (upstream `_descriptors()`): a dict with `title`, `columns`
/// (rows of name, type, source, computed with, comment, with the table's own columns first), `separators` (row
/// indexes after which the source changes), `indexes` (rows as dicts), `comment` and `custom_metadata`.
#[pyfunction]
#[pyo3(signature = (tbl_id, version=None))]
pub fn tbl_describe<'py>(py: Python<'py>, tbl_id: &str, version: Option<i64>) -> Result<Bound<'py, PyDict>> {
    let id = parse_uuid(tbl_id)?;
    with_catalog(py, |cat| {
        let md = cat.table(id)?.clone();
        let path = cat.table_path(id)?;
        let kind = if version.is_some() { "snapshot" } else { md.kind.display_name() };
        let mut title = format!("{kind} {}", py_repr(&path.to_string()));
        // bases and their effective versions
        let chain = cat.base_chain(id);
        let mut bases = Vec::new();
        let tbl_path = cat.tbl_path(id, version)?;
        let skip = if md.is_pure_snapshot() { 0 } else { 1 };
        for e in tbl_path.iter().skip(skip) {
            let base_path = cat.table_path(e.tbl_id)?.to_string();
            bases.push(match e.pinned {
                Some(ver) => py_repr(&format!("{base_path}:{ver}")),
                None => py_repr(&base_path),
            });
        }
        if !bases.is_empty() {
            title.push_str(&format!(" (of {})", bases.join(", ")));
        }
        if let Some(p) = md.view.as_ref().and_then(|v| v.predicate.as_ref()) {
            title.push_str(&format!("\nWhere: {}", crate::expr::display::display(&*crate::expr::serde::from_json(p)?)));
        }
        let cols = cat.resolve_columns(id, version)?;
        let rows = PyList::empty(py);
        let mut separators = Vec::new();
        let mut prev_source: Option<String> = None;
        let mut i = 0usize;
        for owner in &chain {
            let source = cat.table(*owner)?.name.clone();
            for c in cols.iter().filter(|c| c.tbl_id == *owner) {
                let computed_with = match &c.md.value_expr {
                    Some(e) => crate::expr::display::display(&*crate::expr::serde::from_json(e)?),
                    None => String::new(),
                };
                if prev_source.as_ref().is_some_and(|p| *p != source) {
                    separators.push(i - 1);
                }
                prev_source = Some(source.clone());
                rows.append((c.name.clone(), c.md.col_type.repr(), source.clone(), computed_with, c.md.comment.clone().unwrap_or_default()))?;
                i += 1;
            }
        }
        let idx_rows = PyList::empty(py);
        let visible = cat.visible_indexes(id)?;
        let show_type = visible.iter().any(|(_, i)| i.is_btree());
        let show_embedding = visible.iter().any(|(_, i)| !i.is_btree());
        for (col_name, idx) in &visible {
            let row = PyDict::new(py);
            row.set_item("Index Name", &idx.name)?;
            row.set_item("Column", col_name)?;
            if show_type {
                row.set_item("Type", if idx.is_btree() { "btree" } else { "embedding" })?;
            }
            if show_embedding {
                match &idx.kind {
                    IndexKind::Embedding { metric, val_col_id, .. } => {
                        row.set_item("Metric", metric)?;
                        let embedding = match md.column(*val_col_id).and_then(|c| c.value_expr.as_ref()) {
                            Some(e) => crate::expr::display::display(&*crate::expr::serde::from_json(e)?),
                            None => String::new(),
                        };
                        row.set_item("Embedding", embedding)?;
                    }
                    IndexKind::Btree { .. } => {
                        row.set_item("Metric", "")?;
                        row.set_item("Embedding", "")?;
                    }
                }
            }
            idx_rows.append(row)?;
        }
        let sv = md.schema_version_md(md.schema_version_at(version)?)?;
        let d = PyDict::new(py);
        d.set_item("title", title)?;
        d.set_item("columns", rows)?;
        d.set_item("separators", separators)?;
        d.set_item("indexes", idx_rows)?;
        d.set_item("comment", sv.comment.clone())?;
        d.set_item("custom_metadata", crate::value::py::json_to_py(py, &sv.custom_metadata)?)?;
        Ok(d)
    })
}
