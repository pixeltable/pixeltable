//! Query planning and execution.
//!
//! `QuerySpec` is the state behind a Python `Query` (from table, select list, where clause, order by, limit). It is
//! validated as it is built (the builder methods raise upstream's errors) and executed in two steps:
//! `QueryInput::snapshot()` collects what the query reads while the catalog lock is held, and `execute()` runs the
//! scan -> filter -> sort -> limit -> project pipeline without the lock (so UDFs can later call into Python).
//!
//! Extension points: joins (more than one source in `QueryInput`), group by / aggregates and window functions
//! (a grouping stage between filter and project), distinct, sample.

pub mod compute;
pub mod compute_rows;
pub mod insert;

use std::sync::Arc;
use uuid::Uuid;

use crate::catalog::Catalog;
use crate::catalog::path::{fold_identifier, is_valid_identifier};
use crate::catalog::table::ResolvedColumn;
use crate::catalog::views::ScanSource;
use crate::error::{Error, Result, py_repr};
use crate::expr::eval::{RowContext, eval, eval_predicate};
use crate::expr::{ColumnRef, Expr, ExprKind, ExprRef};
use crate::store::rows::{StoredRow, TableData};
use crate::types::ColumnType;
use crate::value::{Value, cmp_values};

#[derive(Debug, Clone)]
pub struct QuerySpec {
    pub tbl_id: Uuid,
    /// the version of the from-table (None: current)
    pub version: Option<i64>,
    /// None: all columns of the table (resolved at execution time)
    pub select: Option<Vec<(ExprRef, Option<String>)>>,
    pub where_: Option<ExprRef>,
    pub order_by: Vec<(ExprRef, bool)>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// The output of a query: column names and types, and rows of values.
#[derive(Debug, Clone, Default)]
pub struct ResultData {
    pub names: Vec<String>,
    pub types: Vec<ColumnType>,
    pub rows: Vec<Vec<Value>>,
}

impl QuerySpec {
    pub fn new(tbl_id: Uuid, version: Option<i64>) -> Self {
        QuerySpec { tbl_id, version, select: None, where_: None, order_by: Vec::new(), limit: None, offset: None }
    }

    fn tbl_name(&self, cat: &Catalog) -> String {
        let name = cat.table(self.tbl_id).map(|t| t.name.clone()).unwrap_or_default();
        match self.version {
            Some(v) => format!("{name}:{v}"),
            None => name,
        }
    }

    /// Fails unless every column the expression references belongs to the query's table (or one of its bases).
    pub fn check_bound(&self, cat: &Catalog, e: &Expr) -> Result<()> {
        let chain = cat.base_chain(self.tbl_id);
        if e.column_refs().iter().all(|c| chain.contains(&c.tbl_id)) {
            return Ok(());
        }
        crate::bail!(
            UnsupportedOperation,
            "That expression cannot be evaluated in the context of this query's tables ({}): {}",
            self.tbl_name(cat),
            crate::expr::display::display(e)
        );
    }

    /// `Query.select()`
    pub fn with_select(&self, cat: &Catalog, items: Vec<(ExprRef, Option<String>)>) -> Result<QuerySpec> {
        if self.select.is_some() {
            crate::bail!(InvalidState, "Select list already specified");
        }
        for (_, name) in &items {
            if let Some(n) = name {
                if !is_valid_identifier(n, false) {
                    crate::bail!(InvalidArgument, "Invalid name: {n}");
                }
            }
        }
        if items.is_empty() {
            return Ok(self.clone());
        }
        let items: Vec<(ExprRef, Option<String>)> = items.into_iter().map(|(e, n)| (e, n.map(|n| fold_identifier(&n)))).collect();
        for (e, _) in &items {
            if e.col_type.is_invalid() && !matches!(e.as_literal(), Some(Value::Null)) {
                crate::bail!(UnsupportedOperation, "Invalid type: {}", crate::expr::display::display(e));
            }
            self.check_bound(cat, e)?;
        }
        let names = output_names(&items);
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            if !seen.insert(name.clone()) {
                let positions: Vec<String> =
                    names.iter().enumerate().filter(|(_, n)| *n == name).map(|(i, _)| i.to_string()).collect();
                crate::bail!(
                    UnsupportedOperation,
                    "Repeated column name {} in select() at positions: {}",
                    py_repr(name),
                    positions.join(", ")
                );
            }
        }
        let mut q = self.clone();
        q.select = Some(items);
        Ok(q)
    }

    /// `Query.where()`
    pub fn with_where(&self, cat: &Catalog, pred: ExprRef) -> Result<QuerySpec> {
        if self.where_.is_some() {
            crate::bail!(InvalidState, "where() clause already specified");
        }
        if !pred.col_type.is_bool() {
            crate::bail!(
                UnsupportedOperation,
                "where() expression needs to return `Bool`, but instead returns `{}`",
                pred.col_type.repr()
            );
        }
        self.check_bound(cat, &pred)?;
        let mut q = self.clone();
        q.where_ = Some(pred);
        Ok(q)
    }

    /// `Query.order_by()`
    pub fn with_order_by(&self, cat: &Catalog, exprs: Vec<ExprRef>, asc: bool) -> Result<QuerySpec> {
        for e in &exprs {
            self.check_bound(cat, e)?;
        }
        let mut q = self.clone();
        q.order_by.extend(exprs.into_iter().map(|e| (e, asc)));
        Ok(q)
    }

    /// `Query.limit()`
    pub fn with_limit(&self, n: i64, offset: Option<i64>) -> Result<QuerySpec> {
        if n < 0 {
            crate::bail!(InvalidArgument, "'limit()' parameter must be >= 0");
        }
        if offset.is_some_and(|o| o < 0) {
            crate::bail!(InvalidArgument, "'offset' parameter must be >= 0");
        }
        let mut q = self.clone();
        q.limit = Some(n);
        q.offset = offset;
        Ok(q)
    }

    pub fn check_head_tail(&self, which: &str) -> Result<()> {
        if !self.order_by.is_empty() {
            crate::bail!(UnsupportedOperation, "{which}() cannot be used with order_by()");
        }
        Ok(())
    }

    /// The select list of `select *`.
    pub fn star_select(cat: &Catalog, tbl_id: Uuid, version: Option<i64>) -> Result<Vec<(ExprRef, Option<String>)>> {
        Ok(cat
            .resolve_columns(tbl_id, version)?
            .into_iter()
            .map(|c| (column_ref_expr(&c, version), None))
            .collect())
    }

    /// The effective select list with output names.
    pub fn effective_select(&self, cat: &Catalog) -> Result<Vec<(ExprRef, String)>> {
        let items = match &self.select {
            Some(items) => items.clone(),
            None => Self::star_select(cat, self.tbl_id, self.version)?,
        };
        let names = output_names(&items);
        Ok(items.into_iter().map(|(e, _)| e).zip(names).collect())
    }

    /// Fails if an expression of the query references a column that no longer exists (it was dropped after the
    /// query was built).
    pub fn check_columns_exist(&self, cat: &Catalog) -> Result<()> {
        let visible: std::collections::HashSet<(Uuid, crate::catalog::table::ColId)> =
            cat.resolve_columns(self.tbl_id, self.version)?.iter().map(|c| (c.tbl_id, c.md.id)).collect();
        let mut exprs: Vec<&ExprRef> = Vec::new();
        if let Some(items) = &self.select {
            exprs.extend(items.iter().map(|(e, _)| e));
        }
        exprs.extend(self.where_.iter());
        exprs.extend(self.order_by.iter().map(|(e, _)| e));
        for e in exprs {
            for r in e.column_refs() {
                if !visible.contains(&(r.tbl_id, r.col_id)) {
                    crate::bail!(
                        ColumnNotFound,
                        "Column was dropped (no record for column ID {} in table {})",
                        r.col_id,
                        py_repr(&self.tbl_name(cat))
                    );
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------------- serialization

    pub fn to_json(&self) -> Result<serde_json::Value> {
        use crate::expr::serde::to_json as e2j;
        Ok(serde_json::json!({
            "_classname": "Query",
            "from_clause": {"tbls": [{"id": self.tbl_id.to_string(), "effective_version": self.version}], "join_clauses": []},
            "select_list": match &self.select {
                None => serde_json::Value::Null,
                Some(items) => serde_json::Value::Array(items.iter().map(|(e, n)| Ok(serde_json::json!([e2j(e)?, n]))).collect::<Result<_>>()?),
            },
            "where_clause": self.where_.as_ref().map(|e| e2j(e)).transpose()?,
            "order_by_clause": serde_json::Value::Array(self.order_by.iter().map(|(e, asc)| Ok(serde_json::json!([e2j(e)?, asc]))).collect::<Result<_>>()?),
            "limit_val": self.limit,
            "offset_val": self.offset,
        }))
    }

    pub fn from_json(d: &serde_json::Value) -> Result<QuerySpec> {
        use crate::expr::serde::from_json as j2e;
        let bad = || Error::internal(format!("malformed serialized query: {d}"));
        if d.get("_classname").and_then(|c| c.as_str()) != Some("Query") {
            return Err(Error::internal("Expected a serialized Query"));
        }
        let tbl = d.pointer("/from_clause/tbls/0").ok_or_else(bad)?;
        let tbl_id = tbl.get("id").and_then(|v| v.as_str()).and_then(|s| Uuid::parse_str(s).ok()).ok_or_else(bad)?;
        let version = tbl.get("effective_version").and_then(|v| v.as_i64());
        let select = match d.get("select_list") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Array(items)) => Some(
                items
                    .iter()
                    .map(|item| {
                        let e = j2e(item.get(0).ok_or_else(bad)?)?;
                        let n = item.get(1).and_then(|n| n.as_str()).map(str::to_owned);
                        Ok((e, n))
                    })
                    .collect::<Result<_>>()?,
            ),
            _ => return Err(bad()),
        };
        let where_ = match d.get("where_clause") {
            None | Some(serde_json::Value::Null) => None,
            Some(w) => Some(j2e(w)?),
        };
        let order_by = match d.get("order_by_clause") {
            Some(serde_json::Value::Array(items)) => items
                .iter()
                .map(|item| Ok((j2e(item.get(0).ok_or_else(bad)?)?, item.get(1).and_then(|a| a.as_bool()).unwrap_or(true))))
                .collect::<Result<_>>()?,
            _ => Vec::new(),
        };
        Ok(QuerySpec {
            tbl_id,
            version,
            select,
            where_,
            order_by,
            limit: d.get("limit_val").and_then(|v| v.as_i64()),
            offset: d.get("offset_val").and_then(|v| v.as_i64()),
        })
    }
}

pub fn column_ref_expr(c: &ResolvedColumn, version: Option<i64>) -> ExprRef {
    let _ = version;
    Expr::new(
        ExprKind::ColumnRef(ColumnRef { tbl_id: c.tbl_id, col_id: c.md.id, name: c.name.clone(), tbl_version: None }),
        c.md.col_type.clone(),
    )
}

/// Output column names (upstream `_normalize_select_list()`): explicit names, else the expression's default name
/// (de-duplicated with `_1`, `_2`, ...), else `col_<i>`.
pub fn output_names(items: &[(ExprRef, Option<String>)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (i, (e, name)) in items.iter().enumerate() {
        let column_name = match name {
            Some(n) => n.clone(),
            None => match e.default_column_name().map(|n| fold_identifier(&n)) {
                Some(default) => {
                    let mut candidate = default.clone();
                    if seen.contains(&default) {
                        for j in 1..=out.len() {
                            candidate = format!("{default}_{j}");
                            if !seen.contains(&candidate) {
                                break;
                            }
                        }
                    }
                    candidate
                }
                None => format!("col_{i}"),
            },
        };
        seen.insert(column_name.clone());
        out.push(column_name);
    }
    out
}

/// Everything a query reads, captured under the catalog lock.
pub struct QueryInput {
    pub source: ScanSource,
    pub version: i64,
    pub data: Arc<TableData>,
    pub select: Vec<(ExprRef, String)>,
    pub predicates: Vec<ExprRef>,
    /// value expressions of unstored computed columns of the scanned table, evaluated on read
    pub unstored: std::collections::HashMap<crate::catalog::table::ColId, ExprRef>,
    /// media columns of the scanned table that are validated on read
    pub read_validation: Arc<crate::media::ReadValidation>,
}

impl QueryInput {
    pub fn snapshot(cat: &mut Catalog, q: &QuerySpec) -> Result<QueryInput> {
        q.check_columns_exist(cat)?;
        let source = cat.scan_source(q.tbl_id, q.version)?;
        let base = cat.table(source.base_tbl_id)?;
        let version = source.base_version.unwrap_or(base.current_version);
        let select = q.effective_select(cat)?;
        let predicates = source
            .predicates
            .iter()
            .map(crate::expr::serde::from_json)
            .collect::<Result<Vec<_>>>()?;
        let mut unstored = std::collections::HashMap::new();
        for c in &cat.table(source.base_tbl_id)?.columns {
            if !c.stored {
                if let Some(e) = &c.value_expr {
                    unstored.insert(c.id, crate::expr::serde::from_json(e)?);
                }
            }
        }
        let read_validation = Arc::new(crate::media::ReadValidation::for_table(cat.table(source.base_tbl_id)?)?);
        let data = cat.data(source.base_tbl_id)?;
        Ok(QueryInput { source, version, data, select, predicates, unstored, read_validation })
    }
}

struct StoredRowCtx<'a> {
    row: &'a StoredRow,
    unstored: &'a std::collections::HashMap<crate::catalog::table::ColId, ExprRef>,
    read_validation: &'a crate::media::ReadValidation,
}

impl<'a> StoredRowCtx<'a> {
    fn new(row: &'a StoredRow, input: &'a QueryInput) -> Self {
        StoredRowCtx { row, unstored: &input.unstored, read_validation: &input.read_validation }
    }

    /// For a read-validated media column with a file: the validation error, if any.
    fn read_validation_error(&self, col: &ColumnRef) -> Option<Option<crate::store::rows::CellError>> {
        let kind = self.read_validation.kind(col.col_id)?;
        match self.row.cell(col.col_id) {
            Value::Media(s) => Some(self.read_validation.error(s, kind)),
            _ => Some(None),
        }
    }
}

impl RowContext for StoredRowCtx<'_> {
    fn column_value(&self, col: &ColumnRef) -> Result<Value> {
        if let Some(e) = self.unstored.get(&col.col_id) {
            return eval(e, self);
        }
        if let Some(Some(_)) = self.read_validation_error(col) {
            // an invalid file reads as NULL; its error is the cell's error
            return Ok(Value::Null);
        }
        Ok(self.row.cell(col.col_id).clone())
    }

    fn column_error(&self, col: &ColumnRef) -> Result<Option<crate::store::rows::CellError>> {
        if let Some(e) = self.read_validation_error(col) {
            return Ok(e);
        }
        Ok(self.row.error(col.col_id).cloned())
    }
}

/// The rows visible to the query (version + view predicates + where clause), in rowid order.
fn filtered_rows<'a>(input: &'a QueryInput, q: &QuerySpec) -> Result<Vec<&'a StoredRow>> {
    let mut rows: Vec<&StoredRow> = input.data.visible_at(input.version).collect();
    rows.sort_by_key(|r| r.rowid);
    let mut out = Vec::with_capacity(rows.len());
    'rows: for r in rows {
        let ctx = StoredRowCtx::new(r, input);
        for p in &input.predicates {
            if !eval_predicate(p, &ctx)? {
                continue 'rows;
            }
        }
        if let Some(w) = &q.where_ {
            if !eval_predicate(w, &ctx)? {
                continue;
            }
        }
        out.push(r);
    }
    Ok(out)
}

/// Runs the query. `rowid_desc`: for `tail()`, order by rowid descending instead of ascending.
pub fn execute(input: &QueryInput, q: &QuerySpec, rowid_desc: bool) -> Result<ResultData> {
    let mut rows = filtered_rows(input, q)?;
    if rowid_desc {
        rows.reverse();
    }
    if !q.order_by.is_empty() {
        let mut keyed: Vec<(Vec<Value>, &StoredRow)> = rows
            .into_iter()
            .map(|r| {
                let ctx = StoredRowCtx::new(r, input);
                let keys = q.order_by.iter().map(|(e, _)| eval(e, &ctx)).collect::<Result<Vec<_>>>()?;
                Ok((keys, r))
            })
            .collect::<Result<_>>()?;
        keyed.sort_by(|(a, _), (b, _)| {
            for (i, (_, asc)) in q.order_by.iter().enumerate() {
                // Postgres null placement: ASC puts NULLs last, DESC puts them first
                let ord = cmp_values(&a[i], &b[i]);
                let ord = if *asc { ord } else { ord.reverse() };
                if ord.is_ne() {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
        rows = keyed.into_iter().map(|(_, r)| r).collect();
    }
    let offset = q.offset.unwrap_or(0).max(0) as usize;
    let rows: Vec<&StoredRow> = match q.limit {
        Some(n) => rows.into_iter().skip(offset).take(n.max(0) as usize).collect(),
        None => rows.into_iter().skip(offset).collect(),
    };
    let mut out_rows = Vec::with_capacity(rows.len());
    for r in rows {
        let ctx = StoredRowCtx::new(r, input);
        out_rows.push(input.select.iter().map(|(e, _)| eval(e, &ctx)).collect::<Result<Vec<_>>>()?);
    }
    Ok(ResultData {
        names: input.select.iter().map(|(_, n)| n.clone()).collect(),
        types: input.select.iter().map(|(e, _)| e.col_type.clone()).collect(),
        rows: out_rows,
    })
}

/// `Query.count()`
pub fn count(input: &QueryInput, q: &QuerySpec) -> Result<usize> {
    if q.limit.is_some() || q.offset.is_some() {
        crate::bail!(
            UnsupportedOperation,
            "count() cannot be used with limit() or offset(). Use `select(pxtf.count())` instead."
        );
    }
    Ok(filtered_rows(input, q)?.len())
}
