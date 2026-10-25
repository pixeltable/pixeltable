//! Query planning and execution.
//!
//! `QuerySpec` is the state behind a Python `Query`: the from-table and its joins, select list, where clause,
//! group by, order by, limit/offset and sample clause. It is validated as it is built (the builder methods raise
//! upstream's errors) and executed in two steps (`query.rs`): `QueryInput::snapshot()` collects what the query
//! reads while the catalog lock is held, and `execute()` runs scan -> join -> filter -> sample -> aggregate ->
//! window functions -> sort -> limit -> project without the lock (so UDFs can call into Python).

pub mod compute;
pub mod compute_rows;
pub mod dispatch;
pub mod fast;
pub mod index;
pub mod insert;
pub mod par;
pub mod prefetch;
pub mod propagate;
pub mod query;
pub mod sample;
pub mod scan;
pub mod txn;
pub mod update;

use uuid::Uuid;

use crate::catalog::Catalog;
use crate::catalog::path::{fold_identifier, is_valid_identifier};
use crate::catalog::table::ResolvedColumn;
use crate::error::{Error, Result, py_repr};
use crate::expr::{ColumnRef, Expr, ExprKind, ExprRef};
use crate::value::Value;

pub use query::{QueryInput, count, execute};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    FullOuter,
    Cross,
}

impl JoinType {
    pub fn parse(how: &str) -> Result<JoinType> {
        Ok(match how.to_ascii_uppercase().as_str() {
            "INNER" => JoinType::Inner,
            "LEFT" => JoinType::Left,
            "FULL_OUTER" => JoinType::FullOuter,
            "CROSS" => JoinType::Cross,
            _ => crate::bail!(InvalidArgument, "`how` must be one of: ['inner', 'left', 'full_outer', 'cross']"),
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            JoinType::Inner => "INNER",
            JoinType::Left => "LEFT",
            JoinType::FullOuter => "FULL_OUTER",
            JoinType::Cross => "CROSS",
        }
    }
}

/// A table joined to the query's from-clause.
#[derive(Debug, Clone)]
pub struct JoinClause {
    pub tbl_id: Uuid,
    pub version: Option<i64>,
    pub how: JoinType,
    /// None for a cross join
    pub on: Option<ExprRef>,
}

/// `Query.sample()` (upstream `SampleClause`).
#[derive(Debug, Clone)]
pub struct SampleClause {
    pub n: Option<i64>,
    pub n_per_stratum: Option<i64>,
    pub fraction: Option<f64>,
    pub seed: Option<i64>,
    pub stratify_by: Vec<ExprRef>,
}

impl SampleClause {
    pub fn is_stratified(&self) -> bool {
        !self.stratify_by.is_empty()
    }

    /// A fractional, unstratified sample is a deterministic function of the rows (it can define a view).
    pub fn is_repeatable(&self) -> bool {
        !self.is_stratified() && self.fraction.is_some()
    }

    /// The serialized form (as in `Query.as_dict()`).
    pub fn to_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "_classname": "SampleClause",
            "version": 1,
            "n": self.n,
            "n_per_stratum": self.n_per_stratum,
            "fraction": self.fraction,
            "seed": self.seed,
            "stratify_exprs": self.stratify_by.iter().map(|e| crate::expr::serde::to_json(e)).collect::<Result<Vec<_>>>()?,
        }))
    }

    pub fn from_json(s: &serde_json::Value) -> Result<SampleClause> {
        use serde_json::Value as J;
        let stratify_by = match s.get("stratify_exprs") {
            Some(J::Array(items)) => items.iter().map(crate::expr::serde::from_json).collect::<Result<Vec<_>>>()?,
            _ => Vec::new(),
        };
        Ok(SampleClause {
            n: s.get("n").and_then(J::as_i64),
            n_per_stratum: s.get("n_per_stratum").and_then(J::as_i64),
            fraction: s.get("fraction").and_then(J::as_f64),
            seed: s.get("seed").and_then(J::as_i64),
            stratify_by,
        })
    }

    /// Upstream `SampleClause.__repr__()`.
    pub fn display(&self) -> String {
        let opt_i = |v: Option<i64>| v.map_or("None".to_string(), |n| n.to_string());
        let frac = self.fraction.map_or("None".to_string(), crate::value::py_float_repr);
        let strata: Vec<String> = self.stratify_by.iter().map(|e| crate::expr::display::display(e)).collect();
        format!(
            "sample(n={}, n_per_stratum={}, fraction={frac}, seed={}, [{}])",
            opt_i(self.n),
            opt_i(self.n_per_stratum),
            opt_i(self.seed),
            strata.join(",")
        )
    }
}

#[derive(Debug, Clone)]
pub struct QuerySpec {
    pub tbl_id: Uuid,
    /// the version of the from-table (None: current)
    pub version: Option<i64>,
    pub joins: Vec<JoinClause>,
    /// None: all columns of the from-tables (resolved at execution time)
    pub select: Option<Vec<(ExprRef, Option<String>)>>,
    pub where_: Option<ExprRef>,
    pub group_by: Option<Vec<ExprRef>>,
    /// `group_by(base_table)` of a view: group by the rows of that base
    pub grouping_tbl: Option<Uuid>,
    pub order_by: Vec<(ExprRef, bool)>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    /// in a `@pxt.query` template: the parameters that give the limit / offset (bound per call, `bind_params()`)
    pub limit_param: Option<String>,
    pub offset_param: Option<String>,
    pub sample: Option<SampleClause>,
}

/// The output of a query: column names and types, and rows of values.
#[derive(Debug, Clone, Default)]
pub struct ResultData {
    pub names: Vec<String>,
    pub types: Vec<crate::types::ColumnType>,
    pub rows: Vec<Vec<Value>>,
}

impl QuerySpec {
    pub fn new(tbl_id: Uuid, version: Option<i64>) -> Self {
        QuerySpec {
            tbl_id,
            version,
            joins: Vec::new(),
            select: None,
            where_: None,
            group_by: None,
            grouping_tbl: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
            limit_param: None,
            offset_param: None,
            sample: None,
        }
    }

    /// (table id, version) of each from-table: the first table, then the joined ones.
    pub fn from_tbls(&self) -> Vec<(Uuid, Option<i64>)> {
        let mut out = vec![(self.tbl_id, self.version)];
        out.extend(self.joins.iter().map(|j| (j.tbl_id, j.version)));
        out
    }

    pub fn has_joins(&self) -> bool {
        !self.joins.is_empty()
    }

    pub fn is_grouped(&self) -> bool {
        self.group_by.is_some() || self.grouping_tbl.is_some()
    }

    fn one_tbl_name(cat: &Catalog, tbl_id: Uuid, version: Option<i64>) -> String {
        let name = cat.table(tbl_id).map(|t| t.name.clone()).unwrap_or_default();
        match version {
            Some(v) => format!("{name}:{v}"),
            None => name,
        }
    }

    fn tbl_name(&self, cat: &Catalog) -> String {
        Self::one_tbl_name(cat, self.tbl_id, self.version)
    }

    /// Names of the from-tables (for messages and `repr()`).
    pub fn tbl_names(&self, cat: &Catalog) -> Vec<String> {
        self.from_tbls().into_iter().map(|(id, v)| Self::one_tbl_name(cat, id, v)).collect()
    }

    /// True if every column `e` references belongs to one of `tbls` (or their bases).
    fn is_bound_by(cat: &Catalog, tbls: &[(Uuid, Option<i64>)], e: &Expr) -> bool {
        let chains: Vec<Vec<Uuid>> = tbls.iter().map(|(id, _)| cat.base_chain(*id)).collect();
        e.column_refs().iter().all(|c| chains.iter().any(|ch| ch.contains(&c.tbl_id)))
    }

    /// Fails unless every column the expression references belongs to the query's tables (or their bases).
    pub fn check_bound(&self, cat: &Catalog, e: &Expr) -> Result<()> {
        if Self::is_bound_by(cat, &self.from_tbls(), e) {
            return Ok(());
        }
        crate::bail!(
            UnsupportedOperation,
            "That expression cannot be evaluated in the context of this query's tables ({}): {}",
            self.tbl_names(cat).join(", "),
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
    pub fn with_where(&self, pred: ExprRef) -> Result<QuerySpec> {
        if self.where_.is_some() {
            crate::bail!(InvalidState, "where() clause already specified");
        }
        if self.sample.is_some() {
            crate::bail!(UnsupportedOperation, "where() cannot be used after sample()");
        }
        if !pred.col_type.is_bool() {
            crate::bail!(
                UnsupportedOperation,
                "where() expression needs to return `Bool`, but instead returns `{}`",
                pred.col_type.repr()
            );
        }
        // as upstream, the predicate's tables are checked when the query runs (`QueryInput::snapshot()`)
        let mut q = self.clone();
        q.where_ = Some(pred);
        Ok(q)
    }

    /// `Query.order_by()`
    pub fn with_order_by(&self, exprs: Vec<ExprRef>, asc: bool) -> Result<QuerySpec> {
        if self.sample.is_some() {
            crate::bail!(UnsupportedOperation, "order_by() cannot be used with sample()");
        }
        let mut q = self.clone();
        q.order_by.extend(exprs.into_iter().map(|e| (e, asc)));
        Ok(q)
    }

    /// `Query.limit()`
    pub fn with_limit(&self, n: i64, offset: Option<i64>) -> Result<QuerySpec> {
        if self.sample.is_some() {
            crate::bail!(UnsupportedOperation, "limit() cannot be used with sample()");
        }
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

    /// `Query.limit()` in a `@pxt.query` template: each of limit and offset is a constant or a parameter.
    pub fn with_limit_args(&self, n: LimitArg, offset: Option<LimitArg>) -> Result<QuerySpec> {
        let mut q = self.with_limit(n.constant().unwrap_or(0), offset.as_ref().and_then(LimitArg::constant))?;
        q.limit = n.constant();
        q.limit_param = n.param();
        q.offset_param = offset.as_ref().and_then(LimitArg::param);
        Ok(q)
    }

    /// Binds the limit / offset parameters of a template to the call's values (`value(name)`).
    pub fn bind_params(&self, value: &dyn Fn(&str) -> Option<Value>) -> Result<QuerySpec> {
        let mut q = self.clone();
        let resolve = |p: &str, role: &str| -> Result<i64> {
            let Some(v) = value(p).and_then(|v| v.as_i64()) else {
                crate::bail!(InvalidArgument, "{} parameter must be an int; got None", py_repr(role));
            };
            if v < 0 {
                crate::bail!(InvalidArgument, "{} parameter must be >= 0", py_repr(role));
            }
            Ok(v)
        };
        if let Some(p) = q.limit_param.take() {
            q.limit = Some(resolve(&p, "limit")?);
        }
        if let Some(p) = q.offset_param.take() {
            q.offset = Some(resolve(&p, "offset")?);
        }
        Ok(q)
    }

    /// `Query.group_by(*exprs)`
    pub fn with_group_by(&self, exprs: Vec<ExprRef>) -> Result<QuerySpec> {
        self.check_group_by_allowed()?;
        let mut q = self.clone();
        q.group_by = Some(exprs);
        Ok(q)
    }

    /// `Query.group_by(base_table)` (a view grouped by the rows of one of its bases)
    pub fn with_grouping_tbl(&self, cat: &Catalog, grouping_tbl: Uuid) -> Result<QuerySpec> {
        self.check_group_by_allowed()?;
        if self.has_joins() {
            crate::bail!(UnsupportedOperation, "group_by() with Table not supported for joins");
        }
        let chain = cat.base_chain(self.tbl_id);
        if grouping_tbl == self.tbl_id || !chain.contains(&grouping_tbl) {
            crate::bail!(
                UnsupportedOperation,
                "group_by(): {} is not a base table of {}",
                py_repr(&cat.table(grouping_tbl).map(|t| t.name.clone()).unwrap_or_default()),
                py_repr(&self.tbl_name(cat))
            );
        }
        let mut q = self.clone();
        q.grouping_tbl = Some(grouping_tbl);
        Ok(q)
    }

    fn check_group_by_allowed(&self) -> Result<()> {
        if self.is_grouped() {
            crate::bail!(InvalidState, "group_by() already specified");
        }
        if self.sample.is_some() {
            crate::bail!(UnsupportedOperation, "group_by() cannot be used with sample()");
        }
        Ok(())
    }

    /// `Query.distinct()`: group by the (effective) select list.
    pub fn with_distinct(&self, cat: &Catalog) -> Result<QuerySpec> {
        let exprs = self.effective_select(cat)?.into_iter().map(|(e, _)| e).collect();
        self.with_group_by(exprs)
    }

    /// `Query.join()`; `on` is a predicate or a list of column references (already converted).
    pub fn with_join(&self, cat: &Catalog, other: (Uuid, Option<i64>), on: JoinOn, how: &str) -> Result<QuerySpec> {
        let self_versioned = cat.table(self.tbl_id)?.is_data_versioned;
        let other_versioned = cat.table(other.0)?.is_data_versioned;
        if self_versioned != other_versioned {
            crate::bail!(UnsupportedOperation, "join is not supported between data-versioned and operational tables");
        }
        if self.sample.is_some() {
            crate::bail!(UnsupportedOperation, "join() cannot be used with sample()");
        }
        let on = if how == "cross" {
            if !matches!(on, JoinOn::None) {
                crate::bail!(UnsupportedOperation, "`on` not allowed for cross join");
            }
            None
        } else {
            if matches!(on, JoinOn::None) {
                crate::bail!(UnsupportedOperation, "`how={}` requires `on` to be present", py_repr(how));
            }
            Some(self.join_predicate(cat, other, on)?)
        };
        let how = JoinType::parse(how)?;
        let mut q = self.clone();
        q.joins.push(JoinClause { tbl_id: other.0, version: other.1, how, on });
        Ok(q)
    }

    /// Upstream `_create_join_predicate()`.
    fn join_predicate(&self, cat: &Catalog, other: (Uuid, Option<i64>), on: JoinOn) -> Result<ExprRef> {
        let mut joined = self.from_tbls();
        joined.push(other);
        let col_refs: Vec<ExprRef> = match on {
            JoinOn::None => return Err(Error::internal("join predicate without `on`")),
            JoinOn::Expr(e) if e.as_column_ref().is_none() => {
                if !Self::is_bound_by(cat, &joined, &e) {
                    crate::bail!(
                        UnsupportedOperation,
                        "`on` expression cannot be evaluated in the context of the joined tables: {}",
                        crate::expr::display::display(&e)
                    );
                }
                if !e.col_type.is_bool() {
                    crate::bail!(
                        TypeMismatch,
                        "`on` expects an expression of type `Bool`, but got one of type `{}`: {}",
                        e.col_type.repr(),
                        crate::expr::display::display(&e)
                    );
                }
                return Ok(e);
            }
            JoinOn::Expr(e) => vec![e],
            JoinOn::Columns(cols) => cols,
        };
        let mut preds = Vec::new();
        for col_ref in col_refs {
            if !Self::is_bound_by(cat, &joined, &col_ref) {
                crate::bail!(
                    UnsupportedOperation,
                    "`on` expression cannot be evaluated in the context of the joined tables: {}",
                    crate::expr::display::display(&col_ref)
                );
            }
            let c = col_ref.as_column_ref().ok_or_else(|| Error::internal("join column is not a column reference"))?;
            let col_name = cat
                .resolve_columns(c.tbl_id, None)?
                .into_iter()
                .find(|r| r.tbl_id == c.tbl_id && r.md.id == c.col_id)
                .map(|r| r.name)
                .unwrap_or_else(|| c.name.clone());
            let rhs = cat.resolve_column_by_name(other.0, other.1, &col_name)?.ok_or_else(|| {
                crate::pxt_err!(ColumnNotFound, "`on` column {} not found in joined table", py_repr(&col_name))
            })?;
            let rhs_ref = column_ref_expr(&rhs, other.1);
            let in_from = self.from_tbls().iter().any(|(id, v)| {
                cat.resolve_columns(*id, *v).map(|cols| cols.iter().any(|r| r.tbl_id == c.tbl_id && r.md.id == c.col_id)).unwrap_or(false)
            });
            let lhs_ref = if in_from {
                col_ref.clone()
            } else {
                let mut found: Option<ExprRef> = None;
                for (id, v) in self.from_tbls() {
                    if let Some(r) = cat.resolve_column_by_name(id, v, &col_name)? {
                        if found.is_some() {
                            crate::bail!(UnsupportedOperation, "`on`: ambiguous column reference: {}", py_repr(&col_name));
                        }
                        found = Some(column_ref_expr(&r, v));
                    }
                }
                found.ok_or_else(|| {
                    crate::pxt_err!(
                        ColumnNotFound,
                        "`on`: column {} not found in any of: {}",
                        py_repr(&col_name),
                        self.tbl_names(cat).join(" ")
                    )
                })?
            };
            preds.push(crate::expr::build::comparison(crate::expr::CmpOp::Eq, lhs_ref, rhs_ref)?);
        }
        if preds.len() == 1 {
            return Ok(preds.remove(0));
        }
        crate::expr::build::compound(crate::expr::LogicalOp::And, preds)
    }

    /// `Query.sample()` (parameters already type-checked by the binding).
    pub fn with_sample(&self, cat: &Catalog, s: SampleClause) -> Result<QuerySpec> {
        if self.sample.is_some() {
            crate::bail!(UnsupportedOperation, "Multiple sample() clauses not allowed");
        }
        if self.is_grouped() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with group_by()");
        }
        if !self.order_by.is_empty() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with order_by()");
        }
        if self.limit.is_some() || self.limit_param.is_some() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with limit()");
        }
        if self.has_joins() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with join()");
        }
        for e in &s.stratify_by {
            if !e.col_type.is_scalar() {
                crate::bail!(InvalidArgument, "Invalid type: expression must be a scalar type (not `{}`)", e.col_type.repr());
            }
            if !Self::is_bound_by(cat, &self.from_tbls(), e) {
                crate::bail!(
                    UnsupportedOperation,
                    "That expression cannot be evaluated in the context of this query's tables ({}): {}",
                    self.tbl_names(cat).join(","),
                    crate::expr::display::display(e)
                );
            }
        }
        let mut q = self.clone();
        q.sample = Some(s);
        Ok(q)
    }

    /// head()/tail(): not with order_by, joins, sample or group_by.
    pub fn check_head_tail(&self, which: &str) -> Result<()> {
        if !self.order_by.is_empty() {
            crate::bail!(UnsupportedOperation, "{which}() cannot be used with order_by()");
        }
        if self.has_joins() {
            crate::bail!(UnsupportedOperation, "{which}() not supported for joins");
        }
        if self.sample.is_some() {
            crate::bail!(UnsupportedOperation, "{which}() cannot be used with sample()");
        }
        if self.is_grouped() {
            crate::bail!(UnsupportedOperation, "{which}() cannot be used with group_by()");
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
            None => {
                let mut items = Vec::new();
                for (id, v) in self.from_tbls() {
                    items.extend(Self::star_select(cat, id, v)?);
                }
                items
            }
        };
        let names = output_names(&items);
        Ok(items.into_iter().map(|(e, _)| e).zip(names).collect())
    }

    /// All expressions of the query's clauses.
    pub fn clause_exprs(&self) -> Vec<&ExprRef> {
        let mut exprs: Vec<&ExprRef> = Vec::new();
        if let Some(items) = &self.select {
            exprs.extend(items.iter().map(|(e, _)| e));
        }
        exprs.extend(self.where_.iter());
        if let Some(g) = &self.group_by {
            exprs.extend(g.iter());
        }
        exprs.extend(self.order_by.iter().map(|(e, _)| e));
        exprs.extend(self.joins.iter().filter_map(|j| j.on.as_ref()));
        if let Some(s) = &self.sample {
            exprs.extend(s.stratify_by.iter());
        }
        exprs
    }

    /// Fails if an expression of the query references a column that no longer exists (it was dropped after the
    /// query was built).
    pub fn check_columns_exist(&self, cat: &Catalog) -> Result<()> {
        self.check_columns(cat, true)
    }

    /// Fails as upstream fails to deserialize the query (`Query.from_dict()`) if a table it reads or a column it
    /// references no longer exists.
    pub fn check_loadable(&self, cat: &Catalog) -> Result<()> {
        for (id, _) in self.from_tbls() {
            cat.table(id)?;
        }
        self.check_columns(cat, false)
    }

    /// `with_drop_info`: name the version that dropped the column (upstream `ColumnHandle.get()`; deserialization
    /// does not).
    fn check_columns(&self, cat: &Catalog, with_drop_info: bool) -> Result<()> {
        let mut visible: std::collections::HashSet<(Uuid, crate::catalog::table::ColId)> = std::collections::HashSet::new();
        for (id, v) in self.from_tbls() {
            visible.extend(cat.path_columns(&cat.tbl_path(id, v)?)?.iter().map(|c| (c.tbl_id, c.md.id)));
        }
        for e in self.clause_exprs() {
            for r in e.column_refs() {
                if !visible.contains(&(r.tbl_id, r.col_id)) {
                    // upstream `ColumnHandle.get()`: the column's table (versioned name) and when it was dropped
                    let (name, dropped) = match cat.table(r.tbl_id) {
                        Ok(md) => (
                            r.tbl_version.map_or_else(|| md.name.clone(), |v| format!("{}:{v}", md.name)),
                            md.column(r.col_id)
                                .and_then(|c| c.schema_version_drop)
                                .filter(|_| md.is_data_versioned && with_drop_info)
                                .map(|v| format!("; it was dropped in table version {v}"))
                                .unwrap_or_default(),
                        ),
                        Err(_) => (self.tbl_name(cat), String::new()),
                    };
                    crate::bail!(
                        ColumnNotFound,
                        "Column was dropped (no record for column ID {} in table {}{dropped})",
                        r.col_id,
                        py_repr(&name)
                    );
                }
            }
        }
        Ok(())
    }

    /// Rows of upstream's query descriptor (`From`, `Where`, `Group By`, `Order By`, `Limit`, `Sample`).
    pub fn descriptor_rows(&self, cat: &Catalog) -> Vec<(String, String)> {
        use crate::expr::display::display;
        let mut out: Vec<(String, String)> = Vec::new();
        for (i, name) in self.tbl_names(cat).into_iter().enumerate() {
            out.push((if i == 0 { "From".into() } else { String::new() }, name));
        }
        if let Some(w) = &self.where_ {
            out.push(("Where".into(), display(w)));
        }
        if let Some(g) = &self.group_by {
            for (i, e) in g.iter().enumerate() {
                out.push((if i == 0 { "Group By".into() } else { String::new() }, display(e)));
            }
        }
        for (i, (e, asc)) in self.order_by.iter().enumerate() {
            let label = if i == 0 { "Order By".into() } else { String::new() };
            out.push((label, format!("{} {}", display(e), if *asc { "asc" } else { "desc" })));
        }
        if let Some(mut s) = self.limit.map(|n| n.to_string()).or_else(|| self.limit_param.clone()) {
            if let Some(o) = self.offset.map(|o| o.to_string()).or_else(|| self.offset_param.clone()) {
                s.push_str(&format!(",{o}"));
            }
            out.push(("Limit".into(), s));
        }
        if let Some(s) = &self.sample {
            out.push(("Sample".into(), s.display()));
        }
        out
    }

    /// The query with every expression replaced by `f(expr)` (used to bind the parameters of a `@pxt.query`
    /// template). Every expression-holding field must be mapped here (the destructuring makes a new field a compile
    /// error until it is).
    pub fn map_exprs(&self, f: &dyn Fn(&ExprRef) -> Result<ExprRef>) -> Result<QuerySpec> {
        let QuerySpec {
            tbl_id,
            version,
            joins,
            select,
            where_,
            group_by,
            grouping_tbl,
            order_by,
            limit,
            offset,
            limit_param,
            offset_param,
            sample,
        } = self;
        let map_all = |items: &[ExprRef]| -> Result<Vec<ExprRef>> { items.iter().map(f).collect() };
        Ok(QuerySpec {
            tbl_id: *tbl_id,
            version: *version,
            joins: joins
                .iter()
                .map(|j| Ok(JoinClause { on: j.on.as_ref().map(f).transpose()?, ..j.clone() }))
                .collect::<Result<_>>()?,
            select: select
                .as_ref()
                .map(|items| items.iter().map(|(e, n)| Ok((f(e)?, n.clone()))).collect::<Result<Vec<_>>>())
                .transpose()?,
            where_: where_.as_ref().map(f).transpose()?,
            group_by: group_by.as_ref().map(|g| map_all(g)).transpose()?,
            grouping_tbl: *grouping_tbl,
            order_by: order_by.iter().map(|(e, asc)| Ok((f(e)?, *asc))).collect::<Result<_>>()?,
            limit: *limit,
            offset: *offset,
            limit_param: limit_param.clone(),
            offset_param: offset_param.clone(),
            sample: sample
                .as_ref()
                .map(|s| Ok::<_, Error>(SampleClause { stratify_by: map_all(&s.stratify_by)?, ..s.clone() }))
                .transpose()?,
        })
    }

    /// Every expression of the query.
    pub fn exprs(&self) -> Vec<ExprRef> {
        let out = std::cell::RefCell::new(Vec::new());
        let _ = self.map_exprs(&|e| {
            out.borrow_mut().push(e.clone());
            Ok(e.clone())
        });
        out.into_inner()
    }

    // ------------------------------------------------------------------------------------------- serialization

    pub fn to_json(&self) -> Result<serde_json::Value> {
        use crate::expr::serde::to_json as e2j;
        use serde_json::{Value as J, json};
        let exprs = |items: &[ExprRef]| -> Result<J> { Ok(J::Array(items.iter().map(|e| e2j(e)).collect::<Result<_>>()?)) };
        let mut tbls = vec![json!({"id": self.tbl_id.to_string(), "effective_version": self.version})];
        let mut join_clauses = Vec::new();
        for j in &self.joins {
            tbls.push(json!({"id": j.tbl_id.to_string(), "effective_version": j.version}));
            join_clauses.push(json!({
                "join_type": j.how.name(),
                "join_predicate": j.on.as_ref().map(|e| e2j(e)).transpose()?,
            }));
        }
        Ok(json!({
            "_classname": "Query",
            "from_clause": {"tbls": tbls, "join_clauses": join_clauses},
            "select_list": match &self.select {
                None => J::Null,
                Some(items) => J::Array(items.iter().map(|(e, n)| Ok(json!([e2j(e)?, n]))).collect::<Result<_>>()?),
            },
            "where_clause": self.where_.as_ref().map(|e| e2j(e)).transpose()?,
            "group_by_clause": self.group_by.as_ref().map(|g| exprs(g)).transpose()?,
            "grouping_tbl": self.grouping_tbl.map(|id| json!({"id": id.to_string()})),
            "order_by_clause": if self.order_by.is_empty() {
                J::Null
            } else {
                J::Array(self.order_by.iter().map(|(e, asc)| Ok(json!([e2j(e)?, asc]))).collect::<Result<_>>()?)
            },
            "limit_val": self.limit,
            "offset_val": self.offset,
            "limit_param": self.limit_param,
            "offset_param": self.offset_param,
            "sample_clause": match &self.sample {
                None => J::Null,
                Some(s) => s.to_json()?,
            },
        }))
    }

    pub fn from_json(d: &serde_json::Value) -> Result<QuerySpec> {
        use crate::expr::serde::from_json as j2e;
        use serde_json::Value as J;
        let bad = || Error::internal(format!("malformed serialized query: {d}"));
        if d.get("_classname").and_then(|c| c.as_str()) != Some("Query") {
            return Err(Error::internal("Expected a serialized Query"));
        }
        let parse_tbl = |tbl: &J| -> Result<(Uuid, Option<i64>)> {
            let id = tbl.get("id").and_then(|v| v.as_str()).and_then(|s| Uuid::parse_str(s).ok()).ok_or_else(bad)?;
            Ok((id, tbl.get("effective_version").and_then(|v| v.as_i64())))
        };
        let exprs = |j: Option<&J>| -> Result<Option<Vec<ExprRef>>> {
            match j {
                None | Some(J::Null) => Ok(None),
                Some(J::Array(items)) => Ok(Some(items.iter().map(j2e).collect::<Result<_>>()?)),
                _ => Err(bad()),
            }
        };
        let tbls: Vec<&J> = match d.pointer("/from_clause/tbls") {
            Some(J::Array(t)) => t.iter().collect(),
            _ => return Err(bad()),
        };
        let (tbl_id, version) = parse_tbl(tbls.first().ok_or_else(bad)?)?;
        let mut joins = Vec::new();
        if let Some(J::Array(clauses)) = d.pointer("/from_clause/join_clauses") {
            for (i, c) in clauses.iter().enumerate() {
                let (id, v) = parse_tbl(tbls.get(i + 1).ok_or_else(bad)?)?;
                let how = JoinType::parse(c.get("join_type").and_then(|j| j.as_str()).ok_or_else(bad)?)?;
                let on = match c.get("join_predicate") {
                    None | Some(J::Null) => None,
                    Some(p) => Some(j2e(p)?),
                };
                joins.push(JoinClause { tbl_id: id, version: v, how, on });
            }
        }
        let select = match d.get("select_list") {
            None | Some(J::Null) => None,
            Some(J::Array(items)) => Some(
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
            None | Some(J::Null) => None,
            Some(w) => Some(j2e(w)?),
        };
        let order_by = match d.get("order_by_clause") {
            Some(J::Array(items)) => items
                .iter()
                .map(|item| Ok((j2e(item.get(0).ok_or_else(bad)?)?, item.get(1).and_then(|a| a.as_bool()).unwrap_or(true))))
                .collect::<Result<_>>()?,
            _ => Vec::new(),
        };
        let grouping_tbl = d
            .get("grouping_tbl")
            .and_then(|g| g.get("id"))
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok());
        let sample = match d.get("sample_clause") {
            None | Some(J::Null) => None,
            Some(s) => Some(SampleClause::from_json(s)?),
        };
        Ok(QuerySpec {
            tbl_id,
            version,
            joins,
            select,
            where_,
            group_by: exprs(d.get("group_by_clause"))?,
            grouping_tbl,
            order_by,
            limit: d.get("limit_val").and_then(|v| v.as_i64()),
            offset: d.get("offset_val").and_then(|v| v.as_i64()),
            limit_param: d.get("limit_param").and_then(|v| v.as_str()).map(str::to_owned),
            offset_param: d.get("offset_param").and_then(|v| v.as_str()).map(str::to_owned),
            sample,
        })
    }
}

/// A limit or offset: a constant, or (in a `@pxt.query` template) a parameter.
pub enum LimitArg {
    Const(i64),
    Param(String),
}

impl LimitArg {
    fn constant(&self) -> Option<i64> {
        match self {
            LimitArg::Const(c) => Some(*c),
            LimitArg::Param(_) => None,
        }
    }

    fn param(&self) -> Option<String> {
        match self {
            LimitArg::Param(p) => Some(p.clone()),
            LimitArg::Const(_) => None,
        }
    }
}

/// The `on` argument of `join()`.
pub enum JoinOn {
    None,
    Expr(ExprRef),
    Columns(Vec<ExprRef>),
}

pub fn column_ref_expr(c: &ResolvedColumn, version: Option<i64>) -> ExprRef {
    let _ = version;
    Expr::new(
        ExprKind::ColumnRef(ColumnRef { tbl_id: c.tbl_id, col_id: c.md.id, name: c.name.clone(), tbl_version: c.tbl_version }),
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
