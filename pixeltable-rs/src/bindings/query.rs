//! `pixeltable._core.QuerySpec` (the state behind `pixeltable.Query`) and `pixeltable._core.ResultSet`.

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString};

use super::catalog::parse_uuid;
use super::expr::{as_expr, from_object};
use super::types::wrap_type;
use crate::catalog::path::fold_identifier;
use crate::catalog::runtime::with_catalog;
use crate::error::{Error, Result};
use crate::exec::{JoinOn, QueryInput, QuerySpec, ResultData, SampleClause, count, execute};
use crate::value::Value;
use crate::value::py as vpy;

#[pyclass(name = "QuerySpec", module = "pixeltable._core", frozen)]
pub struct PyQuerySpec {
    pub q: QuerySpec,
}

fn wrap(q: QuerySpec) -> PyQuerySpec {
    PyQuerySpec { q }
}

fn obj_str(obj: &Bound<'_, PyAny>) -> String {
    obj.str().map(|s| s.to_string()).unwrap_or_default()
}

fn int_param(obj: &Bound<'_, PyAny>, name: &str) -> Result<i64> {
    if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
        return Ok(obj.extract()?);
    }
    let t = vpy::infer_literal(obj)?.map(|(t, _)| t.repr()).unwrap_or_else(|| "Invalid".to_string());
    crate::bail!(TypeMismatch, "{} parameter must be of type `Int`; got `{t}`", crate::error::py_repr(name))
}

fn run(py: Python<'_>, q: &QuerySpec, rowid_desc: bool) -> Result<ResultData> {
    let input = with_catalog(py, |cat| QueryInput::snapshot(cat, q))?;
    execute(&input, rowid_desc)
}

/// Upstream `_validate_constant_type_range()` for `sample()` parameters: a constant of the required type, in range.
fn sample_param(
    obj: Option<&Bound<'_, PyAny>>,
    name: &str,
    want_float: bool,
    range: (Option<f64>, Option<f64>),
) -> Result<Option<Value>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else { return Ok(None) };
    let Some(e) = from_object(obj)? else {
        crate::bail!(TypeMismatch, "{} parameter must be of type `{}`; got `Invalid`", crate::error::py_repr(name), if want_float { "Float" } else { "Int" });
    };
    let ok = if want_float { e.col_type.is_float() } else { e.col_type.is_int() };
    if !ok {
        crate::bail!(
            TypeMismatch,
            "{} parameter must be of type `{}`; got `{}`",
            crate::error::py_repr(name),
            if want_float { "Float" } else { "Int" },
            e.col_type.repr()
        );
    }
    let Some(v) = e.as_literal().cloned() else {
        crate::bail!(InvalidArgument, "{} parameter must be a constant; got: {}", crate::error::py_repr(name), crate::expr::display::display(&e));
    };
    let x = v.as_f64().unwrap_or(0.0);
    let fmt = |b: f64| if want_float { crate::value::py_float_repr(b) } else { (b as i64).to_string() };
    if let Some(lo) = range.0 {
        if x < lo {
            crate::bail!(InvalidArgument, "{} parameter must be >= {}", crate::error::py_repr(name), fmt(lo));
        }
    }
    if let Some(hi) = range.1 {
        if x > hi {
            crate::bail!(InvalidArgument, "{} parameter must be <= {}", crate::error::py_repr(name), fmt(hi));
        }
    }
    Ok(Some(v))
}

#[pymethods]
impl PyQuerySpec {
    #[new]
    #[pyo3(signature = (tbl_id, version=None))]
    fn py_new(py: Python<'_>, tbl_id: &str, version: Option<i64>) -> Result<Self> {
        let id = parse_uuid(tbl_id)?;
        // find out about dropped tables early
        with_catalog(py, |cat| cat.table(id).map(|_| ()))?;
        Ok(wrap(QuerySpec::new(id, version)))
    }

    /// (table id, version) of the from-table
    fn tbl_ref(&self) -> (String, Option<i64>) {
        (self.q.tbl_id.to_string(), self.q.version)
    }

    #[getter]
    fn has_select_list(&self) -> bool {
        self.q.select.is_some()
    }

    #[getter]
    fn has_order_by(&self) -> bool {
        !self.q.order_by.is_empty()
    }

    #[getter]
    fn has_limit(&self) -> bool {
        self.q.limit.is_some()
    }

    #[getter]
    fn has_where(&self) -> bool {
        self.q.where_.is_some()
    }

    /// `items`: (expression or literal, name or None) pairs.
    fn select(&self, py: Python<'_>, items: Vec<(Bound<'_, PyAny>, Option<String>)>) -> Result<Self> {
        let mut exprs = Vec::with_capacity(items.len());
        for (raw, name) in items {
            let e = from_object(&raw)?.ok_or_else(|| crate::pxt_err!(InvalidExpression, "Invalid expression: {}", obj_str(&raw)))?;
            exprs.push((e, name));
        }
        with_catalog(py, |cat| Ok(wrap(self.q.with_select(cat, exprs)?)))
    }

    #[pyo3(name = "where_")]
    fn where_clause(&self, py: Python<'_>, pred: &Bound<'_, PyAny>) -> Result<Self> {
        if self.q.where_.is_some() {
            crate::bail!(InvalidState, "where() clause already specified");
        }
        let e = as_expr(pred).ok_or_else(|| {
            crate::pxt_err!(InvalidExpression, "where() expects a Pixeltable expression; got: {}", obj_str(pred))
        })?;
        with_catalog(py, |cat| Ok(wrap(self.q.with_where(cat, e)?)))
    }

    #[pyo3(signature = (exprs, asc=true))]
    fn order_by(&self, py: Python<'_>, exprs: Vec<Bound<'_, PyAny>>, asc: bool) -> Result<Self> {
        let mut es = Vec::new();
        for raw in exprs {
            es.push(as_expr(&raw).ok_or_else(|| crate::pxt_err!(InvalidExpression, "Invalid expression in order_by(): {}", obj_str(&raw)))?);
        }
        with_catalog(py, |cat| Ok(wrap(self.q.with_order_by(cat, es, asc)?)))
    }

    #[pyo3(signature = (n, offset=None))]
    fn limit(&self, n: &Bound<'_, PyAny>, offset: Option<&Bound<'_, PyAny>>) -> Result<Self> {
        if n.is_none() {
            crate::bail!(MissingRequired, "'limit()' parameter must be present");
        }
        if n.is_instance_of::<PyInt>() && !n.is_instance_of::<PyBool>() && n.extract::<i64>()? < 0 {
            crate::bail!(InvalidArgument, "'limit()' parameter must be >= 0");
        }
        let n = int_param(n, "limit()")?;
        let offset = match offset {
            None => None,
            Some(o) if o.is_none() => None,
            Some(o) => {
                if o.is_instance_of::<PyInt>() && !o.is_instance_of::<PyBool>() && o.extract::<i64>()? < 0 {
                    crate::bail!(InvalidArgument, "'offset' parameter must be >= 0");
                }
                Some(int_param(o, "offset")?)
            }
        };
        Ok(wrap(self.q.with_limit(n, offset)?))
    }

    fn collect(&self, py: Python<'_>) -> Result<PyResultSet> {
        Ok(PyResultSet::new(run(py, &self.q, false)?))
    }

    /// `show(n)`: not after `sample()`.
    fn show(&self, py: Python<'_>, n: &Bound<'_, PyAny>) -> Result<PyResultSet> {
        if self.q.sample.is_some() {
            crate::bail!(UnsupportedOperation, "show() cannot be used with sample()");
        }
        let q = self.limit(n, None)?;
        q.collect(py)
    }

    #[getter]
    fn has_group_by(&self) -> bool {
        self.q.is_grouped()
    }

    #[getter]
    fn has_joins(&self) -> bool {
        self.q.has_joins()
    }

    #[getter]
    fn has_sample(&self) -> bool {
        self.q.sample.is_some()
    }

    /// `items`: expressions, or a single table id (group by a base table's rows).
    #[pyo3(signature = (exprs, grouping_tbl=None))]
    fn group_by(&self, py: Python<'_>, exprs: Vec<Bound<'_, PyAny>>, grouping_tbl: Option<String>) -> Result<Self> {
        if let Some(t) = grouping_tbl {
            let id = parse_uuid(&t)?;
            return with_catalog(py, |cat| Ok(wrap(self.q.with_grouping_tbl(cat, id)?)));
        }
        if self.q.is_grouped() {
            crate::bail!(InvalidState, "group_by() already specified");
        }
        let mut es = Vec::new();
        for raw in exprs {
            es.push(as_expr(&raw).ok_or_else(|| crate::pxt_err!(InvalidExpression, "Invalid expression in group_by(): {}", obj_str(&raw)))?);
        }
        with_catalog(py, |cat| {
            for e in &es {
                self.q.check_bound(cat, e)?;
            }
            Ok(wrap(self.q.with_group_by(es)?))
        })
    }

    fn distinct(&self, py: Python<'_>) -> Result<Self> {
        with_catalog(py, |cat| Ok(wrap(self.q.with_distinct(cat)?)))
    }

    /// `on`: None, an expression or a sequence of column references; `other`: (table id, version).
    fn join(&self, py: Python<'_>, other: (String, Option<i64>), on: &Bound<'_, PyAny>, how: &str) -> Result<Self> {
        let other_id = parse_uuid(&other.0)?;
        let on = if on.is_none() {
            JoinOn::None
        } else if let Some(e) = as_expr(on) {
            JoinOn::Expr(e)
        } else {
            let bad = || crate::pxt_err!(InvalidArgument, "`on` must be a sequence of column references or a boolean expression");
            let is_seq = on.is_instance_of::<PyList>() || on.is_instance_of::<pyo3::types::PyTuple>();
            if !is_seq {
                return Err(bad());
            }
            let mut cols = Vec::new();
            for item in on.try_iter()? {
                let item = item?;
                match as_expr(&item) {
                    Some(e) if e.as_column_ref().is_some() => cols.push(e),
                    _ => return Err(bad()),
                }
            }
            if cols.is_empty() {
                return Err(bad());
            }
            JoinOn::Columns(cols)
        };
        with_catalog(py, |cat| Ok(wrap(self.q.with_join(cat, (other_id, other.1), on, how)?)))
    }

    #[pyo3(signature = (n=None, n_per_stratum=None, fraction=None, seed=None, stratify_by=None))]
    fn sample(
        &self,
        py: Python<'_>,
        n: Option<&Bound<'_, PyAny>>,
        n_per_stratum: Option<&Bound<'_, PyAny>>,
        fraction: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
        stratify_by: Option<&Bound<'_, PyAny>>,
    ) -> Result<Self> {
        if self.q.sample.is_some() {
            crate::bail!(UnsupportedOperation, "Multiple sample() clauses not allowed");
        }
        if self.q.is_grouped() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with group_by()");
        }
        if !self.q.order_by.is_empty() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with order_by()");
        }
        if self.q.limit.is_some() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with limit()");
        }
        if self.q.has_joins() {
            crate::bail!(UnsupportedOperation, "sample() cannot be used with join()");
        }
        let given = |o: Option<&Bound<'_, PyAny>>| o.is_some_and(|o| !o.is_none());
        if [given(n), given(n_per_stratum), given(fraction)].iter().filter(|g| **g).count() != 1 {
            crate::bail!(MissingRequired, "Exactly one of `n`, `n_per_stratum`, or `fraction` must be specified.");
        }
        let has_strata = given(stratify_by);
        if given(n_per_stratum) && !has_strata {
            crate::bail!(UnsupportedOperation, "Must specify `stratify_by` to use `n_per_stratum`");
        }
        let n = sample_param(n, "n", false, (Some(1.0), None))?;
        let n_per_stratum = sample_param(n_per_stratum, "n_per_stratum", false, (Some(1.0), None))?;
        let fraction = sample_param(fraction, "fraction", true, (Some(0.0), Some(1.0)))?;
        let seed = sample_param(seed, "seed", false, (None, None))?;
        let mut strata = Vec::new();
        if let Some(sb) = stratify_by.filter(|o| !o.is_none()) {
            let items: Vec<Bound<'_, PyAny>> = if as_expr(sb).is_some() {
                vec![sb.clone()]
            } else if sb.is_instance_of::<PyList>() || sb.is_instance_of::<pyo3::types::PyTuple>() {
                sb.try_iter()?.collect::<PyResult<_>>()?
            } else {
                crate::bail!(InvalidArgument, "`stratify_by` must be a list of scalar expressions");
            };
            for it in items {
                match as_expr(&it) {
                    Some(e) => strata.push(e),
                    None => crate::bail!(InvalidExpression, "Invalid expression: {}", obj_str(&it)),
                }
            }
        }
        let clause = SampleClause {
            n: n.and_then(|v| v.as_i64()),
            n_per_stratum: n_per_stratum.and_then(|v| v.as_i64()),
            fraction: fraction.and_then(|v| v.as_f64()),
            seed: seed.and_then(|v| v.as_i64()),
            stratify_by: strata,
        };
        with_catalog(py, |cat| Ok(wrap(self.q.with_sample(cat, clause)?)))
    }

    /// (heading, info) rows of the query descriptor (From, Where, Group By, Order By, Limit, Sample).
    fn query_descriptor(&self, py: Python<'_>) -> Result<Vec<(String, String)>> {
        with_catalog(py, |cat| Ok(self.q.descriptor_rows(cat)))
    }

    /// Checks for update()/delete()/recompute_columns() (upstream `_validate_mutable()`), except snapshot checks.
    fn validate_mutable(&self, op_name: &str) -> Result<()> {
        if self.q.is_grouped() {
            crate::bail!(UnsupportedOperation, "Cannot use `{op_name}` after `group_by`.");
        }
        if !self.q.order_by.is_empty() {
            crate::bail!(UnsupportedOperation, "Cannot use `{op_name}` after `order_by`.");
        }
        if self.q.select.is_some() {
            crate::bail!(UnsupportedOperation, "Cannot use `{op_name}` after `select`.");
        }
        if self.q.limit.is_some() {
            crate::bail!(UnsupportedOperation, "Cannot use `{op_name}` after `limit`.");
        }
        if self.q.has_joins() {
            crate::bail!(UnsupportedOperation, "Cannot use `{op_name}` after `join`.");
        }
        Ok(())
    }

    fn head(&self, py: Python<'_>, n: i64) -> Result<PyResultSet> {
        self.q.check_head_tail("head")?;
        let q = self.q.with_limit(n, None)?;
        Ok(PyResultSet::new(run(py, &q, false)?))
    }

    fn tail(&self, py: Python<'_>, n: i64) -> Result<PyResultSet> {
        self.q.check_head_tail("tail")?;
        let q = self.q.with_limit(n, None)?;
        let mut data = run(py, &q, true)?;
        data.rows.reverse();
        Ok(PyResultSet::new(data))
    }

    fn count(&self, py: Python<'_>) -> Result<usize> {
        let input = with_catalog(py, |cat| QueryInput::snapshot(cat, &self.q))?;
        count(&input)
    }

    /// name -> ColumnType of the result columns
    fn schema<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>> {
        let items = with_catalog(py, |cat| self.q.effective_select(cat))?;
        let d = PyDict::new(py);
        for (e, name) in items {
            d.set_item(name, wrap_type(py, &e.col_type)?)?;
        }
        Ok(d)
    }

    /// Rows of the description table: (name, type repr, expression)
    fn describe_rows(&self, py: Python<'_>) -> Result<Vec<(String, String, String)>> {
        let items = with_catalog(py, |cat| self.q.effective_select(cat))?;
        Ok(items
            .into_iter()
            .map(|(e, n)| (n, e.col_type.repr(), crate::expr::display::display(&e)))
            .collect())
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>> {
        Ok(vpy::json_to_py(py, &self.q.to_json()?)?)
    }

    /// `from_dict()` without checking the catalog (for deserialization that runs under the catalog lock).
    #[staticmethod]
    fn _from_dict_unchecked(py: Python<'_>, d: &Bound<'_, PyAny>) -> Result<Self> {
        let s: String = py.import("json")?.call_method1("dumps", (d,))?.extract()?;
        let j: serde_json::Value = serde_json::from_str(&s)?;
        Ok(wrap(QuerySpec::from_json(&j)?))
    }

    #[staticmethod]
    fn from_dict(py: Python<'_>, d: &Bound<'_, PyAny>) -> Result<Self> {
        let s: String = py.import("json")?.call_method1("dumps", (d,))?.extract()?;
        let j: serde_json::Value = serde_json::from_str(&s)?;
        let q = QuerySpec::from_json(&j)?;
        with_catalog(py, |cat| cat.table(q.tbl_id).map(|_| ()))?;
        Ok(wrap(q))
    }
}

/// The rows of a query result, held in Rust and converted to Python objects on access.
#[pyclass(name = "ResultSet", module = "pixeltable._core", frozen)]
pub struct PyResultSet {
    pub data: ResultData,
}

impl PyResultSet {
    pub fn new(data: ResultData) -> Self {
        PyResultSet { data }
    }

    fn col_idx(&self, name: &str) -> Result<usize> {
        let folded = fold_identifier(name);
        self.data
            .names
            .iter()
            .position(|n| *n == folded)
            .ok_or_else(|| crate::pxt_err!(InvalidColumnName, "Invalid column name: {name}"))
    }

    fn row_idx(&self, i: i64) -> PyResult<usize> {
        let n = self.data.rows.len() as i64;
        let idx = if i < 0 { i + n } else { i };
        if idx < 0 || idx >= n {
            return Err(PyIndexError::new_err("list index out of range"));
        }
        Ok(idx as usize)
    }

    fn row_dict<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        if let Some(row) = self.data.rows.get(i) {
            for ((name, t), v) in self.data.names.iter().zip(self.data.types.iter()).zip(row.iter()) {
                d.set_item(name, vpy::to_py(py, v, t)?)?;
            }
        }
        Ok(d)
    }
}

#[pymethods]
impl PyResultSet {
    fn __len__(&self) -> usize {
        self.data.rows.len()
    }

    fn column_names(&self) -> Vec<String> {
        self.data.names.clone()
    }

    /// name -> ColumnType
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (n, t) in self.data.names.iter().zip(self.data.types.iter()) {
            d.set_item(n, wrap_type(py, t)?)?;
        }
        Ok(d)
    }

    fn row<'py>(&self, py: Python<'py>, i: i64) -> PyResult<Bound<'py, PyDict>> {
        let idx = self.row_idx(i)?;
        self.row_dict(py, idx)
    }

    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for i in 0..self.data.rows.len() {
            list.append(self.row_dict(py, i)?)?;
        }
        Ok(list)
    }

    fn column<'py>(&self, py: Python<'py>, name: &str) -> Result<Bound<'py, PyList>> {
        let c = self.col_idx(name)?;
        let t = &self.data.types[c];
        let list = PyList::empty(py);
        for row in &self.data.rows {
            list.append(vpy::to_py(py, &row[c], t)?)?;
        }
        Ok(list)
    }

    /// `result[i, col]` with `col` a name or a position
    fn cell<'py>(&self, py: Python<'py>, i: i64, col: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
        let r = self.row_idx(i)?;
        let c = if col.is_instance_of::<PyString>() {
            self.col_idx(&col.extract::<String>()?)?
        } else {
            let ci: i64 = col.extract()?;
            let n = self.data.names.len() as i64;
            let ci = if ci < 0 { ci + n } else { ci };
            if ci < 0 || ci >= n {
                return Err(Error::Python(PyIndexError::new_err("list index out of range")));
            }
            ci as usize
        };
        Ok(vpy::to_py(py, &self.data.rows[r][c], &self.data.types[c])?)
    }

    /// `result[index]`: a row (int), a column (str), or a cell ((int, str | int)).
    fn getitem<'py>(&self, py: Python<'py>, index: &Bound<'py, PyAny>) -> Result<Bound<'py, PyAny>> {
        if index.is_instance_of::<PyString>() {
            return Ok(self.column(py, &index.extract::<String>()?)?.into_any());
        }
        if index.is_instance_of::<PyInt>() {
            return Ok(self.row(py, index.extract()?)?.into_any());
        }
        if let Ok(t) = index.cast::<pyo3::types::PyTuple>() {
            if t.len() == 2 {
                let (i, c) = (t.get_item(0)?, t.get_item(1)?);
                if !i.is_instance_of::<PyInt>() || !(c.is_instance_of::<PyString>() || c.is_instance_of::<PyInt>()) {
                    crate::bail!(
                        UnsupportedOperation,
                        "Bad index, expected [<row idx>, <column name | column index>]: {}",
                        obj_str(index)
                    );
                }
                return self.cell(py, i.extract()?, &c);
            }
        }
        crate::bail!(UnsupportedOperation, "Bad index: {}", obj_str(index))
    }

    /// Upstream `ResultSet.to_pydantic()`: one model instance per row; model field names are matched against the
    /// folded column names.
    fn to_pydantic<'py>(&self, py: Python<'py>, model: &Bound<'py, PyAny>) -> Result<Bound<'py, PyList>> {
        let model_name: String = model.getattr("__name__")?.extract()?;
        let fields = model.getattr("model_fields")?.cast_into::<PyDict>().map_err(|_| Error::internal("model_fields is not a dict"))?;
        let forbid_extra = model
            .getattr("model_config")
            .ok()
            .and_then(|c| c.get_item("extra").ok())
            .and_then(|e| e.extract::<String>().ok())
            .is_some_and(|e| e == "forbid");
        let set_repr = |items: &[String]| -> String {
            let parts: Vec<String> = items.iter().map(|s| crate::error::py_repr(s)).collect();
            format!("{{{}}}", parts.join(", "))
        };
        // folded field name -> original name
        let mut folded_to_orig: Vec<(String, String)> = Vec::new();
        let mut required: Vec<String> = Vec::new();
        for (name, field) in fields.iter() {
            let name: String = name.extract()?;
            let folded = fold_identifier(&name);
            if let Some((_, prev)) = folded_to_orig.iter().find(|(f, _)| *f == folded) {
                crate::bail!(
                    UnsupportedOperation,
                    "Result column names are case-insensitive, but model {model_name} has fields {} and {} which are the same",
                    crate::error::py_repr(prev),
                    crate::error::py_repr(&name)
                );
            }
            if field.call_method0("is_required")?.is_truthy()? {
                required.push(folded.clone());
            }
            folded_to_orig.push((folded, name));
        }
        let missing: Vec<String> = required
            .iter()
            .filter(|f| !self.data.names.contains(f))
            .map(|f| folded_to_orig.iter().find(|(x, _)| x == f).map(|(_, o)| o.clone()).unwrap_or_default())
            .collect();
        if !missing.is_empty() {
            crate::bail!(
                UnsupportedOperation,
                "Required model fields ({}) are missing from result set columns ({})",
                set_repr(&missing),
                self.data.names.join(", ")
            );
        }
        if forbid_extra {
            let extra: Vec<String> = self.data.names.iter().filter(|n| !folded_to_orig.iter().any(|(f, _)| f == *n)).cloned().collect();
            if !extra.is_empty() {
                crate::bail!(UnsupportedOperation, "Extra fields {} are not allowed in model with extra='forbid'", set_repr(&extra));
            }
        }
        let validation_error = py.import("pydantic")?.getattr("ValidationError")?;
        let out = PyList::empty(py);
        for row in &self.data.rows {
            let kwargs = PyDict::new(py);
            for ((name, t), v) in self.data.names.iter().zip(self.data.types.iter()).zip(row.iter()) {
                let key = folded_to_orig.iter().find(|(f, _)| f == name).map(|(_, o)| o.as_str()).unwrap_or(name);
                kwargs.set_item(key, vpy::to_py(py, v, t)?)?;
            }
            match model.call((), Some(&kwargs)) {
                Ok(m) => out.append(m)?,
                Err(e) if e.is_instance(py, &validation_error) => {
                    crate::bail!(UnsupportedOperation, "{}", e.value(py).str()?);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(out)
    }

    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let records = self.rows(py)?;
        let pd = py.import("pandas")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", PyList::new(py, &self.data.names)?)?;
        pd.getattr("DataFrame")?.call_method("from_records", (records,), Some(&kwargs))
    }
}

enum CursorState {
    Pending,
    Open { data: ResultData, pos: usize, columns: Py<PyDict>, col_types: Py<PyDict> },
    Closed,
}

/// The backend of `pixeltable.ResultCursor`: yields `pixeltable.Row` objects.
///
/// The query runs when the cursor is opened; rows are converted to Python objects one at a time.
#[pyclass(name = "ResultCursor", module = "pixeltable._core", frozen)]
pub struct PyCursor {
    q: QuerySpec,
    state: parking_lot::Mutex<CursorState>,
}

impl PyCursor {
    fn open_inner(&self, py: Python<'_>, state: &mut CursorState) -> Result<()> {
        match state {
            CursorState::Open { .. } => crate::bail!(InvalidState, "Cursor is already open."),
            CursorState::Closed => crate::bail!(InvalidState, "Cursor is closed and cannot be reopened."),
            CursorState::Pending => {}
        }
        let data = run(py, &self.q, false)?;
        let columns = PyDict::new(py);
        let col_types = PyDict::new(py);
        for (i, (n, t)) in data.names.iter().zip(data.types.iter()).enumerate() {
            columns.set_item(n, i)?;
            col_types.set_item(n, wrap_type(py, t)?)?;
        }
        *state = CursorState::Open { data, pos: 0, columns: columns.unbind(), col_types: col_types.unbind() };
        Ok(())
    }
}

#[pymethods]
impl PyCursor {
    fn open(&self, py: Python<'_>) -> Result<()> {
        let mut state = self.state.lock();
        self.open_inner(py, &mut state)
    }

    fn close(&self) {
        *self.state.lock() = CursorState::Closed;
    }

    #[getter]
    fn state_name(&self) -> &'static str {
        match *self.state.lock() {
            CursorState::Pending => "pending",
            CursorState::Open { .. } => "open",
            CursorState::Closed => "closed",
        }
    }

    #[getter]
    fn _closed(&self) -> bool {
        matches!(*self.state.lock(), CursorState::Closed)
    }

    fn __iter__(slf: Bound<'_, Self>) -> Result<Bound<'_, Self>> {
        let py = slf.py();
        {
            let this = slf.get();
            let mut state = this.state.lock();
            match *state {
                CursorState::Closed => crate::bail!(InvalidState, "Cursor is closed and cannot be iterated upon."),
                CursorState::Pending => this.open_inner(py, &mut state)?,
                CursorState::Open { .. } => {}
            }
        }
        Ok(slf)
    }

    fn __next__<'py>(&self, py: Python<'py>) -> Result<Option<Bound<'py, PyAny>>> {
        let mut state = self.state.lock();
        let row = match &mut *state {
            CursorState::Open { data, pos, columns, col_types } => {
                if *pos >= data.rows.len() {
                    None
                } else {
                    let i = *pos;
                    *pos += 1;
                    let vals: Vec<Bound<'py, PyAny>> = data.rows[i]
                        .iter()
                        .zip(data.types.iter())
                        .map(|(v, t)| vpy::to_py(py, v, t))
                        .collect::<PyResult<_>>()?;
                    let row_cls = py.import("pixeltable._row")?.getattr("Row")?;
                    Some(row_cls.call1((pyo3::types::PyTuple::new(py, vals)?, columns.bind(py), col_types.bind(py)))?)
                }
            }
            _ => None,
        };
        if row.is_none() {
            *state = CursorState::Closed;
        }
        Ok(row)
    }
}

#[pymethods]
impl PyQuerySpec {
    fn cursor(&self) -> PyCursor {
        PyCursor { q: self.q.clone(), state: parking_lot::Mutex::new(CursorState::Pending) }
    }
}
