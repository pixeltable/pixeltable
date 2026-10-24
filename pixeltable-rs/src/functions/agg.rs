//! Aggregate functions: the `AggregateFunction` / `AggState` traits, the builtin aggregates (`count`, `sum`, `min`,
//! `max`, `mean` of `pixeltable.functions` and `pixeltable.functions.json.make_list`), and the construction of
//! aggregate calls with upstream's rules for `order_by` / `group_by` and init parameters.
//!
//! An aggregate call is a `FunctionCall` whose `func` is `FuncRef::Agg`. The executor (`exec/agg.rs`) creates one
//! `AggState` per group (or per window partition), feeds it the evaluated arguments of each row with `update()` and
//! reads the result with `value()`.
//!
//! Python UDAs plug in by implementing `AggregateFunction` (signatures built from the Aggregator class's type
//! hints, `create()` instantiating the class, `AggState::update()` / `value()` calling its methods under the GIL)
//! and building calls with `make_agg_call()` (from Python objects: `bindings::expr::agg_call_from_py()`).
//! Deserialization of a stored UDA call goes through `resolve_uda()`.

use std::fmt::Debug;
use std::sync::{Arc, OnceLock};

use serde_json::json;

use super::FuncRef;
use super::signature::{Bound, Signature};
use crate::error::{Error, Result, py_repr};
use crate::expr::{Expr, ExprKind, ExprRef, FunctionCall};
use crate::types::ColumnType;
use crate::value::{Json, Value, cmp_values};

pub type AggRef = Arc<dyn AggregateFunction>;

/// The running state of one aggregation (one group, or one window partition).
pub trait AggState: Send {
    /// Adds a row; `args` are the positional arguments, `kwargs` the keyword arguments other than init parameters.
    fn update(&mut self, args: &[Value], kwargs: &[(String, Value)]) -> Result<()>;
    fn value(&mut self) -> Result<Value>;
}

pub trait AggregateFunction: Send + Sync + Debug {
    /// display name (`sum`, or a UDA's class name)
    fn name(&self) -> &str;
    fn qualified_name(&self) -> String;
    /// one or more signatures (type substitutions), tried in order; init parameters are keyword-only parameters
    fn signatures(&self) -> &[Signature];
    /// names of the parameters passed to the aggregator's constructor rather than to `update()`
    fn init_param_names(&self) -> &[String];
    fn requires_order_by(&self) -> bool;
    fn allows_std_agg(&self) -> bool;
    fn allows_window(&self) -> bool;
    /// true if upstream evaluates the aggregate in SQL for arguments of these types
    fn is_sql(&self, _arg_types: &[ColumnType]) -> bool {
        false
    }
    /// A fresh state for signature `sig_idx`, with the init arguments of the call.
    fn create(&self, sig_idx: usize, init_args: &[(String, Value)]) -> Result<Box<dyn AggState>>;
    fn to_json(&self) -> Result<Json>;
}

/// Picks the signature for a call: a single signature reports binding/type errors; with several, the first one
/// that accepts the arguments wins.
pub fn select_signature(
    f: &dyn AggregateFunction,
    args: &[ExprRef],
    kwargs: &[(String, ExprRef)],
) -> Result<(usize, Bound<ExprRef>)> {
    let sigs = f.signatures();
    if sigs.len() == 1 {
        let bound = sigs[0].bind_exprs(args, kwargs)?;
        sigs[0].validate_args(&bound, f.name())?;
        return Ok((0, bound));
    }
    for (i, sig) in sigs.iter().enumerate() {
        if let Ok(bound) = sig.bind_exprs(args, kwargs) {
            if sig.validate_args(&bound, f.name()).is_ok() {
                return Ok((i, bound));
            }
        }
    }
    crate::bail!(UnsupportedOperation, "Function {} has no matching signature for arguments", py_repr(f.name()))
}

/// Builds an aggregate call (upstream `AggregateFunction.__call__`). `kwargs` may contain `order_by` / `group_by`
/// entries; `order_by` / `group_by` given separately take precedence (callers that already extracted them).
pub fn make_agg_call(
    f: AggRef,
    mut args: Vec<ExprRef>,
    mut kwargs: Vec<(String, ExprRef)>,
    order_by: Option<ExprRef>,
    group_by: Vec<ExprRef>,
) -> Result<ExprRef> {
    let name = f.name().to_string();
    let mut order_by = order_by;
    if let Some(pos) = kwargs.iter().position(|(k, _)| k == "order_by") {
        if f.requires_order_by() {
            crate::bail!(
                UnsupportedOperation,
                "{name}(): order_by invalid, this function requires the first argument to be the ordering expression"
            );
        }
        if !f.allows_window() {
            crate::bail!(
                UnsupportedOperation,
                "{name}(): order_by invalid with an aggregate function that does not allow windows"
            );
        }
        order_by = Some(kwargs.remove(pos).1);
    } else if order_by.is_none() && f.requires_order_by() {
        if args.is_empty() {
            crate::bail!(UnsupportedOperation, "{name}(): requires an ordering expression as its first argument");
        }
        order_by = Some(args.remove(0));
    }
    let mut group_by = group_by;
    if let Some(pos) = kwargs.iter().position(|(k, _)| k == "group_by") {
        if !f.allows_window() {
            crate::bail!(
                UnsupportedOperation,
                "{name}(): group_by invalid with an aggregate function that does not allow windows"
            );
        }
        group_by.push(kwargs.remove(pos).1);
    }
    let (sig_idx, bound) = select_signature(f.as_ref(), &args, &kwargs)?;
    let sig = &f.signatures()[sig_idx];
    for p in f.init_param_names() {
        if let Some((i, _)) = sig.param(p) {
            if let Some(a) = &bound.args[i] {
                if a.as_literal().is_none() {
                    crate::bail!(InvalidConfiguration, "{name}(): init() parameter {} must be a constant value", py_repr(p));
                }
            }
        }
    }
    let ret = sig.call_return_type(sig.return_type.clone(), &bound);
    let mut fc = FunctionCall::new(FuncRef::Agg(f), args, kwargs);
    fc.order_by = order_by.into_iter().collect();
    fc.group_by = group_by;
    Ok(Expr::new(ExprKind::FunctionCall(fc), ret))
}

/// How the executor feeds rows to the states of one aggregate call.
pub struct AggPlan {
    pub func: AggRef,
    pub sig_idx: usize,
    pub init_args: Vec<(String, Value)>,
    /// for each positional argument: skip the row if the value is NULL (non-nullable parameter)
    skip_null_args: Vec<bool>,
    /// for each keyword argument: its name and Some(skip-if-NULL) for an update() argument, None for an init one
    kwarg_roles: Vec<(String, Option<bool>)>,
}

impl AggPlan {
    pub fn new(fc: &FunctionCall) -> Result<AggPlan> {
        let FuncRef::Agg(f) = &fc.func else { return Err(Error::internal("AggPlan of a non-aggregate call")) };
        let (sig_idx, _) = select_signature(f.as_ref(), &fc.args, &fc.kwargs)?;
        let sig = &f.signatures()[sig_idx];
        let init_names = f.init_param_names();
        let mut init_args = Vec::new();
        let mut kwarg_roles = Vec::new();
        for (k, e) in &fc.kwargs {
            if init_names.contains(k) {
                init_args.push((k.clone(), e.as_literal().cloned().unwrap_or(Value::Null)));
                kwarg_roles.push((k.clone(), None));
            } else {
                let nullable = sig.param(k).map(|(_, p)| p.col_type.nullable || p.kind.is_var()).unwrap_or(true);
                kwarg_roles.push((k.clone(), Some(!nullable)));
            }
        }
        // positional arguments bind to the positional parameters in order
        let mut positional = sig.params.iter().filter(|p| {
            matches!(p.kind, super::signature::ParamKind::PositionalOnly | super::signature::ParamKind::PositionalOrKeyword)
        });
        let skip_null_args = fc.args.iter().map(|_| positional.next().is_some_and(|p| !p.col_type.nullable)).collect();
        Ok(AggPlan { func: f.clone(), sig_idx, init_args, skip_null_args, kwarg_roles })
    }

    pub fn new_state(&self) -> Result<Box<dyn AggState>> {
        self.func.create(self.sig_idx, &self.init_args)
    }

    /// True if a row with these arguments is skipped (a NULL argument for a non-nullable parameter).
    pub fn skips(&self, args: &[Value], kwargs: &[Value]) -> bool {
        args.iter().zip(self.skip_null_args.iter()).any(|(v, skip)| *skip && v.is_null())
            || kwargs.iter().zip(self.kwarg_roles.iter()).any(|(v, (_, role))| *role == Some(true) && v.is_null())
    }

    /// Feeds one row (the evaluated positional and keyword arguments, in call order) to `state`; a row with a NULL
    /// argument for a non-nullable parameter is skipped.
    pub fn update(&self, state: &mut dyn AggState, args: Vec<Value>, kwargs: Vec<Value>) -> Result<()> {
        if self.skips(&args, &kwargs) {
            return Ok(());
        }
        let mut kw = Vec::new();
        for (v, (k, role)) in kwargs.into_iter().zip(self.kwarg_roles.iter()) {
            if let Some(skip) = role {
                if *skip && v.is_null() {
                    return Ok(());
                }
                kw.push((k.clone(), v));
            }
        }
        state.update(&args, &kw)
    }
}

// ------------------------------------------------------------------------------------------- builtin aggregates

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuiltinKind {
    Count,
    Sum,
    Min,
    Max,
    Mean,
    MakeList,
}

#[derive(Debug)]
pub struct BuiltinAgg {
    kind: BuiltinKind,
    name: &'static str,
    module: &'static str,
    sigs: Vec<Signature>,
    allows_window: bool,
}

impl AggregateFunction for BuiltinAgg {
    fn name(&self) -> &str {
        self.name
    }
    fn qualified_name(&self) -> String {
        format!("{}.{}", self.module, self.name)
    }
    fn signatures(&self) -> &[Signature] {
        &self.sigs
    }
    fn init_param_names(&self) -> &[String] {
        &[]
    }
    fn requires_order_by(&self) -> bool {
        false
    }
    fn allows_std_agg(&self) -> bool {
        true
    }
    fn allows_window(&self) -> bool {
        self.allows_window
    }
    fn is_sql(&self, arg_types: &[ColumnType]) -> bool {
        match self.kind {
            BuiltinKind::MakeList => false,
            // Postgres has no min/max of booleans; upstream computes those in Python
            BuiltinKind::Min | BuiltinKind::Max => !arg_types.first().is_some_and(|t| t.is_bool()),
            _ => true,
        }
    }
    fn create(&self, _sig_idx: usize, _init_args: &[(String, Value)]) -> Result<Box<dyn AggState>> {
        Ok(match self.kind {
            BuiltinKind::Count => Box::new(CountState(0)),
            BuiltinKind::Sum => Box::new(SumState(None)),
            BuiltinKind::Min => Box::new(MinMaxState { val: None, is_min: true }),
            BuiltinKind::Max => Box::new(MinMaxState { val: None, is_min: false }),
            BuiltinKind::Mean => Box::new(MeanState { int_sum: 0, float_sum: 0.0, is_float: false, count: 0 }),
            BuiltinKind::MakeList => Box::new(MakeListState(Vec::new())),
        })
    }
    fn to_json(&self) -> Result<Json> {
        Ok(json!({ "builtin_agg": self.qualified_name() }))
    }
}

struct CountState(i64);

impl AggState for CountState {
    fn update(&mut self, args: &[Value], _: &[(String, Value)]) -> Result<()> {
        if !args.first().is_none_or(Value::is_null) {
            self.0 += 1;
        }
        Ok(())
    }
    fn value(&mut self) -> Result<Value> {
        Ok(Value::Int(self.0))
    }
}

struct SumState(Option<Value>);

impl AggState for SumState {
    fn update(&mut self, args: &[Value], _: &[(String, Value)]) -> Result<()> {
        let v = match args.first().map(Value::json_scalar) {
            None | Some(Value::Null) => return Ok(()),
            Some(Value::Bool(b)) => Value::Int(b as i64),
            Some(v) => v,
        };
        self.0 = Some(match (self.0.take(), v) {
            (None, v) => v,
            (Some(Value::Int(a)), Value::Int(b)) => Value::Int(
                a.checked_add(b).ok_or_else(|| crate::pxt_err!(UnsupportedOperation, "integer out of range in sum()"))?,
            ),
            (Some(a), b) => Value::Float(a.as_f64().unwrap_or(f64::NAN) + b.as_f64().unwrap_or(f64::NAN)),
        });
        Ok(())
    }
    fn value(&mut self) -> Result<Value> {
        Ok(self.0.clone().unwrap_or(Value::Null))
    }
}

struct MinMaxState {
    val: Option<Value>,
    is_min: bool,
}

impl AggState for MinMaxState {
    fn update(&mut self, args: &[Value], _: &[(String, Value)]) -> Result<()> {
        let v = match args.first() {
            None | Some(Value::Null) => return Ok(()),
            Some(v) => v.json_scalar(),
        };
        let replace = match &self.val {
            None => true,
            Some(cur) => {
                let ord = cmp_values(&v, cur);
                if self.is_min { ord.is_lt() } else { ord.is_gt() }
            }
        };
        if replace {
            self.val = Some(v);
        }
        Ok(())
    }
    fn value(&mut self) -> Result<Value> {
        Ok(self.val.clone().unwrap_or(Value::Null))
    }
}

struct MeanState {
    int_sum: i128,
    float_sum: f64,
    is_float: bool,
    count: i64,
}

impl AggState for MeanState {
    fn update(&mut self, args: &[Value], _: &[(String, Value)]) -> Result<()> {
        match args.first().map(Value::json_scalar) {
            None | Some(Value::Null) => return Ok(()),
            Some(Value::Int(i)) => self.int_sum += i as i128,
            Some(Value::Bool(b)) => self.int_sum += b as i128,
            Some(v) => {
                self.is_float = true;
                self.float_sum += v.as_f64().unwrap_or(f64::NAN);
            }
        }
        self.count += 1;
        Ok(())
    }
    fn value(&mut self) -> Result<Value> {
        if self.count == 0 {
            return Ok(Value::Null);
        }
        let sum = self.int_sum as f64 + self.float_sum;
        Ok(Value::Float(sum / self.count as f64))
    }
}

struct MakeListState(Vec<Json>);

impl AggState for MakeListState {
    fn update(&mut self, args: &[Value], _: &[(String, Value)]) -> Result<()> {
        match args.first() {
            None | Some(Value::Null) => {}
            Some(v) => self.0.push(crate::value::value_to_json(v)?),
        }
        Ok(())
    }
    fn value(&mut self) -> Result<Value> {
        Ok(Value::Json(Json::Array(self.0.clone())))
    }
}

fn sigs(specs: &[&str]) -> Vec<Signature> {
    specs.iter().map(|s| Signature::parse(s).unwrap_or_else(|e| panic!("{e:?}"))).collect()
}

fn builtin_aggs() -> &'static Vec<AggRef> {
    static AGGS: OnceLock<Vec<AggRef>> = OnceLock::new();
    AGGS.get_or_init(|| {
        const GLOBALS: &str = "pixeltable.functions.globals";
        let ordered = ["String", "Int", "Float", "Bool", "Timestamp"];
        let minmax: Vec<String> = ordered.iter().map(|t| format!("(val: {t} | None) -> {t} | None")).collect();
        let minmax: Vec<&str> = minmax.iter().map(String::as_str).collect();
        let mk = |kind, name, module, specs: &[&str], allows_window| -> AggRef {
            Arc::new(BuiltinAgg { kind, name, module, sigs: sigs(specs), allows_window })
        };
        vec![
            mk(BuiltinKind::Sum, "sum", GLOBALS, &["(val: Int | None) -> Int | None", "(val: Float | None) -> Float | None"], true),
            mk(BuiltinKind::Count, "count", GLOBALS, &["(val: Any | None) -> Int"], true),
            mk(BuiltinKind::Min, "min", GLOBALS, &minmax, true),
            mk(BuiltinKind::Max, "max", GLOBALS, &minmax, true),
            mk(
                BuiltinKind::Mean,
                "mean",
                GLOBALS,
                &["(val: Int | None) -> Float | None", "(val: Float | None) -> Float | None"],
                false,
            ),
            mk(BuiltinKind::MakeList, "make_list", "pixeltable.functions.json", &["(obj: Json) -> Json"], false),
        ]
    })
}

pub fn lookup_builtin_agg(qualified_name: &str) -> Option<AggRef> {
    builtin_aggs().iter().find(|a| a.qualified_name() == qualified_name).cloned()
}

pub fn builtin_agg_names() -> Vec<String> {
    builtin_aggs().iter().map(|a| a.qualified_name()).collect()
}

/// Resolves a stored reference to a Python UDA (`{"uda": <descriptor>, ...}`); the functions work provides the
/// resolution (through `pixeltable.func`).
pub fn resolve_uda(j: &Json) -> Result<AggRef> {
    Err(Error::not_implemented(format!("resolving the stored Python UDA {j}")))
}
