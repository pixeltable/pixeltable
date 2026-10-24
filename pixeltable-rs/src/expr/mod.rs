//! The expression IR.
//!
//! Every Python expression object wraps an `ExprRef` (an immutable, shareable node). Nodes are built by the
//! constructors in `build.rs`, which type the node at construction time and raise the same errors upstream raises
//! there. `display.rs` renders upstream's `str(expr)`, `serde.rs` converts to and from JSON (computed columns, view
//! predicates, `Query.as_dict()`), and `eval.rs` evaluates a node against a row.
//!
//! To add a node kind: add a variant to `ExprKind`, a constructor in `build.rs`, and the matching arms in
//! `display.rs`, `serde.rs`, `eval.rs` and `ExprKind::children()`.

pub mod build;
pub mod display;
pub mod eval;
pub mod serde;

use std::sync::Arc;
use uuid::Uuid;

use crate::catalog::table::ColId;
use crate::functions::FuncRef;
use crate::types::{ArrayIndex, ColumnType};
use crate::value::Value;

pub type ExprRef = Arc<Expr>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Mod = 4,
    FloorDiv = 5,
    Pow = 6,
}

impl ArithOp {
    pub fn symbol(self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Div => "/",
            ArithOp::Mod => "%",
            ArithOp::FloorDiv => "//",
            ArithOp::Pow => "**",
        }
    }
    pub fn from_i64(v: i64) -> Option<ArithOp> {
        Some(match v {
            0 => ArithOp::Add,
            1 => ArithOp::Sub,
            2 => ArithOp::Mul,
            3 => ArithOp::Div,
            4 => ArithOp::Mod,
            5 => ArithOp::FloorDiv,
            6 => ArithOp::Pow,
            _ => return None,
        })
    }
    pub fn from_name(name: &str) -> Option<ArithOp> {
        Some(match name {
            "add" => ArithOp::Add,
            "sub" => ArithOp::Sub,
            "mul" => ArithOp::Mul,
            "truediv" => ArithOp::Div,
            "mod" => ArithOp::Mod,
            "floordiv" => ArithOp::FloorDiv,
            "pow" => ArithOp::Pow,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt = 0,
    Le = 1,
    Eq = 2,
    Ne = 3,
    Gt = 4,
    Ge = 5,
}

impl CmpOp {
    pub fn symbol(self) -> &'static str {
        match self {
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
    pub fn reverse(self) -> CmpOp {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Ge => CmpOp::Le,
            other => other,
        }
    }
    pub fn from_i64(v: i64) -> Option<CmpOp> {
        Some(match v {
            0 => CmpOp::Lt,
            1 => CmpOp::Le,
            2 => CmpOp::Eq,
            3 => CmpOp::Ne,
            4 => CmpOp::Gt,
            5 => CmpOp::Ge,
            _ => return None,
        })
    }
    pub fn from_name(name: &str) -> Option<CmpOp> {
        Some(match name {
            "lt" => CmpOp::Lt,
            "le" => CmpOp::Le,
            "eq" => CmpOp::Eq,
            "ne" => CmpOp::Ne,
            "gt" => CmpOp::Gt,
            "ge" => CmpOp::Ge,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And = 0,
    Or = 1,
    Not = 2,
}

impl LogicalOp {
    pub fn symbol(self) -> &'static str {
        match self {
            LogicalOp::And => "&",
            LogicalOp::Or => "|",
            LogicalOp::Not => "~",
        }
    }
    pub fn from_i64(v: i64) -> Option<LogicalOp> {
        Some(match v {
            0 => LogicalOp::And,
            1 => LogicalOp::Or,
            2 => LogicalOp::Not,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringOp {
    Concat = 0,
    Repeat = 1,
}

impl StringOp {
    pub fn symbol(self) -> &'static str {
        match self {
            StringOp::Concat => "+",
            StringOp::Repeat => "*",
        }
    }
}

/// A reference to a column of a table. `tbl_version` pins the reference to a table version (snapshots / `t:3`);
/// None reads the version the query runs against.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnRef {
    pub tbl_id: Uuid,
    pub col_id: ColId,
    /// the column name when the reference was created (for display)
    pub name: String,
    pub tbl_version: Option<i64>,
}

/// One step of a JSON path (`t.c['a'][0]`, `t.c.a[1:3]`, `t.c['*']`): extension point for the JSON work.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonPathElem {
    Key(String),
    Index(i64),
    Slice(Option<i64>, Option<i64>, Option<i64>),
    Wildcard,
}

/// A property of a column (`t.c.errortype`, `t.img.localpath`, ...), upstream `ColumnPropertyRef.Property`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnProp {
    ErrorType = 0,
    ErrorMsg = 1,
    FileUrl = 2,
    LocalPath = 3,
    CellMd = 4,
}

impl ColumnProp {
    pub fn name(self) -> &'static str {
        match self {
            ColumnProp::ErrorType => "errortype",
            ColumnProp::ErrorMsg => "errormsg",
            ColumnProp::FileUrl => "fileurl",
            ColumnProp::LocalPath => "localpath",
            ColumnProp::CellMd => "cellmd",
        }
    }
    pub fn from_name(name: &str) -> Option<ColumnProp> {
        Some(match name {
            "errortype" => ColumnProp::ErrorType,
            "errormsg" => ColumnProp::ErrorMsg,
            "fileurl" => ColumnProp::FileUrl,
            "localpath" => ColumnProp::LocalPath,
            "cellmd" => ColumnProp::CellMd,
            _ => return None,
        })
    }
    pub fn from_i64(v: i64) -> Option<ColumnProp> {
        Some(match v {
            0 => ColumnProp::ErrorType,
            1 => ColumnProp::ErrorMsg,
            2 => ColumnProp::FileUrl,
            3 => ColumnProp::LocalPath,
            4 => ColumnProp::CellMd,
            _ => return None,
        })
    }
    /// True for the properties read from a cell's error (errortype, errormsg, cellmd).
    pub fn is_cellmd(self) -> bool {
        matches!(self, ColumnProp::ErrorType | ColumnProp::ErrorMsg | ColumnProp::CellMd)
    }
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub func: FuncRef,
    pub args: Vec<ExprRef>,
    pub kwargs: Vec<(String, ExprRef)>,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    ColumnRef(ColumnRef),
    Literal(Value),
    Arithmetic { op: ArithOp, lhs: ExprRef, rhs: ExprRef },
    StringOp { op: StringOp, lhs: ExprRef, rhs: ExprRef },
    Comparison { op: CmpOp, lhs: ExprRef, rhs: ExprRef },
    Compound { op: LogicalOp, operands: Vec<ExprRef> },
    IsNull(ExprRef),
    /// `lhs.isin([...])` with a constant value list (already filtered to values of lhs's type)
    InList { lhs: ExprRef, values: Vec<Value>, display: String },
    /// `lhs.isin(json_expr)`
    InExpr { lhs: ExprRef, rhs: ExprRef },
    /// `operand.astype(t)`: changes the static type only; the target type is the node's `col_type`
    TypeCast(ExprRef),
    InlineList(Vec<ExprRef>),
    InlineDict(Vec<(String, ExprRef)>),
    InlineArray(Vec<ExprRef>),
    JsonPath { base: ExprRef, path: Vec<JsonPathElem> },
    ArraySlice { base: ExprRef, index: Vec<ArrayIndex> },
    FunctionCall(FunctionCall),
    /// `col.<prop>`: `col` is a ColumnRef node
    ColumnProperty { col: ExprRef, prop: ColumnProp },
    /// a named parameter of an expression template (`@pxt.expr_udf`, `Function.using()`, `@pxt.query`), replaced by
    /// an argument when the template is instantiated (`crate::functions::udf::substitute`)
    Variable(String),
    /// `col.similarity(item)` against an embedding index of `col` (a ColumnRef node)
    Similarity(Similarity),
}

/// A similarity lookup against an embedding index (upstream `SimilarityExpr`).
#[derive(Debug, Clone)]
pub struct Similarity {
    pub col: ExprRef,
    /// the query value (a constant)
    pub item: ExprRef,
    pub idx_name: String,
    /// the table holding the index and its value column
    pub idx_tbl_id: Uuid,
    /// the index's hidden value column (the embeddings of the column's values)
    pub val_col_id: ColId,
    /// `cosine`, `ip` or `l2`
    pub metric: String,
    /// the embedding of `item` (the index's embedding function applied to it)
    pub item_embedding: ExprRef,
    /// `item_embedding`'s value, computed once per expression
    pub cache: Arc<std::sync::OnceLock<Value>>,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub col_type: ColumnType,
}

impl ExprKind {
    /// Direct sub-expressions.
    pub fn children(&self) -> Vec<&ExprRef> {
        match self {
            ExprKind::ColumnRef(_) | ExprKind::Literal(_) | ExprKind::Variable(_) => vec![],
            ExprKind::Arithmetic { lhs, rhs, .. }
            | ExprKind::StringOp { lhs, rhs, .. }
            | ExprKind::Comparison { lhs, rhs, .. }
            | ExprKind::InExpr { lhs, rhs } => vec![lhs, rhs],
            ExprKind::Compound { operands, .. } => operands.iter().collect(),
            ExprKind::IsNull(e) | ExprKind::TypeCast(e) => vec![e],
            ExprKind::ColumnProperty { col, .. } => vec![col],
            ExprKind::Similarity(sim) => vec![&sim.col, &sim.item],
            ExprKind::InList { lhs, .. } => vec![lhs],
            ExprKind::InlineList(items) | ExprKind::InlineArray(items) => items.iter().collect(),
            ExprKind::InlineDict(items) => items.iter().map(|(_, e)| e).collect(),
            ExprKind::JsonPath { base, .. } | ExprKind::ArraySlice { base, .. } => vec![base],
            ExprKind::FunctionCall(fc) => fc.args.iter().chain(fc.kwargs.iter().map(|(_, e)| e)).collect(),
        }
    }

    /// The same node with its direct sub-expressions replaced (`children` in the order of `children()`). Function
    /// calls of Python functions are rebuilt by `crate::functions::udf::substitute`, which re-binds them.
    pub fn with_children(&self, children: Vec<ExprRef>) -> ExprKind {
        let mut it = children.into_iter();
        let mut next = || it.next().expect("with_children(): too few children");
        match self {
            ExprKind::ColumnRef(_) | ExprKind::Literal(_) | ExprKind::Variable(_) => self.clone(),
            ExprKind::Arithmetic { op, .. } => ExprKind::Arithmetic { op: *op, lhs: next(), rhs: next() },
            ExprKind::StringOp { op, .. } => ExprKind::StringOp { op: *op, lhs: next(), rhs: next() },
            ExprKind::Comparison { op, .. } => ExprKind::Comparison { op: *op, lhs: next(), rhs: next() },
            ExprKind::InExpr { .. } => ExprKind::InExpr { lhs: next(), rhs: next() },
            ExprKind::Compound { op, operands } => {
                ExprKind::Compound { op: *op, operands: operands.iter().map(|_| next()).collect() }
            }
            ExprKind::IsNull(_) => ExprKind::IsNull(next()),
            ExprKind::TypeCast(_) => ExprKind::TypeCast(next()),
            ExprKind::ColumnProperty { prop, .. } => ExprKind::ColumnProperty { col: next(), prop: *prop },
            ExprKind::Similarity(sim) => ExprKind::Similarity(Similarity { col: next(), item: next(), ..sim.clone() }),
            ExprKind::InList { values, display, .. } => {
                ExprKind::InList { lhs: next(), values: values.clone(), display: display.clone() }
            }
            ExprKind::InlineList(items) => ExprKind::InlineList(items.iter().map(|_| next()).collect()),
            ExprKind::InlineArray(items) => ExprKind::InlineArray(items.iter().map(|_| next()).collect()),
            ExprKind::InlineDict(items) => ExprKind::InlineDict(items.iter().map(|(k, _)| (k.clone(), next())).collect()),
            ExprKind::JsonPath { path, .. } => ExprKind::JsonPath { base: next(), path: path.clone() },
            ExprKind::ArraySlice { index, .. } => ExprKind::ArraySlice { base: next(), index: index.clone() },
            ExprKind::FunctionCall(fc) => ExprKind::FunctionCall(FunctionCall {
                func: fc.func.clone(),
                args: fc.args.iter().map(|_| next()).collect(),
                kwargs: fc.kwargs.iter().map(|(k, _)| (k.clone(), next())).collect(),
            }),
        }
    }
}

impl Expr {
    pub fn new(kind: ExprKind, col_type: ColumnType) -> ExprRef {
        Arc::new(Expr { kind, col_type })
    }

    pub fn literal(v: Value, t: ColumnType) -> ExprRef {
        Expr::new(ExprKind::Literal(v), t)
    }

    pub fn as_literal(&self) -> Option<&Value> {
        match &self.kind {
            ExprKind::Literal(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_column_ref(&self) -> Option<&ColumnRef> {
        match &self.kind {
            ExprKind::ColumnRef(c) => Some(c),
            _ => None,
        }
    }

    /// All column references in this expression (pre-order, with duplicates).
    pub fn column_refs(&self) -> Vec<&ColumnRef> {
        let mut out = Vec::new();
        self.visit(&mut |e| {
            if let ExprKind::ColumnRef(c) = &e.kind {
                out.push(c);
            }
        });
        out
    }

    pub fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Expr)) {
        f(self);
        for c in self.kind.children() {
            c.visit(f);
        }
    }

    /// Ids of the tables this expression references.
    pub fn tbl_ids(&self) -> Vec<Uuid> {
        let mut ids: Vec<Uuid> = self.column_refs().iter().map(|c| c.tbl_id).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// The default result column name (`Expr.default_column_name()`).
    pub fn default_column_name(&self) -> Option<String> {
        match &self.kind {
            ExprKind::ColumnRef(c) => Some(c.name.clone()),
            ExprKind::Literal(_) => Some("Literal".to_string()),
            ExprKind::FunctionCall(fc) => Some(fc.func.name().to_string()),
            ExprKind::ColumnProperty { .. } => Some(crate::expr::display::display(self).replace('.', "_")),
            ExprKind::Similarity(_) => Some("similarity".to_string()),
            _ => None,
        }
    }

    /// Structural equality of two expressions (upstream `Expr.equals()`).
    pub fn equals(&self, other: &Expr) -> bool {
        crate::expr::serde::to_json(self).ok() == crate::expr::serde::to_json(other).ok()
    }
}
