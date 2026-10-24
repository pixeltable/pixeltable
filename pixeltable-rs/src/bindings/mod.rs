//! PyO3 bindings: conversion of arguments and results only; the logic lives in the other modules.
//!
//! One file per Python-facing concern (`types`, `expr`, `catalog`, `table`, `query`); `register()` adds every class
//! and function to `pixeltable._core`.

pub mod catalog;
pub mod expr;
pub mod mutate;
pub mod query;
pub mod table;
pub mod types;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<types::PyColumnType>()?;
    m.add_class::<expr::PyExpr>()?;
    m.add_class::<expr::PyMethodRef>()?;
    m.add_class::<query::PyCursor>()?;
    m.add_class::<query::PyQuerySpec>()?;
    m.add_class::<query::PyResultSet>()?;

    m.add_function(wrap_pyfunction!(expr::expr_from_object, m)?)?;
    m.add_function(wrap_pyfunction!(expr::array, m)?)?;
    m.add_function(wrap_pyfunction!(expr::call_builtin, m)?)?;
    m.add_function(wrap_pyfunction!(expr::builtin_names, m)?)?;

    m.add_function(wrap_pyfunction!(catalog::init, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::home, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::reload_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::reset_db, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::register_reserved_names, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::create_dir, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::drop_dir, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::move_, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::get_dir_contents, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::list_tables, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::list_dirs, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::get_dir_tree, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::ls_rows, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::create_table, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::create_view, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::get_table, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::drop_table, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::default_time_zone, m)?)?;

    m.add_function(wrap_pyfunction!(table::tbl_kind, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_name, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_path, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_version, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_columns, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_schema, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_column_ref, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_list_views, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_insert, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_add_columns, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_add_column, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_add_computed_column, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_drop_column, m)?)?;
    m.add_function(wrap_pyfunction!(table::tbl_rename_column, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_versions, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_update, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_batch_update, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_recompute_columns, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_delete, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_revert, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_alter_column, m)?)?;
    m.add_function(wrap_pyfunction!(mutate::tbl_alter_computed_column, m)?)?;
    Ok(())
}
