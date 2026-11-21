/** Aggregate functions, from `pixeltable.functions.globals`, for queries with `groupBy()`. */

import { FloatExpr, IntExpr, type AnyExpr, type Expr } from '../expr.ts';
import { callNode } from './call.ts';

function aggregate(name: string, arg: AnyExpr) {
  return callNode(`pixeltable.functions.globals.${name}`, [arg], {}, true);
}

/** The number of rows in which `expr` is not null. */
export function count(expr: AnyExpr): IntExpr<false> {
  return new IntExpr(aggregate('count', expr));
}

export function sum<N extends boolean>(expr: Expr<number, N>): FloatExpr<true> {
  return new FloatExpr(aggregate('sum', expr));
}

export function mean<N extends boolean>(expr: Expr<number, N>): FloatExpr<true> {
  return new FloatExpr(aggregate('mean', expr));
}

export function min<N extends boolean>(expr: Expr<number, N>): FloatExpr<true> {
  return new FloatExpr(aggregate('min', expr));
}

export function max<N extends boolean>(expr: Expr<number, N>): FloatExpr<true> {
  return new FloatExpr(aggregate('max', expr));
}
