/** Wrappers for `pixeltable.functions.string`. */

import { StringExpr, type AnyExpr, type Operand } from '../expr.ts';
import { callNode } from './call.ts';

/** `format('{0} ({1})', t.title, t.year)`: Python's `str.format()`, evaluated per row. */
export function format(template: Operand<string>, ...args: (AnyExpr | string | number | boolean)[]): StringExpr<false> {
  return new StringExpr(callNode('pixeltable.functions.string.format', [template, ...args]));
}
