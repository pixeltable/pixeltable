import { UnknownExpr, type Node } from '../expr.ts';

export function callNode(
  path: string,
  args: readonly unknown[],
  kwargs: Record<string, unknown> = {},
  aggregate = false,
): Node {
  return { kind: 'call', path, aggregate, args, kwargs };
}

/** Calls the Pixeltable function at a Python path, untyped: `pxt.fn('pixeltable.functions.string.format')(...)`. */
export function fn(path: string): (...args: unknown[]) => UnknownExpr {
  return (...args) => new UnknownExpr(callNode(path, args));
}
