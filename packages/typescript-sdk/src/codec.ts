/**
 * Encodes models and expressions as name-based JSON nodes. Nothing in the output is a table id, column id or
 * resolved type: the Python side replays it through the Pixeltable DSL.
 */

import { columnDef } from './columns.ts';
import { Expr, type AnyExpr, type ColumnOptions, type TypeSpec } from './expr.ts';
import { layout, type DefinedModel, type Layout } from './model.ts';

export const EXPORT_VERSION = 1;

export type ColumnSpec = { readonly type: TypeSpec } | { readonly value: unknown };

export interface ModelSpec {
  readonly cls: string;
  readonly name: string;
  readonly options: Record<string, unknown>;
  readonly columns: readonly (readonly [string, ColumnSpec & Record<string, unknown>])[];
  readonly indexes: readonly unknown[];
}

export interface ModelExport {
  readonly v: number;
  readonly models: readonly ModelSpec[];
}

interface Context {
  readonly binding: Layout;
  /** The expression being defined, which encodes as itself even though a field holds it. */
  readonly root: AnyExpr;
  readonly field: string;
}

export function modelSpec(cls: DefinedModel): ModelSpec {
  const binding = layout(cls);
  const columns = binding.fields.map(([field, expr]) => [field, columnSpec(binding, field, expr)] as const);
  return { cls: cls.name, name: binding.info.name, options: defined(binding.info.options), columns, indexes: [] };
}

function columnSpec(binding: Layout, field: string, expr: AnyExpr): ColumnSpec & Record<string, unknown> {
  const node = expr.node;
  if (node.kind === 'col') {
    return { type: node.def.type, ...defined(node.def.options) };
  }
  if (node.kind === 'column') {
    return { value: encodeExpr(node.of, { binding, root: node.of, field }), ...defined(node.options) };
  }
  return { value: encodeExpr(expr, { binding, root: expr, field }) };
}

export function encodeExpr(expr: AnyExpr, ctx: Context): unknown {
  if (expr !== ctx.root) {
    const name = ctx.binding.names.get(expr);
    if (name !== undefined) {
      return { col: name };
    }
  }
  const node = expr.node;
  const child = (e: AnyExpr): unknown => encodeExpr(e, ctx);
  switch (node.kind) {
    case 'col': {
      const def = columnDef(expr);
      if (def.model !== ctx.binding.cls || def.name === undefined) {
        throw new TypeError(
          `${ctx.binding.cls.name}.${ctx.field} refers to a column that is not a field of ${ctx.binding.cls.name}`,
        );
      }
      return { col: def.name };
    }
    case 'lit':
      return { lit: literalValue(node.value), type: node.type };
    case 'op':
      return { op: node.op, args: node.args.map((a) => encodeValue(a, ctx)) };
    case 'method':
      return {
        method: node.name,
        on: child(node.on),
        args: node.args.map((a) => encodeValue(a, ctx)),
        kwargs: encodeKwargs(node.kwargs, ctx),
      };
    case 'prop':
      return { prop: node.name, on: child(node.on) };
    case 'call':
      return {
        call: node.path,
        args: node.args.map((a) => encodeValue(a, ctx)),
        kwargs: encodeKwargs(node.kwargs, ctx),
      };
    case 'cast':
      return { cast: node.type, on: child(node.on) };
    case 'get':
      return { get: node.key, on: child(node.on) };
    case 'at':
      return { at: node.index, on: child(node.on) };
    case 'slice':
      return { slice: [node.start, node.stop, node.step], on: child(node.on) };
    case 'isIn':
      return { isIn: encodeValue(node.values, ctx), on: child(node.on) };
    case 'column':
      return child(node.of);
  }
}

/** A plain value: bare where JSON represents it unambiguously, a typed literal otherwise. */
export function encodeValue(value: unknown, ctx: Context): unknown {
  if (value instanceof Expr) {
    return encodeExpr(value as AnyExpr, ctx);
  }
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new TypeError(`${ctx.binding.cls.name}.${ctx.field}: ${value} has no Pixeltable literal`);
    }
    return value;
  }
  if (typeof value === 'bigint') {
    return { lit: literalValue(value), type: 'int' };
  }
  if (value instanceof Date) {
    return { lit: literalValue(value), type: 'timestamp' };
  }
  if (Array.isArray(value)) {
    return value.map((v) => encodeValue(v, ctx));
  }
  if (typeof value === 'object') {
    // an object is always wrapped, so that it cannot read as a node
    return { lit: value, type: 'json' };
  }
  throw new TypeError(`${ctx.binding.cls.name}.${ctx.field}: cannot encode a ${typeof value}`);
}

function literalValue(value: unknown): unknown {
  if (value instanceof Date) {
    return { $pxt: 'timestamp', v: value.toISOString() };
  }
  if (typeof value === 'bigint') {
    return { $pxt: 'int', v: value.toString() };
  }
  return value;
}

function encodeKwargs(kwargs: Readonly<Record<string, unknown>>, ctx: Context): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(kwargs)) {
    if (value !== undefined) {
      out[key] = encodeValue(value, ctx);
    }
  }
  return out;
}

function defined(options: ColumnOptions | Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(options).filter(([, v]) => v !== undefined));
}
