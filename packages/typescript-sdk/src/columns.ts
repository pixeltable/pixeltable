/**
 * Column factories. A base column is an expression that reads the column, plus the options that define it; each
 * option returns a new column, so a field holds the result of the whole chain.
 */

import {
  BoolExpr,
  ColumnDef,
  DateExpr,
  FloatExpr,
  IntExpr,
  JsonExpr,
  StringExpr,
  TimestampExpr,
  UuidExpr,
  type AnyExpr,
  type ColumnOptions,
  type Node,
  type TypeName,
} from './expr.ts';

/** Marks the expressions that define a base column, as opposed to a computed one. */
export interface BaseColumn {
  readonly __column: true;
}

function colNode(def: ColumnDef): Node {
  return { kind: 'col', def };
}

export function columnDef(expr: AnyExpr): ColumnDef {
  if (expr.node.kind !== 'col') {
    throw new TypeError('not a base column');
  }
  return expr.node.def;
}

function rebuilt<C extends AnyExpr>(col: C, type: { nullable?: boolean }, options: ColumnOptions = {}): C {
  const ctor = col.constructor as new (node: Node) => C;
  return new ctor(colNode(columnDef(col).with(type, options)));
}

export class StringColumn<N extends boolean = false> extends StringExpr<N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): StringColumn<true> {
    return rebuilt(this, { nullable: true }) as unknown as StringColumn<true>;
  }

  primaryKey(): this {
    return rebuilt(this, {}, { primaryKey: true });
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

export class IntColumn<N extends boolean = false> extends IntExpr<N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): IntColumn<true> {
    return rebuilt(this, { nullable: true }) as unknown as IntColumn<true>;
  }

  primaryKey(): this {
    return rebuilt(this, {}, { primaryKey: true });
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

export class FloatColumn<N extends boolean = false> extends FloatExpr<N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): FloatColumn<true> {
    return rebuilt(this, { nullable: true }) as unknown as FloatColumn<true>;
  }

  primaryKey(): this {
    return rebuilt(this, {}, { primaryKey: true });
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

export class BoolColumn<N extends boolean = false> extends BoolExpr<N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): BoolColumn<true> {
    return rebuilt(this, { nullable: true }) as unknown as BoolColumn<true>;
  }

  primaryKey(): this {
    return rebuilt(this, {}, { primaryKey: true });
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

export class TimestampColumn<N extends boolean = false> extends TimestampExpr<N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): TimestampColumn<true> {
    return rebuilt(this, { nullable: true }) as unknown as TimestampColumn<true>;
  }

  primaryKey(): this {
    return rebuilt(this, {}, { primaryKey: true });
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

export class DateColumn<N extends boolean = false> extends DateExpr<N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): DateColumn<true> {
    return rebuilt(this, { nullable: true }) as unknown as DateColumn<true>;
  }

  primaryKey(): this {
    return rebuilt(this, {}, { primaryKey: true });
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

export class UuidColumn<N extends boolean = false> extends UuidExpr<N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): UuidColumn<true> {
    return rebuilt(this, { nullable: true }) as unknown as UuidColumn<true>;
  }

  primaryKey(): this {
    return rebuilt(this, {}, { primaryKey: true });
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

export class JsonColumn<T = unknown, N extends boolean = false> extends JsonExpr<T, N> implements BaseColumn {
  declare readonly __column: true;

  orNull(): JsonColumn<T, true> {
    return rebuilt(this, { nullable: true }) as unknown as JsonColumn<T, true>;
  }

  comment(text: string): this {
    return rebuilt(this, {}, { comment: text });
  }
}

function base<C>(ctor: new (node: Node) => C, type: TypeName): C {
  return new ctor(colNode(new ColumnDef({ type, nullable: false })));
}

export function string(): StringColumn {
  return base<StringColumn<false>>(StringColumn, 'string');
}

export function int(): IntColumn {
  return base<IntColumn<false>>(IntColumn, 'int');
}

export function float(): FloatColumn {
  return base<FloatColumn<false>>(FloatColumn, 'float');
}

export function bool(): BoolColumn {
  return base<BoolColumn<false>>(BoolColumn, 'bool');
}

export function timestamp(): TimestampColumn {
  return base<TimestampColumn<false>>(TimestampColumn, 'timestamp');
}

export function date(): DateColumn {
  return base<DateColumn<false>>(DateColumn, 'date');
}

export function uuid(): UuidColumn {
  return base<UuidColumn<false>>(UuidColumn, 'uuid');
}

export function json<T = unknown>(): JsonColumn<T> {
  return base<JsonColumn<T, false>>(JsonColumn, 'json');
}

/** A computed column with options, the counterpart of Python's `pxt.Column(value=..., stored=...)`. */
export function column<E extends AnyExpr>(expr: E, options: Omit<ColumnOptions, 'mediaValidation'>): E {
  const ctor = expr.constructor as new (node: Node) => E;
  return new ctor({ kind: 'column', of: expr, options });
}
