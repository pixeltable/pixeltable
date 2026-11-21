/**
 * Expressions over a model's columns. Each one records what it applies, by name, and nothing more: the Python side
 * replays it through the Pixeltable DSL, which resolves functions and checks types. The static types here mirror
 * Python's result types, so that row types come out right.
 */

export type TypeName = 'string' | 'int' | 'float' | 'bool' | 'timestamp' | 'date' | 'uuid' | 'json';

export interface TypeSpec {
  readonly type: TypeName;
  readonly nullable: boolean;
}

export interface ColumnOptions {
  readonly primaryKey?: boolean;
  readonly comment?: string;
  readonly stored?: boolean;
  readonly mediaValidation?: 'on_read' | 'on_write';
  readonly customMetadata?: unknown;
  readonly destination?: string;
}

/** A base column's type and options. Laying out its model records the field it is assigned to. */
export class ColumnDef {
  readonly type: TypeSpec;
  readonly options: ColumnOptions;
  name: string | undefined;
  model: object | undefined;

  constructor(type: TypeSpec, options: ColumnOptions = {}) {
    this.type = type;
    this.options = options;
    this.name = undefined;
    this.model = undefined;
  }

  with(type: Partial<TypeSpec>, options: ColumnOptions = {}): ColumnDef {
    return new ColumnDef({ ...this.type, ...type }, { ...this.options, ...options });
  }
}

export type OpName =
  | 'eq'
  | 'ne'
  | 'lt'
  | 'lte'
  | 'gt'
  | 'gte'
  | 'and'
  | 'or'
  | 'not'
  | 'add'
  | 'sub'
  | 'mul'
  | 'div'
  | 'mod'
  | 'floorDiv'
  | 'pow'
  | 'neg'
  | 'abs'
  | 'round'
  | 'concat'
  | 'isNull'
  | 'isNotNull';

export type Node =
  | { readonly kind: 'col'; readonly def: ColumnDef }
  | { readonly kind: 'lit'; readonly value: unknown; readonly type: TypeName }
  | { readonly kind: 'op'; readonly op: OpName; readonly args: readonly unknown[] }
  | {
      readonly kind: 'method';
      readonly name: string;
      /** The Python function the method calls, such as `pixeltable.functions.string.upper`. */
      readonly path: string;
      readonly on: AnyExpr;
      readonly args: readonly unknown[];
      readonly kwargs: Readonly<Record<string, unknown>>;
    }
  | { readonly kind: 'prop'; readonly name: string; readonly path: string; readonly on: AnyExpr }
  | {
      readonly kind: 'call';
      readonly path: string;
      readonly aggregate: boolean;
      readonly args: readonly unknown[];
      readonly kwargs: Readonly<Record<string, unknown>>;
    }
  | { readonly kind: 'cast'; readonly type: TypeSpec; readonly on: AnyExpr }
  | { readonly kind: 'get'; readonly key: string; readonly on: AnyExpr }
  | { readonly kind: 'at'; readonly index: number; readonly on: AnyExpr }
  | {
      readonly kind: 'slice';
      readonly start: number | null;
      readonly stop: number | null;
      readonly step: number | null;
      readonly on: AnyExpr;
    }
  | { readonly kind: 'isIn'; readonly values: unknown; readonly on: AnyExpr }
  | { readonly kind: 'column'; readonly of: AnyExpr; readonly options: ColumnOptions };

export type AnyExpr = Expr<unknown, boolean>;

/** An expression of value type T, or a plain value that becomes a literal. */
export type Operand<T> = Expr<T, boolean> | T;

/** Whether an operand can be null: an expression's own flag, true for a null literal, false for any other value. */
export type NullOf<O> = O extends Expr<unknown, infer M> ? M : O extends null | undefined ? true : false;

export type Or<A extends boolean, B extends boolean> = true extends A ? true : B;

/** The value a row holds for an expression. */
export type ValueOf<E> = E extends Expr<infer T, infer N> ? (true extends N ? T | null : T) : never;

export abstract class Expr<T, N extends boolean> {
  declare readonly __value: T;
  declare readonly __nullable: N;
  declare readonly __kind: string;
  readonly node: Node;

  constructor(node: Node) {
    this.node = node;
  }

  eq(other: Operand<T> | null): BoolExpr<false> {
    return new BoolExpr(op('eq', this, other));
  }

  ne(other: Operand<T> | null): BoolExpr<false> {
    return new BoolExpr(op('ne', this, other));
  }

  lt(other: Operand<T>): BoolExpr<false> {
    return new BoolExpr(op('lt', this, other));
  }

  lte(other: Operand<T>): BoolExpr<false> {
    return new BoolExpr(op('lte', this, other));
  }

  gt(other: Operand<T>): BoolExpr<false> {
    return new BoolExpr(op('gt', this, other));
  }

  gte(other: Operand<T>): BoolExpr<false> {
    return new BoolExpr(op('gte', this, other));
  }

  isNull(): BoolExpr<false> {
    return new BoolExpr(op('isNull', this));
  }

  isNotNull(): BoolExpr<false> {
    return new BoolExpr(op('isNotNull', this));
  }

  isIn(values: readonly T[] | AnyExpr): BoolExpr<false> {
    return new BoolExpr({ kind: 'isIn', values, on: this });
  }

  /** Converts to the type of a column factory's result: `t.year.cast(pxt.string())`. */
  cast<C extends AnyExpr>(target: C): Retyped<C> {
    if (target.node.kind !== 'col') {
      throw new TypeError('cast() takes a column type, such as pxt.float() or pxt.float().orNull()');
    }
    const type = target.node.def.type;
    return newExpr(type.type, { kind: 'cast', type, on: this }) as Retyped<C>;
  }
}

export class BoolExpr<N extends boolean> extends Expr<boolean, N> {
  declare readonly __kind: 'bool';

  and<O extends Operand<boolean>>(other: O): BoolExpr<Or<N, NullOf<O>>> {
    return new BoolExpr(op('and', this, other));
  }

  or<O extends Operand<boolean>>(other: O): BoolExpr<Or<N, NullOf<O>>> {
    return new BoolExpr(op('or', this, other));
  }

  not(): BoolExpr<N> {
    return new BoolExpr(op('not', this));
  }
}

export class StringExpr<N extends boolean> extends Expr<string, N> {
  declare readonly __kind: 'string';

  concat<O extends Operand<string>>(other: O): StringExpr<Or<N, NullOf<O>>> {
    return new StringExpr(op('concat', this, other));
  }

  upper(): StringExpr<N> {
    return new StringExpr(method(this, 'string', 'upper'));
  }

  lower(): StringExpr<N> {
    return new StringExpr(method(this, 'string', 'lower'));
  }

  capitalize(): StringExpr<N> {
    return new StringExpr(method(this, 'string', 'capitalize'));
  }

  strip(chars?: Operand<string>): StringExpr<N> {
    return new StringExpr(method(this, 'string', 'strip', chars === undefined ? [] : [chars]));
  }

  len(): IntExpr<N> {
    return new IntExpr(method(this, 'string', 'len'));
  }

  contains(substr: Operand<string>, options: { case?: boolean } = {}): BoolExpr<N> {
    return new BoolExpr(method(this, 'string', 'contains', [substr], options));
  }

  startswith(substr: Operand<string>): BoolExpr<N> {
    return new BoolExpr(method(this, 'string', 'startswith', [substr]));
  }

  endswith(substr: Operand<string>): BoolExpr<N> {
    return new BoolExpr(method(this, 'string', 'endswith', [substr]));
  }

  replace(substr: Operand<string>, repl: Operand<string>, options: { n?: number } = {}): StringExpr<N> {
    return new StringExpr(method(this, 'string', 'replace', [substr, repl], options));
  }
}

type NumOperand = Expr<number, boolean> | number;

/** Whether an operand is an Int: an Int expression, or a number literal written without a fraction. */
type IntOperand<O> = O extends { readonly __kind: 'int' }
  ? true
  : O extends AnyExpr
    ? false
    : O extends number
      ? number extends O
        ? false
        : `${O}` extends `${bigint}`
          ? true
          : false
      : false;

type NonNegIntLiteral<O> = O extends number
  ? number extends O
    ? false
    : `${O}` extends `-${string}`
      ? false
      : `${O}` extends `${bigint}`
        ? true
        : false
  : false;

type Num<IsInt extends boolean, M extends boolean> = IsInt extends true ? IntExpr<M> : FloatExpr<M>;

export class IntExpr<N extends boolean> extends Expr<number, N> {
  declare readonly __kind: 'int';

  add<O extends NumOperand>(other: O): Num<IntOperand<O>, Or<N, NullOf<O>>> {
    return arith(this, 'add', other, isInt(other));
  }

  sub<O extends NumOperand>(other: O): Num<IntOperand<O>, Or<N, NullOf<O>>> {
    return arith(this, 'sub', other, isInt(other));
  }

  mul<O extends NumOperand>(other: O): Num<IntOperand<O>, Or<N, NullOf<O>>> {
    return arith(this, 'mul', other, isInt(other));
  }

  mod<O extends NumOperand>(other: O): Num<IntOperand<O>, Or<N, NullOf<O>>> {
    return arith(this, 'mod', other, isInt(other));
  }

  floorDiv<O extends NumOperand>(other: O): Num<IntOperand<O>, Or<N, NullOf<O>>> {
    return arith(this, 'floorDiv', other, isInt(other));
  }

  /** An Int only for a non-negative integer literal exponent, as in Python. */
  pow<O extends NumOperand>(other: O): Num<NonNegIntLiteral<O>, Or<N, NullOf<O>>> {
    return arith(this, 'pow', other, typeof other === 'number' && Number.isInteger(other) && other >= 0);
  }

  div<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'div', other, false);
  }

  neg(): IntExpr<N> {
    return new IntExpr(op('neg', this));
  }

  abs(): FloatExpr<N> {
    return new FloatExpr(op('abs', this));
  }

  round(digits?: number): FloatExpr<N> {
    return new FloatExpr(op('round', this, ...(digits === undefined ? [] : [digits])));
  }
}

export class FloatExpr<N extends boolean> extends Expr<number, N> {
  declare readonly __kind: 'float';

  add<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'add', other, false);
  }

  sub<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'sub', other, false);
  }

  mul<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'mul', other, false);
  }

  mod<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'mod', other, false);
  }

  floorDiv<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'floorDiv', other, false);
  }

  pow<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'pow', other, false);
  }

  div<O extends NumOperand>(other: O): FloatExpr<Or<N, NullOf<O>>> {
    return arith(this, 'div', other, false);
  }

  neg(): FloatExpr<N> {
    return new FloatExpr(op('neg', this));
  }

  abs(): FloatExpr<N> {
    return new FloatExpr(op('abs', this));
  }

  round(digits?: number): FloatExpr<N> {
    return new FloatExpr(op('round', this, ...(digits === undefined ? [] : [digits])));
  }

  floor(): FloatExpr<N> {
    return new FloatExpr(method(this, 'math', 'floor'));
  }

  ceil(): FloatExpr<N> {
    return new FloatExpr(method(this, 'math', 'ceil'));
  }
}

export class TimestampExpr<N extends boolean> extends Expr<Date, N> {
  declare readonly __kind: 'timestamp';

  get year(): IntExpr<N> {
    return new IntExpr(prop(this, 'timestamp', 'year'));
  }

  get month(): IntExpr<N> {
    return new IntExpr(prop(this, 'timestamp', 'month'));
  }

  get day(): IntExpr<N> {
    return new IntExpr(prop(this, 'timestamp', 'day'));
  }

  get hour(): IntExpr<N> {
    return new IntExpr(prop(this, 'timestamp', 'hour'));
  }

  get minute(): IntExpr<N> {
    return new IntExpr(prop(this, 'timestamp', 'minute'));
  }

  get second(): IntExpr<N> {
    return new IntExpr(prop(this, 'timestamp', 'second'));
  }

  weekday(): IntExpr<N> {
    return new IntExpr(method(this, 'timestamp', 'weekday'));
  }

  strftime(format: Operand<string>): StringExpr<N> {
    return new StringExpr(method(this, 'timestamp', 'strftime', [format]));
  }
}

/** A date; a row holds it as a `YYYY-MM-DD` string. */
export class DateExpr<N extends boolean> extends Expr<string, N> {
  declare readonly __kind: 'date';

  get year(): IntExpr<N> {
    return new IntExpr(prop(this, 'date', 'year'));
  }

  get month(): IntExpr<N> {
    return new IntExpr(prop(this, 'date', 'month'));
  }

  get day(): IntExpr<N> {
    return new IntExpr(prop(this, 'date', 'day'));
  }
}

export class UuidExpr<N extends boolean> extends Expr<string, N> {
  declare readonly __kind: 'uuid';
}

export class JsonExpr<T, N extends boolean> extends Expr<T, N> {
  declare readonly __kind: 'json';

  get(key: string): JsonExpr<unknown, true> {
    return new JsonExpr({ kind: 'get', key, on: this });
  }

  at(index: number): JsonExpr<unknown, true> {
    return new JsonExpr({ kind: 'at', index, on: this });
  }

  slice(start?: number, stop?: number, step?: number): JsonExpr<unknown[], true> {
    return new JsonExpr({ kind: 'slice', start: start ?? null, stop: stop ?? null, step: step ?? null, on: this });
  }
}

/** The result of an untyped call; `cast()` gives it a type. */
export class UnknownExpr extends Expr<unknown, boolean> {
  declare readonly __kind: 'unknown';
}

export interface ExprKinds<N extends boolean> {
  string: StringExpr<N>;
  int: IntExpr<N>;
  float: FloatExpr<N>;
  bool: BoolExpr<N>;
  timestamp: TimestampExpr<N>;
  date: DateExpr<N>;
  uuid: UuidExpr<N>;
  json: JsonExpr<unknown, N>;
}

/** The expression type for the column type of C, with C's nullability. */
export type Retyped<C> = C extends { readonly __kind: infer K; readonly __nullable: infer M extends boolean }
  ? K extends keyof ExprKinds<M>
    ? ExprKinds<M>[K]
    : never
  : never;

const EXPR_CLASSES: Record<TypeName, new (node: Node) => AnyExpr> = {
  string: StringExpr,
  int: IntExpr,
  float: FloatExpr,
  bool: BoolExpr,
  timestamp: TimestampExpr,
  date: DateExpr,
  uuid: UuidExpr,
  json: JsonExpr,
};

export function newExpr(type: TypeName, node: Node): AnyExpr {
  return new EXPR_CLASSES[type](node);
}

/** A literal of the given column type: `pxt.lit(1, 'float')`. */
export function lit<K extends TypeName>(value: unknown, type: K): ExprKinds<false>[K] {
  return newExpr(type, { kind: 'lit', value, type }) as ExprKinds<false>[K];
}

function op(name: OpName, ...args: unknown[]): Node {
  return { kind: 'op', op: name, args };
}

function method(
  on: AnyExpr,
  module: string,
  name: string,
  args: readonly unknown[] = [],
  kwargs: Record<string, unknown> = {},
): Node {
  return { kind: 'method', name, path: `pixeltable.functions.${module}.${name}`, on, args, kwargs };
}

function prop(on: AnyExpr, module: string, name: string): Node {
  return { kind: 'prop', name, path: `pixeltable.functions.${module}.${name}`, on };
}

function isInt(operand: NumOperand): boolean {
  return operand instanceof IntExpr || (typeof operand === 'number' && Number.isInteger(operand));
}

function arith<R>(self: AnyExpr, name: OpName, other: NumOperand, intResult: boolean): R {
  const node = op(name, self, other);
  return (intResult ? new IntExpr(node) : new FloatExpr(node)) as unknown as R;
}
