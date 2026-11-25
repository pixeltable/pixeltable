/**
 * Class-based schemas, as in Python: `const TableModel = pxt.modelBase()`, then `class Films extends
 * TableModel('films') { ... }` declares a table whose fields are its columns, in declaration order; a computed column's
 * initializer refers to earlier fields through `this`.
 *
 * `await TableModel.bindAll('pxt://org:db')` binds the base's models to their tables, as `TableModel.bind_all()` does
 * in Python. A model's table then reads and writes rows: `const t = Films.table(); await t.where(t.is_hit).collect()`.
 */

import type { BaseColumn } from './columns.ts';
import { PixeltableError } from './errors.ts';
import { Expr, type AnyExpr, type ValueOf } from './expr.ts';
import { Table, TableHandle, withColumns } from './table.ts';

const MODEL_INFO: unique symbol = Symbol.for('pixeltable.model');

export interface ModelOptions {
  readonly comment?: string;
  readonly customMetadata?: unknown;
  readonly mediaValidation?: 'on_read' | 'on_write';
}

export interface ModelInfo {
  readonly name: string;
  readonly options: ModelOptions;
}

/** A model class: one that extends the result of a model base's `TableModel(name)`. */
export type DefinedModel = (new () => object) & { readonly [MODEL_INFO]: ModelInfo };

/** The table each TableModel(name) base was bound to, by the model base's bindAll(). */
const boundHandles = new WeakMap<object, TableHandle>();

/** Each model's table, and the binding it was made from; a later bindAll() makes a new one. */
const modelTables = new WeakMap<object, { readonly handle: TableHandle; readonly table: Table<unknown, unknown> }>();

/** A bound model's table: its columns are the model's fields, typed as the class body declares them. */
export type ModelTable<M> = Table<Row<M>, Insert<M>> & Fields<M>;

/** The base of every model, with the operations of a bound model. */
export abstract class ModelBase {
  /** The table this model is bound to, as `Films.table` in Python: `const t = Films.table(); t.where(t.is_hit)`. */
  static table<M extends object>(this: new () => M): ModelTable<M> {
    const model = this as unknown as DefinedModel;
    const handle = boundHandles.get(nameBase(model));
    if (handle === undefined) {
      throw new PixeltableError(
        'NOT_BOUND',
        `${model.name} is not bound to a table; call bindAll() on its model base, with the target of ` +
          `'pxt schema update', first`,
      );
    }
    const cached = modelTables.get(model);
    if (cached?.handle === handle) {
      return cached.table as ModelTable<M>;
    }
    // the base learns which class extends it only now, when that class asks for its table
    const l = layout(model);
    const missing = l.fields.map(([field]) => field).filter((field) => handle.column(field) === undefined);
    if (missing.length > 0) {
      throw new PixeltableError(
        'SCHEMA_MISMATCH',
        `${handle.uri} has no column(s) ${missing.join(', ')} that ${model.name} defines; run 'pxt schema update'`,
      );
    }
    const table = withColumns(new Table(handle, new Map(l.fields), (e) => l.names.get(e)));
    modelTables.set(model, { handle, table });
    return table as ModelTable<M>;
  }
}

export type ModelClass = typeof ModelBase & { readonly [MODEL_INFO]: ModelInfo };

/** The base class that a model base's TableModel(name) returned for model, which holds the table name. */
function nameBase(model: DefinedModel): object {
  let cls: object | null = Object.getPrototypeOf(model) as object | null;
  while (cls !== null && !Object.prototype.hasOwnProperty.call(cls, MODEL_INFO)) {
    cls = Object.getPrototypeOf(cls) as object | null;
  }
  return cls ?? model;
}

/** What `pxt.modelBase()` returns: `TableModel(name)` for each model's base class, and `TableModel.bindAll(dir)`. */
export interface ModelBaseFactory {
  (name: string, options?: ModelOptions): ModelClass;

  /**
   * Bind every model on this base to the table it defines under dir, the target that `pxt schema update` applied it
   * to, as `TableModel.bind_all(dir)` does in Python. Raises if a table does not exist.
   */
  bindAll(dir: string): Promise<void>;
}

/**
 * A model base, as `pxt.model_base()` creates one in Python: `const TableModel = pxt.modelBase()`. Each model on it
 * extends `TableModel(name)`, and its table names must be distinct.
 */
export function modelBase(): ModelBaseFactory {
  const registered = new Map<string, ModelClass>(); // by folded table name
  const factory = (name: string, options: ModelOptions = {}): ModelClass => {
    const key = name.toLowerCase();
    if (registered.has(key)) {
      throw new PixeltableError('INVALID_SCHEMA', `two models on one model base have the table name '${name}'`);
    }
    abstract class Model extends ModelBase {
      static readonly [MODEL_INFO]: ModelInfo = { name, options };
    }
    registered.set(key, Model);
    return Model;
  };
  const bindAll = async (dir: string): Promise<void> => {
    const prefix = dir.endsWith('/') ? dir.slice(0, -1) : dir;
    // open every table before binding any, so that a missing table leaves the base as it was
    const opened = await Promise.all(
      [...registered.values()].map(async (base) => [base, await TableHandle.open(`${prefix}/${base[MODEL_INFO].name}`)] as const),
    );
    for (const [base, handle] of opened) {
      boundHandles.set(base, handle);
    }
  };
  return Object.assign(factory, { bindAll });
}

/** Whether value is a model class, rather than the base class that TableModel(name) returned. */
export function isModelClass(value: unknown): value is DefinedModel {
  return (
    typeof value === 'function' && MODEL_INFO in value && !Object.prototype.hasOwnProperty.call(value, MODEL_INFO)
  );
}

export interface Layout {
  readonly cls: DefinedModel;
  readonly info: ModelInfo;
  /** The one instance, whose fields are the columns of the model's table. */
  readonly instance: object;
  /** The model's fields, in declaration order. */
  readonly fields: readonly (readonly [string, AnyExpr])[];
  /** The field holding each expression object, so that a reference to it encodes as that column. */
  readonly names: ReadonlyMap<AnyExpr, string>;
}

const layouts = new WeakMap<object, Layout>();

/** Construct the model once, and record for each column the field that holds it. */
export function layout(cls: DefinedModel): Layout {
  const cached = layouts.get(cls);
  if (cached !== undefined) {
    return cached;
  }
  const instance = new cls() as Record<string, unknown>;
  const fields: [string, AnyExpr][] = [];
  const names = new Map<AnyExpr, string>();
  for (const key of Object.keys(instance)) {
    const value = instance[key];
    if (!(value instanceof Expr)) {
      throw new TypeError(`${cls.name}.${key}: a model field must be a column or an expression`);
    }
    const expr = value as AnyExpr;
    const other = names.get(expr);
    if (other !== undefined) {
      throw new TypeError(`${cls.name}.${key} and ${cls.name}.${other} hold the same column object`);
    }
    if (expr.node.kind === 'col') {
      const def = expr.node.def;
      if (def.model !== undefined) {
        throw new TypeError(`${cls.name}.${key} holds a column of another model`);
      }
      def.name = key;
      def.model = cls;
    }
    names.set(expr, key);
    fields.push([key, expr]);
  }
  const result: Layout = { cls, info: cls[MODEL_INFO], instance, fields, names };
  layouts.set(cls, result);
  return result;
}

type ExprKeys<M> = { [K in keyof M]: M[K] extends AnyExpr ? K : never }[keyof M];

/** The fields of model M that are columns. */
export type Fields<M> = Pick<M, ExprKeys<M>>;
type BaseKeys<M> = { [K in keyof M]: M[K] extends BaseColumn ? K : never }[keyof M];
type NullableKeys<M> = { [K in keyof M]: M[K] extends Expr<unknown, true> ? K : never }[keyof M];

/** A row of model M: every column, with null where the column is nullable. */
export type Row<M> = { [K in ExprKeys<M>]: ValueOf<M[K]> };

/** A row to insert into model M: its base columns, nullable ones optional. */
export type Insert<M> = { [K in Exclude<BaseKeys<M>, NullableKeys<M>>]: ValueOf<M[K]> } & {
  [K in Extract<BaseKeys<M>, NullableKeys<M>>]?: ValueOf<M[K]>;
};
