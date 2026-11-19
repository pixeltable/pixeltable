/**
 * Tables and queries over /rpc. A table handle holds the table's metadata, as the Python client does: column ids for
 * the expressions it sends, and the table version that a write is checked against.
 */

import { PixeltableError } from './errors.ts';
import {
  ColumnDef,
  Expr,
  newExpr,
  UnknownExpr,
  type AnyExpr,
  type BoolExpr,
  type TypeName,
  type TypeSpec,
  type ValueOf,
} from './expr.ts';
import { encode, literal, type PyDict, type Scope } from './pyexpr.ts';
import { call, decodeValue, encodeValue } from './rpc.ts';
import { parseUri, transportFor, type PxtUri, type Transport } from './transport.ts';

interface ColumnMd {
  readonly name: string;
  readonly tblId: string;
  readonly colId: number;
  readonly type: TypeSpec | undefined;
  readonly isComputed: boolean;
}

interface TableMd {
  /** The table and its bases, as (id, version) pairs, the table first. */
  readonly chain: readonly { readonly id: string; readonly version: number }[];
  readonly columns: ReadonlyMap<string, ColumnMd>;
}

const TYPE_NAMES: Record<string, TypeName> = {
  StringType: 'string',
  IntType: 'int',
  FloatType: 'float',
  BoolType: 'bool',
  TimestampType: 'timestamp',
  DateType: 'date',
  UUIDType: 'uuid',
  JsonType: 'json',
};

function parseMd(mdList: unknown[]): TableMd {
  const chain: { id: string; version: number }[] = [];
  const columns = new Map<string, ColumnMd>();
  for (const tagged of mdList) {
    const md = (tagged as { v: Record<string, Record<string, unknown>> }).v;
    const tblMd = md['tbl_md']!;
    const tblId = String(tblMd['tbl_id']);
    chain.push({ id: tblId, version: Number(md['version_md']!['version']) });
    const columnMd = tblMd['column_md'] as Record<string, { schema_version_drop: number | null }>;
    const schemaCols = md['schema_version_md']!['columns'] as Record<
      string,
      { name: string | null; col_type: { _classname: string; nullable: boolean }; value_expr: unknown }
    >;
    for (const [id, col] of Object.entries(schemaCols)) {
      if (col.name === null || columnMd[id]?.schema_version_drop != null || columns.has(col.name)) {
        continue;
      }
      const typeName = TYPE_NAMES[col.col_type._classname];
      columns.set(col.name, {
        name: col.name,
        tblId,
        colId: Number(id),
        type: typeName === undefined ? undefined : { type: typeName, nullable: col.col_type.nullable },
        isComputed: col.value_expr !== null,
      });
    }
  }
  return { chain, columns };
}

/** The result of a write: how many rows it touched, and how many cells failed to compute. */
export interface UpdateStatus {
  readonly insertedRows: number;
  readonly updatedRows: number;
  readonly deletedRows: number;
  readonly numExcs: number;
}

function updateStatus(result: unknown): UpdateStatus {
  const stats = (result as { v: { row_count_stats: Record<string, number> } }).v.row_count_stats;
  return {
    insertedRows: stats['inserted_rows'] ?? 0,
    updatedRows: stats['updated_rows'] ?? 0,
    deletedRows: stats['deleted_rows'] ?? 0,
    numExcs: stats['num_excs'] ?? 0,
  };
}

/** One table in one database, and the metadata the client last read for it. */
export class TableHandle {
  readonly uri: string;
  readonly #transport: Transport;
  #md: TableMd;

  private constructor(uri: string, transport: Transport, md: TableMd) {
    this.uri = uri;
    this.#transport = transport;
    this.#md = md;
  }

  static async open(uri: string): Promise<TableHandle> {
    const parsed: PxtUri = parseUri(uri);
    if (parsed.path.length === 0) {
      throw new PixeltableError('INVALID_ARGUMENT', `${uri} names a database, not a table`);
    }
    const transport = transportFor(parsed);
    const response = await call(transport, {
      class_name: 'CatalogBase',
      method: 'get_table',
      args: {
        path: { $pxt: 'Path', v: { components: parsed.path, version: null } },
        if_not_exists: { $pxt: 'IfNotExistsParam', v: 'ERROR' },
      },
    });
    return new TableHandle(uri, transport, parseMd(response.result as unknown[]));
  }

  columnNames(): string[] {
    return [...this.#md.columns.keys()];
  }

  column(name: string): ColumnMd | undefined {
    return this.#md.columns.get(name.toLowerCase());
  }

  columnRef(name: string): PyDict {
    const col = this.column(name);
    if (col === undefined) {
      throw new PixeltableError('NOT_FOUND', `${this.uri} has no column '${name}'`);
    }
    return {
      _classname: 'ColumnRef',
      tbl_id: this.#md.chain[0]!.id,
      effective_version: null,
      col_tbl_id: col.tblId,
      col_tbl_effective_version: null,
      col_id: col.colId,
      perform_validation: false,
    };
  }

  /** TablePathKey.as_dict(): the live table, or the versions this client last read. */
  pathKey(snapshot: boolean): PyDict | null {
    let key: PyDict | null = null;
    for (const entry of [...this.#md.chain].reverse()) {
      key = { tbl_version: { id: entry.id, effective_version: snapshot ? entry.version : null }, base: key };
    }
    return key;
  }

  /** A write, retried against refreshed metadata when the daemon reports that this client's is behind. */
  async mutate(method: string, args: Record<string, unknown>): Promise<unknown> {
    for (;;) {
      const response = await call(this.#transport, {
        class_name: 'Table',
        method,
        args,
        path_key: this.pathKey(false),
        snapshot_path_key: this.pathKey(true),
      });
      if (response.current_md != null) {
        this.#md = parseMd(response.current_md);
      }
      if (response.is_stale_md !== true) {
        return response.result;
      }
    }
  }

  async insert(rows: readonly object[]): Promise<UpdateStatus> {
    const result = await this.mutate('insert', {
      rows: rows.map(encodeValue),
      on_error: 'abort',
      if_exists: 'error',
      print_stats: false,
      return_rows: false,
    });
    return updateStatus(result);
  }

  async delete(where: PyDict | null): Promise<UpdateStatus> {
    return updateStatus(await this.mutate('delete', { where: where === null ? null : { $pxt: 'Expr', v: where } }));
  }

  async query(method: 'collect' | 'count', query: PyDict): Promise<unknown> {
    const response = await call(this.#transport, {
      class_name: 'Query',
      method,
      args: method === 'collect' ? { query, args: null } : { query },
    });
    return response.result;
  }
}

/** Which column of a table each of its column expressions reads. */
export type ColumnNames = (expr: AnyExpr) => string | undefined;

type Selected<S> = { [K in keyof S]: ValueOf<S[K]> };

interface Clauses {
  readonly where?: AnyExpr;
  readonly select?: readonly (readonly [AnyExpr, string | null])[];
  readonly groupBy?: readonly AnyExpr[];
  readonly orderBy?: readonly (readonly [AnyExpr, boolean])[];
  readonly limit?: number;
}

/** A query over one table, built like a Python Query and run on the daemon. */
export class Query<R> {
  readonly #table: TableHandle;
  readonly #columnNames: ColumnNames;
  readonly #clauses: Clauses;

  constructor(table: TableHandle, columnNames: ColumnNames, clauses: Clauses = {}) {
    this.#table = table;
    this.#columnNames = columnNames;
    this.#clauses = clauses;
  }

  #with<R2 = R>(clauses: Clauses): Query<R2> {
    return new Query<R2>(this.#table, this.#columnNames, { ...this.#clauses, ...clauses });
  }

  where(predicate: Expr<boolean, boolean>): Query<R> {
    const prior = this.#clauses.where as BoolExpr<boolean> | undefined;
    return this.#with({ where: prior === undefined ? predicate : prior.and(predicate) });
  }

  /** The named outputs of `select({ name: expr, ... })` are typed; those of `select(expr, ...)` are not. */
  select<S extends Record<string, AnyExpr>>(columns: S): Query<Selected<S>>;
  select(...exprs: AnyExpr[]): Query<Record<string, unknown>>;
  select(...args: unknown[]): Query<unknown> {
    const named = args.length === 1 && !(args[0] instanceof Expr);
    const select: (readonly [AnyExpr, string | null])[] = named
      ? Object.entries(args[0] as Record<string, AnyExpr>).map(([name, e]) => [e, name] as const)
      : (args as AnyExpr[]).map((e) => [e, null] as const);
    return this.#with({ select });
  }

  groupBy(...exprs: AnyExpr[]): Query<R> {
    return this.#with({ groupBy: exprs });
  }

  orderBy(expr: AnyExpr, options: { asc?: boolean } = {}): Query<R> {
    return this.#with({ orderBy: [...(this.#clauses.orderBy ?? []), [expr, options.asc ?? true] as const] });
  }

  limit(n: number): Query<R> {
    return this.#with({ limit: n });
  }

  async collect(): Promise<R[]> {
    const result = (await this.#table.query('collect', this.#dict())) as {
      schema: Record<string, unknown>;
      rows: unknown[][];
    };
    const names = Object.keys(result.schema);
    return result.rows.map((row) => Object.fromEntries(names.map((name, i) => [name, decodeValue(row[i])])) as R);
  }

  async count(): Promise<number> {
    return (await this.#table.query('count', this.#dict())) as number;
  }

  /** Python's Query.as_dict(). */
  #dict(): PyDict {
    const enc = (e: AnyExpr): PyDict => encodeFor(this.#table, this.#columnNames, e);
    const c = this.#clauses;
    return {
      _classname: 'Query',
      from_clause: { tbls: [this.#table.pathKey(false)], join_clauses: [] },
      select_list: c.select?.map(([e, name]) => ({ $pxt: 'tuple', v: [enc(e), name] })) ?? null,
      where_clause: c.where === undefined ? null : enc(c.where),
      group_by_clause: c.groupBy?.map(enc) ?? null,
      grouping_tbl: null,
      order_by_clause: c.orderBy?.map(([e, asc]) => ({ $pxt: 'tuple', v: [enc(e), asc] })) ?? null,
      limit_val: c.limit === undefined ? null : literal(c.limit, 'int'),
      offset_val: null,
      sample_clause: null,
    };
  }
}

function encodeFor(table: TableHandle, columnNames: ColumnNames, expr: AnyExpr): PyDict {
  const scope: Scope = { columnName: columnNames, columnRef: (name) => table.columnRef(name) };
  return encode(expr, scope);
}


/**
 * A column of a table opened with getTable(), whose type is known only at run time. It offers every expression
 * method; one that does not apply to the column's type fails when called, as an attribute error does in Python.
 */
export interface Column extends Expr<any, boolean> {
  and(other: unknown): BoolExpr<boolean>;
  or(other: unknown): BoolExpr<boolean>;
  not(): BoolExpr<boolean>;
  concat(other: unknown): Column;
  upper(): Column;
  lower(): Column;
  capitalize(): Column;
  strip(chars?: unknown): Column;
  len(): Column;
  contains(substr: unknown, options?: { case?: boolean }): BoolExpr<boolean>;
  startswith(substr: unknown): BoolExpr<boolean>;
  endswith(substr: unknown): BoolExpr<boolean>;
  replace(substr: unknown, repl: unknown, options?: { n?: number }): Column;
  add(other: unknown): Column;
  sub(other: unknown): Column;
  mul(other: unknown): Column;
  div(other: unknown): Column;
  mod(other: unknown): Column;
  floorDiv(other: unknown): Column;
  pow(other: unknown): Column;
  neg(): Column;
  abs(): Column;
  round(digits?: number): Column;
  floor(): Column;
  ceil(): Column;
  readonly year: Column;
  readonly month: Column;
  readonly day: Column;
  readonly hour: Column;
  readonly minute: Column;
  readonly second: Column;
  weekday(): Column;
  strftime(format: unknown): Column;
  get(key: string): Column;
  at(index: number): Column;
  slice(start?: number, stop?: number, step?: number): Column;
}

export type Columns = { readonly [name: string]: Column };

/**
 * A table: the one `pxt.getTable()` opens, with columns of run-time types, or the one a bound model's `table()`
 * returns, with the model's own column types. Its columns are also its properties (see TableOf).
 */
export class Table<R = Record<string, unknown>, I = Record<string, unknown>> {
  readonly #handle: TableHandle;
  readonly #columns: ReadonlyMap<string, AnyExpr>;
  readonly #columnNames: ColumnNames;

  constructor(handle: TableHandle, columns: ReadonlyMap<string, AnyExpr>, columnNames: ColumnNames) {
    this.#handle = handle;
    this.#columns = columns;
    this.#columnNames = columnNames;
  }

  get uri(): string {
    return this.#handle.uri;
  }

  columns(): string[] {
    return this.#handle.columnNames();
  }

  /** The column of that name, for one whose name a method of Table takes, as `t['name']` in Python. */
  col(name: string): Column {
    const expr = this.#columns.get(name) ?? this.#columns.get(name.toLowerCase());
    if (expr === undefined) {
      throw new PixeltableError('NOT_FOUND', `${this.uri} has no column '${name}'`);
    }
    return expr as Column;
  }

  #query(): Query<R> {
    return new Query<R>(this.#handle, this.#columnNames);
  }

  insert(rows: readonly I[]): Promise<UpdateStatus> {
    return this.#handle.insert(rows as readonly object[]);
  }

  delete(where?: Expr<boolean, boolean>): Promise<UpdateStatus> {
    return this.#handle.delete(where === undefined ? null : encodeFor(this.#handle, this.#columnNames, where));
  }

  where(predicate: Expr<boolean, boolean>): Query<R> {
    return this.#query().where(predicate);
  }

  select<S extends Record<string, AnyExpr>>(columns: S): Query<Selected<S>>;
  select(...exprs: AnyExpr[]): Query<Record<string, unknown>>;
  select(...args: unknown[]): Query<unknown> {
    const query = this.#query();
    return (query.select as (...a: unknown[]) => Query<unknown>).apply(query, args);
  }

  groupBy(...exprs: AnyExpr[]): Query<R> {
    return this.#query().groupBy(...exprs);
  }

  orderBy(expr: AnyExpr, options?: { asc?: boolean }): Query<R> {
    return this.#query().orderBy(expr, options);
  }

  limit(n: number): Query<R> {
    return this.#query().limit(n);
  }

  collect(): Promise<R[]> {
    return this.#query().collect();
  }

  count(): Promise<number> {
    return this.#query().count();
  }
}

/** A table whose columns are also its properties, `t.title`, as in Python; a method of Table wins over a column. */
export function withColumns<T extends Table<unknown, unknown>, C>(table: T): T & C {
  return new Proxy(table, {
    get(target, prop) {
      if (typeof prop === 'symbol' || prop in target) {
        // the target, not the proxy, has the private fields its getters and methods read
        const value = Reflect.get(target, prop, target) as unknown;
        return typeof value === 'function' ? (value as (...a: unknown[]) => unknown).bind(target) : value;
      }
      if (prop === 'then') {
        return undefined; // not a thenable
      }
      return target.col(prop);
    },
  }) as T & C;
}

/** The table at a `pxt://org:db/dir/table` URI, as `pxt.get_table()` returns it in Python: `t.year.gt(2000)`. */
export async function getTable(uri: string): Promise<Table & Columns> {
  const handle = await TableHandle.open(uri);
  const columns = new Map<string, AnyExpr>();
  const names = new Map<AnyExpr, string>();
  for (const name of handle.columnNames()) {
    const md = handle.column(name)!;
    const def = new ColumnDef(md.type ?? { type: 'json', nullable: true });
    def.name = name;
    const node = { kind: 'col', def } as const;
    const expr = md.type === undefined ? new UnknownExpr(node) : newExpr(md.type.type, node);
    def.model = handle;
    columns.set(name, expr);
    names.set(expr, name);
  }
  return withColumns<Table, Columns>(new Table(handle, columns, (e) => names.get(e)));
}
