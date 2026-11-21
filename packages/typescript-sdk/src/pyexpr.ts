/**
 * Encodes expressions the way Python's `Expr.as_dict()` does, for queries sent over /rpc. A column becomes a
 * ColumnRef by table and column id, taken from the table's metadata. A function call carries only its Python path:
 * the daemon resolves the function and computes the call's type.
 */

import { PixeltableError } from './errors.ts';
import { Expr, type AnyExpr, type TypeName, type TypeSpec } from './expr.ts';

const CALLABLE = 'pixeltable.func.callable_function.CallableFunction';
const AGGREGATE = 'pixeltable.func.aggregate_function.AggregateFunction';

const COMPARISON: Record<string, number> = { lt: 0, lte: 1, eq: 2, ne: 3, gt: 4, gte: 5 };
const ARITHMETIC: Record<string, number> = { add: 0, sub: 1, mul: 2, div: 3, mod: 4, floorDiv: 5, pow: 6 };
const LOGICAL = { and: 0, or: 1, not: 2 } as const;
const CONCAT = 0;

const TYPE_CLASSES: Record<TypeName, string> = {
  string: 'StringType',
  int: 'IntType',
  float: 'FloatType',
  bool: 'BoolType',
  timestamp: 'TimestampType',
  date: 'DateType',
  uuid: 'UUIDType',
  json: 'JsonType',
};

export type PyDict = Record<string, unknown>;

export interface Scope {
  /** The column an expression object stands for, if it is one of the table's columns. */
  columnName(expr: AnyExpr): string | undefined;
  /** The ColumnRef for a column name. */
  columnRef(name: string): PyDict;
}

export function columnType(spec: TypeSpec): PyDict {
  return { _classname: TYPE_CLASSES[spec.type], nullable: spec.nullable };
}

export function encode(expr: AnyExpr, scope: Scope): PyDict {
  const name = scope.columnName(expr);
  if (name !== undefined) {
    return scope.columnRef(name);
  }
  const node = expr.node;
  const operand = (value: unknown): PyDict => (value instanceof Expr ? encode(value as AnyExpr, scope) : literal(value));
  switch (node.kind) {
    case 'col':
      throw new PixeltableError('INVALID_ARGUMENT', `column '${node.def.name ?? '?'}' is not a column of this table`);
    case 'column':
      return encode(node.of, scope);
    case 'lit':
      return literal(node.value, node.type);
    case 'op': {
      const [a, b] = node.args;
      if (node.op in COMPARISON) {
        if (b === null || b === undefined) {
          // as in Python, comparing with None tests for null
          const isNull = { _classname: 'IsNull', components: [operand(a)] };
          return node.op === 'eq' ? isNull : node.op === 'ne' ? not(isNull) : invalid(`${node.op}(null)`);
        }
        return { _classname: 'Comparison', operator: COMPARISON[node.op], components: [operand(a), operand(b)] };
      }
      if (node.op in ARITHMETIC) {
        return { _classname: 'ArithmeticExpr', operator: ARITHMETIC[node.op], components: [operand(a), operand(b)] };
      }
      switch (node.op) {
        case 'and':
        case 'or':
          return { _classname: 'CompoundPredicate', operator: LOGICAL[node.op], components: node.args.map(operand) };
        case 'not':
          return not(operand(a));
        case 'isNull':
          return { _classname: 'IsNull', components: [operand(a)] };
        case 'isNotNull':
          return not({ _classname: 'IsNull', components: [operand(a)] });
        case 'concat':
          return { _classname: 'StringOp', operator: CONCAT, components: [operand(a), operand(b)] };
        case 'neg':
          // Python's -e is e * -1
          return { _classname: 'ArithmeticExpr', operator: ARITHMETIC['mul'], components: [operand(a), literal(-1)] };
        case 'abs':
        case 'round':
          return functionCall(`pixeltable.functions.math.${node.op}`, false, node.args.map(operand), {}, false);
      }
      return invalid(`operator ${node.op}`);
    }
    case 'method':
      return functionCall(node.path, false, [encode(node.on, scope), ...node.args.map(operand)], kwargs(node.kwargs, operand), true);
    case 'prop':
      return functionCall(node.path, false, [encode(node.on, scope)], {}, false);
    case 'call':
      return functionCall(node.path, node.aggregate, node.args.map(operand), kwargs(node.kwargs, operand), false);
    case 'cast':
      return { _classname: 'TypeCast', new_type: columnType(node.type), components: [encode(node.on, scope)] };
    case 'get':
    case 'at':
    case 'slice': {
      const element = node.kind === 'get' ? node.key : node.kind === 'at' ? node.index : [node.start, node.stop, node.step];
      const base = encode(node.on, scope);
      // a path into a path extends it, as e['a']['b'] does in Python
      if (base['_classname'] === 'JsonPath') {
        return { ...base, path_elements: [...(base['path_elements'] as unknown[]), element] };
      }
      return { _classname: 'JsonPath', path_elements: [element], root_type: null, components: [base] };
    }
    case 'isIn': {
      if (node.values instanceof Expr) {
        return invalid('isIn() with an expression');
      }
      return { _classname: 'InPredicate', value_list: node.values, components: [encode(node.on, scope)] };
    }
  }
}

function not(component: PyDict): PyDict {
  return { _classname: 'CompoundPredicate', operator: LOGICAL.not, components: [component] };
}

function kwargs(values: Readonly<Record<string, unknown>>, operand: (v: unknown) => PyDict): Record<string, PyDict> {
  const out: Record<string, PyDict> = {};
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined) {
      out[key] = operand(value);
    }
  }
  return out;
}

function functionCall(
  path: string,
  aggregate: boolean,
  args: PyDict[],
  kwargs: Record<string, PyDict>,
  isMethodCall: boolean,
): PyDict {
  const kwargIdxs: Record<string, number> = {};
  const components = [...args];
  for (const [name, value] of Object.entries(kwargs)) {
    kwargIdxs[name] = components.length;
    components.push(value);
  }
  return {
    _classname: 'FunctionCall',
    fn: { _classpath: aggregate ? AGGREGATE : CALLABLE, path },
    // the daemon computes the return type from the function it resolves
    return_type: null,
    arg_idxs: args.map((_, i) => i),
    kwarg_idxs: kwargIdxs,
    group_by_start_idx: 0,
    group_by_stop_idx: 0,
    order_by_start_idx: components.length,
    is_method_call: isMethodCall,
    components,
  };
}

/** A literal, typed as Python types a plain value: an integral number is an Int, any other number a Float. */
export function literal(value: unknown, type?: TypeName): PyDict {
  const inferred: TypeName =
    type ??
    (typeof value === 'string'
      ? 'string'
      : typeof value === 'boolean'
        ? 'bool'
        : typeof value === 'number'
          ? Number.isInteger(value)
            ? 'int'
            : 'float'
          : typeof value === 'bigint'
            ? 'int'
            : value instanceof Date
              ? 'timestamp'
              : 'json');
  const val = value instanceof Date ? value.toISOString() : typeof value === 'bigint' ? Number(value) : value;
  return { _classname: 'Literal', val, col_type: { _classname: TYPE_CLASSES[inferred], nullable: false } };
}

function invalid(what: string): never {
  throw new PixeltableError('UNSUPPORTED_OPERATION', `${what} is not supported in queries yet`);
}
