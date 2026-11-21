import { layout, type DefinedModel, type Insert, type Row } from './model.ts';

const INSERT_ROUTE = Symbol('pixeltable.insertRoute');

export interface InsertRoute {
  readonly [INSERT_ROUTE]: true;
  readonly service: string;
  readonly path: string;
  readonly model: DefinedModel;
  readonly inputs: readonly string[];
  readonly outputs: readonly string[];
}

export function insertRoute<M extends DefinedModel>(
  model: M,
  options: {
    service: string;
    path: `/${string}`;
    inputs: readonly Extract<keyof Insert<InstanceType<M>>, string>[];
    outputs: readonly Extract<keyof Row<InstanceType<M>>, string>[];
  },
): InsertRoute {
  return { ...options, model, [INSERT_ROUTE]: true };
}

export function isInsertRoute(value: unknown): value is InsertRoute {
  return typeof value === 'object' && value !== null && INSERT_ROUTE in value && value[INSERT_ROUTE] === true;
}

export function routeSpec({ model, service, path, inputs, outputs }: InsertRoute) {
  return { table: layout(model).info.name, service, path, inputs, outputs };
}
