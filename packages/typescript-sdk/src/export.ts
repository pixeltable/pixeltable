/**
 * What `pxt schema` runs to read a TypeScript schema file: Node imports the file and prints each exported model as
 * data, which the Python CLI turns into TableModel classes.
 */

import { pathToFileURL } from 'node:url';

import { EXPORT_VERSION, modelSpec, type ModelExport } from './codec.ts';
import { isModelClass } from './model.ts';
import { hookTsx } from './register.ts';

/** Every model class the file at appFile exports, as data for `pxt schema`. */
export async function exportModels(appFile: string): Promise<ModelExport> {
  hookTsx();
  const module = (await import(pathToFileURL(appFile).href)) as Record<string, unknown>;
  const models = Object.values(module).filter(isModelClass);
  return { v: EXPORT_VERSION, models: models.map(modelSpec) };
}
