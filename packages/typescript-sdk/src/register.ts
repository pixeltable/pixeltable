/**
 * Lets Node import .tsx files the way it imports .ts files, with their types stripped; a schema file holds no JSX.
 * `node --import @pixeltable/sdk/register main.ts` runs a program that imports `app.tsx`.
 */

import { readFileSync } from 'node:fs';
import { registerHooks, stripTypeScriptTypes } from 'node:module';
import { fileURLToPath } from 'node:url';

let hooked = false;

export function hookTsx(): void {
  if (hooked) {
    return;
  }
  hooked = true;
  registerHooks({
    load(url, context, nextLoad) {
      if (url.startsWith('file:') && url.endsWith('.tsx')) {
        const source = readFileSync(fileURLToPath(url), 'utf8');
        return { format: 'module', source: stripTypeScriptTypes(source), shortCircuit: true };
      }
      return nextLoad(url, context);
    },
  });
}

hookTsx();
