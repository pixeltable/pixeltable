import { expect, it } from 'vitest';
import { connectionResultSchema } from './connection';

it('opens the same catalog from an entrypoint result and a browse page', () => {
  const value = { connection: { organization: 'demo', db: 'main' } };
  expect(connectionResultSchema.parse(value)).toEqual(value);
  expect(connectionResultSchema.parse({ page: value })).toEqual(value);
});

it('retains a database choice and refuses unrelated or malformed results', () => {
  const value = { connection: { organization: 'demo', db: null }, databases: [{ db: 'main' }] };
  expect(connectionResultSchema.parse(value)).toEqual(value);
  for (const item of [{}, { page: {} }, { connection: { organization: '../other', db: 'main' } }]) {
    expect(connectionResultSchema.safeParse(item).success).toBe(false);
  }
});
