import { z } from 'zod';

const slug = z.string().regex(/^[a-z][a-z0-9_-]*$/);
const connection = z.object({
  connection: z.object({ organization: slug, db: slug.nullable() }),
  databases: z.array(z.object({ db: slug })).optional(),
});
export const connectionResultSchema = connection.or(z.object({ page: connection }).transform((value) => value.page));
