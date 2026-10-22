import type { App } from '@modelcontextprotocol/ext-apps';
import { z } from 'zod';

const referenceSchema = z.object({
  organization: z.string(), db: z.string(),
  path: z.string().min(1).max(1024), table_id: z.string().min(1).max(128),
  version: z.number().int().nonnegative().nullable(),
  offset: z.number().int().min(0).max(100000),
  fingerprint: z.string().regex(/^[a-f0-9]{64}$/),
  errors_only: z.boolean(), column: z.string().min(1).max(256),
  order_by: z.string().max(256).optional(), order_desc: z.boolean().optional(),
});
const previewSchema = z.object({ _meta: z.object({ preview: z.object({
  data_uri: z.string().max(248000).regex(/^data:(image\/jpeg|audio\/[a-z0-9.+-]+|video\/[a-z0-9.+-]+);base64,/),
}) }) });

export function createMcpMediaResolver(app: Pick<App, 'callServerTool'>, catalog: string) {
  return async (location: string): Promise<string> => {
    if (!location.startsWith('pxt-preview:') || location.length > 12000) {
      throw new Error('This media cannot be previewed in chat. Open it in Pixeltable Cloud.');
    }
    const reference = referenceSchema.parse(JSON.parse(decodeURIComponent(location.slice('pxt-preview:'.length))));
    if (`pxt://${reference.organization}:${reference.db}` !== catalog) throw new Error('This media belongs to another catalog.');
    const result = await app.callServerTool({ name: 'preview_collection_media', arguments: reference });
    if (result.isError) {
      const message = result.content?.find((item) => item.type === 'text');
      throw new Error(message?.type === 'text' ? message.text : 'The media preview could not be read.');
    }
    return previewSchema.parse(result)._meta.preview.data_uri;
  };
}
