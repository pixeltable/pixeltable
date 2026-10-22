import { describe, expect, it, vi } from 'vitest';
import { createMcpMediaResolver } from './media';

const reference = { organization: 'org', db: 'main', path: 'video/frames', table_id: 'table-1', version: 2, offset: 4, fingerprint: 'a'.repeat(64), errors_only: false, column: 'frame', order_by: 'timestamp', order_desc: true };
const location = (value = reference) => 'pxt-preview:' + encodeURIComponent(JSON.stringify(value));

describe('embedded media boundary', () => {
  it('passes the scoped, sorted record reference and returns only bounded inline media', async () => {
    const callServerTool = vi.fn().mockResolvedValue({ _meta: { preview: { data_uri: 'data:image/jpeg;base64,YQ==' } } });
    const resolve = createMcpMediaResolver({ callServerTool }, 'pxt://org:main');
    expect(await resolve(location())).toBe('data:image/jpeg;base64,YQ==');
    expect(callServerTool).toHaveBeenCalledWith({ name: 'preview_collection_media', arguments: reference });
    callServerTool.mockResolvedValue({ _meta: { preview: { data_uri: 'https://private.example/image' } } });
    await expect(resolve(location())).rejects.toThrow();
  });
  it('rejects foreign references, arbitrary URLs and HTML before calling a tool', async () => {
    const callServerTool = vi.fn();
    const resolve = createMcpMediaResolver({ callServerTool }, 'pxt://org:main');
    for (const value of ['https://private.example/image', 'data:text/html;base64,YQ==', location({ ...reference, db: 'other' }), 'pxt-preview:%FF']) {
      await expect(resolve(value)).rejects.toThrow();
    }
    expect(callServerTool).not.toHaveBeenCalled();
  });
});
