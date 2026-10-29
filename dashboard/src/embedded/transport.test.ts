import { expect, it, vi } from 'vitest'
import { createMcpDashboardTransport } from './transport'

it('uses an app tool and strips only the connected catalog prefix', async () => {
  const callServerTool = vi.fn().mockResolvedValue({ structuredContent: { data: { rows: [] } } })
  const read = createMcpDashboardTransport({ callServerTool }, 'pxt://org:main')
  expect(await (await read('/api/dashboard/tables/data?path=pxt%3A%2F%2Forg%3Amain%2Fdemo%2Fitems&limit=25')).json()).toEqual({ rows: [] })
  expect(callServerTool).toHaveBeenCalledWith({ name: 'read_dashboard', arguments: { db: 'main', endpoint: 'table_data', parameters: { path: 'demo/items', limit: '25' } } })
})

it('refuses foreign catalogs, exports, URLs and repeated scope parameters before calling the host', async () => {
  const callServerTool = vi.fn()
  const read = createMcpDashboardTransport({ callServerTool }, 'pxt://org:main')
  for (const url of [
    'https://attacker.example/api/dirs', '/api/dashboard/tables/export',
    '/api/dirs?path=pxt%3A%2F%2Fother%3Amain', '/api/dirs?path=one&path=two',
    '/api/dashboard/search?q=x&catalogs=pxt%3A%2F%2Fother%3Amain',
  ]) await expect(read(url)).rejects.toThrow()
  expect(callServerTool).not.toHaveBeenCalled()
})

it('rejects a stripped path that still carries a URI or climbs out of the catalog', async () => {
  const callServerTool = vi.fn()
  const read = createMcpDashboardTransport({ callServerTool }, 'pxt://org:main')
  for (const target of ['pxt://org:main/pxt://other:db/secret', 'pxt://org:main/../x', 'a/../b']) {
    await expect(read(`/api/dirs?path=${encodeURIComponent(target)}`)).rejects.toThrow('Choose an item in the connected catalog.')
  }
  expect(callServerTool).not.toHaveBeenCalled()
})

it('reports host refusals and missing dashboard capabilities without falling back to HTTP', async () => {
  const callServerTool = vi.fn().mockResolvedValueOnce({ isError: true, content: [{ type: 'text', text: 'Refresh this catalog.' }] }).mockResolvedValueOnce({})
  const read = createMcpDashboardTransport({ callServerTool }, 'pxt://org:main')
  await expect(read('/api/dirs')).rejects.toThrow('Refresh this catalog.')
  await expect(read('/api/dirs')).rejects.toThrow('Update the connection')
})
