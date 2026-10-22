import { afterEach, expect, it, vi } from 'vitest'
import { createDashboardClient } from './client'

afterEach(() => vi.unstubAllGlobals())

it('keeps the default HTTP URL, sorting, pagination and error filter contract', async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json({ columns: [], rows: [], total_count: 0 }))
  vi.stubGlobal('fetch', fetch)
  await createDashboardClient().getTableData('pxt://org:main/demo/items', { offset: 25, limit: 25, orderBy: 'name', orderDesc: true, errorsOnly: true })
  const url = new URL(fetch.mock.calls[0][0], 'http://localhost')
  expect(url.pathname).toBe('/api/dashboard/tables/data')
  expect(Object.fromEntries(url.searchParams)).toEqual({ path: 'pxt://org:main/demo/items', offset: '25', limit: '25', order_by: 'name', order_desc: 'true', errors_only: 'true' })
})

it('keeps clients isolated and does not replace the browser fetch function', async () => {
  const fetch = vi.fn().mockRejectedValue(new Error('unexpected HTTP'))
  vi.stubGlobal('fetch', fetch)
  const one = createDashboardClient(async () => Response.json({ tree: { entries: [{ name: 'one' }] } }))
  const two = createDashboardClient(async () => Response.json({ tree: { entries: [{ name: 'two' }] } }))
  expect(await one.getDirectoryTree()).toEqual([{ name: 'one' }])
  expect(await two.getDirectoryTree()).toEqual([{ name: 'two' }])
  expect(fetch).not.toHaveBeenCalled()
})

it('preserves useful HTTP errors through either transport', async () => {
  const client = createDashboardClient(async () => Response.json({ error_code: 'TABLE_NOT_FOUND', detail: 'private trace' }, { status: 404 }))
  await expect(client.getTableMetadata('gone')).rejects.toThrow('Not found.')
})
