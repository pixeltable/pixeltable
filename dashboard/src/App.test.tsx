import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'
import { createDashboardClient } from '@/api/client'
import { DashboardRuntimeContext, type DashboardRuntime } from '@/lib/dashboard-runtime'
import { createMemoryPreferences } from '@/lib/preferences'

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
const mounted: Array<ReturnType<typeof createRoot>> = []
afterEach(async () => {
  await act(async () => { mounted.splice(0).forEach((root) => root.unmount()) })
  document.body.replaceChildren()
  document.documentElement.classList.remove('dark')
  vi.unstubAllGlobals()
})

it('renders the embedded catalog with denied browser storage, no status request and host-controlled theme', async () => {
  const storage = { getItem: vi.fn(() => { throw new Error('storage denied') }), setItem: vi.fn(() => { throw new Error('storage denied') }) }
  vi.stubGlobal('localStorage', storage)
  const read = vi.fn(async (_url: string) => Response.json({ tree: { entries: [] } }))
  const runtime = { kind: 'embedded', catalog: 'pxt://org:main', client: createDashboardClient(read), preferences: createMemoryPreferences(), resolveMedia: async () => '', theme: 'dark' } satisfies DashboardRuntime
  const container = document.createElement('div'); document.body.append(container)
  const root = createRoot(container); mounted.push(root)
  const render = (value: DashboardRuntime) => root.render(<DashboardRuntimeContext.Provider value={value}><MemoryRouter><App /></MemoryRouter></DashboardRuntimeContext.Provider>)
  await act(async () => { render(runtime) })
  expect(container.textContent).toContain('pxt://org:main')
  expect(container.textContent).not.toContain('Add cloud')
  expect(container.querySelector('button[title="Light mode"]')).toBeNull()
  expect(document.documentElement.classList.contains('dark')).toBe(true)
  expect(read.mock.calls.every(([url]) => !url.includes('/status'))).toBe(true)
  expect(storage.getItem).not.toHaveBeenCalled()
  expect(storage.setItem).not.toHaveBeenCalled()
  await act(async () => { render({ ...runtime, theme: 'light' }) })
  expect(document.documentElement.classList.contains('dark')).toBe(false)
})

it('keeps local catalog controls, persisted theme and status reads available', async () => {
  const preferences = createMemoryPreferences()
  preferences.setItem('pxt-theme', 'dark')
  vi.stubGlobal('localStorage', preferences)
  const read = vi.fn(async (url: string) => url.includes('/status')
    ? Response.json({ pxt_version: '0.7.19', home: '/test', total_tables: 0, total_errors: 0 })
    : Response.json({ tree: { entries: [] } }))
  const runtime = { kind: 'local', client: createDashboardClient(read), preferences } satisfies DashboardRuntime
  const container = document.createElement('div'); document.body.append(container)
  const root = createRoot(container); mounted.push(root)
  await act(async () => root.render(<DashboardRuntimeContext.Provider value={runtime}><MemoryRouter><App /></MemoryRouter></DashboardRuntimeContext.Provider>))
  expect(read.mock.calls.some(([url]) => url === '/api/status')).toBe(true)
  expect(container.textContent).toContain('Local')
  expect(container.querySelector('button[title="Light mode"]')).not.toBeNull()
  expect(document.documentElement.classList.contains('dark')).toBe(true)
  expect(preferences.getItem('pxt-theme')).toBe('dark')
})
