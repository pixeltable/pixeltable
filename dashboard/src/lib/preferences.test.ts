import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { Window } from 'happy-dom'
import { createMemoryPreferences } from './preferences'
import { loadActiveCatalog, loadExtraCatalogs, saveActiveCatalog, saveExtraCatalogs } from './utils'

beforeEach(() => vi.stubGlobal('localStorage', new Window({ url: 'http://localhost' }).localStorage))
afterEach(() => vi.unstubAllGlobals())

it('keeps local catalog choices in the existing browser keys', () => {
  saveExtraCatalogs(['pxt://org:main'])
  saveActiveCatalog('pxt://org:main')
  expect(loadActiveCatalog()).toBe('pxt://org:main')
  expect(JSON.parse(localStorage.getItem('pxt-catalogs') ?? 'null')).toEqual(['pxt://org:main'])
})

it('supports isolated in-memory preferences when browser storage is denied', () => {
  vi.stubGlobal('localStorage', { getItem() { throw new Error('denied') }, setItem() { throw new Error('denied') } })
  const one = createMemoryPreferences()
  const two = createMemoryPreferences()
  saveExtraCatalogs(['pxt://org:main'], one)
  saveActiveCatalog('pxt://org:main', one)
  expect(loadActiveCatalog(one)).toBe('pxt://org:main')
  expect(loadActiveCatalog(two)).toBe('local')
  expect(loadExtraCatalogs(two)).toEqual([])
})
