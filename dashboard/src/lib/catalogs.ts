// The user's set of catalogs, persisted in the browser. 'local' is the in-process catalog (always present,
// not removable); the persisted list holds the hosted catalog uris (e.g. 'pxt://org:db') the user has added.

export const LOCAL_CATALOG = 'local'

// localStorage key holding the JSON array of hosted catalog uris (excludes 'local').
const CATALOGS_KEY = 'pxt-catalogs'
const ACTIVE_KEY = 'pxt-active-catalog'

export function loadExtraCatalogs(): string[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(CATALOGS_KEY) ?? '[]')
    if (Array.isArray(parsed)) return parsed.filter((u): u is string => typeof u === 'string')
  } catch {
    // ignore a malformed persisted value and fall back to none
  }
  return []
}

export function saveExtraCatalogs(uris: string[]): void {
  localStorage.setItem(CATALOGS_KEY, JSON.stringify(uris))
}

export function loadActiveCatalog(): string {
  const saved = localStorage.getItem(ACTIVE_KEY)
  if (saved === null || saved === '' || saved === LOCAL_CATALOG) return LOCAL_CATALOG
  if (loadExtraCatalogs().includes(saved)) return saved
  return LOCAL_CATALOG
}

export function saveActiveCatalog(uri: string): void {
  localStorage.setItem(ACTIVE_KEY, uri)
}

/** Short label for the switcher chip. */
export function catalogLabel(uri: string): string {
  if (uri === LOCAL_CATALOG) return 'Local'
  if (uri.startsWith('pxt://')) return uri.slice('pxt://'.length)
  return uri
}

/**
 * Normalize a user-entered hosted catalog URI.
 * Accepts `pxt://org:db…` or bare `org:db` (no spaces); returns null when invalid.
 */
export function normalizeCloudCatalogUri(raw: string): string | null {
  const trimmed = raw.trim()
  if (trimmed === '' || trimmed === LOCAL_CATALOG) return null

  let uri = trimmed
  if (!uri.startsWith('pxt://')) {
    // Bare org:db form — single colon separating two non-empty segments, no spaces.
    if (!/^[^:\s]+:[^:\s]+$/.test(uri)) return null
    uri = `pxt://${uri}`
  }

  const rest = uri.slice('pxt://'.length)
  if (!/^[^:\s]+:[^:\s]+/.test(rest)) return null
  return uri
}
