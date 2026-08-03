// The user's set of catalogs, persisted in the browser. 'local' is the in-process catalog (always present,
// not removable); the persisted list holds the hosted catalog uris (e.g. 'pxt://org:db') the user has added.

export const LOCAL_CATALOG = 'local'

// localStorage key holding the JSON array of hosted catalog uris (excludes 'local').
const CATALOGS_KEY = 'pxt-catalogs'

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
