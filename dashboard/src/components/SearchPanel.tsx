import { useState, useEffect, useRef, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { search } from '@/api/client'
import { useDebounce } from '@/hooks/useApi'
import { cn } from '@/lib/utils'
import type { SearchResults } from '@/types'
import {
  Search,
  Folder,
  Table2,
  Eye,
  Camera,
  Copy,
  Hash,
  X,
  ArrowUp,
  ArrowDown,
  CornerDownLeft,
  Zap,
  Loader2,
} from 'lucide-react'

// ── Types ────────────────────────────────────────────────────────────────────

interface SearchPanelProps {
  isOpen: boolean
  onClose: () => void
  onSelect: (path: string, type: string) => void
}

type ResultType = 'directory' | 'table' | 'column'

interface SearchResultItem {
  type: ResultType
  name: string
  path: string
  subtype?: string
  extra?: string
}

// ── Icon + color mapping ─────────────────────────────────────────────────────

const RESULT_META: Record<string, {
  icon: typeof Table2
  color: string
  bg: string
}> = {
  directory:  { icon: Folder,  color: 'text-k-yellow',      bg: 'bg-k-yellow/10' },
  table:      { icon: Table2,  color: 'text-blue-400',      bg: 'bg-blue-400/10' },
  view:       { icon: Eye,     color: 'text-purple-400',    bg: 'bg-purple-400/10' },
  snapshot:   { icon: Camera,  color: 'text-orange-400',    bg: 'bg-orange-400/10' },
  replica:    { icon: Copy,    color: 'text-muted-foreground', bg: 'bg-muted' },
  column:     { icon: Hash,    color: 'text-emerald-400',   bg: 'bg-emerald-400/10' },
  computed:   { icon: Zap,     color: 'text-k-yellow',      bg: 'bg-k-yellow/10' },
}

function getResultMeta(item: SearchResultItem) {
  if (item.type === 'column') return RESULT_META.column
  if (item.type === 'directory') return RESULT_META.directory
  return RESULT_META[item.subtype ?? 'table'] ?? RESULT_META.table
}

// ── Result item ──────────────────────────────────────────────────────────────

function ResultItem({
  item,
  isSelected,
  onClick,
  onHover,
}: {
  item: SearchResultItem
  isSelected: boolean
  onClick: () => void
  onHover: () => void
}) {
  const ref = useRef<HTMLButtonElement>(null)
  const meta = getResultMeta(item)
  const Icon = meta.icon

  useEffect(() => {
    if (isSelected) ref.current?.scrollIntoView({ block: 'nearest' })
  }, [isSelected])

  return (
    <button
      ref={ref}
      className={cn(
        'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all duration-100',
        isSelected
          ? 'bg-accent/80 ring-1 ring-border/60'
          : 'hover:bg-accent/40',
      )}
      onClick={onClick}
      onMouseEnter={onHover}
    >
      {/* Icon badge */}
      <div className={cn(
        'flex h-8 w-8 shrink-0 items-center justify-center rounded-md',
        meta.bg,
      )}>
        <Icon className={cn('h-3.5 w-3.5', meta.color)} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={cn(
            'text-[13px] font-medium truncate',
            isSelected ? 'text-foreground' : 'text-foreground/80',
          )}>
            {item.name}
          </span>
          {item.subtype && item.type === 'table' && item.subtype !== 'table' && (
            <span className={cn(
              'text-[10px] font-medium px-1.5 py-0.5 rounded',
              meta.bg,
              meta.color,
            )}>
              {item.subtype}
            </span>
          )}
        </div>
        <div className="text-[11px] text-muted-foreground truncate font-mono mt-0.5">
          {item.type === 'column' ? (
            <>
              <span className="text-muted-foreground/80">in</span>{' '}
              <span className="text-foreground/70">{item.path}</span>
              {item.extra && (
                <span className="text-muted-foreground/70 ml-1.5">· {item.extra}</span>
              )}
            </>
          ) : (
            item.path
          )}
        </div>
      </div>

      {/* Type label */}
      <span className={cn(
        'text-[11px] font-medium capitalize shrink-0 transition-opacity',
        isSelected ? 'text-muted-foreground' : 'text-muted-foreground/70',
      )}>
        {item.type}
      </span>
    </button>
  )
}

// ── Section header ───────────────────────────────────────────────────────────

function SectionHeader({ label, count }: { label: string; count: number }) {
  return (
    <div className="flex items-center gap-2 px-3 pt-3 pb-1.5">
      <span className="text-[11px] font-semibold text-muted-foreground/70 uppercase tracking-wider">
        {label}
      </span>
      <span className="text-[11px] text-muted-foreground/60 tabular-nums">
        {count}
      </span>
      <div className="flex-1 h-px bg-border/30" />
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

export function SearchPanel({ isOpen, onClose, onSelect }: SearchPanelProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResults | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const debouncedQuery = useDebounce(query, 200)

  const flattenedResults: SearchResultItem[] = results
    ? [
        ...results.directories.map(d => ({ type: 'directory' as const, name: d.name, path: d.path })),
        ...results.tables.map(t => ({ type: 'table' as const, name: t.name, path: t.path, subtype: t.type })),
        ...results.columns.map(c => ({ type: 'column' as const, name: c.name, path: c.table, extra: c.type })),
      ]
    : []

  useEffect(() => {
    if (!debouncedQuery.trim()) { setResults(null); return }
    setLoading(true)
    search(debouncedQuery)
      .then(setResults)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [debouncedQuery])

  useEffect(() => { setSelectedIndex(0) }, [results])

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 0)
    else { setQuery(''); setResults(null) }
  }, [isOpen])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex(i => Math.min(i + 1, flattenedResults.length - 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex(i => Math.max(i - 1, 0))
        break
      case 'Enter':
        e.preventDefault()
        if (flattenedResults[selectedIndex]) {
          const item = flattenedResults[selectedIndex]
          onSelect(item.path, item.type === 'column' ? 'table' : item.type)
        }
        break
      case 'Escape':
        onClose()
        break
    }
  }, [flattenedResults, selectedIndex, onSelect, onClose])

  const handleItemClick = useCallback((item: SearchResultItem) => {
    onSelect(item.path, item.type === 'column' ? 'table' : item.type)
  }, [onSelect])

  if (!isOpen) return null

  let runningIdx = 0

  return createPortal(
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        style={{ animationDuration: '150ms' }}
        onClick={onClose}
      />

      {/* Dialog */}
      <div className="fixed inset-0 flex items-start justify-center pt-[12vh] px-4">
        <div
          className="w-full max-w-[560px] bg-card/95 backdrop-blur-xl border border-border/50 rounded-xl shadow-2xl shadow-black/40 overflow-hidden relative z-10"
          style={{ animation: 'search-enter 200ms ease-out' }}
        >
          {/* ── Search input ──────────────────────────────────────────── */}
          <div className="flex items-center gap-3 px-4 h-14 border-b border-border/40">
            {loading ? (
              <Loader2 className="h-4 w-4 text-k-yellow animate-spin shrink-0" />
            ) : (
              <Search className="h-4 w-4 text-muted-foreground/70 shrink-0" />
            )}
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search directories, tables, columns…"
              className="flex-1 bg-transparent text-foreground placeholder-muted-foreground/70 outline-none text-[15px] font-light"
              autoComplete="off"
              spellCheck={false}
            />
            {query && (
              <button
                onClick={() => setQuery('')}
                className="p-1 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
            <div className="pl-2 border-l border-border/30">
              <kbd className="text-[11px] bg-accent/80 px-2 py-1 rounded text-muted-foreground border border-border/40 font-mono">
              ESC
            </kbd>
            </div>
          </div>

          {/* ── Results ────────────────────────────────────────────────── */}
          <div className="max-h-[50vh] overflow-auto px-2 pb-2">
            {/* Loading skeleton */}
            {loading && !results && (
              <div className="space-y-2 p-3">
                {[1, 2, 3].map(i => (
                  <div key={i} className="flex items-center gap-3 animate-pulse">
                    <div className="h-8 w-8 rounded-md bg-accent" />
                    <div className="flex-1 space-y-1.5">
                      <div className="h-3 w-24 rounded bg-accent" />
                      <div className="h-2.5 w-40 rounded bg-accent/60" />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* No results */}
            {!loading && query && flattenedResults.length === 0 && (
              <div className="py-12 text-center">
                <Search className="h-8 w-8 text-muted-foreground/30 mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">
                  No results for "<span className="text-foreground/90 font-medium">{query}</span>"
                </p>
                <p className="text-xs text-muted-foreground/70 mt-1">
                  Try a broader query or different terms
                </p>
              </div>
            )}

            {/* Grouped results */}
            {!loading && flattenedResults.length > 0 && results && (
              <div className="animate-fade-in" style={{ animationDuration: '150ms' }}>
                {results.directories.length > 0 && (
                  <div>
                    <SectionHeader label="Directories" count={results.directories.length} />
                    <div className="space-y-0.5">
                      {results.directories.map((d) => {
                        const idx = runningIdx++
                        return (
                      <ResultItem
                        key={`dir-${d.path}`}
                        item={{ type: 'directory', name: d.name, path: d.path }}
                        isSelected={selectedIndex === idx}
                            onClick={() => handleItemClick({ type: 'directory', name: d.name, path: d.path })}
                            onHover={() => setSelectedIndex(idx)}
                      />
                        )
                      })}
                    </div>
                  </div>
                )}

                {results.tables.length > 0 && (
                  <div>
                    <SectionHeader label="Tables" count={results.tables.length} />
                    <div className="space-y-0.5">
                      {results.tables.map((t) => {
                        const idx = runningIdx++
                        return (
                          <ResultItem
                            key={`tbl-${t.path}`}
                            item={{ type: 'table', name: t.name, path: t.path, subtype: t.type }}
                            isSelected={selectedIndex === idx}
                            onClick={() => handleItemClick({ type: 'table', name: t.name, path: t.path, subtype: t.type })}
                            onHover={() => setSelectedIndex(idx)}
                          />
                        )
                      })}
                    </div>
                  </div>
                )}

                {results.columns.length > 0 && (
                  <div>
                    <SectionHeader label="Columns" count={results.columns.length} />
                    <div className="space-y-0.5">
                      {results.columns.map((c) => {
                        const idx = runningIdx++
                        return (
                          <ResultItem
                            key={`col-${c.table}-${c.name}`}
                            item={{ type: 'column', name: c.name, path: c.table, extra: c.type }}
                            isSelected={selectedIndex === idx}
                            onClick={() => handleItemClick({ type: 'column', name: c.name, path: c.table, extra: c.type })}
                            onHover={() => setSelectedIndex(idx)}
                          />
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {!loading && !query && (
              <div className="py-10 text-center">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/80 mx-auto mb-4">
                  <Search className="h-4.5 w-4.5 text-muted-foreground/60" />
                </div>
                <p className="text-[13px] text-muted-foreground font-light">
                  Search across your Pixeltable instance
                </p>
                <p className="text-xs text-muted-foreground/70 mt-1">
                  Directories, tables, views, snapshots, and columns
                </p>
              </div>
            )}
          </div>

          {/* ── Footer / keyboard hints ────────────────────────────────── */}
          <div className="flex items-center gap-4 px-4 py-2.5 border-t border-border/30 bg-accent/30">
            <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground/70">
              <div className="flex gap-0.5">
                <kbd className="inline-flex h-[18px] w-[18px] items-center justify-center rounded border border-border/50 bg-accent/60">
                  <ArrowUp className="h-2.5 w-2.5" />
                </kbd>
                <kbd className="inline-flex h-[18px] w-[18px] items-center justify-center rounded border border-border/50 bg-accent/60">
                  <ArrowDown className="h-2.5 w-2.5" />
                </kbd>
              </div>
              <span>navigate</span>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground/70">
              <kbd className="inline-flex h-[18px] px-1.5 items-center justify-center rounded border border-border/50 bg-accent/60">
                <CornerDownLeft className="h-2.5 w-2.5" />
              </kbd>
              <span>select</span>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground/70">
              <kbd className="inline-flex h-[18px] px-1.5 items-center justify-center rounded border border-border/50 bg-accent/60 text-[10px] font-mono">
                esc
              </kbd>
              <span>close</span>
            </div>
            {flattenedResults.length > 0 && (
              <span className="ml-auto text-[11px] text-muted-foreground/70 tabular-nums">
                {flattenedResults.length} result{flattenedResults.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Animation keyframes */}
      <style>{`
        @keyframes search-enter {
          from {
            opacity: 0;
            transform: translateY(-8px) scale(0.98);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
      `}</style>
    </div>,
    document.body,
  )
}
