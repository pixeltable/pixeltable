import { useState, useEffect, useLayoutEffect, useMemo, useRef, useCallback } from 'react'
import { flushSync } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
  type ImperativePanelGroupHandle,
} from 'react-resizable-panels'
import { getTableMetadata, getTableData, getPipeline } from '@/api/client'
import { useDebounce } from '@/hooks/useDebounce'
import type {
  PipelineColumn, CellError, DataRow,
  TableMetadata, TableData, DataColumn, ColumnInfo, IndexInfo,
  PipelineNode as PipelineNodeType, PipelineEdge, PipelineVersion,
} from '@/types'
import { cn, tableHref } from '@/lib/utils'
import {
  ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ChevronUp, ChevronDown,
  ImageIcon, Film, Music, FileText,
  Rows3, Table2, Filter, X, Search,
  RefreshCw, Key, Download, SquareFunction,
  Copy, Eye, Camera,
  GitBranch, ArrowRight, ExternalLink,
  AlertTriangle, Clock,
} from 'lucide-react'
import { ColumnFlowDiagram } from './ColumnFlowDiagram'
import { ColumnTypeBadge, formatColumnTypeDisplay, getColumnTypeMeta } from '@/lib/column-types'
import { formatPythonExpr, PythonExpr } from '@/lib/python-highlight'

function KindIcon({ kind, className = 'h-4 w-4' }: { kind: string; className?: string }) {
  switch (kind) {
    case 'view':
      return <Eye className={`${className} text-purple-400 shrink-0`} />
    case 'snapshot':
      return <Camera className={`${className} text-orange-400 shrink-0`} />
    case 'replica':
      return <Copy className={`${className} text-muted-foreground shrink-0`} />
    default:
      return <Table2 className={`${className} text-blue-400 shrink-0`} />
  }
}

function kindWord(kind: string): string {
  switch (kind) {
    case 'view':
      return 'View'
    case 'snapshot':
      return 'Snapshot'
    case 'replica':
      return 'Replica'
    default:
      return 'Table'
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

type ViewMode = 'table' | 'gallery'
type MediaType = 'image' | 'video' | 'audio' | 'document'
type FilterValue = string | number | boolean | null
type ColumnFilter =
  | { type: 'contains'; value: string }
  | { type: 'values'; selected: FilterValue[] }
  | { type: 'range'; min: string; max: string }
  | { type: 'dateRange'; from: string; to: string }
type Filters = Record<string, ColumnFilter>

const getMediaType = (colType: string): MediaType => {
  const t = (colType || '').toLowerCase()
  if (t.includes('image')) return 'image'
  if (t.includes('video')) return 'video'
  if (t.includes('audio')) return 'audio'
  return 'document'
}

// ── Media Thumbnail (inline cell) ─────────────────────────────────────────

function MediaPreview({ url, type, onExpand }: { url: string; type: MediaType; onExpand?: () => void }) {
  const [error, setError] = useState(false)

  if (error || !url) {
    const Icon = { image: ImageIcon, video: Film, audio: Music, document: FileText }[type]
    return (
      <div className="flex items-center gap-1 text-muted-foreground text-[11px]">
        <Icon className="h-3.5 w-3.5" />
        <span className="truncate max-w-24">{url?.split('/').pop() || 'N/A'}</span>
      </div>
    )
  }

  if (type === 'image') return (
    <img src={url} alt="" className="max-h-16 max-w-32 rounded cursor-pointer hover:ring-2 ring-k-yellow object-cover" onError={() => setError(true)} onClick={onExpand} />
  )

  if (type === 'video') return (
    <div className="relative group cursor-pointer" onClick={onExpand}>
      <video src={url} className="max-h-16 max-w-32 rounded object-cover hover:ring-2 ring-k-yellow" muted preload="metadata" onError={() => setError(true)} />
      <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded opacity-0 group-hover:opacity-100 transition-opacity">
        <div className="w-7 h-7 bg-k-yellow/90 rounded-full flex items-center justify-center">
          <svg className="w-3 h-3 text-black ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
        </div>
      </div>
    </div>
  )

  if (type === 'audio') return <audio controls className="h-8 w-32"><source src={url} /></audio>

  const isExternal = /^https?:\/\//i.test(url)
  if (isExternal) {
    return (
      <a href={url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-[11px] text-foreground hover:underline cursor-pointer">
        <ExternalLink className="h-3 w-3" />
        <span className="truncate max-w-24">{url.split('/').pop()}</span>
      </a>
    )
  }

  return (
    <button onClick={onExpand} className="flex items-center gap-1 text-[11px] text-foreground hover:underline cursor-pointer">
      <FileText className="h-3.5 w-3.5" />
      <span className="truncate max-w-24">{url.split('/').pop()}</span>
    </button>
  )
}

// ── Media Lightbox (table-level, with row navigation) ─────────────────────

function MediaLightbox({ url, type, index, total, onClose, onPrev, onNext }: {
  url: string; type: MediaType; index: number; total: number
  onClose: () => void; onPrev: () => void; onNext: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); onPrev() }
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); onNext() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onPrev, onNext])

  const isPdf = /\.pdf(\?|$)/i.test(url)

  return (
    <div className="fixed inset-0 bg-black/90 flex items-center justify-center z-50" onClick={onClose}>
      {/* Close */}
      <button className="absolute top-4 right-4 text-white/70 hover:text-foreground transition-colors z-10" onClick={onClose}>
        <X className="h-7 w-7" />
      </button>

      {/* Counter */}
      <div className="absolute top-4 left-1/2 -translate-x-1/2 text-[11px] text-white/50 tabular-nums z-10">
        {index + 1} / {total}
      </div>

      {/* Prev */}
      <button
        onClick={e => { e.stopPropagation(); onPrev() }}
        disabled={index === 0}
        className="absolute left-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white disabled:opacity-20 disabled:cursor-default transition-all z-10"
      >
        <ChevronLeft className="h-5 w-5" />
      </button>

      {/* Next */}
      <button
        onClick={e => { e.stopPropagation(); onNext() }}
        disabled={index >= total - 1}
        className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white disabled:opacity-20 disabled:cursor-default transition-all z-10"
      >
        <ChevronRight className="h-5 w-5" />
      </button>

      {/* Content */}
      <div onClick={e => e.stopPropagation()}>
        {type === 'image' && <img src={url} alt="" className="max-h-[90vh] max-w-[90vw] rounded-lg" />}
        {type === 'video' && <video src={url} controls autoPlay className="max-h-[85vh] max-w-[90vw] rounded-lg" />}
        {type === 'document' && (/^https?:\/\//i.test(url) ? (
          <div className="flex flex-col items-center gap-4 bg-card rounded-lg p-10 border border-border/60">
            <FileText className="h-12 w-12 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">External documents cannot be previewed inline</p>
            <a href={url} target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-sm text-foreground hover:underline">
              <ExternalLink className="h-4 w-4" />Open in new tab
            </a>
          </div>
        ) : (
          <iframe src={url} className="w-[85vw] h-[85vh] rounded-lg bg-white" title="Document preview" sandbox={isPdf ? undefined : 'allow-same-origin allow-scripts'} />
        ))}
      </div>
    </div>
  )
}

// ── JSON Tree Viewer ──────────────────────────────────────────────────────

function countJsonMatches(value: unknown, q: string): number {
  if (!q) return 0
  let count = 0
  if (value === null || value === undefined) return 0
  if (typeof value !== 'object') {
    if (String(value).toLowerCase().includes(q)) count++
    return count
  }
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value as Record<string, unknown>)
  for (const [k, v] of entries) {
    if (k.toLowerCase().includes(q)) count++
    count += countJsonMatches(v, q)
  }
  return count
}

function subtreeHasMatch(value: unknown, key: string | undefined, q: string): boolean {
  if (!q) return false
  if (key?.toLowerCase().includes(q)) return true
  if (value === null || value === undefined) return false
  if (typeof value !== 'object') return String(value).toLowerCase().includes(q)
  const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value as Record<string, unknown>)
  return entries.some(([k, v]) => subtreeHasMatch(v, k, q))
}

function JsonNode({ keyName, value, depth, expandLevel, searchMatch, path }: {
  keyName?: string; value: unknown; depth: number; expandLevel: number; searchMatch: string; path: string
}) {
  const isObj = value !== null && typeof value === 'object'
  const isArray = Array.isArray(value)
  const hasMatch = searchMatch ? subtreeHasMatch(value, keyName, searchMatch) : false
  const [manualOpen, setManualOpen] = useState<boolean | null>(null)
  const [pathCopied, setPathCopied] = useState(false)

  const isOpen = searchMatch
    ? hasMatch
    : manualOpen !== null ? manualOpen : depth < expandLevel

  useEffect(() => { setManualOpen(null) }, [expandLevel, searchMatch])

  const keyMatches = searchMatch && keyName?.toLowerCase().includes(searchMatch)
  const valStr = !isObj ? String(value ?? 'null') : ''
  const valMatches = searchMatch && valStr.toLowerCase().includes(searchMatch)
  const HL = 'bg-k-yellow/25 rounded-sm px-0.5 -mx-0.5'

  const currentPath = path + (keyName !== undefined ? (path ? '.' : '') + keyName : '')

  const copyPath = (e: React.MouseEvent) => {
    e.stopPropagation()
    navigator.clipboard.writeText(currentPath).then(() => {
      setPathCopied(true)
      setTimeout(() => setPathCopied(false), 1000)
    })
  }

  if (!isObj) {
    const color =
      value === null ? 'text-muted-foreground italic' :
      typeof value === 'string' ? 'text-muted-foreground' :
      typeof value === 'number' ? 'text-muted-foreground' :
      typeof value === 'boolean' ? 'text-muted-foreground' : 'text-foreground'
    const display = typeof value === 'string' ? `"${valStr}"` : valStr
    return (
      <div className="group/node flex items-baseline gap-1 py-[1px] hover:bg-accent/20 rounded-sm" style={{ paddingLeft: depth * 16 }}>
        {keyName !== undefined && (
          <span className={cn('text-muted-foreground opacity-80 shrink-0', keyMatches && HL)} title={currentPath} onClick={copyPath} role="button">
            {keyName}<span className="text-muted-foreground">:</span>
          </span>
        )}
        <span className={cn(color, 'break-all', valMatches && HL)}>{display}</span>
        <button onClick={copyPath} className="opacity-0 group-hover/node:opacity-100 ml-1 p-0.5 rounded hover:bg-accent transition-opacity shrink-0" title={`Copy path: ${currentPath}`}>
          <Copy className="h-2.5 w-2.5 text-muted-foreground" />
        </button>
        {pathCopied && <span className="text-[9px] text-foreground animate-in fade-in">Copied</span>}
      </div>
    )
  }

  const entries = isArray ? value.map((v, i) => [i, v] as const) : Object.entries(value as Record<string, unknown>)
  const bracket = isArray ? ['[', ']'] : ['{', '}']
  const summary = `${entries.length} ${isArray ? (entries.length === 1 ? 'item' : 'items') : (entries.length === 1 ? 'key' : 'keys')}`

  return (
    <div>
      <div
        className="group/node flex items-baseline gap-1 py-[1px] cursor-pointer hover:bg-accent/20 rounded-sm transition-colors"
        style={{ paddingLeft: depth * 16 }}
        onClick={() => setManualOpen(isOpen ? false : true)}
      >
        <ChevronRight className={cn('h-3 w-3 shrink-0 text-muted-foreground transition-transform duration-150', isOpen && 'rotate-90')} />
        {keyName !== undefined && (
          <span className={cn('text-muted-foreground opacity-80 shrink-0', keyMatches && HL)} title={currentPath}>
            {keyName}<span className="text-muted-foreground">:</span>
          </span>
        )}
        <span className="text-muted-foreground">
          {bracket[0]}{!isOpen && <span className="text-muted-foreground"> {summary} {bracket[1]}</span>}
        </span>
        {keyName !== undefined && (
          <button onClick={copyPath} className="opacity-0 group-hover/node:opacity-100 ml-1 p-0.5 rounded hover:bg-accent transition-opacity shrink-0" title={`Copy path: ${currentPath}`}>
            <Copy className="h-2.5 w-2.5 text-muted-foreground" />
          </button>
        )}
      </div>
      {isOpen && (
        <>
          {entries.map(([k, v]) => (
            <JsonNode key={String(k)} keyName={isArray ? undefined : String(k)} value={v} depth={depth + 1} expandLevel={expandLevel} searchMatch={searchMatch} path={isArray ? `${currentPath}[${k}]` : currentPath} />
          ))}
          <div className="text-muted-foreground py-[1px]" style={{ paddingLeft: depth * 16 + 16 }}>{bracket[1]}</div>
        </>
      )}
    </div>
  )
}

// ── Expandable Cell Detail ─────────────────────────────────────────────────

function CellDetail({ value, onClose, pythonHighlight = false }: {
  value: string; onClose: () => void; pythonHighlight?: boolean
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const [copied, setCopied] = useState(false)
  const [search, setSearch] = useState('')
  const [viewRaw, setViewRaw] = useState(false)
  const [expandLevel, setExpandLevel] = useState(2)

  const handleCopy = () => {
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  let parsed: unknown = null
  let isJson = false
  try {
    parsed = JSON.parse(value)
    isJson = typeof parsed === 'object' && parsed !== null
  } catch { /* not JSON */ }

  const formatted = isJson ? JSON.stringify(parsed, null, 2) : value
  const searchLower = search.toLowerCase().trim()
  const matchCount = isJson && searchLower ? countJsonMatches(parsed, searchLower) : 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="relative bg-card border border-border rounded-lg shadow-2xl max-w-3xl w-full max-h-[85vh] flex flex-col m-4"
        onClick={e => e.stopPropagation()}
      >
        {/* Toolbar */}
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border/60 shrink-0">
          <span className="text-[10px] font-medium text-muted-foreground shrink-0">
            {value.length.toLocaleString()} chars
          </span>

          {isJson && (
            <div className="relative flex-1 max-w-xs">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
              <input
                type="text"
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search keys & values…"
                className="h-7 w-full pl-7 pr-2 text-[11px] rounded border border-border/40 bg-background/50 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
              />
            </div>
          )}

          {isJson && searchLower && (
            <span className="text-[10px] text-muted-foreground shrink-0 tabular-nums">
              {matchCount} {matchCount === 1 ? 'match' : 'matches'}
            </span>
          )}

          <div className="flex items-center gap-1 ml-auto shrink-0">
            {isJson && !viewRaw && (
              <>
                <button
                  onClick={() => setExpandLevel(99)}
                  className="text-[10px] px-1.5 py-1 rounded text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
                  title="Expand all"
                >
                  Expand
                </button>
                <button
                  onClick={() => setExpandLevel(1)}
                  className="text-[10px] px-1.5 py-1 rounded text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
                  title="Collapse all"
                >
                  Collapse
                </button>
                <div className="w-px h-4 bg-border/40 mx-0.5" />
              </>
            )}
            {(isJson || pythonHighlight) && (
              <button
                onClick={() => setViewRaw(!viewRaw)}
                className={cn('text-[10px] px-1.5 py-1 rounded transition-colors', viewRaw ? 'bg-accent text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-accent/50')}
              >
                Raw
              </button>
            )}
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors px-1.5 py-1 rounded hover:bg-accent"
            >
              <Copy className="h-3 w-3" />
              {copied ? 'Copied' : 'Copy'}
            </button>
            <button onClick={onClose} className="p-1 rounded hover:bg-accent transition-colors">
              <X className="h-3.5 w-3.5 text-muted-foreground" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-3 text-[11px] font-mono leading-relaxed select-text">
          {isJson && !viewRaw ? (
            <JsonNode value={parsed} depth={0} expandLevel={expandLevel} searchMatch={searchLower} path="$" />
          ) : pythonHighlight && !viewRaw ? (
            <div className="whitespace-pre text-foreground">
              <PythonExpr code={formatPythonExpr(value)} />
            </div>
          ) : pythonHighlight && viewRaw ? (
            <div className="whitespace-pre-wrap break-words text-foreground">
              <PythonExpr code={value} />
            </div>
          ) : (
            <pre className="whitespace-pre-wrap break-words text-foreground">{formatted}</pre>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Cell Renderer ──────────────────────────────────────────────────────────

const TRUNCATE_CHARS = 100
const TRUNCATE_JSON_CHARS = 80

function Cell({ value, column, error, onMediaExpand }: { value: unknown; column: DataColumn; error?: CellError; onMediaExpand?: () => void }) {
  const [expanded, setExpanded] = useState(false)

  if (!column.is_stored) {
    return <span className="text-muted-foreground italic text-[11px]" title="Unstored column — value not loaded">unstored</span>
  }
  if (error) {
    return (
      <div className="group/err relative flex items-center gap-1.5 rounded px-1.5 py-0.5 bg-destructive/10 border border-destructive/30 cursor-default" title={`${error.error_type}: ${error.error_msg}`}>
        <AlertTriangle className="h-3 w-3 text-destructive shrink-0" />
        <span className="text-[11px] text-destructive font-medium truncate max-w-[140px]">{error.error_type}</span>
        <div className="absolute bottom-full left-0 mb-1 hidden group-hover/err:block z-50 w-max max-w-xs">
          <div className="rounded-lg border border-destructive/40 bg-card shadow-lg px-3 py-2 text-xs">
            <div className="font-semibold text-destructive mb-0.5">{error.error_type}</div>
            <div className="text-muted-foreground break-words">{error.error_msg || 'No message'}</div>
          </div>
        </div>
      </div>
    )
  }
  if (value === null || value === undefined) return <span className="text-muted-foreground italic text-[11px]">null</span>
  if (column.is_media && typeof value === 'string') return <MediaPreview url={value} type={getMediaType(column.type)} onExpand={onMediaExpand} />

  if (typeof value === 'object') {
    const full = JSON.stringify(value, null, 2)
    const preview = JSON.stringify(value)
    const isLong = preview.length > TRUNCATE_JSON_CHARS
    return (
      <>
        <button
          onClick={() => isLong && setExpanded(true)}
          className={cn(
            'text-left text-[11px] bg-accent/50 px-2 py-1 rounded max-w-xs font-mono block',
            isLong ? 'cursor-pointer hover:bg-accent/80 transition-colors' : 'cursor-default',
          )}
          title={isLong ? 'Click to expand' : undefined}
        >
          <span className="line-clamp-2 break-all">{isLong ? preview.slice(0, TRUNCATE_JSON_CHARS) + ' …' : preview}</span>
        </button>
        {expanded && <CellDetail value={full} onClose={() => setExpanded(false)} />}
      </>
    )
  }

  if (typeof value === 'boolean') return <span className={cn('text-[11px] font-medium', value ? 'text-muted-foreground' : 'text-destructive')}>{String(value)}</span>
  if (typeof value === 'number') return <span className="font-mono text-xs text-foreground tabular-nums">{value.toLocaleString()}</span>

  const str = String(value)
  if (str.length <= TRUNCATE_CHARS) return <span className="text-xs">{str}</span>

  return (
    <>
      <button
        onClick={() => setExpanded(true)}
        className="text-left text-xs cursor-pointer hover:text-foreground transition-colors group/cell"
        title="Click to expand"
      >
        <span>{str.slice(0, TRUNCATE_CHARS)}</span>
        <span className="text-muted-foreground group-hover/cell:text-foreground transition-colors"> …more</span>
      </button>
      {expanded && <CellDetail value={str} onClose={() => setExpanded(false)} />}
    </>
  )
}

// ── Gallery Card ───────────────────────────────────────────────────────────

function GalleryCard({ row, columns, mediaCol, onClick }: {
  row: DataRow; columns: DataColumn[]; mediaCol: DataColumn
  onClick: () => void
}) {
  const url = row[mediaCol.name] as string | null
  const type = getMediaType(mediaCol.type)
  const otherCols = columns.filter(c => c.name !== mediaCol.name).slice(0, 3)
  const hasErrors = row._errors && Object.keys(row._errors).length > 0

  return (
    <div className={cn(
      'relative group rounded-lg border overflow-hidden transition-all',
      hasErrors ? 'border-destructive/40 hover:border-destructive/60' : 'border-border/60 hover:border-border',
    )}>
      <div className="aspect-square bg-background cursor-pointer" onClick={onClick}>
        {url ? (
          type === 'image' ? <img src={url} alt="" className="w-full h-full object-cover" /> :
          type === 'video' ? <video src={url} className="w-full h-full object-cover" muted preload="metadata" /> :
          <div className="w-full h-full flex items-center justify-center text-muted-foreground"><FileText className="h-12 w-12" /></div>
        ) : <div className="w-full h-full flex items-center justify-center text-muted-foreground"><ImageIcon className="h-12 w-12" /></div>}
        {hasErrors && (
          <div className="absolute top-1.5 right-1.5 flex items-center gap-1 bg-destructive/90 text-white text-[10px] font-medium rounded px-1.5 py-0.5">
            <AlertTriangle className="h-2.5 w-2.5" />
            {Object.keys(row._errors!).length}
          </div>
        )}
      </div>
      <div className="p-2 text-[11px] space-y-0.5">
        {otherCols.map(c => {
          const cellErr = row._errors?.[c.name]
          return (
            <div key={c.name} className="flex justify-between gap-2 truncate">
              <span className="text-muted-foreground">{c.name}:</span>
              {cellErr ? (
                <span className="truncate text-destructive flex items-center gap-0.5" title={`${cellErr.error_type}: ${cellErr.error_msg}`}>
                  <AlertTriangle className="h-2.5 w-2.5 shrink-0" />{cellErr.error_type}
                </span>
              ) : (
                <span className="truncate text-foreground">{row[c.name] != null ? String(row[c.name]).slice(0, 20) : 'null'}</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Filter Panel (Excel-style) ─────────────────────────────────────────────

function getColumnFilterType(colType: string): 'text' | 'numeric' | 'date' | 'bool' | 'enum' {
  const t = colType.toLowerCase()
  if (t.includes('bool')) return 'bool'
  if (t.includes('int') || t.includes('float')) return 'numeric'
  if (t.includes('timestamp') || t.includes('date')) return 'date'
  if (t.includes('string')) return 'text'
  return 'text'
}

const MAX_ENUM_VALUES = 50

function FilterControl({ col, filter, rows, onUpdate, onClear }: {
  col: DataColumn
  filter: ColumnFilter | undefined
  rows: DataRow[]
  onUpdate: (f: ColumnFilter) => void
  onClear: () => void
}) {
  const filterType = getColumnFilterType(col.type)
  const [enumSearch, setEnumSearch] = useState('')

  if (filterType === 'text') {
    const current = filter?.type === 'contains' ? filter.value : ''
    return (
      <input
        type="text"
        value={current}
        onChange={e => e.target.value ? onUpdate({ type: 'contains', value: e.target.value }) : onClear()}
        placeholder="Contains…"
        className="h-7 w-full px-2 text-[11px] rounded border border-border/40 bg-background/50 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
      />
    )
  }

  if (filterType === 'numeric') {
    const min = filter?.type === 'range' ? filter.min : ''
    const max = filter?.type === 'range' ? filter.max : ''
    return (
      <div className="flex items-center gap-1.5">
        <input
          type="number"
          value={min}
          onChange={e => {
            const v = e.target.value
            if (!v && !max) onClear()
            else onUpdate({ type: 'range', min: v, max })
          }}
          placeholder="Min"
          className="h-7 w-full px-2 text-[11px] rounded border border-border/40 bg-background/50 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
        />
        <span className="text-muted-foreground text-[10px]">–</span>
        <input
          type="number"
          value={max}
          onChange={e => {
            const v = e.target.value
            if (!v && !min) onClear()
            else onUpdate({ type: 'range', min, max: v })
          }}
          placeholder="Max"
          className="h-7 w-full px-2 text-[11px] rounded border border-border/40 bg-background/50 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
        />
      </div>
    )
  }

  if (filterType === 'date') {
    const from = filter?.type === 'dateRange' ? filter.from : ''
    const to = filter?.type === 'dateRange' ? filter.to : ''
    return (
      <div className="flex flex-col gap-1">
        <input
          type="datetime-local"
          value={from}
          onChange={e => {
            const v = e.target.value
            if (!v && !to) onClear()
            else onUpdate({ type: 'dateRange', from: v, to })
          }}
          className="h-7 w-full px-2 text-[11px] rounded border border-border/40 bg-background/50 text-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
        />
        <input
          type="datetime-local"
          value={to}
          onChange={e => {
            const v = e.target.value
            if (!v && !from) onClear()
            else onUpdate({ type: 'dateRange', from, to: v })
          }}
          className="h-7 w-full px-2 text-[11px] rounded border border-border/40 bg-background/50 text-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
        />
      </div>
    )
  }

  // Bool & enum: searchable checklist
  const uniqueVals = useMemo(() => {
    const set = new Set<FilterValue>()
    for (const row of rows) {
      const v = row[col.name]
      if (v !== null && v !== undefined && typeof v !== 'object') set.add(v as FilterValue)
      if (set.size >= MAX_ENUM_VALUES) break
    }
    return [...set].sort((a, b) => String(a).localeCompare(String(b)))
  }, [rows, col.name])

  const selected = filter?.type === 'values' ? filter.selected : []
  const filtered = enumSearch
    ? uniqueVals.filter(v => String(v).toLowerCase().includes(enumSearch.toLowerCase()))
    : uniqueVals

  const toggle = (val: FilterValue) => {
    const next = selected.includes(val) ? selected.filter(v => v !== val) : [...selected, val]
    next.length ? onUpdate({ type: 'values', selected: next }) : onClear()
  }

  return (
    <div className="space-y-1">
      {uniqueVals.length > 6 && (
        <input
          type="text"
          value={enumSearch}
          onChange={e => setEnumSearch(e.target.value)}
          placeholder="Search values…"
          className="h-6 w-full px-2 text-[10px] rounded border border-border/30 bg-background/50 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
        />
      )}
      <div className="max-h-32 overflow-y-auto space-y-px">
        {filtered.map(val => {
          const active = selected.includes(val)
          return (
            <label key={String(val)} className="flex items-center gap-1.5 px-1 py-0.5 rounded hover:bg-accent/30 cursor-pointer text-[11px]">
              <input
                type="checkbox"
                checked={active}
                onChange={() => toggle(val)}
                className="rounded border-border/60 h-3 w-3 accent-k-yellow"
              />
              <span className={cn('truncate', active ? 'text-foreground' : 'text-muted-foreground')}>
                {String(val)}
              </span>
            </label>
          )
        })}
        {filtered.length === 0 && (
          <span className="text-[10px] text-muted-foreground px-1">No values</span>
        )}
        {uniqueVals.length >= MAX_ENUM_VALUES && (
          <span className="text-[10px] text-muted-foreground px-1 block">Showing top {MAX_ENUM_VALUES}</span>
        )}
      </div>
    </div>
  )
}

function FilterPanel({ columns, data, filters, onChange, onClose }: {
  columns: DataColumn[]; data: TableData; filters: Filters
  onChange: (f: Filters) => void; onClose: () => void
}) {
  const filterableColumns = useMemo(() =>
    columns.filter(c => !c.is_media && !c.type.toLowerCase().includes('json')).slice(0, 8),
    [columns]
  )

  const activeCount = Object.keys(filters).length

  return (
    <div className="w-64 border-l border-border/60 bg-card/40 flex flex-col shrink-0 overflow-hidden">
      <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-border/40 shrink-0">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-foreground">Filters</span>
          {activeCount > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-k-yellow text-black font-medium">{activeCount}</span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {activeCount > 0 && (
            <button onClick={() => onChange({})} className="text-[10px] text-muted-foreground hover:text-foreground transition-colors px-1.5 py-0.5 rounded hover:bg-accent">
              Clear all
            </button>
          )}
          <button onClick={onClose} className="p-0.5 hover:bg-accent rounded transition-colors">
            <X className="h-3.5 w-3.5 text-muted-foreground" />
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {filterableColumns.map(col => {
          const ft = getColumnFilterType(col.type)
          const hasFilter = !!filters[col.name]
          return (
            <div key={col.name} className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-muted-foreground font-medium truncate">{col.name}</span>
                <span className="text-[9px] text-muted-foreground uppercase shrink-0">{ft}</span>
              </div>
              <FilterControl
                col={col}
                filter={filters[col.name]}
                rows={data.rows}
                onUpdate={f => onChange({ ...filters, [col.name]: f })}
                onClear={() => {
                  const next = { ...filters }
                  delete next[col.name]
                  onChange(next)
                }}
              />
              {hasFilter && (
                <button
                  onClick={() => { const next = { ...filters }; delete next[col.name]; onChange(next) }}
                  className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  Clear
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Column Chips ───────────────────────────────────────────────────────────

const SCHEMA_FILTER_THRESHOLD = 20

type SchemaColKey = 'name' | 'type'
type SchemaColWidths = Record<SchemaColKey, number>
const SCHEMA_COL_DEFAULTS: SchemaColWidths = { name: 120, type: 100 }
const SCHEMA_COL_MIN: SchemaColWidths = { name: 72, type: 72 }
const SCHEMA_COL_MAX = 800
const SCHEMA_COL_INFO_WIDTH = 140
/** Bumped when defaults change so stale localStorage widths do not stick. */
const SCHEMA_COLS_STORAGE_KEY = 'pxt-schema-cols-v3'

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

function loadSchemaColWidths(): SchemaColWidths {
  try {
    const raw = localStorage.getItem(SCHEMA_COLS_STORAGE_KEY)
    if (!raw) return SCHEMA_COL_DEFAULTS
    const parsed = JSON.parse(raw) as Partial<SchemaColWidths>
    if (!parsed || typeof parsed !== 'object') return SCHEMA_COL_DEFAULTS
    return {
      name: Number.isFinite(parsed.name) ? clamp(parsed.name as number, SCHEMA_COL_MIN.name, SCHEMA_COL_MAX) : SCHEMA_COL_DEFAULTS.name,
      type: Number.isFinite(parsed.type) ? clamp(parsed.type as number, SCHEMA_COL_MIN.type, SCHEMA_COL_MAX) : SCHEMA_COL_DEFAULTS.type,
    }
  } catch {
    return SCHEMA_COL_DEFAULTS
  }
}

function ChipTypeLabel({ type }: { type: string }) {
  const meta = getColumnTypeMeta(type)
  const Icon = meta.icon
  const label = formatColumnTypeDisplay(type, 24)
  return (
    <span
      className={cn(
        'inline-flex items-center gap-0.5 text-[10px] font-mono truncate max-w-[7rem]',
        meta.emphasis === 'media' ? meta.color : 'text-muted-foreground opacity-70',
      )}
      title={label !== type ? type : undefined}
    >
      <Icon className="h-2.5 w-2.5 shrink-0" />
      <span className="truncate">{label}</span>
    </span>
  )
}

function ColResizeHandle({ atMin, getStartWidth, onResize, onReset }: {
  atMin: boolean
  getStartWidth: () => number
  onResize: (newWidth: number) => void
  onReset: () => void
}) {
  return (
    <div
      onPointerDown={(e) => {
        e.preventDefault()
        e.stopPropagation()
        const el = e.currentTarget
        const pointerId = e.pointerId
        el.setPointerCapture(pointerId)
        const startX = e.clientX
        const startW = getStartWidth()
        const onMove = (m: PointerEvent) => onResize(startW + (m.clientX - startX))
        const cleanup = () => {
          el.removeEventListener('pointermove', onMove)
          el.removeEventListener('pointerup', cleanup)
          el.removeEventListener('pointercancel', cleanup)
          if (el.hasPointerCapture(pointerId)) el.releasePointerCapture(pointerId)
        }
        el.addEventListener('pointermove', onMove)
        el.addEventListener('pointerup', cleanup)
        el.addEventListener('pointercancel', cleanup)
      }}
      onDoubleClick={(e) => { e.preventDefault(); e.stopPropagation(); onReset() }}
      className="group absolute right-0 top-0 h-full w-2 cursor-col-resize select-none flex items-center justify-end"
      title="Drag to resize · double-click to reset"
    >
      <div className={cn(
        'w-px h-full transition-all',
        atMin
          ? 'bg-destructive/70 group-hover:bg-destructive group-hover:w-0.5'
          : 'bg-border group-hover:bg-foreground/60 group-hover:w-0.5',
      )} />
    </div>
  )
}

function ColumnChips({ columns, indices, tableMediaValidation, expanded, onToggle }: {
  columns: ColumnInfo[]; indices: IndexInfo[]; tableMediaValidation: 'on_read' | 'on_write'; expanded: boolean; onToggle: () => void
}) {
  const [filter, setFilter] = useState('')
  const [expandedExpr, setExpandedExpr] = useState<string | null>(null)
  const [colWidths, setColWidths] = useState<SchemaColWidths>(() => loadSchemaColWidths())
  useEffect(() => {
    localStorage.setItem(SCHEMA_COLS_STORAGE_KEY, JSON.stringify(colWidths))
  }, [colWidths])

  const handleResize = useCallback(
    (key: SchemaColKey) => (newWidth: number) =>
      setColWidths(w => {
        const next = clamp(newWidth, SCHEMA_COL_MIN[key], SCHEMA_COL_MAX)
        return w[key] === next ? w : { ...w, [key]: next }
      }),
    [],
  )
  const resetCol = useCallback(
    (key: SchemaColKey) => setColWidths(w => ({ ...w, [key]: SCHEMA_COL_DEFAULTS[key] })),
    [],
  )
  const isDerived = (c: ColumnInfo) => c.is_computed || c.is_iterator_col
  const mutableCount = columns.filter(c => !isDerived(c)).length
  const computedCount = columns.filter(isDerived).length
  const computedStoredCount = columns.filter(c => isDerived(c) && c.is_stored).length
  const showFilter = columns.length >= SCHEMA_FILTER_THRESHOLD

  const filtered = useMemo(() => {
    if (!filter) return columns
    const q = filter.toLowerCase()
    return columns.filter(c => c.name.toLowerCase().includes(q) || c.type_.toLowerCase().includes(q))
  }, [columns, filter])

  const hasInfo = useMemo(
    () => columns.some(col =>
      Boolean(col.comment)
      || Boolean(col.media_validation && col.media_validation !== tableMediaValidation)
      || Boolean(col.is_computed && !col.is_stored)
      || Boolean(col.destination),
    ),
    [columns, tableMediaValidation],
  )

  return (
    <div className="border-b border-border/40 flex flex-col h-full min-h-0 bg-card">
      <div data-schema-chrome className="flex items-center gap-2 px-4 py-2 shrink-0">
        <button
          onClick={onToggle}
          className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors"
        >
          <ChevronDown className={cn('h-3 w-3 transition-transform', !expanded && '-rotate-90')} />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Columns
          </span>
        </button>
        <span className="text-[11px] text-muted-foreground tabular-nums">{columns.length} columns</span>
        <span className="text-[11px] text-muted-foreground">·</span>
        <span className="text-[11px] text-muted-foreground tabular-nums">{mutableCount} mutable</span>
        {computedCount > 0 && (
          <>
            <span className="text-[11px] text-muted-foreground">·</span>
            <span className="text-[11px] text-muted-foreground tabular-nums">
              {computedCount} computed ({computedStoredCount} stored)
            </span>
          </>
        )}
        {indices.length > 0 && (
          <>
            <span className="text-[11px] text-muted-foreground">·</span>
            <span className="text-[11px] text-muted-foreground tabular-nums">
              {indices.length} {indices.length === 1 ? 'index' : 'indices'}
            </span>
          </>
        )}
        {showFilter && (
          <div className="ml-auto relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
            <input
              type="text"
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder={`Filter ${columns.length} columns…`}
              className="h-6 w-44 pl-7 pr-2 text-[11px] rounded border border-border/40 bg-background/50 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring/30"
            />
            {filter && (
              <button onClick={() => setFilter('')} className="absolute right-1.5 top-1/2 -translate-y-1/2">
                <X className="h-3 w-3 text-muted-foreground hover:text-foreground" />
              </button>
            )}
          </div>
        )}
      </div>

      {!expanded && (
        <div data-schema-scroll className="overflow-y-auto px-4 pb-2.5 flex-1 min-h-0">
          <div className="flex flex-wrap gap-1.5">
            {filtered.map(col => (
              <div
                key={col.name}
                className={cn(
                  'group flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px] border transition-colors',
                  col.is_computed
                    ? 'border-amber-500/20 bg-amber-500/5 text-muted-foreground hover:bg-amber-500/10'
                    : 'border-border/40 bg-muted/20 text-muted-foreground hover:bg-muted/40',
                )}
                title={[
                  col.is_computed && col.computed_with ? `${col.name}: ${col.computed_with}` : col.name,
                  col.destination ? `→ ${col.destination}` : '',
                  col.media_validation ? `validation: ${col.media_validation}` : '',
                ].filter(Boolean).join('\n')}
              >
                {col.is_primary_key && <Key className="h-2.5 w-2.5 text-foreground shrink-0" />}
                {!col.is_primary_key && col.is_iterator_col && (
                  <SquareFunction
                    className="h-2.5 w-2.5 text-violet-400 shrink-0"
                    aria-label="iterator"
                  />
                )}
                {!col.is_primary_key && !col.is_iterator_col && col.is_computed && (
                  <SquareFunction
                    className={cn('h-2.5 w-2.5 text-k-yellow shrink-0', !col.is_stored && 'opacity-50')}
                    aria-label={col.is_stored ? 'computed' : 'computed on demand (not stored)'}
                  />
                )}
                <span className="font-mono font-medium">{col.name}</span>
                <ChipTypeLabel type={col.type_} />
                {col.destination && <ExternalLink className="h-2.5 w-2.5 text-muted-foreground shrink-0" />}
              </div>
            ))}
            {filter && filtered.length === 0 && (
              <span className="text-[11px] text-muted-foreground py-1">No columns match "{filter}"</span>
            )}
          </div>
        </div>
      )}

      {expanded && (
        <div data-schema-scroll className="border-t border-border/30 overflow-y-auto overflow-x-auto flex-1 min-h-0">
          <div className="px-4 py-1">
            <table className="w-full text-[11px] table-fixed">
              <colgroup>
                <col style={{ width: colWidths.name }} />
                <col style={{ width: colWidths.type }} />
                <col />
                {hasInfo && <col style={{ width: SCHEMA_COL_INFO_WIDTH }} />}
              </colgroup>
              <thead className="sticky top-0 bg-card z-10">
                <tr className="border-b border-border/30 text-left text-muted-foreground">
                  <th className="relative py-1.5 px-2 font-medium overflow-visible">
                    Name
                    <ColResizeHandle
                      atMin={colWidths.name <= SCHEMA_COL_MIN.name}
                      getStartWidth={() => colWidths.name}
                      onResize={handleResize('name')}
                      onReset={() => resetCol('name')}
                    />
                  </th>
                  <th className="relative py-1.5 px-2 font-medium overflow-visible">
                    Type
                    <ColResizeHandle
                      atMin={colWidths.type <= SCHEMA_COL_MIN.type}
                      getStartWidth={() => colWidths.type}
                      onResize={handleResize('type')}
                      onReset={() => resetCol('type')}
                    />
                  </th>
                  <th className="py-1.5 px-2 font-medium">Computed With</th>
                  {hasInfo && (
                    <th className="py-1.5 px-2 font-medium truncate" title="Comment, validation, storage, destination">
                      Info
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {filtered.map(col => (
                  <tr key={col.name} className="border-b border-border/20 hover:bg-accent/20 transition-colors">
                    <td className="py-1.5 px-2 overflow-hidden" title={col.name}>
                      <div className="flex items-center gap-1.5">
                        {col.is_primary_key && <Key className="h-3 w-3 text-foreground shrink-0" />}
                        {col.is_iterator_col && (
                          <SquareFunction
                            className="h-3 w-3 text-violet-400 shrink-0"
                            aria-label="iterator"
                          />
                        )}
                        {col.is_computed && (
                          <SquareFunction
                            className={cn('h-3 w-3 text-k-yellow shrink-0', !col.is_stored && 'opacity-50')}
                            aria-label={col.is_stored ? 'computed' : 'computed on demand (not stored)'}
                          />
                        )}
                        <span className="font-mono font-medium text-foreground truncate">{col.name}</span>
                      </div>
                    </td>
                    <td className="py-1.5 px-2 overflow-hidden" title={col.type_}>
                      <ColumnTypeBadge type={col.type_} />
                    </td>
                    <td className="py-1.5 px-2 overflow-hidden">
                      {col.is_computed && col.computed_with ? (() => {
                        const expr = col.computed_with
                        const isLong = expr.includes('\n') || expr.length > 60
                        return (
                          <div
                            className={cn(
                              'bg-accent/50 px-2 py-0.5 rounded line-clamp-3 overflow-hidden',
                              isLong && 'cursor-pointer hover:bg-accent/80 transition-colors',
                            )}
                            title={isLong ? 'Click to expand' : expr}
                            onClick={isLong ? () => setExpandedExpr(expr) : undefined}
                          >
                            <PythonExpr code={expr} className="text-[11px] font-mono leading-relaxed whitespace-pre-wrap" />
                          </div>
                        )
                      })() : (
                        <span className="text-muted-foreground text-[11px]">—</span>
                      )}
                    </td>
                    {hasInfo && (
                      <td className="py-1.5 px-2 text-[11px] text-muted-foreground overflow-hidden">
                        <div className="flex flex-col gap-y-0.5 min-w-0">
                          {col.comment && (
                            <span className="italic truncate" title={col.comment}>{col.comment}</span>
                          )}
                          {col.media_validation && col.media_validation !== tableMediaValidation && (
                            <span className="truncate">media validation: {col.media_validation.replace(/_/g, ' ')}</span>
                          )}
                          {col.is_computed && !col.is_stored && (
                            <span>stored: False</span>
                          )}
                          {col.destination && (
                            <span className="font-mono truncate" title={col.destination}>destination: {col.destination}</span>
                          )}
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
                {filter && filtered.length === 0 && (
                  <tr><td colSpan={hasInfo ? 4 : 3} className="py-3 text-center text-muted-foreground text-[11px]">
                    No columns match "{filter}"
                  </td></tr>
                )}
              </tbody>
            </table>

            {indices.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border/30">
                <div className="flex items-center gap-1.5 mb-2">
                  <Search className="h-3 w-3 text-muted-foreground" />
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Indices
                  </span>
                  <span className="text-[11px] text-muted-foreground tabular-nums">{indices.length}</span>
                </div>
                <div className="space-y-1.5">
                  {indices.map(idx => (
                    <div key={idx.name} className="flex items-center gap-2 text-[11px] px-2 py-1 rounded bg-accent/30">
                      <span className="font-mono font-medium text-foreground">{idx.name}</span>
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-500/10 text-muted-foreground">{idx.index_type}</span>
                      <span className="text-muted-foreground">on</span>
                      <span className="font-mono text-foreground">{idx.columns.join(', ')}</span>
                      {idx.parameters && Object.keys(idx.parameters).length > 0 && (
                        <span className="text-[10px] text-muted-foreground ml-auto">
                          {Object.entries(idx.parameters)
                            .filter(([k]) => k === 'metric' || k === 'embedding')
                            .map(([k, v]) => `${k}: ${String(v).slice(0, 30)}`)
                            .join(' · ')}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {filter && (
        <div className="px-4 pb-1.5 text-[10px] text-muted-foreground">
          Showing {filtered.length} of {columns.length} columns
        </div>
      )}
      {expandedExpr && <CellDetail value={expandedExpr} onClose={() => setExpandedExpr(null)} pythonHighlight />}
    </div>
  )
}

// ── Table Header ───────────────────────────────────────────────────────────

function SdkSnippet({ metadata }: { metadata: TableMetadata }) {
  const [copied, setCopied] = useState(false)
  const path = metadata.path
  const cols = Object.values(metadata.columns).filter(c => !c.is_primary_key).slice(0, 3).map(c => c.name)
  const snippet = [
    `import pixeltable as pxt`,
    ``,
    `t = pxt.get_table('${path}')`,
    `t.select(${cols.map(c => `t.${c}`).join(', ')}).head(5)`,
  ].join('\n')

  const handleCopy = () => {
    navigator.clipboard.writeText(snippet).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <div className="mt-2 rounded-md border border-border/60 bg-background/80 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-border/40 bg-accent/30">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Python SDK</span>
        <button onClick={handleCopy} className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors">
          <Copy className="h-2.5 w-2.5" />
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="px-3 py-2 text-[11px] font-mono text-foreground leading-relaxed select-text overflow-x-auto">{snippet}</pre>
    </div>
  )
}

function TableHeader({ metadata, rowCount }: { metadata: TableMetadata; rowCount: number | null }) {
  const [showSnippet, setShowSnippet] = useState(false)
  const kind = metadata.kind

  return (
    <div className="px-4 pt-3 pb-2.5 border-b border-border/40 shrink-0 bg-card">
      <div className="flex items-center gap-2.5 mb-0.5">
        <KindIcon kind={kind} className="h-4 w-4" />
        <span className="text-[11px] font-medium text-muted-foreground shrink-0">{kindWord(kind)}</span>
        <h2 className="text-sm font-semibold text-foreground">{metadata.name}</h2>
        <span className="text-xs font-mono text-muted-foreground">({metadata.path})</span>
        <span className="text-[11px] text-muted-foreground">·</span>
        <span className="text-[11px] text-muted-foreground tabular-nums">
          {rowCount != null ? rowCount.toLocaleString() : '—'} rows
        </span>
        <button
          onClick={() => setShowSnippet(!showSnippet)}
          className={cn(
            'ml-auto flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium transition-colors border',
            showSnippet
              ? 'bg-k-yellow/10 text-foreground border-k-yellow/20'
              : 'bg-accent/50 text-muted-foreground border-border/40 hover:text-foreground hover:bg-accent',
          )}
          title="Show Python SDK snippet"
        >
          {'</>'}
        </button>
      </div>
      {metadata.comment && (
        <div className="ml-6.5 mt-0.5 text-[11px] text-muted-foreground">— {metadata.comment}</div>
      )}
      {showSnippet && <SdkSnippet metadata={metadata} />}
    </div>
  )
}

// ── Lineage Panel ─────────────────────────────────────────────────────────

function LineagePanel({ mode, tablePath, pipelineData, pipelineColumns, onTableClick, onViewFullLineage }: {
  mode: 'table' | 'column'
  tablePath: string
  pipelineData: { nodes: PipelineNodeType[]; edges: PipelineEdge[] } | null
  pipelineColumns: PipelineColumn[] | null
  onTableClick: (path: string) => void
  onViewFullLineage: () => void
}) {
  if (!pipelineData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-5 h-5 border-2 border-k-yellow border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const nodeByPath = new Map(pipelineData.nodes.map(n => [n.path, n]))
  const currentNode = nodeByPath.get(tablePath) ?? null
  // Walk base chain root-first.
  const ancestorChain: PipelineNodeType[] = []
  let cursor = currentNode?.base ? nodeByPath.get(currentNode.base) : null
  while (cursor) {
    ancestorChain.unshift(cursor)
    cursor = cursor.base ? nodeByPath.get(cursor.base) ?? null : null
  }
  const childrenOf = (path: string) => pipelineData.nodes.filter(n => n.base === path)
  const hasColumnDeps = pipelineColumns?.some(c => c.depends_on && c.depends_on.length > 0)

  const NodeCard = ({ node, label, isCurrent }: { node: PipelineNodeType; label?: string; isCurrent?: boolean }) => {
    return (
      <div
        className={cn(
          'rounded-lg border p-3 min-w-[180px] transition-colors',
          isCurrent
            ? 'border-foreground/40 bg-muted/40'
            : 'border-border/60 bg-card/50 hover:border-border cursor-pointer',
        )}
        onClick={() => !isCurrent && onTableClick(node.path)}
      >
        {label && <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">{label}</div>}
        <div className="flex items-center gap-2">
          <KindIcon kind={node.is_view ? 'view' : 'table'} className="h-3.5 w-3.5" />
          <span className={cn('text-xs font-medium font-mono', isCurrent ? 'text-foreground' : 'text-foreground hover:underline')}>
            {node.name}
          </span>
        </div>
        <div className="flex items-center gap-3 mt-1.5 text-[11px] text-muted-foreground">
          <span className="tabular-nums">{node.row_count.toLocaleString()} rows</span>
        </div>
        <div className="flex items-center gap-1 mt-1 text-[11px] text-muted-foreground">
          <span className="tabular-nums">{node.insertable_count} mutable</span>
          {node.computed_count > 0 && (
            <>
              <span className="text-muted-foreground">·</span>
              <span className="tabular-nums">{node.computed_count} computed</span>
            </>
          )}
        </div>
      </div>
    )
  }

  if (mode === 'table') {
    const DescendantTree = ({ path }: { path: string }) => {
      const children = childrenOf(path)
      if (children.length === 0) return null
      return (
        <div className="flex flex-col gap-2">
          {children.map(child => (
            <div key={child.path} className="flex items-start gap-3">
              <NodeCard node={child} />
              {childrenOf(child.path).length > 0 && (
                <>
                  <ArrowRight className="h-4 w-4 text-muted-foreground mt-6 shrink-0" />
                  <DescendantTree path={child.path} />
                </>
              )}
            </div>
          ))}
        </div>
      )
    }
    const hasAncestors = ancestorChain.length > 0
    const hasDescendants = currentNode != null && childrenOf(currentNode.path).length > 0

    return (
      <div className="flex flex-col h-full overflow-auto">
        <div className="px-5 py-4">
          <div className="flex items-center justify-end mb-3">
            <button onClick={onViewFullLineage} className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
              <GitBranch className="h-3 w-3" />
              View full pipeline
            </button>
          </div>
          <div className="flex items-start gap-3 flex-wrap">
            {ancestorChain.map((n, i) => (
              <div key={n.path} className="flex items-start gap-3">
                <NodeCard node={n} label={i === 0 ? 'Root' : 'Base'} />
                <ArrowRight className="h-4 w-4 text-muted-foreground mt-6 shrink-0" />
              </div>
            ))}
            {currentNode && <NodeCard node={currentNode} label={hasAncestors ? 'Current' : undefined} isCurrent />}
            {hasDescendants && currentNode && (
              <>
                <ArrowRight className="h-4 w-4 text-muted-foreground mt-6 shrink-0" />
                <DescendantTree path={currentNode.path} />
              </>
            )}
          </div>
          {!hasAncestors && !hasDescendants && currentNode && (
            <p className="text-[11px] text-muted-foreground mt-2">This is a standalone table with no base or derived views.</p>
          )}
        </div>
      </div>
    )
  }

  // mode === 'column'
  if (hasColumnDeps && pipelineColumns) {
    return (
      <div className="h-full min-h-0">
        <ColumnFlowDiagram columns={pipelineColumns} />
      </div>
    )
  }
  return (
    <div className="h-full flex flex-col items-center justify-center py-16 text-muted-foreground">
      <GitBranch className="h-8 w-8 text-muted-foreground mb-2" />
      <p className="text-xs">No computed column dependencies to visualize</p>
    </div>
  )
}

// ── History Panel ─────────────────────────────────────────────────────────

function HistoryPanel({ versions }: { versions: Pick<PipelineVersion, 'version' | 'created_at' | 'change_type' | 'inserts' | 'updates' | 'deletes' | 'errors'>[] }) {
  if (versions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
        <Clock className="h-8 w-8 text-muted-foreground mb-2" />
        <p className="text-xs">No version history available</p>
      </div>
    )
  }

  const totalInserts = versions.reduce((s, v) => s + v.inserts, 0)
  const totalErrors = versions.reduce((s, v) => s + v.errors, 0)

  return (
    <div className="flex flex-col h-full overflow-auto">
      <div className="px-5 py-3 border-b border-border/30 flex items-center gap-4">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Version History
        </span>
        <span className="text-[11px] text-muted-foreground tabular-nums">{versions.length} versions</span>
        <span className="text-[11px] text-muted-foreground tabular-nums">{totalInserts.toLocaleString()} total inserts</span>
        {totalErrors > 0 && (
          <span className="text-[11px] text-destructive tabular-nums flex items-center gap-1">
            <AlertTriangle className="h-3 w-3" />{totalErrors.toLocaleString()} errors
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto px-5 py-3">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-card z-10">
            <tr className="border-b border-border/30 text-left text-muted-foreground">
              <th className="py-1.5 px-2 font-medium w-16">Version</th>
              <th className="py-1.5 px-2 font-medium">Change</th>
              <th className="py-1.5 px-2 font-medium text-right">Inserts</th>
              <th className="py-1.5 px-2 font-medium text-right">Updates</th>
              <th className="py-1.5 px-2 font-medium text-right">Deletes</th>
              <th className="py-1.5 px-2 font-medium text-right">Errors</th>
              <th className="py-1.5 px-2 font-medium text-right">Created</th>
            </tr>
          </thead>
          <tbody>
            {versions.map(v => (
              <tr key={v.version} className="border-b border-border/20 hover:bg-accent/20 transition-colors">
                <td className="py-1.5 px-2 font-mono font-medium text-foreground">v{v.version}</td>
                <td className="py-1.5 px-2">
                  {v.change_type ? (
                    <span className={cn(
                      'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium',
                      v.change_type === 'schema' ? 'bg-purple-500/10 text-muted-foreground' :
                      v.change_type === 'data' ? 'bg-emerald-500/10 text-muted-foreground' :
                      'bg-accent text-muted-foreground'
                    )}>
                      {v.change_type}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </td>
                <td className="py-1.5 px-2 text-right tabular-nums">
                  {v.inserts > 0 ? <span className="text-muted-foreground">+{v.inserts.toLocaleString()}</span> : <span className="text-muted-foreground">0</span>}
                </td>
                <td className="py-1.5 px-2 text-right tabular-nums">
                  {v.updates > 0 ? <span className="text-muted-foreground">{v.updates.toLocaleString()}</span> : <span className="text-muted-foreground">0</span>}
                </td>
                <td className="py-1.5 px-2 text-right tabular-nums">
                  {v.deletes > 0 ? <span className="text-muted-foreground">-{v.deletes.toLocaleString()}</span> : <span className="text-muted-foreground">0</span>}
                </td>
                <td className="py-1.5 px-2 text-right tabular-nums">
                  {v.errors > 0 ? (
                    <span className="text-destructive flex items-center justify-end gap-1">
                      <AlertTriangle className="h-3 w-3" />{v.errors.toLocaleString()}
                    </span>
                  ) : <span className="text-muted-foreground">0</span>}
                </td>
                <td className="py-1.5 px-2 text-right text-muted-foreground">
                  {v.created_at ? new Date(v.created_at).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' }) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────

const SCHEMA_PANEL_ABS_MAX = 70
const SCHEMA_PANEL_MIN = 5
/** Extra pixels so the last schema row isn’t flush against the resize handle. */
const SCHEMA_PANEL_PAD_PX = 8
/** Persists COLUMNS expanded (table) vs collapsed (chips) across table switches. */
const SCHEMA_EXPANDED_KEY = 'pxt-schema-expanded'

function loadSchemaExpanded(): boolean {
  try {
    return localStorage.getItem(SCHEMA_EXPANDED_KEY) !== '0'
  } catch {
    return true
  }
}

/** Natural schema panel height: fixed chrome + inner content (not the flex-grown scrollport). */
function measureSchemaNaturalHeight(
  groupEl: HTMLElement,
  wrapperEl: HTMLElement,
): { groupHeight: number; naturalHeight: number } | null {
  const scrollableEl = wrapperEl.querySelector<HTMLElement>('[data-schema-scroll]')
  const contentEl = scrollableEl?.firstElementChild as HTMLElement | null
  if (!scrollableEl || !contentEl) return null
  const groupHeight = groupEl.clientHeight
  // Prefer explicit chrome height; fall back to wrapper − scrollport (unreliable while the panel
  // is still at the prior chip size mid-toggle).
  const chromeEl = wrapperEl.querySelector<HTMLElement>('[data-schema-chrome]')
  const fixedHeight = chromeEl
    ? chromeEl.getBoundingClientRect().height
    : Math.max(0, wrapperEl.clientHeight - scrollableEl.clientHeight)
  // scrollHeight survives overflow clipping; getBoundingClientRect can under-report mid-layout.
  const contentHeight = Math.max(
    contentEl.scrollHeight,
    contentEl.offsetHeight,
    contentEl.getBoundingClientRect().height,
  )
  const naturalHeight = fixedHeight + contentHeight
  if (groupHeight === 0 || naturalHeight === 0) return null
  return { groupHeight, naturalHeight }
}

function schemaHeightToPct(naturalHeight: number, groupHeight: number, padPx = 0): number {
  return Math.min(
    SCHEMA_PANEL_ABS_MAX,
    Math.max(SCHEMA_PANEL_MIN, ((naturalHeight + padPx) / groupHeight) * 100),
  )
}

export function TableDetailView({ tablePath }: { tablePath: string }) {
  const navigate = useNavigate()
  // Metadata
  const [metadata, setMetadata] = useState<TableMetadata | null>(null)
  const [metaLoading, setMetaLoading] = useState(true)
  const [metaError, setMetaError] = useState<string | null>(null)

  // Data
  const [data, setData] = useState<TableData | null>(null)
  const [dataLoading, setDataLoading] = useState(true)
  const [dataError, setDataError] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const [orderBy, setOrderBy] = useState<string | null>(null)
  const [orderDesc, setOrderDesc] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('table')
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState<Filters>({})
  const [expandedRow, setExpandedRow] = useState<number | null>(null)
  const [lightboxCell, setLightboxCell] = useState<{ rowIdx: number; colName: string } | null>(null)
  useEffect(() => {
    if (expandedRow === null) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpandedRow(null)
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); setExpandedRow(r => r !== null ? Math.max(0, r - 1) : null) }
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); setExpandedRow(r => r !== null && data ? Math.min(data.rows.length - 1, r + 1) : null) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [expandedRow, data])
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [errorsOnly, setErrorsOnly] = useState(false)
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null)
  const [schemaExpanded, setSchemaExpanded] = useState(loadSchemaExpanded)
  const [schemaMinSize, setSchemaMinSize] = useState(SCHEMA_PANEL_MIN)
  const [schemaMaxSize, setSchemaMaxSize] = useState(SCHEMA_PANEL_ABS_MAX)
  const groupRef = useRef<ImperativePanelGroupHandle>(null)
  const groupContainerRef = useRef<HTMLDivElement>(null)
  const schemaContentRef = useRef<HTMLDivElement>(null)
  const fittingRef = useRef(false)
  const mountedRef = useRef(false)
  /** While true, ResizeObserver may grow an undersized panel (defeats stale autoSave clamp). */
  const forceSchemaFitRef = useRef(false)
  const schemaFitRafRef = useRef<number | null>(null)

  const fitSchemaToContent = useCallback((lockToContent: boolean) => {
    const group = groupRef.current
    const groupEl = groupContainerRef.current
    const wrapperEl = schemaContentRef.current
    if (!group || !groupEl || !wrapperEl) return

    if (schemaFitRafRef.current != null) {
      cancelAnimationFrame(schemaFitRafRef.current)
      schemaFitRafRef.current = null
    }

    forceSchemaFitRef.current = true
    // Unlock first so setLayout is not clamped by chip-mode maxSize from the prior render.
    flushSync(() => {
      setSchemaMinSize(SCHEMA_PANEL_MIN)
      setSchemaMaxSize(SCHEMA_PANEL_ABS_MAX)
    })

    const applyFit = () => {
      const g = groupRef.current
      const gEl = groupContainerRef.current
      const wEl = schemaContentRef.current
      if (!g || !gEl || !wEl) return
      const measured = measureSchemaNaturalHeight(gEl, wEl)
      if (!measured) return
      const pct = schemaHeightToPct(measured.naturalHeight, measured.groupHeight, SCHEMA_PANEL_PAD_PX)
      flushSync(() => {
        setSchemaMaxSize(pct)
        setSchemaMinSize(lockToContent ? pct : SCHEMA_PANEL_MIN)
      })
      fittingRef.current = true
      g.setLayout([pct, 100 - pct])
    }

    // Pass 1: immediate. Pass 2–3: after the panel grows so the table can lay out at full height
    // (and after autoSaveId may briefly restore a stale chip-sized layout).
    applyFit()
    schemaFitRafRef.current = requestAnimationFrame(() => {
      applyFit()
      schemaFitRafRef.current = requestAnimationFrame(() => {
        applyFit()
        schemaFitRafRef.current = null
        forceSchemaFitRef.current = false
        fittingRef.current = false
      })
    })
  }, [])

  // Chevron toggles expanded table ↔ chip summary; panel refits via the metadata/expand effect.
  const toggleSchema = useCallback(() => {
    setSchemaExpanded(prev => {
      const next = !prev
      try {
        localStorage.setItem(SCHEMA_EXPANDED_KEY, next ? '1' : '0')
      } catch { /* ignore */ }
      return next
    })
  }, [])
  const [pipelineColumns, setPipelineColumns] = useState<PipelineColumn[] | null>(null)
  const [pipelineData, setPipelineData] = useState<{ nodes: PipelineNodeType[]; edges: PipelineEdge[] } | null>(null)
  const [contentTab, setContentTab] = useState<'data' | 'lineage-table' | 'lineage-column' | 'history'>('data')
  const [searchQuery, setSearchQuery] = useState('')
  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const PAGE_SIZE_OPTIONS = viewMode === 'gallery' ? [12, 24, 48, 96] : [10, 25, 50, 100]
  const [pageSize, setPageSize] = useState(viewMode === 'gallery' ? 24 : 25)

  // Fetch metadata
  useEffect(() => {
    setMetaLoading(true)
    setMetaError(null)
    getTableMetadata(tablePath)
      .then(setMetadata)
      .catch(e => setMetaError(e instanceof Error ? e.message : 'Failed to load metadata'))
      .finally(() => setMetaLoading(false))
  }, [tablePath])

  // Fetch scoped pipeline data for the lineage/history tabs.
  useEffect(() => {
    if (pipelineData) return
    getPipeline(tablePath)
      .then(p => {
        setPipelineData(p)
        const node = p.nodes.find(n => n.path === tablePath)
        if (node) setPipelineColumns(node.columns)
      })
      .catch(() => {})
  }, [tablePath, pipelineData])

  // Fetch data
  const fetchData = useCallback(() => {
    setDataLoading(true)
    setDataError(null)
    getTableData(tablePath, { offset: page * pageSize, limit: pageSize, orderBy: orderBy || undefined, orderDesc, errorsOnly })
      .then(d => { setData(d); setLastRefreshed(new Date()) })
      .catch(e => setDataError(e instanceof Error ? e.message : 'Failed to load data'))
      .finally(() => setDataLoading(false))
  }, [tablePath, page, pageSize, orderBy, orderDesc, errorsOnly])

  useEffect(() => { fetchData() }, [fetchData])

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      refreshIntervalRef.current = setInterval(fetchData, 10_000)
    }
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }
  }, [autoRefresh, fetchData])

  // Reset on table change (keep schemaExpanded — preference persists across tables).
  useEffect(() => {
    setPage(0); setFilters({}); setAutoRefresh(false); setErrorsOnly(false)
    setSchemaMinSize(SCHEMA_PANEL_MIN)
    setSchemaMaxSize(SCHEMA_PANEL_ABS_MAX)
    forceSchemaFitRef.current = false
    setPipelineColumns(null); setPipelineData(null)
    setContentTab('data'); setSearchQuery('')
    mountedRef.current = false
    // Cancel pending content-fit RAF on table switch and on unmount.
    return () => {
      if (schemaFitRafRef.current != null) {
        cancelAnimationFrame(schemaFitRafRef.current)
        schemaFitRafRef.current = null
      }
    }
  }, [tablePath])

  // Always content-fit when metadata loads or COLUMNS expand mode changes (same path as chevron).
  // Do not skip for pxt-table-layout-customized — table switch must land under the last row/chip.
  useLayoutEffect(() => {
    if (!metadata) return
    fitSchemaToContent(!schemaExpanded)
  }, [metadata, schemaExpanded, fitSchemaToContent])

  // Cap drag so the schema panel cannot grow past its natural content height.
  // In chip-summary mode, also lock minSize so the summarize row can’t be clipped.
  // When expanded, grow if still pinned at max (stale clamp / late content measure) but do not
  // fight an intentional shrink below content.
  const schemaMaxSizeRef = useRef(schemaMaxSize)
  schemaMaxSizeRef.current = schemaMaxSize

  useLayoutEffect(() => {
    if (!metadata) return
    const groupEl = groupContainerRef.current
    if (!groupEl) return

    const updateMax = () => {
      const group = groupRef.current
      const wrapperEl = schemaContentRef.current
      if (!group || !wrapperEl) return

      const measured = measureSchemaNaturalHeight(groupEl, wrapperEl)
      if (!measured) return
      const pct = schemaHeightToPct(measured.naturalHeight, measured.groupHeight, SCHEMA_PANEL_PAD_PX)
      const prevMax = schemaMaxSizeRef.current
      setSchemaMaxSize(pct)
      setSchemaMinSize(schemaExpanded ? SCHEMA_PANEL_MIN : pct)

      const layout = group.getLayout()
      const current = layout[0]
      if (current == null) return
      const over = current > pct + 0.01
      const under = current < pct - 0.01
      const atMax = current >= prevMax - 0.5
      // forceSchemaFitRef: grow even when autoSave left us below max after a chevron toggle.
      if (over || (under && (!schemaExpanded || atMax || forceSchemaFitRef.current))) {
        fittingRef.current = true
        group.setLayout([pct, 100 - pct])
        if (!forceSchemaFitRef.current) {
          queueMicrotask(() => { fittingRef.current = false })
        }
      }
    }

    updateMax()
    const ro = new ResizeObserver(updateMax)
    ro.observe(groupEl)
    const wrapperEl = schemaContentRef.current
    if (wrapperEl) ro.observe(wrapperEl)
    return () => ro.disconnect()
  }, [metadata, schemaExpanded])

  // ⌘F shortcut → focus inline search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'f') {
        e.preventDefault()
        searchInputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const mediaColumn = useMemo(() => data?.columns.find(c => c.is_media), [data])
  const debouncedSearch = useDebounce(searchQuery, 200)

  // Apply both faceted filters and text search
  const filteredRows = useMemo(() => {
    if (!data) return []
    const query = debouncedSearch.toLowerCase().trim()
    return data.rows.filter(row => {
      // Type-aware filtering
      const passesFilters = Object.entries(filters).every(([col, filter]) => {
        const v = row[col]
        if (!filter) return true
        if (filter.type === 'contains') {
          return v !== null && v !== undefined && String(v).toLowerCase().includes(filter.value.toLowerCase())
        }
        if (filter.type === 'values') {
          return filter.selected.includes(v as FilterValue)
        }
        if (filter.type === 'range') {
          const num = typeof v === 'number' ? v : parseFloat(String(v))
          if (isNaN(num)) return false
          if (filter.min !== '' && num < parseFloat(filter.min)) return false
          if (filter.max !== '' && num > parseFloat(filter.max)) return false
          return true
        }
        if (filter.type === 'dateRange') {
          const d = String(v ?? '')
          if (filter.from && d < filter.from) return false
          if (filter.to && d > filter.to) return false
          return true
        }
        return true
      })
      if (!passesFilters) return false
      // Text search
      if (!query) return true
      return data.columns.some(col => {
        const v = row[col.name]
        if (v === null || v === undefined) return false
        return String(v).toLowerCase().includes(query)
      })
    })
  }, [data, filters, debouncedSearch])

  const handleSort = (col: string) => {
    if (orderBy === col) { orderDesc ? (setOrderBy(null), setOrderDesc(false)) : setOrderDesc(true) }
    else { setOrderBy(col); setOrderDesc(false) }
    setPage(0)
  }

  // Loading state
  if (metaLoading && !metadata) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-5 h-5 border-2 border-k-yellow border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  // Error state
  if (metaError) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-destructive">
        <p className="text-sm font-medium">Error loading table</p>
        <p className="text-xs text-muted-foreground mt-1">{metaError}</p>
      </div>
    )
  }

  if (!metadata) return null

  const totalPages = data ? Math.ceil(data.total_count / pageSize) : 0

  return (
    <div className="flex flex-col h-full animate-fade-in bg-card">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <TableHeader metadata={metadata} rowCount={metadata.row_count} />

      <div ref={groupContainerRef} className="flex-1 min-h-0 flex flex-col">
      <PanelGroup
        key={metadata.id}
        ref={groupRef}
        direction="vertical"
        autoSaveId={`pxt-table-layout-${metadata.id}`}
        onLayout={() => {
          if (!mountedRef.current) {
            mountedRef.current = true
            return
          }
          if (fittingRef.current) return
          localStorage.setItem(`pxt-table-layout-customized-${metadata.id}`, '1')
        }}
        className="flex-1 min-h-0"
      >
        {/* ── Schema Panel ──────────────────────────────────────────── */}
        <Panel
          defaultSize={25}
          minSize={schemaMinSize}
          maxSize={schemaMaxSize}
          className="flex flex-col min-h-0 overflow-hidden bg-card"
        >
          <div ref={schemaContentRef} className="flex flex-col h-full min-h-0 bg-card">
          <ColumnChips
            columns={Object.values(metadata.columns)}
            indices={Object.values(metadata.indexes ?? {}).filter(idx => idx.index_type === 'embedding')}
            tableMediaValidation={metadata.media_validation}
            expanded={schemaExpanded}
            onToggle={toggleSchema}
          />
          </div>
        </Panel>

        <PanelResizeHandle className="h-px bg-border/60 hover:h-1 hover:bg-accent transition-all data-[resize-handle-state=drag]:bg-accent data-[resize-handle-state=drag]:h-1 cursor-row-resize" />

        {/* ── Data Panel ────────────────────────────────────────────── */}
        <Panel className="flex flex-col min-h-0 overflow-hidden bg-card">
      {/* ── Content Tab Toggle + Toolbar ──────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2 border-b border-border/40 gap-x-3 gap-y-2 shrink-0 bg-card">
        {/* Left: tab toggle + row count/search */}
        <div className="flex flex-wrap items-center gap-3 flex-1 min-w-0">
          {/* Data / Lineage tabs */}
          <div className="flex bg-accent/50 rounded-md p-0.5 shrink-0">
            <button
              onClick={() => setContentTab('data')}
              className={cn('flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors',
                contentTab === 'data' ? 'bg-card shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground')}
            >
              <Rows3 className="h-3 w-3" />
              Data
            </button>
            <button
              onClick={() => setContentTab('lineage-table')}
              className={cn('flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors',
                contentTab === 'lineage-table' ? 'bg-card shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground')}
            >
              <GitBranch className="h-3 w-3" />
              Table lineage
            </button>
            <button
              onClick={() => setContentTab('lineage-column')}
              className={cn('flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors',
                contentTab === 'lineage-column' ? 'bg-card shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground')}
            >
              <GitBranch className="h-3 w-3" />
              Column lineage
            </button>
            <button
              onClick={() => setContentTab('history')}
              className={cn('flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors',
                contentTab === 'history' ? 'bg-card shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground')}
            >
              <Clock className="h-3 w-3" />
              History
            </button>
          </div>

          {contentTab === 'data' && data && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground shrink-0">
              <span>
                <span className="text-foreground font-medium tabular-nums">{filteredRows.length}</span>
                {(searchQuery || Object.keys(filters).length > 0) && filteredRows.length !== data.rows.length
                  ? <span> matching</span>
                  : null}
                <span> of </span>
                <span className="text-foreground font-medium tabular-nums">{data.total_count.toLocaleString()}</span>
                {(searchQuery || Object.keys(filters).length > 0) && (
                  <span className="text-muted-foreground ml-1" title="Filters and search apply to the current page of data only">
                    (this page)
                  </span>
                )}
              </span>
              <select
                value={pageSize}
                onChange={e => { setPageSize(Number(e.target.value)); setPage(0) }}
                className="h-6 px-1.5 text-[11px] rounded border border-border/40 bg-background/50 text-foreground cursor-pointer focus:outline-none focus:ring-1 focus:ring-ring/30"
                title="Rows per page"
              >
                {PAGE_SIZE_OPTIONS.map(n => (
                  <option key={n} value={n}>{n} / page</option>
                ))}
              </select>
            </div>
          )}
          {/* Inline search (data tab only) */}
          {contentTab === 'data' && (
            <div className="relative flex-1 max-w-xs">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Filter rows on this page…"
                className="w-full bg-accent/30 border border-border/40 rounded-md pl-7 pr-7 py-1 text-xs text-foreground placeholder-muted-foreground/70 outline-none focus:border-border/60 focus:bg-accent/50 transition-colors"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 rounded hover:bg-accent transition-colors"
                >
                  <X className="h-2.5 w-2.5 text-muted-foreground" />
                </button>
              )}
            </div>
          )}
        </div>

        {/* Right: controls (data tab only) */}
        {contentTab === 'data' && <div className="flex flex-wrap items-center gap-1.5 shrink-0">
          {/* Refresh */}
          {lastRefreshed && (
            <span className="text-[11px] text-muted-foreground tabular-nums">
              {lastRefreshed.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={fetchData}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            title="Refresh now"
          >
            <RefreshCw className={cn('h-3 w-3', dataLoading && 'animate-spin')} />
          </button>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={cn(
              'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors',
              autoRefresh
                ? 'bg-emerald-500/10 text-muted-foreground border border-emerald-500/20'
                : 'text-muted-foreground hover:bg-accent hover:text-foreground',
            )}
          >
            <div className={cn(
              'w-1.5 h-1.5 rounded-full',
              autoRefresh ? 'bg-emerald-400 animate-pulse' : 'bg-muted-foreground/50',
            )} />
            Auto
          </button>

          <div className="w-px h-4 bg-border/30" />

          {/* View mode toggle (gallery/table) */}
          {mediaColumn && (
            <div className="flex bg-accent rounded-md p-0.5">
              <button onClick={() => setViewMode('table')} className={cn('p-1 rounded', viewMode === 'table' && 'bg-card shadow-sm')} title="Table view">
                <Table2 className="h-3.5 w-3.5" />
              </button>
              <button onClick={() => setViewMode('gallery')} className={cn('p-1 rounded', viewMode === 'gallery' && 'bg-card shadow-sm')} title="Gallery view">
                <Rows3 className="h-3.5 w-3.5" />
              </button>
            </div>
          )}

          {/* Error rows only toggle */}
          <button
            onClick={() => { setErrorsOnly(!errorsOnly); setPage(0) }}
            className={cn(
              'flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium transition-colors',
              errorsOnly
                ? 'bg-destructive/15 text-destructive border border-destructive/30'
                : 'hover:bg-accent text-muted-foreground',
            )}
            title="Show only rows with errors"
          >
            <AlertTriangle className="h-3 w-3" />
            {errorsOnly && 'Errors'}
          </button>

          {/* Faceted filter toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              'p-1.5 rounded-md transition-colors relative',
              showFilters || Object.keys(filters).length > 0
                ? 'bg-k-yellow text-black'
                : 'hover:bg-accent text-muted-foreground',
            )}
            title="Column filters (applies to current page)"
          >
            <Filter className="h-3.5 w-3.5" />
          </button>

          {/* CSV export */}
          <a
            href={`/api/dashboard/tables/export?path=${encodeURIComponent(tablePath)}&limit=100000`}
            download
            className="p-1.5 rounded-md hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            title="Export CSV (up to 100k rows)"
          >
            <Download className="h-3.5 w-3.5" />
          </a>

          <div className="w-px h-4 bg-border/30" />

          {/* Pagination */}
          <button onClick={() => setPage(0)} disabled={page === 0} className="p-1 rounded hover:bg-accent disabled:opacity-30 transition-colors" title="First page">
            <ChevronsLeft className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0} className="p-1 rounded hover:bg-accent disabled:opacity-30 transition-colors" title="Previous page">
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <span
            className="text-[11px] text-muted-foreground tabular-nums cursor-pointer hover:text-foreground transition-colors px-0.5"
            title="Go to page…"
            onClick={() => {
              const input = prompt(`Go to page (1–${totalPages || 1}):`, String(page + 1))
              if (input !== null) {
                const n = parseInt(input, 10)
                if (!isNaN(n) && n >= 1 && n <= (totalPages || 1)) setPage(n - 1)
              }
            }}
          >
            {page + 1}/{totalPages || 1}
          </span>
          <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1} className="p-1 rounded hover:bg-accent disabled:opacity-30 transition-colors" title="Next page">
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => setPage(Math.max(0, (totalPages || 1) - 1))} disabled={page >= totalPages - 1} className="p-1 rounded hover:bg-accent disabled:opacity-30 transition-colors" title="Last page">
            <ChevronsRight className="h-3.5 w-3.5" />
          </button>
        </div>}
      </div>

      {/* ── Content Area ─────────────────────────────────────────────── */}
      {contentTab === 'lineage-table' || contentTab === 'lineage-column' ? (
        <div className="flex-1 min-h-0 overflow-hidden">
          <LineagePanel
            mode={contentTab === 'lineage-table' ? 'table' : 'column'}
            tablePath={tablePath}
            pipelineData={pipelineData}
            pipelineColumns={pipelineColumns}
            onTableClick={(path) => navigate(tableHref(path))}
            onViewFullLineage={() => navigate('/lineage')}
          />
        </div>
      ) : contentTab === 'history' ? (
        <div className="flex-1 min-h-0 overflow-hidden">
          <HistoryPanel versions={pipelineData?.nodes.find(n => n.path === tablePath)?.versions ?? []} />
        </div>
      ) : (
      <div className="flex flex-1 min-h-0 overflow-hidden bg-card">
        <div className="flex-1 overflow-auto relative bg-card">
          {dataError ? (
            <div className="flex flex-col items-center justify-center py-16 text-destructive">
              <p className="text-sm font-medium">Error loading data</p>
              <p className="text-xs text-muted-foreground mt-1">{dataError}</p>
            </div>
          ) : !data ? (
            <div className="flex items-center justify-center py-16">
              <div className="w-5 h-5 border-2 border-k-yellow border-t-transparent rounded-full animate-spin" />
            </div>
          ) : viewMode === 'gallery' && mediaColumn ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 p-4">
              {filteredRows.map((row, idx) => (
                <GalleryCard key={idx} row={row} columns={data.columns} mediaCol={mediaColumn}
                  onClick={() => setExpandedRow(idx)} />
              ))}
            </div>
          ) : (
            <div className="min-w-full">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-card/95 backdrop-blur-sm z-10">
                  <tr className="border-b border-border/60">
                    {data.columns.map(col => {
                      const sortable = col.is_sorted
                      return (
                        <th
                          key={col.name}
                          onClick={sortable ? () => handleSort(col.name) : undefined}
                          className={cn(
                            'text-left px-3.5 py-2 font-medium text-muted-foreground transition-colors group whitespace-nowrap font-mono',
                            sortable ? 'cursor-pointer hover:text-foreground' : 'cursor-default text-muted-foreground',
                          )}
                          title={sortable ? undefined : (col.is_stored ? 'Not indexed - cannot sort' : 'Unstored - cannot sort')}
                        >
                        <div className="flex items-center gap-1">
                          {col.name}
                          {orderBy === col.name && (orderDesc ? <ChevronDown className="h-3 w-3 text-muted-foreground" /> : <ChevronUp className="h-3 w-3 text-muted-foreground" />)}
                        </div>
                        </th>
                      )
                    })}
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map((row, idx) => {
                    const rowErrors = (row as DataRow)._errors
                    return (
                      <tr key={idx} className={cn('border-b border-border/20 hover:bg-accent/20 transition-colors', rowErrors && 'bg-destructive/[0.03]')}>
                        {data.columns.map(col => (
                          <td key={col.name} className="px-3.5 py-2 align-top">
                            <Cell value={row[col.name]} column={col} error={rowErrors?.[col.name]}
                              onMediaExpand={col.is_media ? () => setLightboxCell({ rowIdx: idx, colName: col.name }) : undefined} />
                          </td>
                        ))}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          {data && filteredRows.length === 0 && (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Rows3 className="h-8 w-8 text-muted-foreground mb-2" />
              <p className="text-xs">
                {searchQuery ? `No rows matching "${searchQuery}"` :
                 Object.keys(filters).length ? 'No rows match filters' :
                 'This table is empty'}
              </p>
              {searchQuery && (
                <button onClick={() => setSearchQuery('')} className="mt-2 text-[11px] text-foreground hover:underline">
                  Clear search
                </button>
              )}
            </div>
          )}

          {dataLoading && data && (
            <div className="absolute inset-0 bg-background/50 flex items-center justify-center">
              <div className="w-5 h-5 border-2 border-k-yellow border-t-transparent rounded-full animate-spin" />
            </div>
          )}
        </div>

        {/* Faceted Filter Panel */}
        {showFilters && data && (
          <FilterPanel columns={data.columns} data={data} filters={filters} onChange={setFilters} onClose={() => setShowFilters(false)} />
        )}
      </div>
      )}

      {/* Expanded Row Modal with arrow navigation */}
      {expandedRow !== null && data && filteredRows[expandedRow] && (
        <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center" onClick={() => setExpandedRow(null)}>
          <button className="absolute top-4 right-4 text-white/70 hover:text-foreground transition-colors z-10" onClick={() => setExpandedRow(null)}>
            <X className="h-7 w-7" />
          </button>
          <div className="absolute top-4 left-1/2 -translate-x-1/2 text-[11px] text-white/50 tabular-nums z-10">
            {expandedRow + 1} / {filteredRows.length}
          </div>
          <button
            onClick={e => { e.stopPropagation(); setExpandedRow(Math.max(0, expandedRow - 1)) }}
            disabled={expandedRow === 0}
            className="absolute left-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white disabled:opacity-20 disabled:cursor-default transition-all z-10"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
          <button
            onClick={e => { e.stopPropagation(); setExpandedRow(Math.min(filteredRows.length - 1, expandedRow + 1)) }}
            disabled={expandedRow >= filteredRows.length - 1}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white disabled:opacity-20 disabled:cursor-default transition-all z-10"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
          <div className="bg-card rounded-lg max-w-4xl max-h-[90vh] overflow-auto border border-border/60 mx-16" onClick={e => e.stopPropagation()}>
            <div className="grid md:grid-cols-2 gap-4 p-4">
              {mediaColumn && filteredRows[expandedRow][mediaColumn.name] != null && (
                <div className="aspect-square bg-background rounded overflow-hidden">
                  {getMediaType(mediaColumn.type) === 'image' ? (
                    <img src={filteredRows[expandedRow][mediaColumn.name] as string} alt="" className="w-full h-full object-contain" />
                  ) : getMediaType(mediaColumn.type) === 'video' ? (
                    <video src={filteredRows[expandedRow][mediaColumn.name] as string} controls className="w-full h-full object-contain" />
                  ) : null}
                </div>
              )}
              <div className="space-y-2">
                {data.columns.filter(c => c.name !== mediaColumn?.name).map(col => {
                  const rowErrors = (filteredRows[expandedRow] as DataRow)._errors
                  return (
                    <div key={col.name} className="flex flex-col">
                      <span className="text-[11px] text-muted-foreground">{col.name}</span>
                      <Cell value={filteredRows[expandedRow][col.name]} column={col} error={rowErrors?.[col.name]} />
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      )}
        </Panel>
      </PanelGroup>
      </div>

      {/* Media Lightbox (from clicking a media cell in table view) */}
      {lightboxCell && data && (() => {
        const col = data.columns.find(c => c.name === lightboxCell.colName)
        if (!col) return null
        const url = filteredRows[lightboxCell.rowIdx]?.[lightboxCell.colName] as string | undefined
        if (!url) return null
        return (
          <MediaLightbox
            url={url}
            type={getMediaType(col.type)}
            index={lightboxCell.rowIdx}
            total={filteredRows.length}
            onClose={() => setLightboxCell(null)}
            onPrev={() => setLightboxCell(prev => prev && prev.rowIdx > 0 ? { ...prev, rowIdx: prev.rowIdx - 1 } : prev)}
            onNext={() => setLightboxCell(prev => prev && prev.rowIdx < filteredRows.length - 1 ? { ...prev, rowIdx: prev.rowIdx + 1 } : prev)}
          />
        )
      })()}
    </div>
  )
}
