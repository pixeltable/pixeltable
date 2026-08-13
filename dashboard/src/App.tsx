import { useState, useEffect, useRef, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { Panel, PanelGroup, PanelResizeHandle, type ImperativePanelHandle } from 'react-resizable-panels'
import { Routes, Route, useNavigate, useParams, useLocation } from 'react-router-dom'
import { DirectoryTreePanel } from '@/components/DirectoryTree'
import { CatalogSwitcher } from '@/components/CatalogSwitcher'
import { TableDetailView } from '@/components/TableDetailView'
import { SearchPanel } from '@/components/SearchPanel'
import { PipelineInspector } from '@/components/PipelineInspector'
import { getDirectoryTree, getStatus } from '@/api/client'
import type { SystemStatus } from '@/api/client'
import type { TableNode, TreeNode } from '@/types'
import {
  cn,
  tableHref,
  dirHref,
  loadActiveCatalog,
  saveActiveCatalog,
  loadExtraCatalogs,
  saveExtraCatalogs,
  catalogRootFromPath,
} from '@/lib/utils'
import {
  Search,
  GitBranch,
  Table2,
  PanelLeftClose,
  PanelLeftOpen,
  ExternalLink,
  BookOpen,
  CircleDot,
  FolderOpen,
  AlertTriangle,
  MessageSquare,
  Sun,
  Moon,
  Eye,
  Camera,
  Copy,
} from 'lucide-react'

function DirectoryKindIcon({ kind }: { kind: string }) {
  switch (kind) {
    case 'view':
      return <Eye className="h-3.5 w-3.5 text-purple-400 shrink-0" />
    case 'snapshot':
      return <Camera className="h-3.5 w-3.5 text-orange-400 shrink-0" />
    case 'replica':
      return <Copy className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
    default:
      return <Table2 className="h-3.5 w-3.5 text-blue-400 shrink-0" />
  }
}

// ── Table View ──────────────────────────────────────────────────────────────

function TableView() {
  const { '*': tablePath } = useParams()

  if (!tablePath) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <Table2 className="h-12 w-12 text-muted-foreground" />
        <div className="text-center">
          <p className="text-sm font-medium text-muted-foreground">Select a table</p>
          <p className="text-xs text-muted-foreground mt-1">
            Browse from the sidebar to inspect schema and data
          </p>
        </div>
      </div>
    )
  }

  return <TableDetailView tablePath={tablePath} />
}

// ── Directory View ──────────────────────────────────────────────────────────

function collectTables(nodes: TreeNode[]): TableNode[] {
  const tables: TableNode[] = []
  for (const n of nodes) {
    if (n.kind === 'directory') tables.push(...collectTables(n.entries ?? []))
    else tables.push(n)
  }
  return tables
}

function DirectoryView() {
  const { '*': dirPath } = useParams()
  const navigate = useNavigate()
  const [nodes, setNodes] = useState<TreeNode[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Resolve the directory by fetching its contents by path, so a directory in any catalog (local or
  // hosted) is listed the same way; the daemon re-roots each node's path to its catalog.
  useEffect(() => {
    if (!dirPath) return
    setNodes(null)
    setError(null)
    getDirectoryTree(dirPath)
      .then(setNodes)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load directory'))
  }, [dirPath])

  if (!dirPath) return null

  if (error !== null) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-2 text-muted-foreground">
        <FolderOpen className="h-8 w-8 opacity-20" />
        <p className="text-sm">Directory not found</p>
        <p className="text-[11px] text-muted-foreground/60 font-mono">{error}</p>
      </div>
    )
  }

  if (nodes === null) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-5 h-5 border-2 border-k-yellow border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const name = dirPath.split('/').pop() || dirPath
  const objects = collectTables(nodes)
  const totalErrors = objects.reduce((s, t) => s + t.error_count, 0)
  const tableCount = objects.filter(t => t.kind === 'table').length
  const viewCount = objects.filter(t => t.kind === 'view').length
  const snapshotCount = objects.filter(t => t.kind === 'snapshot').length
  const replicaCount = objects.filter(t => t.kind === 'replica').length
  const kindBreakdown = [
    tableCount > 0 && `${tableCount} table${tableCount === 1 ? '' : 's'}`,
    viewCount > 0 && `${viewCount} view${viewCount === 1 ? '' : 's'}`,
    snapshotCount > 0 && `${snapshotCount} snapshot${snapshotCount === 1 ? '' : 's'}`,
    replicaCount > 0 && `${replicaCount} replica${replicaCount === 1 ? '' : 's'}`,
  ].filter(Boolean).join(' · ')

  return (
    <div className="flex flex-col h-full p-6 animate-fade-in bg-card">
      <div className="flex items-center gap-3 mb-6">
        <FolderOpen className="h-5 w-5 text-foreground" />
        <h2 className="text-lg font-semibold text-foreground">{name}</h2>
        <span className="text-xs text-muted-foreground font-mono">{dirPath}</span>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="rounded-lg border border-border/40 bg-background/40 p-4">
          <div className="text-2xl font-semibold tabular-nums">{objects.length}</div>
          <div className="text-xs text-muted-foreground mt-1">Objects</div>
          {kindBreakdown && (
            <div className="text-[11px] text-muted-foreground mt-1">{kindBreakdown}</div>
          )}
        </div>
        <div className="rounded-lg border border-border/40 bg-background/40 p-4">
          <div className={cn('text-2xl font-semibold tabular-nums', totalErrors > 0 && 'text-destructive')}>
            {totalErrors}
          </div>
          <div className="text-xs text-muted-foreground mt-1">Errors</div>
        </div>
      </div>

      {objects.length > 0 && (
        <div className="rounded-lg border border-border/40 overflow-hidden flex-1 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card/95 backdrop-blur-sm z-10">
              <tr className="border-b border-border/30 bg-muted/20">
                <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground">Name</th>
                <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground">Type</th>
                <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Errors</th>
                <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Version</th>
              </tr>
            </thead>
            <tbody>
              {objects.map(t => (
                <tr key={t.path} className="border-b border-border/20 hover:bg-accent/20 transition-colors cursor-pointer"
                  onClick={() => navigate(tableHref(t.path))}>
                  <td className="py-2 px-3 font-mono text-xs font-medium">{t.name}</td>
                  <td className="py-2 px-3 text-xs text-muted-foreground">
                    <span className="inline-flex items-center gap-1.5">
                      <DirectoryKindIcon kind={t.kind} />
                      <span className="capitalize">{t.kind}</span>
                    </span>
                  </td>
                  <td className="py-2 px-3 text-xs tabular-nums text-right">
                    {t.error_count > 0 ? (
                      <span className="text-destructive flex items-center justify-end gap-1">
                        <AlertTriangle className="h-3 w-3" />{t.error_count}
                      </span>
                    ) : '—'}
                  </td>
                  <td className="py-2 px-3 text-xs tabular-nums text-right text-muted-foreground">
                    {t.version != null ? `v${t.version}` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── Welcome View ────────────────────────────────────────────────────────────

function WelcomeView() {
  const navigate = useNavigate()
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <div className="mb-6">
        <img src="/logo.png?v=3" alt="Pixeltable" className="h-14 w-14 rounded-xl" />
      </div>
      <h1 className="text-xl font-semibold text-foreground mb-2">
        Pixeltable Dashboard
      </h1>
      <p className="text-sm text-muted-foreground max-w-md leading-relaxed">
        Explore your directories, tables, views, and snapshots.
        Select an item from the sidebar, or view the full pipeline lineage.
      </p>

      {/* Primary actions */}
      <div className="mt-8 flex flex-col items-center gap-5">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/lineage')}
            className="flex items-center gap-2.5 rounded-lg bg-k-yellow text-primary-foreground px-5 py-2.5 text-sm font-semibold hover:bg-k-yellow/90 transition-colors shadow-sm"
          >
            <GitBranch className="h-4 w-4" />
            View pipeline lineage
          </button>
          <button
            onClick={() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true }))}
            className="flex items-center gap-2 rounded-lg border border-border px-4 py-2.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          >
            <Search className="h-4 w-4" />
            Search tables
            <kbd className="ml-1 px-1.5 py-0.5 bg-accent rounded border border-border text-[10px]">⌘K</kbd>
          </button>
        </div>

        <div className="flex items-center gap-4 mt-1">
          <a
            href="https://docs.pixeltable.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <BookOpen className="h-3.5 w-3.5" />
            Documentation
            <ExternalLink className="h-2.5 w-2.5 opacity-60" />
          </a>
          <a
            href="https://github.com/pixeltable/pixeltable"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
            GitHub
            <ExternalLink className="h-2.5 w-2.5 opacity-60" />
          </a>
        </div>
      </div>
    </div>
  )
}

// ── Main App ────────────────────────────────────────────────────────────────

function useTheme() {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return true
    const stored = localStorage.getItem('pxt-theme')
    if (stored) return stored === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('pxt-theme', dark ? 'dark' : 'light')
  }, [dark])
  return [dark, () => setDark(d => !d)] as const
}

export default function App() {
  const [tree, setTree] = useState<TreeNode[]>([])
  const [loading, setLoading] = useState(true)
  const [treeError, setTreeError] = useState<string | null>(null)
  const [searchOpen, setSearchOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const sidebarPanelRef = useRef<ImperativePanelHandle>(null)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [dark, toggleTheme] = useTheme()
  const [activeCatalog, setActiveCatalog] = useState(loadActiveCatalog)
  const [treeReload, setTreeReload] = useState(0)
  const [connOpen, setConnOpen] = useState(false)
  const [connPos, setConnPos] = useState<{ top: number; left: number } | null>(null)
  /** Brand/home hit target only — not Search/Lineage. */
  const brandRef = useRef<HTMLButtonElement>(null)
  const connCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const openConnection = () => {
    if (connCloseTimer.current) {
      clearTimeout(connCloseTimer.current)
      connCloseTimer.current = null
    }
    const el = brandRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setConnPos({ top: r.bottom - 1, left: r.left })
    setConnOpen(true)
  }

  const scheduleCloseConnection = () => {
    if (connCloseTimer.current) clearTimeout(connCloseTimer.current)
    connCloseTimer.current = setTimeout(() => setConnOpen(false), 120)
  }

  useEffect(() => () => {
    if (connCloseTimer.current) clearTimeout(connCloseTimer.current)
  }, [])

  const toggleSidebar = () => {
    const panel = sidebarPanelRef.current
    if (!panel) return
    if (panel.isCollapsed()) panel.expand()
    else panel.collapse()
  }
  const navigate = useNavigate()
  const location = useLocation()

  const handleCatalogSelect = useCallback((uri: string) => {
    setActiveCatalog(uri)
    saveActiveCatalog(uri)
    navigate('/')
  }, [navigate])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setTreeError(null)
    getDirectoryTree(activeCatalog)
      .then(nodes => {
        if (!cancelled) setTree(nodes)
      })
      .catch(err => {
        if (!cancelled) {
          setTree([])
          setTreeError(err instanceof Error ? err.message : String(err))
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [activeCatalog, treeReload])

  useEffect(() => {
    getStatus().then(setStatus).catch(console.error)
  }, [])

  // Global ⌘K shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setSearchOpen(true)
      }
      if (e.key === 'Escape') setSearchOpen(false)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleSelectItem = (path: string, type: string) => {
    navigate(type === 'directory' ? dirHref(path) : tableHref(path))
  }

  const handleSearchSelect = (path: string, type: string) => {
    setSearchOpen(false)
    // Keep chrome aligned with one active catalog when opening a hosted search hit.
    const root = catalogRootFromPath(path)
    if (root !== null) {
      const extras = loadExtraCatalogs()
      if (!extras.includes(root)) saveExtraCatalogs([...extras, root])
      setActiveCatalog(root)
      saveActiveCatalog(root)
    }
    handleSelectItem(path, type)
  }

  // The path is stored as a single encoded URL segment (see tableHref/dirHref), so decode it back to the
  // raw catalog path the tree nodes carry.
  const selectedPath = location.pathname.startsWith('/table/')
    ? decodeURIComponent(location.pathname.replace('/table/', ''))
    : location.pathname.startsWith('/dir/')
    ? decodeURIComponent(location.pathname.replace('/dir/', ''))
    : null

  const isNavActive = (path: string) => location.pathname === path

  return (
    <div className="h-screen overflow-hidden bg-background">
      <PanelGroup direction="horizontal" autoSaveId="pxt-layout" className="h-full">
        {/* ── Sidebar ─────────────────────────────────────────────────── */}
        <Panel
          ref={sidebarPanelRef}
          defaultSize={18}
          minSize={10}
          maxSize={40}
          collapsible
          collapsedSize={3.5}
          onCollapse={() => setSidebarOpen(false)}
          onExpand={() => setSidebarOpen(true)}
          className="flex flex-col border-r border-border/60 bg-card/40"
        >
        {/* Header: brand left (connection hover); Search + Lineage right (no connection hover) */}
        <div
          className={cn(
            'relative shrink-0 px-3 pt-3 pb-2',
            sidebarOpen ? 'flex items-start justify-between gap-1' : 'flex flex-col items-center pt-3 pb-2',
          )}
        >
          <button
            ref={brandRef}
            onClick={() => navigate('/')}
            className="flex min-w-0 items-center gap-2.5 hover:opacity-80 transition-opacity"
            onMouseEnter={() => { if (sidebarOpen && status?.config) openConnection() }}
            onMouseLeave={scheduleCloseConnection}
          >
            <img src="/logo.png?v=3" alt="Pixeltable" className="h-7 w-7 shrink-0 rounded-lg" />
            {sidebarOpen && (
              <div className="flex flex-col min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-[13px] font-semibold tracking-tight text-foreground leading-tight">Pixeltable</span>
                  {status && (
                    <span className="text-[10px] text-muted-foreground font-mono leading-tight">v{status.version.split('+')[0]}</span>
                  )}
                </div>
                {status?.config?.home && (
                  <span className="flex items-center gap-1 text-[10px] text-muted-foreground leading-tight mt-0.5">
                    <CircleDot className="h-2 w-2 text-emerald-400 shrink-0" />
                    <span className="truncate">{status.config.home.replace(/^\/Users\/[^/]+\//, '~/')}</span>
                  </span>
                )}
              </div>
            )}
          </button>
          {sidebarOpen ? (
            <div className="flex shrink-0 items-center gap-0.5 pt-0.5">
              <button
                onClick={() => setSearchOpen(true)}
                className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-accent/50 hover:text-foreground transition-colors"
                title="Search (⌘K)"
                aria-label="Search"
              >
                <Search className="h-[15px] w-[15px]" />
              </button>
              <button
                onClick={() => navigate('/lineage')}
                className={cn(
                  'flex h-7 w-7 items-center justify-center rounded-md transition-colors',
                  isNavActive('/lineage')
                    ? 'bg-accent text-foreground'
                    : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                )}
                title="Lineage"
                aria-label="Lineage"
              >
                <GitBranch className="h-[15px] w-[15px]" />
              </button>
            </div>
          ) : (
            <div className="mt-1 flex flex-col items-center gap-0.5">
              <button
                onClick={() => setSearchOpen(true)}
                className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-accent/50 hover:text-foreground transition-colors"
                title="Search (⌘K)"
              >
                <Search className="h-[15px] w-[15px]" />
              </button>
              <button
                onClick={() => navigate('/lineage')}
                className={cn(
                  'flex h-7 w-7 items-center justify-center rounded-md transition-colors',
                  isNavActive('/lineage')
                    ? 'bg-accent text-foreground'
                    : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
                )}
                title="Lineage"
              >
                <GitBranch className="h-[15px] w-[15px]" />
              </button>
            </div>
          )}
          {sidebarOpen && status?.config && connOpen && connPos && createPortal(
            <div
              className="fixed z-[200] min-w-[280px] max-w-sm"
              style={{ top: connPos.top, left: connPos.left }}
              onMouseEnter={openConnection}
              onMouseLeave={scheduleCloseConnection}
            >
              <div className="rounded-lg border border-border/60 bg-card shadow-lg px-3.5 py-3 text-[11px] space-y-2.5">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Connection</div>
                {([
                  ['Home', status.config.home],
                  ['Database', status.config.db_url],
                  ['Media', status.config.media_dir],
                  ['Cache', status.config.file_cache_dir],
                  ['Version', status.version],
                ] as const).map(([label, val]) => (
                  <div key={label}>
                    <div className="text-[10px] text-muted-foreground mb-0.5">{label}</div>
                    <div className="text-foreground font-mono text-[11px] break-all select-text leading-snug">{val}</div>
                  </div>
                ))}
                <div className="flex items-center justify-between pt-1 border-t border-border/30">
                  <span className="text-[10px] text-muted-foreground">Tables</span>
                  <span className="text-foreground font-medium tabular-nums">{status.total_tables}</span>
                </div>
                {status.total_errors > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted-foreground">Errors</span>
                    <span className="text-destructive font-medium tabular-nums">{status.total_errors}</span>
                  </div>
                )}
              </div>
            </div>,
            document.body,
          )}
        </div>

        {/* Navigation: hairline under header (same language as footer) */}
        <nav className="flex min-h-0 flex-1 flex-col overflow-hidden border-t border-border/40 px-2 pt-1.5">
          {/* Sticky catalog switcher (Local / Cloud) */}
          <CatalogSwitcher
            activeCatalog={activeCatalog}
            onSelect={handleCatalogSelect}
            collapsed={!sidebarOpen}
            onExpandRequest={() => sidebarPanelRef.current?.expand()}
          />

          {/* Filter sticky under switcher; tree scrolls inside the panel */}
          {loading ? (
            <div className="flex flex-1 items-center justify-center py-8">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-k-yellow border-t-transparent" />
            </div>
          ) : treeError ? (
            sidebarOpen ? (
              <div className="flex-1 space-y-2 overflow-y-auto px-2 py-3">
                <div className="break-words text-[11px] text-destructive">{treeError}</div>
                <button
                  type="button"
                  onClick={() => setTreeReload(n => n + 1)}
                  className="text-[11px] font-medium text-foreground underline-offset-2 hover:underline"
                >
                  Retry
                </button>
              </div>
            ) : null
          ) : sidebarOpen ? (
            <DirectoryTreePanel
              nodes={tree}
              selectedPath={selectedPath}
              onSelect={handleSelectItem}
            />
          ) : null}
        </nav>

        {/* ── Sidebar Footer ─────────────────────────────────────────── */}
        <div className="px-2 pb-2 pt-1 space-y-0.5 shrink-0 border-t border-border/40">
          <a
            href="https://docs.pixeltable.com"
            target="_blank"
            rel="noopener noreferrer"
            title="Docs"
            className={cn(
              'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground',
              !sidebarOpen && 'justify-center',
            )}
          >
            <BookOpen className="h-[15px] w-[15px] shrink-0" />
            {sidebarOpen && <span>Docs</span>}
          </a>
          <a
            href="https://github.com/pixeltable/pixeltable/issues"
            target="_blank"
            rel="noopener noreferrer"
            title="Feedback"
            className={cn(
              'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground',
              !sidebarOpen && 'justify-center',
            )}
          >
            <MessageSquare className="h-[15px] w-[15px] shrink-0" />
            {sidebarOpen && <span>Feedback</span>}
          </a>
          <button
            type="button"
            onClick={toggleTheme}
            title={dark ? 'Light mode' : 'Dark mode'}
            className={cn(
              'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground',
              !sidebarOpen && 'justify-center',
            )}
          >
            {dark ? <Sun className="h-[15px] w-[15px] shrink-0" /> : <Moon className="h-[15px] w-[15px] shrink-0" />}
            {sidebarOpen && <span>{dark ? 'Light mode' : 'Dark mode'}</span>}
          </button>
          <button
            type="button"
            className={cn(
              'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground',
              !sidebarOpen && 'justify-center',
            )}
            onClick={toggleSidebar}
          >
            {sidebarOpen ? (
              <>
                <PanelLeftClose className="h-[15px] w-[15px] shrink-0" />
                <span>Collapse</span>
              </>
            ) : (
              <PanelLeftOpen className="h-[15px] w-[15px] shrink-0" />
            )}
          </button>
        </div>
        </Panel>

        <PanelResizeHandle className="w-px bg-border/60 hover:w-1 hover:bg-accent transition-all data-[resize-handle-state=drag]:bg-accent data-[resize-handle-state=drag]:w-1 cursor-col-resize" />

        {/* ── Main Content ────────────────────────────────────────────── */}
        <Panel className="flex flex-col min-h-0 overflow-hidden bg-card">
        <Routes>
          <Route path="/" element={<div className="flex-1 overflow-auto h-full"><WelcomeView /></div>} />
          <Route path="/lineage" element={<PipelineInspector />} />
          <Route path="/table/*" element={<div className="flex-1 flex flex-col h-full bg-card"><TableView /></div>} />
          <Route path="/dir/*" element={<div className="flex-1 overflow-auto h-full"><DirectoryView /></div>} />
        </Routes>
        </Panel>
      </PanelGroup>

      {/* ── Search Panel ────────────────────────────────────────────── */}
      <SearchPanel
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
        onSelect={handleSearchSelect}
      />
    </div>
  )
}
