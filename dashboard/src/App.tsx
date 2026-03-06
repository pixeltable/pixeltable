import { useState, useEffect } from 'react'
import { Routes, Route, useNavigate, useParams, useLocation } from 'react-router-dom'
import { DirectoryTree } from '@/components/DirectoryTree'
import { TableDetailView } from '@/components/TableDetailView'
import { SearchPanel } from '@/components/SearchPanel'
import { PipelineInspector } from '@/components/PipelineInspector'
import { getDirectoryTree, getStatus } from '@/api/client'
import type { SystemStatus } from '@/api/client'
import type { TreeNode } from '@/types'
import { cn } from '@/lib/utils'
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
} from 'lucide-react'

// ── Table View ──────────────────────────────────────────────────────────────

function TableView() {
  const { '*': tablePath } = useParams()

  if (!tablePath) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <Table2 className="h-12 w-12 text-muted-foreground/15" />
        <div className="text-center">
          <p className="text-sm font-medium text-muted-foreground">Select a table</p>
          <p className="text-xs text-muted-foreground/60 mt-1">
            Browse from the sidebar to inspect schema and data
          </p>
        </div>
      </div>
    )
  }

  return <TableDetailView tablePath={tablePath} />
}

// ── Directory View ──────────────────────────────────────────────────────────

function findTreeNode(nodes: TreeNode[], path: string): TreeNode | null {
  for (const n of nodes) {
    if (n.path === path) return n
    if (n.children) {
      const found = findTreeNode(n.children, path)
      if (found) return found
    }
  }
  return null
}

function flattenTables(node: TreeNode): TreeNode[] {
  const tables: TreeNode[] = []
  for (const c of node.children ?? []) {
    if (c.type === 'directory') tables.push(...flattenTables(c))
    else tables.push(c)
  }
  return tables
}

function DirectoryView({ tree }: { tree: TreeNode[] }) {
  const { '*': dirPath } = useParams()
  const navigate = useNavigate()

  if (!dirPath) return null

  const dirNode = findTreeNode(tree, dirPath)
  if (!dirNode) return (
    <div className="flex flex-col items-center justify-center h-64 gap-2 text-muted-foreground">
      <FolderOpen className="h-8 w-8 opacity-20" />
      <p className="text-sm">Directory not found</p>
    </div>
  )

  const tables = flattenTables(dirNode)
  const totalErrors = tables.reduce((s, t) => s + (t.error_count ?? 0), 0)

  return (
    <div className="flex flex-col h-full p-6 animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <FolderOpen className="h-5 w-5 text-k-yellow/60" />
        <h2 className="text-lg font-semibold text-foreground">{dirNode.name}</h2>
        <span className="text-xs text-muted-foreground font-mono">{dirPath}</span>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="rounded-lg border border-border/40 bg-card/40 p-4">
          <div className="text-2xl font-semibold tabular-nums">{tables.length}</div>
          <div className="text-xs text-muted-foreground mt-1">Tables</div>
        </div>
        <div className="rounded-lg border border-border/40 bg-card/40 p-4">
          <div className={cn('text-2xl font-semibold tabular-nums', totalErrors > 0 && 'text-destructive')}>
            {totalErrors}
          </div>
          <div className="text-xs text-muted-foreground mt-1">Errors</div>
        </div>
      </div>

      {tables.length > 0 && (
        <div className="rounded-lg border border-border/40 overflow-hidden flex-1 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card/95 backdrop-blur-sm z-10">
              <tr className="border-b border-border/30 bg-muted/20">
                <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground">Table</th>
                <th className="text-left py-2 px-3 text-xs font-medium text-muted-foreground">Type</th>
                <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Errors</th>
                <th className="text-right py-2 px-3 text-xs font-medium text-muted-foreground">Version</th>
              </tr>
            </thead>
            <tbody>
              {tables.map(t => (
                <tr key={t.path} className="border-b border-border/20 hover:bg-accent/20 transition-colors cursor-pointer"
                  onClick={() => navigate(`/table/${t.path}`)}>
                  <td className="py-2 px-3 font-mono text-xs font-medium">{t.name}</td>
                  <td className="py-2 px-3 text-xs text-muted-foreground">{t.type}</td>
                  <td className="py-2 px-3 text-xs tabular-nums text-right">
                    {(t.error_count ?? 0) > 0 ? (
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
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <div className="mb-6">
        <img src="/logo.png" alt="Pixeltable" className="h-14 w-14 rounded-xl" />
      </div>
      <h1 className="text-xl font-semibold text-foreground mb-2">
        Pixeltable Dashboard
      </h1>
      <p className="text-sm text-muted-foreground/90 max-w-sm">
        Explore your directories, tables, views, and snapshots.
        Select an item from the sidebar to get started.
      </p>

      {/* Quick actions */}
      <div className="mt-8 flex flex-col items-center gap-3">
        <div className="text-xs text-muted-foreground/90">
          Press{' '}
          <kbd className="px-1.5 py-0.5 bg-accent rounded border border-border text-[11px] text-muted-foreground">⌘K</kbd>
          {' '}to search
        </div>

        <div className="flex items-center gap-4 mt-2">
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
  const [searchOpen, setSearchOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [dark, toggleTheme] = useTheme()
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    getDirectoryTree()
      .then(setTree)
      .catch(console.error)
      .finally(() => setLoading(false))

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
    navigate(type === 'directory' ? `/dir/${path}` : `/table/${path}`)
  }

  const handleSearchSelect = (path: string, type: string) => {
    setSearchOpen(false)
    handleSelectItem(path, type)
  }

  const selectedPath = location.pathname.startsWith('/table/')
    ? location.pathname.replace('/table/', '')
    : location.pathname.startsWith('/dir/')
    ? location.pathname.replace('/dir/', '')
    : null

  const isNavActive = (path: string) => location.pathname === path

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside
        className={cn(
          'flex flex-col border-r border-border/60 bg-card/40 transition-all duration-200 ease-out',
          sidebarOpen ? 'w-[220px]' : 'w-14',
        )}
      >
        {/* Header: logo + connection */}
        <div className={cn('group relative shrink-0 px-3 pt-3 pb-2', !sidebarOpen && 'flex justify-center pt-3 pb-2')}>
          <button onClick={() => navigate('/')} className="flex items-center gap-2.5 hover:opacity-80 transition-opacity">
            <img src="/logo.png" alt="Pixeltable" className="h-7 w-7 shrink-0 rounded-lg" />
            {sidebarOpen && (
              <div className="flex flex-col min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-[13px] font-semibold tracking-tight text-foreground leading-tight">Pixeltable</span>
                  {status && (
                    <span className="text-[10px] text-muted-foreground/50 font-mono leading-tight">v{status.version.split('+')[0]}</span>
                  )}
                </div>
                {status?.config?.home && (
                  <span className="flex items-center gap-1 text-[10px] text-muted-foreground/60 leading-tight mt-0.5">
                    <CircleDot className="h-2 w-2 text-emerald-400 shrink-0" />
                    <span className="truncate">{status.config.home.replace(/^\/Users\/[^/]+\//, '~/')}</span>
                  </span>
                )}
              </div>
            )}
          </button>
          {/* Hover tooltip with full connection details */}
          {sidebarOpen && status?.config && (
            <div className="absolute top-full left-2 mt-0.5 hidden group-hover:block z-50 min-w-[280px] max-w-sm">
              <div className="rounded-lg border border-border/60 bg-card shadow-lg px-3.5 py-3 text-[11px] space-y-2.5">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/50">Connection</div>
                {([
                  ['Home', status.config.home],
                  ['Database', status.config.db_url],
                  ['Media', status.config.media_dir],
                  ['Cache', status.config.file_cache_dir],
                  ['Version', status.version],
                ] as const).map(([label, val]) => (
                  <div key={label}>
                    <div className="text-[10px] text-muted-foreground/60 mb-0.5">{label}</div>
                    <div className="text-foreground font-mono text-[11px] break-all select-text leading-snug">{val}</div>
                  </div>
                ))}
                <div className="flex items-center justify-between pt-1 border-t border-border/30">
                  <span className="text-[10px] text-muted-foreground/60">Tables</span>
                  <span className="text-foreground font-medium tabular-nums">{status.total_tables}</span>
                </div>
                {status.total_errors > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted-foreground/60">Errors</span>
                    <span className="text-destructive font-medium tabular-nums">{status.total_errors}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex flex-1 flex-col px-2 pt-1 min-h-0 overflow-hidden">
          {/* Search button */}
          {sidebarOpen ? (
            <button
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-2.5 w-full rounded-lg px-2.5 py-[7px] mb-1 text-[13px] font-medium text-muted-foreground hover:bg-accent/50 hover:text-foreground transition-colors"
            >
              <Search className="h-[15px] w-[15px] shrink-0" />
              <span className="flex-1 text-left">Search…</span>
              <kbd className="text-[11px] bg-accent px-1.5 py-0.5 rounded border border-border/60 text-muted-foreground">⌘K</kbd>
            </button>
          ) : (
            <button
              onClick={() => setSearchOpen(true)}
              className="flex items-center justify-center rounded-lg px-2.5 py-[7px] mb-1 text-muted-foreground hover:bg-accent/50 hover:text-foreground transition-colors"
              title="Search (⌘K)"
            >
              <Search className="h-[15px] w-[15px]" />
            </button>
          )}

          {/* Lineage nav */}
          <button
            onClick={() => navigate('/lineage')}
            className={cn(
              'group flex items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] font-medium transition-colors',
              sidebarOpen ? '' : 'justify-center',
              isNavActive('/lineage')
                ? 'bg-accent text-foreground'
                : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
            )}
            title={sidebarOpen ? undefined : 'Lineage'}
          >
            <GitBranch className="h-[15px] w-[15px] shrink-0" />
            {sidebarOpen && <span>Lineage</span>}
          </button>

          {/* Divider */}
          <div className={cn('my-1', sidebarOpen ? 'mx-2.5' : 'mx-1')}>
            <div className="h-px bg-border/40" />
          </div>

        {/* Directory tree */}
          <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center py-8">
                <div className="w-5 h-5 border-2 border-k-yellow border-t-transparent rounded-full animate-spin" />
            </div>
            ) : sidebarOpen ? (
            <DirectoryTree
              nodes={tree}
              selectedPath={selectedPath}
              onSelect={handleSelectItem}
            />
            ) : null}
          </div>
        </nav>

        {/* ── Sidebar Footer ─────────────────────────────────────────── */}
        <div className="px-2 pb-2 space-y-0.5 shrink-0">
          {/* Collapse toggle */}
          <button
            className={cn(
              'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] font-medium text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground',
              sidebarOpen ? '' : 'justify-center',
            )}
            onClick={() => setSidebarOpen(!sidebarOpen)}
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
      </aside>

      {/* ── Main Content ────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="flex items-center justify-end gap-1 px-4 py-1.5 border-b border-border/40 shrink-0">
          <a
            href="https://docs.pixeltable.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <BookOpen className="h-3.5 w-3.5" />
            Docs
          </a>
          <a
            href="https://github.com/pixeltable/pixeltable/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <MessageSquare className="h-3.5 w-3.5" />
            Feedback
          </a>
          <div className="w-px h-3.5 bg-border/40 mx-0.5" />
          <button
            onClick={toggleTheme}
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[12px] font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            {dark ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            {dark ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
        <Routes>
          <Route path="/" element={<div className="flex-1 overflow-auto h-full"><WelcomeView /></div>} />
          <Route path="/lineage" element={<PipelineInspector />} />
          <Route path="/table/*" element={<div className="flex-1 flex flex-col h-full"><TableView /></div>} />
          <Route path="/dir/*" element={<div className="flex-1 overflow-auto h-full"><DirectoryView tree={tree} /></div>} />
        </Routes>
      </main>

      {/* ── Search Panel ────────────────────────────────────────────── */}
      <SearchPanel
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
        onSelect={handleSearchSelect}
      />
    </div>
  )
}
