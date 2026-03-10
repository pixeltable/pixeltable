import { useState, useMemo, useEffect } from 'react'
import type { TreeNode } from '@/types'
import { cn } from '@/lib/utils'
import {
  Folder,
  FolderOpen,
  Table2,
  Eye,
  Camera,
  Copy,
  ChevronRight,
  ChevronDown,
  Search,
  X,
  ChevronsDownUp,
  AlertTriangle,
} from 'lucide-react'

interface DirectoryTreeProps {
  nodes: TreeNode[]
  selectedPath: string | null
  onSelect: (path: string, type: string) => void
}

function getNodeIcon(type: string, isOpen: boolean = false) {
  switch (type) {
    case 'directory':
      return isOpen
        ? <FolderOpen className="h-3.5 w-3.5 text-k-yellow shrink-0" />
        : <Folder className="h-3.5 w-3.5 text-k-yellow shrink-0" />
    case 'table':
      return <Table2 className="h-3.5 w-3.5 text-blue-400 shrink-0" />
    case 'view':
      return <Eye className="h-3.5 w-3.5 text-purple-400 shrink-0" />
    case 'snapshot':
      return <Camera className="h-3.5 w-3.5 text-orange-400 shrink-0" />
    case 'replica':
      return <Copy className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
    default:
      return <Table2 className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
  }
}

function countDescendants(node: TreeNode): number {
  if (!node.children || node.children.length === 0) return 0
  return node.children.reduce((sum, child) => sum + 1 + countDescendants(child), 0)
}

function countAllNodes(nodes: TreeNode[]): number {
  return nodes.reduce((sum, n) => sum + 1 + countAllNodes(n.children || []), 0)
}

function nodeMatchesFilter(node: TreeNode, q: string): boolean {
  if (node.name.toLowerCase().includes(q)) return true
  if (node.children) return node.children.some(c => nodeMatchesFilter(c, q))
  return false
}

function TreeItem({ node, level, selectedPath, onSelect, filter, collapsedAll }: {
  node: TreeNode; level: number; selectedPath: string | null
  onSelect: (path: string, type: string) => void; filter: string; collapsedAll: number
}) {
  const [manualOpen, setManualOpen] = useState<boolean | null>(null)
  const hasChildren = node.children && node.children.length > 0
  const isDirectory = node.type === 'directory'
  const descendantCount = useMemo(() => countDescendants(node), [node])
  const hasErrors = (node.error_count ?? 0) > 0

  useEffect(() => {
    if (collapsedAll > 0) setManualOpen(false)
  }, [collapsedAll])

  const isOpen = filter
    ? true
    : manualOpen !== null
      ? manualOpen
      : level === 0

  const isSelected = selectedPath === node.path
  if (filter && !nodeMatchesFilter(node, filter)) return null

  const handleClick = () => {
    if (isDirectory && hasChildren) setManualOpen(!isOpen)
    onSelect(node.path, node.type)
  }

  return (
    <div>
      <button
        className={cn(
          'group flex items-center gap-1.5 w-full rounded-md py-1 px-2 text-left transition-colors',
          isSelected
            ? 'bg-primary/10 text-foreground'
            : 'text-muted-foreground hover:bg-accent hover:text-foreground',
        )}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={handleClick}
        title={`${node.type}: ${node.path}`}
      >
        {isDirectory && hasChildren ? (
          <span className="w-3.5 h-3.5 flex items-center justify-center shrink-0">
            {isOpen
              ? <ChevronDown className="h-3 w-3 text-muted-foreground" />
              : <ChevronRight className="h-3 w-3 text-muted-foreground" />}
          </span>
        ) : (
          <span className="w-3.5 h-3.5 shrink-0" />
        )}

        {getNodeIcon(node.type, isOpen)}
        <span className="flex-1 text-[13px] truncate">{node.name}</span>

        {hasErrors && (
          <span className="flex items-center gap-0.5 text-[10px] text-destructive shrink-0" title={`${node.error_count} errors`}>
            <AlertTriangle className="h-2.5 w-2.5" />
          </span>
        )}

        {isDirectory && descendantCount > 0 && (
          <span className="text-[10px] text-muted-foreground/50 tabular-nums shrink-0">
            {descendantCount}
          </span>
        )}

        {!isDirectory && node.version !== null && node.version !== undefined && (
          <span className="text-[10px] text-muted-foreground/40 tabular-nums shrink-0">
            v{node.version}
          </span>
        )}
      </button>

      {isDirectory && hasChildren && isOpen && (
        <div>
          {node.children!.map((child) => (
            <TreeItem
              key={child.path}
              node={child}
              level={level + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
              filter={filter}
              collapsedAll={collapsedAll}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function DirectoryTree({ nodes, selectedPath, onSelect }: DirectoryTreeProps) {
  const [filter, setFilter] = useState('')
  const [collapsedAll, setCollapsedAll] = useState(0)
  const totalCount = useMemo(() => countAllNodes(nodes), [nodes])
  const showFilter = totalCount >= 10
  const q = filter.toLowerCase()

  if (nodes.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Folder className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p className="text-xs">No directories or tables found</p>
        <p className="text-[11px] mt-1 text-muted-foreground">
          Create tables using the Python SDK
        </p>
      </div>
    )
  }

  return (
    <div>
      {showFilter && (
        <div className="px-2 pb-1.5 flex items-center gap-1">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground/50" />
            <input
              type="text"
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Filter…"
              className="h-6 w-full pl-6 pr-6 text-[11px] rounded border border-border/40 bg-background/50 text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-ring/30"
            />
            {filter && (
              <button onClick={() => setFilter('')} className="absolute right-1.5 top-1/2 -translate-y-1/2">
                <X className="h-3 w-3 text-muted-foreground/50 hover:text-foreground" />
              </button>
            )}
          </div>
          <button
            onClick={() => setCollapsedAll(c => c + 1)}
            className="h-6 w-6 flex items-center justify-center rounded border border-border/40 bg-background/50 text-muted-foreground/50 hover:text-foreground transition-colors shrink-0"
            title="Collapse all"
          >
            <ChevronsDownUp className="h-3 w-3" />
          </button>
        </div>
      )}
      <div className="space-y-px">
        {nodes.map((node) => (
          <TreeItem
            key={node.path}
            node={node}
            level={0}
            selectedPath={selectedPath}
            onSelect={onSelect}
            filter={q}
            collapsedAll={collapsedAll}
          />
        ))}
      </div>
    </div>
  )
}
