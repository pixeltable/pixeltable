import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type EdgeProps,
  Position,
  Handle,
  useNodesState,
  useEdgesState,
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  Loader2,
  Table2,
  Eye,
  AlertTriangle,
  Rows3,
  GitBranch,
  ChevronDown,
  X,
  Zap,
  Search as SearchIcon,
  ArrowLeft,
  RefreshCw,
  ExternalLink,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { getPipeline } from '@/api/client'
import type { PipelineNode as PipelineNodeType, PipelineColumn, PipelineResponse } from '@/types'
import { FUNC_STYLES } from '@/lib/func-styles'
import { ColumnTypeBadge, ColumnTypeIcon } from '@/lib/column-types'
import { ColumnFlowDiagram } from './ColumnFlowDiagram'
import { PythonExpr } from '@/lib/python-highlight'

// ── Custom Edge ──────────────────────────────────────────────────────────────

function LabeledEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 16,
  })

  const edgeData = data as { label?: string; edgeType?: string } | undefined
  const label = edgeData?.label
  const isQuery = edgeData?.edgeType === 'query'

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: isQuery ? '#888' : '#333',
          strokeWidth: isQuery ? 1 : 1.5,
          strokeDasharray: isQuery ? '5 3' : undefined,
          opacity: isQuery ? 0.5 : 0.8,
        }}
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
            className={cn(
              'absolute pointer-events-all px-1.5 py-0.5 rounded border text-[10px] font-mono',
              isQuery
                ? 'bg-card border-border/40 text-muted-foreground'
                : 'bg-card border-border/60 text-foreground',
            )}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

const edgeTypes = { labeled: LabeledEdge }

// ── Custom Node ──────────────────────────────────────────────────────────────

interface TableNodeData extends PipelineNodeType {
  isSelected: boolean
  onSelect: (path: string) => void
  hasIncoming: boolean
  hasOutgoing: boolean
  [key: string]: unknown
}

function PipelineTableNode({ data }: { data: TableNodeData }) {
  const hasErrors = data.total_errors > 0
  const computedCols = data.columns.filter((c) => c.is_computed)
  const insertableCols = data.columns.filter((c) => !c.is_computed)

  return (
    <div
      className={cn(
        'rounded-lg border bg-card shadow-sm min-w-[200px] max-w-[240px] cursor-pointer transition-all',
        hasErrors ? 'border-destructive/30' : 'border-border/60',
        data.isSelected && 'ring-1 ring-k-yellow/50 border-k-yellow/30',
      )}
      onClick={() => data.onSelect(data.path)}
    >
      {data.hasIncoming && (
        <Handle type="target" position={Position.Top} className="!bg-border !w-2 !h-2 !-top-1" />
      )}

      {/* Header */}
      <div className="px-3 py-2 border-b border-border/40">
        <div className="flex items-center gap-1.5">
          {data.is_view ? (
            <Eye className="h-3 w-3 text-muted-foreground shrink-0" />
          ) : (
            <Table2 className="h-3 w-3 text-muted-foreground shrink-0" />
          )}
          <span className="text-xs font-semibold text-foreground truncate">{data.name}</span>
        </div>
        <div className="flex items-center gap-2 mt-1 flex-wrap">
          <span className="text-[10px] text-muted-foreground tabular-nums">{data.row_count.toLocaleString()} rows</span>
          <span className="text-[10px] text-muted-foreground/80">v{data.version}</span>
          {data.indices.length > 0 && (
            <span className="text-[10px] text-muted-foreground flex items-center gap-0.5">
              <SearchIcon className="h-2.5 w-2.5" />{data.indices.length}
            </span>
          )}
          {hasErrors && (
            <span className="text-[10px] text-destructive flex items-center gap-0.5">
              <AlertTriangle className="h-2.5 w-2.5" />{data.total_errors}
            </span>
          )}
          {data.iterator_type && (
            <span className="text-[10px] px-1 rounded bg-accent text-muted-foreground font-mono">
              {data.iterator_type}
            </span>
          )}
        </div>
      </div>

      {/* Columns */}
      <div className="px-2.5 py-1.5 space-y-px">
        {insertableCols.slice(0, 3).map((col) => (
          <div key={col.name} className="flex items-center gap-1.5">
            <ColumnTypeIcon type={col.type} className="h-2.5 w-2.5" />
            <span className="text-[10px] text-muted-foreground truncate">{col.name}</span>
          </div>
        ))}
        {insertableCols.length > 3 && (
          <div className="text-[10px] text-muted-foreground/70 pl-2.5">+{insertableCols.length - 3} more</div>
        )}

        {computedCols.length > 0 && (
          <>
            {insertableCols.length > 0 && <div className="border-t border-border/30 my-1" />}
            {computedCols.slice(0, 4).map((col) => {
              const ft = col.func_type ? FUNC_STYLES[col.func_type] : null
              return (
                <div key={col.name} className="flex items-center gap-1">
                  <Zap className={cn(
                    'h-2 w-2 shrink-0',
                    col.error_count > 0 ? 'text-destructive' : 'text-k-yellow/50',
                  )} />
                  <span className={cn(
                    'text-[10px] truncate',
                    col.error_count > 0 ? 'text-destructive' : 'text-foreground/80',
                  )}>
                    {col.name}
                  </span>
                  {col.func_name && ft && (
                    <span className={cn('text-[9px] shrink-0 font-mono', ft.text)}>
                      {col.func_name}
                    </span>
                  )}
                </div>
              )
            })}
            {computedCols.length > 4 && (
              <div className="text-[10px] text-muted-foreground/70 pl-2.5">+{computedCols.length - 4} more</div>
            )}
          </>
        )}
      </div>

      {data.hasOutgoing && (
        <Handle type="source" position={Position.Bottom} className="!bg-border !w-2 !h-2 !-bottom-1" />
      )}
    </div>
  )
}

const nodeTypes = { tableNode: PipelineTableNode }

// ── Detail Panel ─────────────────────────────────────────────────────────────

function DetailPanel({
  node,
  onClose,
  onShowColumnFlow,
  onViewTable,
}: {
  node: PipelineNodeType
  onClose: () => void
  onShowColumnFlow: () => void
  onViewTable: (path: string) => void
}) {
  const [showVersions, setShowVersions] = useState(false)
  const hasColumnDeps = node.columns.some((c) => c.depends_on && c.depends_on.length > 0)

  const computed = node.columns.filter((c) => c.is_computed)
  const insertable = node.columns.filter((c) => !c.is_computed)

  return (
    <div className="w-[360px] shrink-0 border-l border-border bg-card overflow-y-auto">
      {/* Header */}
      <div className="px-5 py-4 border-b border-border flex items-start justify-between sticky top-0 bg-card z-10">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            {node.is_view ? (
              <Eye className="h-4 w-4 text-muted-foreground shrink-0" />
            ) : (
              <Table2 className="h-4 w-4 text-muted-foreground shrink-0" />
            )}
            <h3 className="text-sm font-semibold truncate">{node.name}</h3>
          </div>
          <div className="flex items-center gap-2 mt-1 ml-6">
            <button
              className="text-xs text-k-yellow hover:underline font-mono flex items-center gap-1"
              onClick={() => onViewTable(node.path)}
            >
              {node.path}
              <ExternalLink className="h-2.5 w-2.5" />
            </button>
            {node.iterator_type && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-accent text-muted-foreground font-mono">
                {node.iterator_type}
              </span>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground transition-colors p-1 -mr-1 -mt-1 rounded hover:bg-accent"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 px-5 py-3 border-b border-border">
        {[
          { label: 'Rows', value: node.row_count.toLocaleString(), isError: false },
          { label: 'Version', value: `v${node.version}`, isError: false },
          { label: 'Computed', value: String(node.computed_count), isError: false },
          { label: 'Errors', value: String(node.total_errors), isError: node.total_errors > 0 },
        ].map((s) => (
          <div key={s.label} className="text-center">
            <div className={cn(
              'text-sm font-semibold tabular-nums',
              s.isError ? 'text-destructive' : 'text-foreground',
            )}>
              {s.value}
            </div>
            <div className="text-[11px] text-muted-foreground mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Column Data Flow button */}
      {hasColumnDeps && (
        <div className="px-5 py-3 border-b border-border">
          <button
            onClick={onShowColumnFlow}
            className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg border border-k-yellow/20 bg-k-yellow/5 hover:bg-k-yellow/10 transition-colors text-left group"
          >
            <GitBranch className="h-4 w-4 text-k-yellow/70 shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="text-[11px] font-semibold text-foreground">Column Data Flow</div>
              <div className="text-[11px] text-muted-foreground">Visualize column dependencies</div>
            </div>
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground -rotate-90 group-hover:text-k-yellow transition-colors shrink-0" />
          </button>
        </div>
      )}

      {/* Lineage */}
      {node.base && (
        <div className="px-5 py-3 border-b border-border flex items-center gap-2">
          <GitBranch className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <span className="text-xs text-muted-foreground">Derived from</span>
          <button className="text-xs text-k-yellow hover:underline font-mono font-medium" onClick={() => onViewTable(node.base!)}>
            {node.base}
          </button>
        </div>
      )}

      {/* Indices */}
      {node.indices.length > 0 && (
        <div className="px-5 py-3 border-b border-border">
          <SectionLabel>Embedding Indices</SectionLabel>
          <div className="space-y-2">
            {node.indices.map((idx) => (
              <div key={idx.name} className="flex items-start gap-2">
                <SearchIcon className="h-3.5 w-3.5 text-muted-foreground mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <div className="text-xs text-foreground">{idx.columns.join(', ')}</div>
                  <div className="mt-0.5 break-words"><PythonExpr code={idx.embedding} /></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Insertable columns */}
      {insertable.length > 0 && (
        <div className="px-5 py-3 border-b border-border">
          <SectionLabel>Insertable Columns ({insertable.length})</SectionLabel>
          <div className="space-y-1">
            {insertable.map((col) => (
              <div key={col.name} className="flex items-center gap-2 text-xs min-h-[22px]">
                <span className="text-foreground shrink-0">{col.name}</span>
                {col.defined_in && !col.defined_in_self && (
                  <span className="text-[10px] text-muted-foreground/60 italic shrink-0">from {col.defined_in}</span>
                )}
                <span className="ml-auto shrink-0"><ColumnTypeBadge type={col.type} /></span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Computed pipeline */}
      {computed.length > 0 && (
        <div className="px-5 py-3 border-b border-border">
          <div className="flex items-center gap-1.5 mb-2">
            <Zap className="h-3 w-3 text-k-yellow" />
            <SectionLabel className="mb-0">Computed Pipeline ({computed.length})</SectionLabel>
          </div>
          <div className="space-y-1.5">
            {computed.map((col, i) => (
              <ComputedColumnRow key={col.name} col={col} step={i + 1} />
            ))}
          </div>
        </div>
      )}

      {/* Version History */}
      {node.versions.length > 0 && (
        <div className="px-5 py-3">
          <button
            onClick={() => setShowVersions(!showVersions)}
            className="flex items-center gap-1.5 mb-2 hover:text-foreground transition-colors"
          >
            <ChevronDown className={cn('h-3 w-3 text-muted-foreground transition-transform', showVersions && 'rotate-180')} />
            <SectionLabel className="mb-0">Version History ({node.versions.length})</SectionLabel>
          </button>
          {showVersions && (
            <div className="space-y-1 ml-1">
              {node.versions.map((v) => (
                <div key={v.version} className="flex items-center gap-3 text-xs">
                  <span className="text-muted-foreground w-6 text-right tabular-nums">v{v.version}</span>
                  <span className={cn(
                    'px-1.5 py-0.5 rounded text-[11px] min-w-[50px] text-center',
                    v.change_type === 'schema' ? 'bg-accent text-muted-foreground' : 'bg-accent/50 text-muted-foreground',
                  )}>
                    {v.change_type}
                  </span>
                  <span className="text-muted-foreground tabular-nums">+{v.inserts}</span>
                  {v.updates > 0 && <span className="text-muted-foreground tabular-nums">~{v.updates}</span>}
                  {v.deletes > 0 && <span className="text-muted-foreground tabular-nums">-{v.deletes}</span>}
                  {v.errors > 0 && <span className="text-destructive tabular-nums">err:{v.errors}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SectionLabel({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn('text-[11px] font-medium text-muted-foreground uppercase tracking-wider mb-2', className)}>
      {children}
    </div>
  )
}

function ComputedColumnRow({ col, step }: { col: PipelineColumn; step: number }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const ft = col.func_type ? FUNC_STYLES[col.func_type] : null

  return (
    <div className="rounded-md border border-border overflow-hidden">
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-accent/50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="text-[11px] text-muted-foreground w-5 shrink-0 tabular-nums">#{step}</span>
        <span className={cn(
          'text-xs font-medium truncate',
          col.error_count > 0 ? 'text-destructive' : 'text-foreground',
        )}>
          {col.name}
        </span>
        {ft && (
          <span className={cn('text-[11px] px-1.5 py-0.5 rounded bg-accent font-mono shrink-0', ft.text)}>
            {ft.label}
          </span>
        )}
        <ChevronDown className={cn(
          'h-3 w-3 text-muted-foreground transition-transform shrink-0 ml-auto',
          isExpanded && 'rotate-180',
        )} />
      </button>
      {isExpanded && (
        <div className="px-3 pb-3 pt-2 space-y-2 border-t border-border">
          {col.func_name && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-medium">
                <span className="text-purple-300">{col.func_name}</span><span className="text-muted-foreground/50">()</span>
              </span>
            </div>
          )}
          {col.computed_with && (
            <div className="bg-accent rounded-md px-3 py-2 whitespace-pre-wrap break-all">
              <PythonExpr code={col.computed_with} />
            </div>
          )}
          {col.depends_on && col.depends_on.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap">
              <GitBranch className="h-3 w-3 text-muted-foreground shrink-0" />
              {col.depends_on.map((d) => (
                <span key={d} className="text-[11px] bg-accent text-muted-foreground px-1.5 py-0.5 rounded">{d}</span>
              ))}
            </div>
          )}
          {col.error_count > 0 && (
            <div className="text-xs text-destructive flex items-center gap-1.5">
              <AlertTriangle className="h-3 w-3" />
              {col.error_count} errors in sampled rows
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Hierarchical Layout ──────────────────────────────────────────────────────

function buildLayout(
  pipelineNodes: PipelineNodeType[],
  pipelineEdges: PipelineResponse['edges'],
): Node[] {
  const childrenMap = new Map<string, string[]>()
  const parentMap = new Map<string, string>()

  for (const e of pipelineEdges) {
    if (e.type !== 'view') continue
    if (!childrenMap.has(e.source)) childrenMap.set(e.source, [])
    childrenMap.get(e.source)!.push(e.target)
    parentMap.set(e.target, e.source)
  }

  const nodeMap = new Map(pipelineNodes.map((n) => [n.path, n]))

  const roots = pipelineNodes.filter((n) => !parentMap.has(n.path) && childrenMap.has(n.path))
  const standalone = pipelineNodes.filter(
    (n) => !parentMap.has(n.path) && !childrenMap.has(n.path),
  )

  const NODE_W = 240
  const NODE_H_BASE = 100
  const H_GAP = 40
  const V_GAP = 100

  const positions = new Map<string, { x: number; y: number }>()
  let globalX = 0

  function getNodeHeight(path: string): number {
    const n = nodeMap.get(path)
    if (!n) return NODE_H_BASE
    return NODE_H_BASE + Math.min(n.columns.length, 10) * 14
  }

  function getTreeWidth(path: string): number {
    const children = childrenMap.get(path) || []
    if (children.length === 0) return NODE_W
    const childWidths = children.map(getTreeWidth)
    return Math.max(NODE_W, childWidths.reduce((sum, w) => sum + w + H_GAP, -H_GAP))
  }

  function layoutTree(path: string, x: number, y: number) {
    if (positions.has(path)) return
    positions.set(path, { x, y })
    const children = childrenMap.get(path) || []
    if (children.length === 0) return

    const treeWidth = getTreeWidth(path)
    let childX = x - treeWidth / 2 + NODE_W / 2

    for (const child of children) {
      const childTreeW = getTreeWidth(child)
      const childCenterX = childX + childTreeW / 2 - NODE_W / 2
      layoutTree(child, childCenterX, y + getNodeHeight(path) + V_GAP)
      childX += childTreeW + H_GAP
    }
  }

  for (const root of roots) {
    const treeW = getTreeWidth(root.path)
    layoutTree(root.path, globalX + treeW / 2 - NODE_W / 2, 0)
    globalX += treeW + H_GAP * 2
  }

  const standaloneY = 0
  for (const node of standalone) {
    positions.set(node.path, { x: globalX, y: standaloneY })
    globalX += NODE_W + H_GAP
  }

  return pipelineNodes.map((n) => ({
    id: n.path,
    type: 'tableNode' as const,
    position: positions.get(n.path) || { x: 0, y: 0 },
    data: n as PipelineNodeType,
  }))
}

// ── Node Finder ──────────────────────────────────────────────────────────────

function NodeFinder({
  nodes,
  onSelect,
}: {
  nodes: PipelineNodeType[]
  onSelect: (path: string) => void
}) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const reactFlow = useReactFlow()

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as HTMLElement)) setOpen(false)
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [])

  const filtered = useMemo(() => {
    if (!query) return nodes
    const q = query.toLowerCase()
    return nodes.filter((n) => n.name.toLowerCase().includes(q) || n.path.toLowerCase().includes(q))
  }, [nodes, query])

  function handlePick(path: string) {
    onSelect(path)
    setOpen(false)
    setQuery('')
    setTimeout(() => {
      reactFlow.fitView({ nodes: [{ id: path }], duration: 400, padding: 0.6 })
    }, 50)
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => { setOpen(!open); setTimeout(() => inputRef.current?.focus(), 0) }}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-border bg-card shadow-sm text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
      >
        <SearchIcon className="h-3 w-3" />
        <span>Find table…</span>
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 w-64 bg-card border border-border rounded-lg shadow-lg z-50 overflow-hidden">
          <div className="px-2 py-1.5 border-b border-border">
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search tables & views…"
              className="w-full bg-transparent text-xs text-foreground placeholder:text-muted-foreground/60 outline-none"
              onKeyDown={(e) => {
                if (e.key === 'Escape') setOpen(false)
                if (e.key === 'Enter' && filtered.length > 0) handlePick(filtered[0].path)
              }}
            />
          </div>
          <div className="max-h-60 overflow-y-auto">
            {filtered.length === 0 ? (
              <div className="px-3 py-4 text-xs text-muted-foreground text-center">No matches</div>
            ) : (
              filtered.map((n) => (
                <button
                  key={n.path}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-left hover:bg-accent transition-colors"
                  onClick={() => handlePick(n.path)}
                >
                  {n.is_view ? (
                    <Eye className="h-3 w-3 text-purple-400 shrink-0" />
                  ) : (
                    <Table2 className="h-3 w-3 text-blue-400 shrink-0" />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="text-xs text-foreground truncate">{n.name}</div>
                    <div className="text-[10px] text-muted-foreground truncate">{n.path}</div>
                  </div>
                  <span className="text-[10px] text-muted-foreground tabular-nums shrink-0">
                    {n.row_count.toLocaleString()}
                  </span>
                  {n.total_errors > 0 && (
                    <AlertTriangle className="h-2.5 w-2.5 text-destructive shrink-0" />
                  )}
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main Component ───────────────────────────────────────────────────────────

export function PipelineInspector() {
  return (
    <ReactFlowProvider>
      <PipelineInspectorInner />
    </ReactFlowProvider>
  )
}

function PipelineInspectorInner() {
  const navigate = useNavigate()
  const [pipeline, setPipeline] = useState<PipelineResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [columnFlowNode, setColumnFlowNode] = useState<PipelineNodeType | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node<TableNodeData>>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchPipeline = useCallback(() => {
    setIsLoading(true)
    getPipeline()
      .then((data) => {
        setPipeline(data)
        setLastRefreshed(new Date())
        setIsLoading(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Failed to load pipeline')
        setIsLoading(false)
      })
  }, [])

  useEffect(() => {
    fetchPipeline()
  }, [fetchPipeline])

  // Auto-refresh interval
  useEffect(() => {
    if (autoRefresh) {
      refreshIntervalRef.current = setInterval(fetchPipeline, 10_000) // 10s
    }
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current)
        refreshIntervalRef.current = null
      }
    }
  }, [autoRefresh, fetchPipeline])

  const handleSelect = useCallback((path: string) => {
    setSelectedPath((prev) => (prev === path ? null : path))
  }, [])

  useEffect(() => {
    if (!pipeline) return

    const hasIncoming = new Set(pipeline.edges.map((e) => e.target))
    const hasOutgoing = new Set(pipeline.edges.map((e) => e.source))

    const nodesWithCallbacks = pipeline.nodes.map((n) => ({
      ...n,
      isSelected: n.path === selectedPath,
      onSelect: handleSelect,
      hasIncoming: hasIncoming.has(n.path),
      hasOutgoing: hasOutgoing.has(n.path),
    }))

    const flowNodes: Node<TableNodeData>[] = buildLayout(pipeline.nodes, pipeline.edges).map((n) => ({
      ...n,
      data: nodesWithCallbacks.find((pn) => pn.path === n.id)! as TableNodeData,
    }))

    const flowEdges: Edge[] = pipeline.edges.map((e, i) => ({
      id: `e-${i}`,
      source: e.source,
      target: e.target,
      type: 'labeled',
      animated: e.type === 'view',
      data: { label: e.label, edgeType: e.type },
    }))

    setNodes(flowNodes)
    setEdges(flowEdges)
  }, [pipeline, selectedPath, handleSelect, setNodes, setEdges])

  const selectedNode = useMemo(
    () => pipeline?.nodes.find((n) => n.path === selectedPath) ?? null,
    [pipeline, selectedPath],
  )

  const stats = useMemo(() => {
    if (!pipeline) return null
    const tables = pipeline.nodes.filter((n) => !n.is_view).length
    const views = pipeline.nodes.filter((n) => n.is_view).length
    const totalRows = pipeline.nodes.reduce((s, n) => s + n.row_count, 0)
    const totalComputed = pipeline.nodes.reduce((s, n) => s + n.computed_count, 0)
    const totalErrors = pipeline.nodes.reduce((s, n) => s + n.total_errors, 0)
    const totalIndices = pipeline.nodes.reduce((s, n) => s + n.indices.length, 0)
    return { tables, views, totalRows, totalComputed, totalErrors, totalIndices }
  }, [pipeline])

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-2">
          <AlertTriangle className="h-6 w-6 text-destructive mx-auto" />
          <p className="text-xs text-muted-foreground">{error}</p>
        </div>
      </div>
    )
  }

  // Column Data Flow drill-down: replaces the main canvas
  if (columnFlowNode) {
    return (
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border/40 shrink-0">
          <button
            onClick={() => setColumnFlowNode(null)}
            className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors rounded-md px-2 py-1 hover:bg-accent"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Pipeline
          </button>
          <span className="text-muted-foreground/60 text-xs">/</span>
          <div className="flex items-center gap-1.5">
            {columnFlowNode.is_view ? (
              <Eye className="h-3 w-3 text-muted-foreground" />
            ) : (
              <Table2 className="h-3 w-3 text-muted-foreground" />
            )}
            <span className="text-xs font-semibold text-foreground">{columnFlowNode.name}</span>
          </div>
          <span className="text-muted-foreground/60 text-xs">/</span>
          <div className="flex items-center gap-1.5">
            <GitBranch className="h-3 w-3 text-k-yellow" />
            <span className="text-xs font-medium text-k-yellow">Column Data Flow</span>
          </div>
        </div>
        <div className="flex-1 min-h-0">
          <ColumnFlowDiagram columns={columnFlowNode.columns} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Stats bar */}
      {stats && (
        <div className="flex items-center gap-4 px-4 py-2 border-b border-border/40 shrink-0">
          {[
            { icon: Table2, value: stats.tables, label: 'tables' },
            { icon: Eye, value: stats.views, label: 'views' },
            { icon: Rows3, value: stats.totalRows.toLocaleString(), label: 'rows' },
            { icon: Zap, value: stats.totalComputed, label: 'computed' },
            { icon: SearchIcon, value: stats.totalIndices, label: 'indices' },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <s.icon className="h-3 w-3 text-muted-foreground" />
              <span className="tabular-nums">{s.value}</span>
              <span className="text-muted-foreground/80">{s.label}</span>
            </div>
          ))}
          {stats.totalErrors > 0 && (
            <div className="flex items-center gap-1.5 text-[11px] text-destructive">
              <AlertTriangle className="h-3 w-3" />
              <span>{stats.totalErrors} errors</span>
            </div>
          )}

          {/* Auto-refresh + manual refresh */}
          <div className="ml-auto flex items-center gap-2">
            {lastRefreshed && (
              <span className="text-[11px] text-muted-foreground tabular-nums">
                {lastRefreshed.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={fetchPipeline}
              className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
              title="Refresh now"
            >
              <RefreshCw className={cn('h-3 w-3', isLoading && 'animate-spin')} />
            </button>
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={cn(
                'flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors',
                autoRefresh
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
            >
              <div className={cn(
                'w-1.5 h-1.5 rounded-full',
                autoRefresh ? 'bg-emerald-400 animate-pulse' : 'bg-muted-foreground/50',
              )} />
              Auto
            </button>
          </div>
        </div>
      )}

      {/* Flow + Detail */}
      <div className="flex-1 flex min-h-0">
        <div className="flex-1 relative min-h-0">
          {/* Floating node finder */}
          <div className="absolute top-3 left-3 z-10">
            <NodeFinder nodes={pipeline?.nodes ?? []} onSelect={handleSelect} />
          </div>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            fitViewOptions={{ padding: 0.2 }}
            minZoom={0.2}
            maxZoom={1.5}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="hsl(var(--border))" gap={24} size={1} />
            <Controls
              className="!bg-card !border-border !rounded-lg [&>button]:!bg-card [&>button]:!border-border/60 [&>button]:!text-muted-foreground [&>button:hover]:!bg-accent"
            />
          </ReactFlow>
        </div>

        {selectedNode && (
          <DetailPanel
            node={selectedNode}
            onClose={() => setSelectedPath(null)}
            onShowColumnFlow={() => setColumnFlowNode(selectedNode)}
            onViewTable={(path) => navigate(`/table/${path}`)}
          />
        )}
      </div>
    </div>
  )
}
