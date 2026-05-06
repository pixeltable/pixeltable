import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  Panel,
  Handle,
  Position,
  BaseEdge,
  getSmoothStepPath,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  type Node,
  type EdgeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  Zap,
  GitBranch,
  ChevronRight,
  Copy,
  Check,
  X,
  Layers,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { PipelineColumn } from '@/types'
import { getColumnTypeMeta } from '@/lib/column-types'
import { FUNC_STYLES } from '@/lib/func-styles'
import {
  type ColumnNodeData,
  buildLineageGraph,
  applyDagreLayout,
  applyEdgeHighlights,
} from '@/lib/column-lineage'

function formatType(type: string): string {
  if (type.startsWith('Required[') && type.endsWith(']')) {
    type = type.slice('Required['.length, -1)
  }
  return type.split('[')[0]
}

// ── Custom Edge ──────────────────────────────────────────────────────────────

function ColumnEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 12,
  })

  const highlight = (data as Record<string, unknown> | undefined)?.highlight as string | undefined

  const isDimmed = highlight === 'dimmed'

  const style = useMemo(() => {
    if (highlight === 'active') return { stroke: 'hsl(42 98% 48% / 0.7)', strokeWidth: 2.5, strokeDasharray: '6 4' }
    if (isDimmed) return { stroke: 'rgba(100, 100, 100, 0.12)', strokeWidth: 1 }
    return { stroke: 'rgba(100, 100, 100, 0.35)', strokeWidth: 1.5, strokeDasharray: '6 4' }
  }, [highlight, isDimmed])

  return <BaseEdge id={id} path={edgePath} style={style} className={isDimmed ? undefined : 'react-flow__edge-path-animated'} />
}

const edgeTypes = { columnEdge: ColumnEdge }

// ── Custom Node ──────────────────────────────────────────────────────────────

function ColumnNodeComponent({ data, selected }: { data: ColumnNodeData; selected?: boolean }) {
  const { icon: Icon } = getColumnTypeMeta(data.type)
  const funcStyle = data.funcType ? FUNC_STYLES[data.funcType] : null
  const inherited = !data.definedInSelf

  return (
    <div className={cn(
      'min-w-[200px] max-w-[240px] transition-all duration-150',
      selected && 'scale-[1.02]',
    )}>
      {data.upstreamColumns.length > 0 && (
        <Handle
          type="target"
          position={Position.Top}
          className={cn(
            '!w-2 !h-2 !border-2 !border-background transition-colors',
            data.isComputed ? '!bg-k-yellow' : '!bg-muted-foreground/50',
          )}
        />
      )}

      {data.downstreamColumns.length > 0 && (
        <Handle
          type="source"
          position={Position.Bottom}
          className="!w-2 !h-2 !bg-muted-foreground/50 !border-2 !border-background"
        />
      )}

      <div className={cn(
        'border rounded-lg shadow-sm overflow-hidden transition-all duration-150',
        inherited
          ? 'border-dashed border-border/50 bg-card/60 opacity-80'
          : cn(
              'bg-card',
              data.isComputed ? 'border-k-yellow/30' : 'border-border/60',
            ),
        selected && 'ring-1 ring-k-yellow/50 shadow-md border-k-yellow/50 opacity-100',
      )}>
        {/* Origin tag for inherited columns */}
        {inherited && data.definedIn && (
          <div className="px-3 py-1 bg-muted/20 border-b border-border/20">
            <span className="text-[8px] uppercase tracking-wider text-muted-foreground/50 font-medium">
              from {data.definedIn}
            </span>
          </div>
        )}

        {/* Header */}
        <div className={cn(
          'px-3 py-2 flex items-center gap-2',
          !inherited && data.isComputed ? 'bg-k-yellow/5' : '',
        )}>
          <div className={cn(
            'w-5 h-5 rounded flex items-center justify-center shrink-0',
            !inherited && data.isComputed ? 'bg-k-yellow/15' : 'bg-muted/40',
          )}>
            <Icon className={cn(
              'h-3 w-3',
              !inherited && data.isComputed ? 'text-k-yellow' : 'text-muted-foreground',
            )} />
          </div>
          <div className="min-w-0 flex-1">
            <span className={cn(
              'text-[11px] font-semibold truncate block leading-tight',
              inherited ? 'text-foreground/70' : 'text-foreground',
            )}>
              {data.name}
            </span>
            <span className={cn(
              'text-[9px] font-mono leading-tight',
              !inherited && data.isComputed ? 'text-k-yellow/60' : 'text-muted-foreground/50',
            )}>
              {formatType(data.type)}
            </span>
          </div>
          {data.isComputed && (
            <Zap className={cn('h-2.5 w-2.5 shrink-0', inherited ? 'text-muted-foreground/30' : 'text-k-yellow/60')} />
          )}
        </div>

        {/* Function badge */}
        {data.isComputed && (data.funcName || funcStyle) && (
          <div className="px-3 py-1 border-t border-border/30">
            <div className="flex items-center gap-1.5">
              {funcStyle && (
                <span className={cn(
                  'text-[8px] font-semibold uppercase tracking-wider px-1 py-0.5 rounded',
                  funcStyle.bg, funcStyle.text,
                )}>
                  {funcStyle.label}
                </span>
              )}
              {data.funcName && (
                <code className="text-[9px] font-mono text-muted-foreground/60 truncate">
                  {data.funcName.length > 24 ? `${data.funcName.slice(0, 22)}…` : data.funcName}()
                </code>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const nodeTypes = { columnNode: ColumnNodeComponent }

// ── Detail Sidebar ───────────────────────────────────────────────────────────

function ColumnDetailPanel({
  node,
  onClose,
  onNavigate,
}: {
  node: ColumnNodeData
  onClose: () => void
  onNavigate: (id: string) => void
}) {
  const [copied, setCopied] = useState(false)
  const { icon: Icon } = getColumnTypeMeta(node.type)
  const funcStyle = node.funcType ? FUNC_STYLES[node.funcType] : null

  const handleCopy = useCallback(async () => {
    if (!node.computeExpression) return
    try {
      await navigator.clipboard.writeText(node.computeExpression)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch { /* ignore */ }
  }, [node.computeExpression])

  return (
    <div
      className="absolute right-0 top-0 bottom-0 w-[280px] bg-card border-l border-border overflow-y-auto shadow-xl z-50"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div className="sticky top-0 bg-card/95 backdrop-blur-sm px-4 py-3 border-b border-border z-[51]">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <div className={cn(
              'w-6 h-6 rounded flex items-center justify-center shrink-0',
              node.isComputed ? 'bg-k-yellow/15' : 'bg-muted/40',
            )}>
              <Icon className={cn(
                'h-3.5 w-3.5',
                node.isComputed ? 'text-k-yellow' : 'text-muted-foreground',
              )} />
            </div>
            <div className="min-w-0">
              <h3 className="text-xs font-semibold truncate">{node.name}</h3>
              <div className="text-[10px] font-mono text-muted-foreground/60">{node.type}</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-accent transition-colors -mr-1 -mt-1"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Kind + Origin */}
      <div className="px-4 py-2.5 border-b border-border/50">
        <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider mb-1">Kind</div>
        <div className="flex items-center gap-1.5 flex-wrap">
          {node.isComputed && funcStyle ? (
            <span className={cn('text-[10px] font-semibold px-1.5 py-0.5 rounded', funcStyle.bg, funcStyle.text)}>
              {funcStyle.label}
            </span>
          ) : (
            <span className="text-[10px] text-muted-foreground/60 bg-muted/30 px-1.5 py-0.5 rounded">Source</span>
          )}
          {!node.definedInSelf && node.definedIn && (
            <span className="text-[10px] text-muted-foreground/50 bg-muted/20 px-1.5 py-0.5 rounded border border-dashed border-border/40">
              from {node.definedIn}
            </span>
          )}
          {node.isIteratorCol && (
            <span className="text-[10px] text-violet-400 bg-violet-400/10 px-1.5 py-0.5 rounded font-medium">
              iterator
            </span>
          )}
        </div>
      </div>

      {/* Expression */}
      {node.computeExpression && (
        <div className="px-4 py-2.5 border-b border-border/50">
          <div className="flex items-center justify-between mb-1.5">
            <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">Expression</div>
            <button
              onClick={handleCopy}
              className="p-0.5 text-muted-foreground/40 hover:text-foreground transition-colors rounded hover:bg-muted/50"
            >
              {copied ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
            </button>
          </div>
          <pre className="text-[10px] font-mono text-muted-foreground bg-accent/50 rounded px-2.5 py-2 whitespace-pre-wrap break-all leading-relaxed max-h-32 overflow-y-auto">
            {node.computeExpression}
          </pre>
        </div>
      )}

      {/* Comment */}
      {node.comment && (
        <div className="px-4 py-2.5 border-b border-border/50">
          <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider mb-1">Note</div>
          <p className="text-[10px] text-muted-foreground/80 italic">{node.comment}</p>
        </div>
      )}

      {/* Upstream */}
      {node.upstreamColumns.length > 0 && (
        <div className="px-4 py-2.5 border-b border-border/50">
          <div className="flex items-center gap-1.5 mb-1.5">
            <GitBranch className="h-3 w-3 text-muted-foreground/50" />
            <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">
              Depends On ({node.upstreamColumns.length})
            </div>
          </div>
          <div className="space-y-px">
            {node.upstreamColumns.map((col) => (
              <button
                key={col}
                onClick={() => onNavigate(col)}
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left hover:bg-accent/50 transition-colors group"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/30 group-hover:bg-k-yellow transition-colors" />
                <span className="text-[10px] text-foreground/80 font-medium truncate">{col}</span>
                <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/30 ml-auto group-hover:text-foreground transition-colors" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Downstream */}
      {node.downstreamColumns.length > 0 && (
        <div className="px-4 py-2.5 border-b border-border/50">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Zap className="h-3 w-3 text-k-yellow/60" />
            <div className="text-[9px] text-muted-foreground/60 uppercase tracking-wider">
              Used By ({node.downstreamColumns.length})
            </div>
          </div>
          <div className="space-y-px">
            {node.downstreamColumns.map((col) => (
              <button
                key={col}
                onClick={() => onNavigate(col)}
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left hover:bg-accent/50 transition-colors group"
              >
                <Zap className="h-2.5 w-2.5 text-k-yellow/30 group-hover:text-k-yellow transition-colors" />
                <span className="text-[10px] text-foreground/80 font-medium truncate">{col}</span>
                <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/30 ml-auto group-hover:text-foreground transition-colors" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Leaf source */}
      {!node.isComputed && node.downstreamColumns.length === 0 && (
        <div className="px-4 py-6 text-center">
          <p className="text-[10px] text-muted-foreground/40">
            This source column is not referenced by any computed columns.
          </p>
        </div>
      )}
    </div>
  )
}

// ── Empty State ──────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="h-40 border border-border/40 rounded-lg bg-accent/20 flex flex-col items-center justify-center text-center p-6">
      <Layers className="h-5 w-5 text-muted-foreground/30 mb-2" />
      <p className="text-[10px] text-muted-foreground/50 max-w-[200px]">
        No column dependencies to visualize. Add computed columns that reference other columns.
      </p>
    </div>
  )
}

// ── Main Content ─────────────────────────────────────────────────────────────

function ColumnFlowContent({ columns }: { columns: PipelineColumn[] }) {
  const { layoutedNodes, layoutedEdges, sourceCount, computedCount } = useMemo(() => {
    const { nodes, edges, sourceCount: sc, computedCount: cc } = buildLineageGraph(columns)
    const positioned = applyDagreLayout(nodes, edges)
    return { layoutedNodes: positioned, layoutedEdges: edges, sourceCount: sc, computedCount: cc }
  }, [columns])

  const [nodes, setNodes, onNodesChange] = useNodesState<Node<ColumnNodeData>>(layoutedNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const nodeClickedRef = useRef(false)

  useEffect(() => {
    setNodes(layoutedNodes)
    setEdges(layoutedEdges)
  }, [layoutedNodes, layoutedEdges, setNodes, setEdges])

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node<ColumnNodeData>) => {
    nodeClickedRef.current = true
    setSelectedNodeId((prev) => (node.id === prev ? null : node.id))
  }, [])

  const onPaneClick = useCallback(() => {
    if (nodeClickedRef.current) {
      nodeClickedRef.current = false
      return
    }
    setSelectedNodeId(null)
  }, [])

  const styledEdges = useMemo(
    () => applyEdgeHighlights(edges, selectedNodeId),
    [edges, selectedNodeId],
  )

  const selectedNodeData = useMemo(() => {
    if (!selectedNodeId) return null
    return nodes.find((n) => n.id === selectedNodeId)?.data ?? null
  }, [nodes, selectedNodeId])

  if (layoutedEdges.length === 0) {
    return <EmptyState />
  }

  return (
    <div className="relative h-full w-full overflow-hidden bg-card">
      <div
        className={cn(
          'h-full transition-all duration-200',
          selectedNodeData ? 'pr-[280px]' : '',
        )}
      >
        <ReactFlow
          nodes={nodes}
          edges={styledEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          fitViewOptions={{ padding: 0.3, duration: 300 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          selectNodesOnDrag={false}
          minZoom={0.3}
          maxZoom={1.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1a1a1a" />
          <Panel position="top-left">
            <div className="bg-card/90 backdrop-blur-sm border border-border/40 rounded px-2.5 py-1.5 flex items-center gap-2.5">
              <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                <span className="font-semibold text-foreground">{sourceCount}</span> source
                <span className="text-muted-foreground/30">&rarr;</span>
                <span className="font-semibold text-k-yellow">{computedCount}</span> computed
              </div>
            </div>
          </Panel>
          <Panel position="top-right">
            <div className="bg-card/90 backdrop-blur-sm border border-border/40 rounded px-2.5 py-1.5 space-y-1">
              <div className="text-[8px] text-muted-foreground/50 uppercase tracking-wider font-semibold">Legend</div>
              <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground/60">
                <div className="w-2.5 h-2.5 rounded-sm border border-border/60 bg-card" />
                Source Column
              </div>
              <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground/60">
                <div className="w-2.5 h-2.5 rounded-sm border border-k-yellow/40 bg-k-yellow/10" />
                Computed Column
              </div>
              <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground/60">
                <div className="w-2.5 h-2.5 rounded-sm border border-dashed border-border/50 bg-card/60 opacity-70" />
                Inherited (base table)
              </div>
            </div>
          </Panel>
          <Controls
            showInteractive={false}
            className="!bg-card !border-border/40 !rounded-lg [&>button]:!bg-card [&>button]:!border-border/40 [&>button]:!text-muted-foreground [&>button:hover]:!bg-accent"
          />
        </ReactFlow>
      </div>

      {selectedNodeData && (
        <ColumnDetailPanel
          node={selectedNodeData}
          onClose={() => setSelectedNodeId(null)}
          onNavigate={(id) => setSelectedNodeId(id)}
        />
      )}
    </div>
  )
}

// ── Exported Component ───────────────────────────────────────────────────────

export function ColumnFlowDiagram({ columns }: { columns: PipelineColumn[] }) {
  return (
    <ReactFlowProvider>
      <ColumnFlowContent columns={columns} />
    </ReactFlowProvider>
  )
}
