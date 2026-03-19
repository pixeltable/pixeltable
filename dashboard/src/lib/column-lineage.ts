import dagre from 'dagre'
import type { Node, Edge } from '@xyflow/react'
import type { PipelineColumn } from '@/types'

// ── Types ────────────────────────────────────────────────────────────────────

type FuncType = 'builtin' | 'custom_udf' | 'query' | 'iterator' | 'unknown' | null

export interface ColumnNodeData extends Record<string, unknown> {
  name: string
  type: string
  isComputed: boolean
  isIteratorCol: boolean
  definedInSelf: boolean
  definedIn: string | null
  computeExpression: string | null
  funcName: string | null
  funcType: FuncType
  errorCount: number
  upstreamColumns: string[]
  downstreamColumns: string[]
  comment?: string
}

interface LineageResult {
  nodes: Node<ColumnNodeData>[]
  edges: Edge[]
  sourceCount: number
  computedCount: number
}

// ── Layout ───────────────────────────────────────────────────────────────────

const LAYOUT = {
  rankdir: 'TB' as const,
  nodesep: 40,
  ranksep: 80,
  edgesep: 20,
} as const

const NODE_SIZE = { width: 220, height: 80 } as const

// ── Graph Building ───────────────────────────────────────────────────────────

function buildDependencyMaps(columns: PipelineColumn[]) {
  const upstream = new Map<string, string[]>()
  const allNames = new Set(columns.map((c) => c.name))

  for (const col of columns) {
    if (!col.is_computed) continue

    if (col.depends_on && col.depends_on.length > 0) {
      const valid = col.depends_on.filter((d) => allNames.has(d))
      if (valid.length > 0) upstream.set(col.name, valid)
    }
  }

  const downstream = new Map<string, string[]>()
  for (const [target, sources] of upstream) {
    for (const src of sources) {
      const list = downstream.get(src) || []
      list.push(target)
      downstream.set(src, list)
    }
  }

  return { upstream, downstream }
}

/**
 * Build ReactFlow nodes + edges from a table's column metadata.
 * Only includes columns that participate in at least one dependency edge,
 * plus all computed columns (even if they have no parsed deps).
 */
export function buildLineageGraph(columns: PipelineColumn[]): LineageResult {
  if (!columns || columns.length === 0) {
    return { nodes: [], edges: [], sourceCount: 0, computedCount: 0 }
  }

  const { upstream, downstream } = buildDependencyMaps(columns)

  // Determine which columns to include: any that are part of the dependency graph
  const included = new Set<string>()
  for (const col of columns) {
    if (col.is_computed) included.add(col.name)
  }
  for (const sources of upstream.values()) {
    for (const s of sources) included.add(s)
  }

  const visibleColumns = columns.filter((c) => included.has(c.name))
  if (visibleColumns.length === 0) {
    return { nodes: [], edges: [], sourceCount: 0, computedCount: 0 }
  }

  const edges: Edge[] = []
  for (const [target, sources] of upstream) {
    for (const src of sources) {
      if (included.has(src)) {
        edges.push({
          id: `${src}→${target}`,
          source: src,
          target,
          type: 'columnEdge',
        })
      }
    }
  }

  const nodes: Node<ColumnNodeData>[] = visibleColumns.map((col) => ({
    id: col.name,
    type: 'columnNode',
    position: { x: 0, y: 0 },
    data: {
      name: col.name,
      type: col.type,
      isComputed: col.is_computed,
      isIteratorCol: col.is_iterator_col,
      definedInSelf: col.defined_in_self,
      definedIn: col.defined_in,
      computeExpression: col.computed_with,
      funcName: col.func_name,
      funcType: col.func_type,
      errorCount: col.error_count,
      upstreamColumns: upstream.get(col.name) || [],
      downstreamColumns: downstream.get(col.name) || [],
      comment: col.comment,
    },
  }))

  const sourceCount = visibleColumns.filter((c) => !c.is_computed).length
  const computedCount = visibleColumns.filter((c) => c.is_computed).length

  return { nodes, edges, sourceCount, computedCount }
}

// ── Dagre Layout ─────────────────────────────────────────────────────────────

export function applyDagreLayout(
  nodes: Node<ColumnNodeData>[],
  edges: Edge[],
): Node<ColumnNodeData>[] {
  if (nodes.length === 0) return nodes

  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph(LAYOUT)

  for (const node of nodes) {
    g.setNode(node.id, { ...NODE_SIZE })
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target)
  }

  dagre.layout(g)

  return nodes.map((node) => {
    const pos = g.node(node.id)
    return {
      ...node,
      position: {
        x: pos.x - NODE_SIZE.width / 2,
        y: pos.y - NODE_SIZE.height / 2,
      },
    }
  })
}

// ── Edge Highlight Styling ───────────────────────────────────────────────────

export function applyEdgeHighlights(
  edges: Edge[],
  selectedNodeId: string | null,
): Edge[] {
  return edges.map((edge) => {
    if (!selectedNodeId) {
      return { ...edge, animated: false, data: { ...edge.data, highlight: 'default' } }
    }
    const isConnected = edge.source === selectedNodeId || edge.target === selectedNodeId
    return {
      ...edge,
      animated: isConnected,
      data: { ...edge.data, highlight: isConnected ? 'active' : 'dimmed' },
    }
  })
}
