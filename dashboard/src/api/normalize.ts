import type { TreeNode } from '@/types'

const warnedPaths = new Set<string>()

function warnOnce(path: string): void {
  if (warnedPaths.has(path)) return
  warnedPaths.add(path)
  console.warn(`Directory '${path}' missing 'entries'; treating as empty`)
}

/**
 * Recursively repair a tree response: directories that arrive without an `entries` array
 * (server bug, schema skew, partial response) are coerced to having an empty array. A
 * one-time warning per offending path is logged so the bug doesn't go silent.
 */
export function normalizeTreeNode(node: TreeNode): TreeNode {
  if (node.kind !== 'directory') return node
  if (node.entries == null) {
    warnOnce(node.path)
    return { ...node, entries: [] }
  }
  return { ...node, entries: node.entries.map(normalizeTreeNode) }
}

export function normalizeTreeNodes(nodes: TreeNode[]): TreeNode[] {
  return nodes.map(normalizeTreeNode)
}
