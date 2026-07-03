import { describe, it, expect, vi, beforeEach } from 'vitest'
import { normalizeTreeNode, normalizeTreeNodes } from './normalize'
import type { TreeNode } from '@/types'

describe('normalizeTreeNode', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('passes through table nodes unchanged', () => {
    const t: TreeNode = {
      kind: 'table', name: 't', path: '/t', version: 1, error_count: 0, base: null,
    }
    expect(normalizeTreeNode(t)).toEqual(t)
  })

  it('passes through directories that already have entries', () => {
    const d: TreeNode = { kind: 'directory', name: 'd', path: '/d', entries: [] }
    expect(normalizeTreeNode(d)).toEqual(d)
  })

  it('fills in missing entries on a directory and warns', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const broken = { kind: 'directory', name: 'd', path: '/missing-entries-1' } as unknown as TreeNode
    const fixed = normalizeTreeNode(broken)
    expect(fixed).toEqual({ kind: 'directory', name: 'd', path: '/missing-entries-1', entries: [] })
    expect(warn).toHaveBeenCalledOnce()
  })

  it('recurses into nested directories', () => {
    const nested = {
      kind: 'directory',
      name: 'a',
      path: '/a',
      entries: [{ kind: 'directory', name: 'b', path: '/a/b' /* missing entries */ }],
    } as unknown as TreeNode
    const fixed = normalizeTreeNode(nested) as Extract<TreeNode, { kind: 'directory' }>
    const child = fixed.entries[0] as Extract<TreeNode, { kind: 'directory' }>
    expect(child.entries).toEqual([])
  })

  it('normalizeTreeNodes handles a list', () => {
    const list = [{ kind: 'directory', name: 'd', path: '/list-1' }] as unknown as TreeNode[]
    const fixed = normalizeTreeNodes(list)
    const first = fixed[0] as Extract<TreeNode, { kind: 'directory' }>
    expect(first.entries).toEqual([])
  })
})
