import { useState, useEffect, useCallback } from 'react'
import type { TreeNode } from '@/types'
import { getDirectoryTree } from '@/api/client'
import { LOCAL_CATALOG, loadExtraCatalogs, saveExtraCatalogs } from '@/lib/catalogs'
import { DirectoryTree } from './DirectoryTree'
import { Database, ChevronRight, ChevronDown, Plus, Check, X } from 'lucide-react'

interface CatalogSectionProps {
  uri: string
  // pre-fetched tree for this catalog; when omitted the tree is fetched the first time the section opens
  nodes?: TreeNode[]
  removable: boolean
  onRemove?: () => void
  defaultOpen: boolean
  selectedPath: string | null
  onSelect: (path: string, type: string) => void
}

function CatalogSection({ uri, nodes: providedNodes, removable, onRemove, defaultOpen, selectedPath, onSelect }: CatalogSectionProps) {
  const [open, setOpen] = useState(defaultOpen)
  const [fetched, setFetched] = useState<TreeNode[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isProvided = providedNodes !== undefined

  const load = useCallback(() => {
    setLoading(true)
    setError(null)
    getDirectoryTree(uri)
      .then(setFetched)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [uri])

  // A hosted catalog's tree is fetched lazily, the first time its section is opened, so an unreachable
  // catalog never blocks the rest of the sidebar.
  useEffect(() => {
    if (!isProvided && open && fetched === null && !loading && error === null) load()
  }, [isProvided, open, fetched, loading, error, load])

  const nodes = isProvided ? providedNodes! : fetched

  return (
    <div>
      <div className="group flex items-center gap-1.5 w-full rounded-md py-1 px-2 text-muted-foreground hover:bg-accent hover:text-foreground">
        <button className="flex flex-1 items-center gap-1.5 min-w-0 text-left" onClick={() => setOpen((o) => !o)}>
          {open ? (
            <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
          )}
          <Database className="h-3.5 w-3.5 text-k-yellow shrink-0" />
          <span className="flex-1 text-[13px] font-medium truncate" title={uri}>{uri}</span>
        </button>
        {removable && (
          <button
            onClick={onRemove}
            title="Remove catalog"
            className="opacity-0 group-hover:opacity-100 shrink-0 text-muted-foreground/60 hover:text-destructive"
          >
            <X className="h-3 w-3" />
          </button>
        )}
      </div>

      {open && (
        <div className="pl-2">
          {loading && <div className="py-2 pl-4 text-[11px] text-muted-foreground/60">Loading...</div>}
          {error !== null && (
            <div className="flex items-start gap-1 py-1.5 pl-4 pr-2 text-[11px] text-destructive">
              <span className="flex-1 break-words">{error}</span>
              <button onClick={load} className="shrink-0 underline hover:no-underline">
                retry
              </button>
            </div>
          )}
          {nodes !== null && error === null && !loading && (
            <DirectoryTree nodes={nodes} selectedPath={selectedPath} onSelect={onSelect} />
          )}
        </div>
      )}
    </div>
  )
}

interface CatalogTreeProps {
  // the in-process catalog's tree, already loaded by the parent and shown under the 'local' root
  localTree: TreeNode[]
  selectedPath: string | null
  onSelect: (path: string, type: string) => void
}

export function CatalogTree({ localTree, selectedPath, onSelect }: CatalogTreeProps) {
  const [extras, setExtras] = useState<string[]>(loadExtraCatalogs)
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState('')

  useEffect(() => {
    saveExtraCatalogs(extras)
  }, [extras])

  const cancelAdd = () => {
    setAdding(false)
    setDraft('')
  }

  const commitAdd = () => {
    const uri = draft.trim()
    if (uri !== '' && uri !== LOCAL_CATALOG && !extras.includes(uri)) {
      setExtras((prev) => [...prev, uri])
    }
    cancelAdd()
  }

  const removeCatalog = (uri: string) => setExtras((prev) => prev.filter((u) => u !== uri))

  return (
    <div className="space-y-px">
      <CatalogSection
        uri={LOCAL_CATALOG}
        nodes={localTree}
        removable={false}
        defaultOpen
        selectedPath={selectedPath}
        onSelect={onSelect}
      />
      {extras.map((uri) => (
        <CatalogSection
          key={uri}
          uri={uri}
          removable
          defaultOpen={false}
          onRemove={() => removeCatalog(uri)}
          selectedPath={selectedPath}
          onSelect={onSelect}
        />
      ))}

      {adding ? (
        <div className="flex items-center gap-1 px-2 py-1">
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commitAdd()
              if (e.key === 'Escape') cancelAdd()
            }}
            onBlur={cancelAdd}
            placeholder="pxt://org:db"
            className="h-6 flex-1 rounded border border-border/40 bg-background/50 px-1.5 text-[11px] text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-ring/30"
          />
          <button
            // onMouseDown fires before the input's onBlur, so the click isn't swallowed by the cancel
            onMouseDown={(e) => {
              e.preventDefault()
              commitAdd()
            }}
            title="Add catalog"
            className="shrink-0 text-muted-foreground/60 hover:text-foreground"
          >
            <Check className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : (
        <button
          onClick={() => setAdding(true)}
          className="flex w-full items-center gap-1.5 rounded-md px-2 py-1 text-[12px] text-muted-foreground/70 transition-colors hover:bg-accent hover:text-foreground"
        >
          <Plus className="h-3 w-3 shrink-0" />
          <span>Add catalog</span>
        </button>
      )}
    </div>
  )
}
