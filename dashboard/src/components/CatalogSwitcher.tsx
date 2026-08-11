import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { cn } from '@/lib/utils'
import {
  LOCAL_CATALOG,
  catalogLabel,
  loadExtraCatalogs,
  saveExtraCatalogs,
} from '@/lib/catalogs'
import {
  Check,
  ChevronDown,
  Cloud,
  HardDrive,
  Plus,
  X,
} from 'lucide-react'

interface CatalogSwitcherProps {
  activeCatalog: string
  onSelect: (uri: string) => void
  collapsed?: boolean
}

export function CatalogSwitcher({ activeCatalog, onSelect, collapsed = false }: CatalogSwitcherProps) {
  const [catalogs, setCatalogs] = useState<string[]>(loadExtraCatalogs)
  const [open, setOpen] = useState(false)
  const [adding, setAdding] = useState(false)
  const [draft, setDraft] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    saveExtraCatalogs(catalogs)
  }, [catalogs])

  // Drop a stale active selection if the saved list no longer contains it.
  useEffect(() => {
    if (activeCatalog !== LOCAL_CATALOG && !catalogs.includes(activeCatalog)) {
      onSelect(LOCAL_CATALOG)
    }
  }, [activeCatalog, catalogs, onSelect])

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false)
        setAdding(false)
        setDraft('')
      }
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open])

  useEffect(() => {
    if (adding) inputRef.current?.focus()
  }, [adding])

  const cancelAdd = () => {
    setAdding(false)
    setDraft('')
  }

  const commitAdd = () => {
    const uri = draft.trim()
    if (uri === '' || uri === LOCAL_CATALOG) {
      cancelAdd()
      return
    }
    // Hosted catalogs must use the pxt:// URI form the daemon expects.
    if (!uri.startsWith('pxt://')) return
    if (!catalogs.includes(uri)) {
      setCatalogs(prev => [...prev, uri])
    }
    onSelect(uri)
    setOpen(false)
    cancelAdd()
  }

  const removeCatalog = (uri: string) => {
    setCatalogs(prev => prev.filter(c => c !== uri))
    if (activeCatalog === uri) onSelect(LOCAL_CATALOG)
  }

  const select = (uri: string) => {
    onSelect(uri)
    setOpen(false)
    cancelAdd()
  }

  const isCloud = activeCatalog !== LOCAL_CATALOG
  const Icon = isCloud ? Cloud : HardDrive

  if (collapsed) {
    return (
      <div ref={rootRef} className="relative mb-1 flex justify-center">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          title={catalogLabel(activeCatalog)}
          className={cn(
            'flex items-center justify-center rounded-lg px-2.5 py-[7px] transition-colors',
            'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
            open && 'bg-accent text-foreground',
          )}
        >
          <Icon className="h-[15px] w-[15px]" />
        </button>
        {open && (
          <Dropdown
            activeCatalog={activeCatalog}
            catalogs={catalogs}
            adding={adding}
            draft={draft}
            inputRef={inputRef}
            onSelect={select}
            onRemove={removeCatalog}
            onStartAdd={() => setAdding(true)}
            onDraftChange={setDraft}
            onCommitAdd={commitAdd}
            onCancelAdd={cancelAdd}
            align="left"
          />
        )}
      </div>
    )
  }

  return (
    <div ref={rootRef} className="relative mb-1 shrink-0">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className={cn(
          'flex w-full items-center gap-2 rounded-lg border border-border/40 bg-background/40 px-2.5 py-[7px]',
          'text-[13px] font-medium text-foreground transition-colors',
          'hover:bg-accent/50',
          open && 'bg-accent/60',
        )}
        title={activeCatalog === LOCAL_CATALOG ? 'Local catalog' : activeCatalog}
      >
        <Icon className={cn('h-3.5 w-3.5 shrink-0', isCloud ? 'text-sky-400' : 'text-k-yellow')} />
        <span className="flex-1 truncate text-left">{catalogLabel(activeCatalog)}</span>
        <ChevronDown className={cn('h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <Dropdown
          activeCatalog={activeCatalog}
          catalogs={catalogs}
          adding={adding}
          draft={draft}
          inputRef={inputRef}
          onSelect={select}
          onRemove={removeCatalog}
          onStartAdd={() => setAdding(true)}
          onDraftChange={setDraft}
          onCommitAdd={commitAdd}
          onCancelAdd={cancelAdd}
        />
      )}
    </div>
  )
}

function Dropdown({
  activeCatalog,
  catalogs,
  adding,
  draft,
  inputRef,
  onSelect,
  onRemove,
  onStartAdd,
  onDraftChange,
  onCommitAdd,
  onCancelAdd,
  align = 'stretch',
}: {
  activeCatalog: string
  catalogs: string[]
  adding: boolean
  draft: string
  inputRef: RefObject<HTMLInputElement>
  onSelect: (uri: string) => void
  onRemove: (uri: string) => void
  onStartAdd: () => void
  onDraftChange: (v: string) => void
  onCommitAdd: () => void
  onCancelAdd: () => void
  align?: 'stretch' | 'left'
}) {
  return (
    <div
      className={cn(
        'absolute z-50 mt-1 rounded-lg border border-border/60 bg-card shadow-lg py-1',
        align === 'stretch' ? 'left-0 right-0' : 'left-0 min-w-[220px]',
      )}
    >
      <CatalogRow
        label="Local"
        icon={<HardDrive className="h-3.5 w-3.5 text-k-yellow shrink-0" />}
        active={activeCatalog === LOCAL_CATALOG}
        onSelect={() => onSelect(LOCAL_CATALOG)}
      />
      {catalogs.map(uri => (
        <CatalogRow
          key={uri}
          label={catalogLabel(uri)}
          title={uri}
          icon={<Cloud className="h-3.5 w-3.5 text-sky-400 shrink-0" />}
          active={activeCatalog === uri}
          removable
          onSelect={() => onSelect(uri)}
          onRemove={() => onRemove(uri)}
        />
      ))}
      <div className="my-1 mx-2 h-px bg-border/40" />
      {adding ? (
        <div className="flex items-center gap-1 px-2 py-1">
          <input
            ref={inputRef}
            value={draft}
            onChange={e => onDraftChange(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') onCommitAdd()
              if (e.key === 'Escape') onCancelAdd()
            }}
            placeholder="pxt://org:db"
            className="h-7 flex-1 rounded border border-border/40 bg-background/50 px-1.5 text-[11px] text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-ring/30"
          />
          <button
            type="button"
            onMouseDown={e => {
              e.preventDefault()
              onCommitAdd()
            }}
            title="Add cloud catalog"
            className="shrink-0 text-muted-foreground/60 hover:text-foreground p-1"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={onStartAdd}
          className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-[12px] text-muted-foreground/80 hover:bg-accent hover:text-foreground transition-colors"
        >
          <Plus className="h-3 w-3 shrink-0" />
          <span>Add cloud catalog…</span>
        </button>
      )}
    </div>
  )
}

function CatalogRow({
  label,
  title,
  icon,
  active,
  removable,
  onSelect,
  onRemove,
}: {
  label: string
  title?: string
  icon: ReactNode
  active: boolean
  removable?: boolean
  onSelect: () => void
  onRemove?: () => void
}) {
  return (
    <div className="group flex items-center gap-1 px-1">
      <button
        type="button"
        onClick={onSelect}
        title={title}
        className={cn(
          'flex flex-1 min-w-0 items-center gap-2 rounded-md px-1.5 py-1.5 text-[12px] transition-colors',
          active ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground',
        )}
      >
        {icon}
        <span className="flex-1 truncate text-left font-medium">{label}</span>
        {active && <Check className="h-3 w-3 shrink-0 text-foreground" />}
      </button>
      {removable && onRemove && (
        <button
          type="button"
          onClick={onRemove}
          title="Remove catalog"
          className="opacity-0 group-hover:opacity-100 shrink-0 p-1 text-muted-foreground/60 hover:text-destructive transition-opacity"
        >
          <X className="h-3 w-3" />
        </button>
      )}
    </div>
  )
}
