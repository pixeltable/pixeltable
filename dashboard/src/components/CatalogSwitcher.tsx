import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { cn } from '@/lib/utils'
import { getDirectoryTree } from '@/api/client'
import {
  LOCAL_CATALOG,
  catalogLabel,
  loadExtraCatalogs,
  normalizeCloudCatalogUri,
  saveExtraCatalogs,
} from '@/lib/catalogs'
import {
  Check,
  ChevronDown,
  Cloud,
  HardDrive,
  Loader2,
  Plus,
  X,
} from 'lucide-react'

const URI_HINT = 'Use a hosted URI like pxt://org:db'

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
  const [addError, setAddError] = useState<string | null>(null)
  const [addingBusy, setAddingBusy] = useState(false)
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
        setAddError(null)
        setAddingBusy(false)
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
    setAddError(null)
    setAddingBusy(false)
  }

  const handleDraftChange = (v: string) => {
    setDraft(v)
    if (addError) setAddError(null)
  }

  const commitAdd = async () => {
    if (addingBusy) return
    const raw = draft.trim()
    if (raw === '') {
      cancelAdd()
      return
    }

    const uri = normalizeCloudCatalogUri(raw)
    if (uri === null) {
      setAddError(URI_HINT)
      return
    }

    setAddingBusy(true)
    setAddError(null)
    try {
      await getDirectoryTree(uri)
      if (!catalogs.includes(uri)) {
        setCatalogs(prev => [...prev, uri])
      }
      onSelect(uri)
      setOpen(false)
      cancelAdd()
    } catch (e) {
      setAddError(e instanceof Error ? e.message : 'Could not open catalog')
      setAddingBusy(false)
    }
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

  const startAdd = () => {
    setAdding(true)
    setAddError(null)
  }

  const isCloud = activeCatalog !== LOCAL_CATALOG
  const Icon = isCloud ? Cloud : HardDrive
  const draftInvalid = draft.trim() !== '' && normalizeCloudCatalogUri(draft) === null

  const dropdown = open ? (
    <Dropdown
      activeCatalog={activeCatalog}
      catalogs={catalogs}
      adding={adding}
      draft={draft}
      addError={addError}
      addingBusy={addingBusy}
      draftInvalid={draftInvalid}
      inputRef={inputRef}
      onSelect={select}
      onRemove={removeCatalog}
      onStartAdd={startAdd}
      onDraftChange={handleDraftChange}
      onCommitAdd={() => { void commitAdd() }}
      onCancelAdd={cancelAdd}
      align={collapsed ? 'left' : 'stretch'}
    />
  ) : null

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
            open && 'bg-accent text-foreground ring-1 ring-k-yellow/30',
          )}
        >
          <Icon className="h-[15px] w-[15px]" />
        </button>
        {dropdown}
      </div>
    )
  }

  return (
    <div ref={rootRef} className="relative mb-1 shrink-0">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className={cn(
          'flex w-full items-center gap-2 rounded-lg border bg-background/40 px-2.5 py-[7px]',
          'text-[13px] font-medium text-foreground transition-colors',
          'hover:bg-accent/50',
          open
            ? 'border-k-yellow/40 bg-accent/60 ring-1 ring-k-yellow/30'
            : 'border-border/40',
        )}
        title={activeCatalog === LOCAL_CATALOG ? 'Local catalog' : activeCatalog}
      >
        <Icon className={cn('h-3.5 w-3.5 shrink-0', isCloud ? 'text-sky-400' : 'text-k-yellow')} />
        <span className="flex-1 truncate text-left">{catalogLabel(activeCatalog)}</span>
        <ChevronDown className={cn('h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')} />
      </button>
      {dropdown}
    </div>
  )
}

function Dropdown({
  activeCatalog,
  catalogs,
  adding,
  draft,
  addError,
  addingBusy,
  draftInvalid,
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
  addError: string | null
  addingBusy: boolean
  draftInvalid: boolean
  inputRef: RefObject<HTMLInputElement>
  onSelect: (uri: string) => void
  onRemove: (uri: string) => void
  onStartAdd: () => void
  onDraftChange: (v: string) => void
  onCommitAdd: () => void
  onCancelAdd: () => void
  align?: 'stretch' | 'left'
}) {
  const showError = addError !== null
  const inputBorder = showError || draftInvalid
    ? 'border-destructive/50 focus:ring-destructive/30'
    : 'border-border/40 focus:ring-ring/30'

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
        <div className="px-2 py-1.5 space-y-1">
          <div className="flex items-center gap-1">
            <input
              ref={inputRef}
              value={draft}
              disabled={addingBusy}
              onChange={e => onDraftChange(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') onCommitAdd()
                if (e.key === 'Escape') onCancelAdd()
              }}
              placeholder="pxt://org:db"
              aria-invalid={showError || draftInvalid}
              className={cn(
                'h-7 flex-1 rounded border bg-background/50 px-1.5 text-[11px] text-foreground',
                'placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 disabled:opacity-60',
                inputBorder,
              )}
            />
            <button
              type="button"
              disabled={addingBusy}
              onMouseDown={e => {
                e.preventDefault()
                onCommitAdd()
              }}
              title="Add cloud catalog"
              className={cn(
                'shrink-0 p-1 transition-colors disabled:opacity-50',
                showError || draftInvalid
                  ? 'text-destructive/70 hover:text-destructive'
                  : 'text-muted-foreground/60 hover:text-foreground',
              )}
            >
              {addingBusy
                ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                : <Plus className="h-3.5 w-3.5" />}
            </button>
          </div>
          <p className={cn(
            'text-[10px] px-0.5 leading-snug',
            showError ? 'text-destructive' : 'text-muted-foreground',
          )}>
            {addError ?? URI_HINT}
          </p>
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
