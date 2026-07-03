import {
  ImageIcon, Film, Music, FileText,
  Hash, Type, Calendar, CalendarDays, ToggleLeft,
  Braces, List, Binary, Fingerprint,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

type TypeMeta = {
  icon: LucideIcon
  color: string
  bg: string
  label: string
}

const TYPE_MAP: [RegExp, TypeMeta][] = [
  [/^image/i,      { icon: ImageIcon,    color: 'text-muted-foreground',    bg: 'bg-pink-500/10',    label: 'Image' }],
  [/^video/i,      { icon: Film,         color: 'text-muted-foreground',  bg: 'bg-violet-500/10',  label: 'Video' }],
  [/^audio/i,      { icon: Music,        color: 'text-muted-foreground',    bg: 'bg-teal-500/10',    label: 'Audio' }],
  [/^document/i,   { icon: FileText,     color: 'text-muted-foreground',  bg: 'bg-orange-500/10',  label: 'Document' }],
  [/^string/i,     { icon: Type,         color: 'text-muted-foreground', bg: 'bg-emerald-500/10', label: 'String' }],
  [/^int/i,        { icon: Hash,         color: 'text-muted-foreground',    bg: 'bg-blue-500/10',    label: 'Int' }],
  [/^float/i,      { icon: Hash,         color: 'text-muted-foreground',     bg: 'bg-sky-500/10',     label: 'Float' }],
  [/^bool/i,       { icon: ToggleLeft,   color: 'text-muted-foreground',   bg: 'bg-amber-500/10',   label: 'Bool' }],
  [/^timestamp/i,  { icon: Calendar,     color: 'text-muted-foreground',  bg: 'bg-orange-500/10',  label: 'Timestamp' }],
  [/^date/i,       { icon: CalendarDays, color: 'text-muted-foreground',  bg: 'bg-orange-500/10',  label: 'Date' }],
  [/^uuid/i,       { icon: Fingerprint,  color: 'text-muted-foreground',  bg: 'bg-indigo-500/10',  label: 'UUID' }],
  [/^json/i,       { icon: Braces,       color: 'text-muted-foreground',  bg: 'bg-yellow-500/10',  label: 'Json' }],
  [/^array/i,      { icon: List,         color: 'text-muted-foreground',    bg: 'bg-cyan-500/10',    label: 'Array' }],
]

const FALLBACK: TypeMeta = { icon: Binary, color: 'text-muted-foreground', bg: 'bg-accent', label: '?' }

export function getColumnTypeMeta(type: string): TypeMeta {
  const clean = (type || '').trim()
  for (const [re, meta] of TYPE_MAP) {
    if (re.test(clean)) return meta
  }
  return FALLBACK
}

export function ColumnTypeIcon({ type, className = 'h-3.5 w-3.5' }: { type: string; className?: string }) {
  const { icon: Icon, color } = getColumnTypeMeta(type)
  return <Icon className={`${className} ${color} shrink-0`} />
}

export function ColumnTypeBadge({ type }: { type: string }) {
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-mono bg-muted/40 text-muted-foreground">
      {type}
    </span>
  )
}

