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
  [/^image/i,      { icon: ImageIcon,    color: 'text-pink-400',    bg: 'bg-pink-500/10',    label: 'Image' }],
  [/^video/i,      { icon: Film,         color: 'text-violet-400',  bg: 'bg-violet-500/10',  label: 'Video' }],
  [/^audio/i,      { icon: Music,        color: 'text-teal-400',    bg: 'bg-teal-500/10',    label: 'Audio' }],
  [/^document/i,   { icon: FileText,     color: 'text-orange-400',  bg: 'bg-orange-500/10',  label: 'Document' }],
  [/^string/i,     { icon: Type,         color: 'text-emerald-400', bg: 'bg-emerald-500/10', label: 'String' }],
  [/^int/i,        { icon: Hash,         color: 'text-blue-400',    bg: 'bg-blue-500/10',    label: 'Int' }],
  [/^float/i,      { icon: Hash,         color: 'text-sky-400',     bg: 'bg-sky-500/10',     label: 'Float' }],
  [/^bool/i,       { icon: ToggleLeft,   color: 'text-amber-400',   bg: 'bg-amber-500/10',   label: 'Bool' }],
  [/^timestamp/i,  { icon: Calendar,     color: 'text-orange-400',  bg: 'bg-orange-500/10',  label: 'Timestamp' }],
  [/^date/i,       { icon: CalendarDays, color: 'text-orange-300',  bg: 'bg-orange-500/10',  label: 'Date' }],
  [/^uuid/i,       { icon: Fingerprint,  color: 'text-indigo-400',  bg: 'bg-indigo-500/10',  label: 'UUID' }],
  [/^json/i,       { icon: Braces,       color: 'text-yellow-400',  bg: 'bg-yellow-500/10',  label: 'Json' }],
  [/^array/i,      { icon: List,         color: 'text-cyan-400',    bg: 'bg-cyan-500/10',    label: 'Array' }],
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
  const { icon: Icon, color, bg } = getColumnTypeMeta(type)
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-mono ${bg} ${color}`}>
      <Icon className="h-3 w-3 shrink-0" />
      {type}
    </span>
  )
}

