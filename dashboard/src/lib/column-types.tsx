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
  /** Media types keep semantic color; scalars use quiet bordered pills. */
  emphasis: 'media' | 'quiet'
}

const QUIET = {
  color: 'text-muted-foreground',
  bg: 'bg-muted/20 border border-border/40',
  emphasis: 'quiet' as const,
}

/** Match bare media names and Required[…] / Optional[…] wrappers (AA: -700/-800 light, -400 dark). */
const MEDIA_PREFIX = '^(?:(?:required|optional)\\s*\\[\\s*)?'

const TYPE_MAP: [RegExp, TypeMeta][] = [
  [new RegExp(`${MEDIA_PREFIX}image`, 'i'), {
    icon: ImageIcon,
    color: 'text-pink-700 dark:text-pink-400',
    bg: 'bg-pink-500/10 border border-pink-500/20',
    label: 'Image',
    emphasis: 'media',
  }],
  [new RegExp(`${MEDIA_PREFIX}video`, 'i'), {
    icon: Film,
    color: 'text-violet-700 dark:text-violet-400',
    bg: 'bg-violet-500/10 border border-violet-500/20',
    label: 'Video',
    emphasis: 'media',
  }],
  [new RegExp(`${MEDIA_PREFIX}audio`, 'i'), {
    icon: Music,
    color: 'text-teal-700 dark:text-teal-400',
    bg: 'bg-teal-500/10 border border-teal-500/20',
    label: 'Audio',
    emphasis: 'media',
  }],
  [new RegExp(`${MEDIA_PREFIX}document`, 'i'), {
    icon: FileText,
    color: 'text-orange-800 dark:text-orange-400',
    bg: 'bg-orange-500/10 border border-orange-500/20',
    label: 'Document',
    emphasis: 'media',
  }],
  [new RegExp(`${MEDIA_PREFIX}string`, 'i'), { icon: Type, ...QUIET, label: 'String' }],
  [new RegExp(`${MEDIA_PREFIX}int`, 'i'), { icon: Hash, ...QUIET, label: 'Int' }],
  [new RegExp(`${MEDIA_PREFIX}float`, 'i'), { icon: Hash, ...QUIET, label: 'Float' }],
  [new RegExp(`${MEDIA_PREFIX}bool`, 'i'), { icon: ToggleLeft, ...QUIET, label: 'Bool' }],
  [new RegExp(`${MEDIA_PREFIX}timestamp`, 'i'), { icon: Calendar, ...QUIET, label: 'Timestamp' }],
  [new RegExp(`${MEDIA_PREFIX}date`, 'i'), { icon: CalendarDays, ...QUIET, label: 'Date' }],
  [new RegExp(`${MEDIA_PREFIX}uuid`, 'i'), { icon: Fingerprint, ...QUIET, label: 'UUID' }],
  [new RegExp(`${MEDIA_PREFIX}json`, 'i'), { icon: Braces, ...QUIET, label: 'Json' }],
  [new RegExp(`${MEDIA_PREFIX}array`, 'i'), { icon: List, ...QUIET, label: 'Array' }],
]

const FALLBACK: TypeMeta = { icon: Binary, ...QUIET, label: '?' }

export function getColumnTypeMeta(type: string): TypeMeta {
  const clean = (type || '').trim()
  for (const [re, meta] of TYPE_MAP) {
    if (re.test(clean)) return meta
  }
  return FALLBACK
}

/** Short label for schema UI — nested Json/Array payloads collapse; full string stays in title. */
export function formatColumnTypeDisplay(type: string, maxLen = 36): string {
  const t = (type || '').trim()
  if (!t) return '?'
  if (/^json\s*[[{]/i.test(t)) return 'Json'
  if (/^array\s*[[]/i.test(t)) return 'Array'
  if (/^required\[\s*json/i.test(t)) return 'Required[Json]'
  if (/^required\[\s*array/i.test(t)) return 'Required[Array]'
  if (/^optional\[\s*json/i.test(t)) return 'Optional[Json]'
  if (/^optional\[\s*array/i.test(t)) return 'Optional[Array]'
  if (t.length > maxLen) return `${t.slice(0, maxLen - 1)}…`
  return t
}

export function ColumnTypeIcon({ type, className = 'h-3.5 w-3.5' }: { type: string; className?: string }) {
  const { icon: Icon, color } = getColumnTypeMeta(type)
  return <Icon className={`${className} ${color} shrink-0`} />
}

export function ColumnTypeBadge({ type }: { type: string }) {
  const { icon: Icon, color, bg } = getColumnTypeMeta(type)
  const display = formatColumnTypeDisplay(type)
  return (
    <span
      className={`inline-flex max-w-full items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-mono ${bg} ${color}`}
      title={display !== type ? type : undefined}
    >
      <Icon className="h-3 w-3 shrink-0" />
      <span className="truncate">{display}</span>
    </span>
  )
}
