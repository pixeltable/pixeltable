export const FUNC_STYLES: Record<string, { text: string; bg: string; label: string }> = {
  builtin: { text: 'text-muted-foreground/80', bg: 'bg-muted/40', label: 'built-in' },
  custom_udf: { text: 'text-k-yellow/90', bg: 'bg-k-yellow/10', label: 'UDF' },
  query: { text: 'text-blue-400/90', bg: 'bg-blue-400/10', label: 'query' },
  iterator: { text: 'text-violet-400/90', bg: 'bg-violet-400/10', label: 'iterator' },
  unknown: { text: 'text-muted-foreground/60', bg: 'bg-muted/30', label: 'fn' },
}
