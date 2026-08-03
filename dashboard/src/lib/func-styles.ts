export const FUNC_STYLES: Record<string, { text: string; bg: string; label: string }> = {
  builtin: { text: 'text-muted-foreground', bg: 'bg-muted/40', label: 'built-in' },
  custom_udf: { text: 'text-foreground', bg: 'bg-k-yellow/10', label: 'UDF' },
  query: { text: 'text-muted-foreground', bg: 'bg-blue-400/10', label: 'query' },
  iterator: { text: 'text-muted-foreground', bg: 'bg-violet-400/10', label: 'iterator' },
  unknown: { text: 'text-muted-foreground', bg: 'bg-muted/30', label: 'fn' },
}
