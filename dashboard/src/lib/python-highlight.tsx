const PY_TOKENS: [RegExp, string][] = [
  [/('[^']*'|"[^"]*")/g, 'text-emerald-400'],
  [/\b(\d+\.?\d*)\b/g, 'text-blue-400'],
  [/\b(True|False|None)\b/g, 'text-amber-400'],
  [/([a-zA-Z_]\w*)(?=\s*\()/g, 'text-purple-300'],
  [/([[\](),=])/g, 'text-muted-foreground/50'],
]

export function PythonExpr({ code, className }: { code: string; className?: string }) {
  const parts: { text: string; cls: string; idx: number }[] = []
  const used = new Array(code.length).fill(false)

  for (const [re, cls] of PY_TOKENS) {
    let m: RegExpExecArray | null
    re.lastIndex = 0
    while ((m = re.exec(code)) !== null) {
      const start = m.index
      const text = m[1] ?? m[0]
      const end = start + text.length
      if (used.slice(start, end).some(Boolean)) continue
      for (let i = start; i < end; i++) used[i] = true
      parts.push({ text, cls, idx: start })
    }
  }

  parts.sort((a, b) => a.idx - b.idx)

  const result: React.ReactNode[] = []
  let cursor = 0
  for (const p of parts) {
    if (p.idx > cursor) result.push(<span key={`t${cursor}`} className="text-foreground/80">{code.slice(cursor, p.idx)}</span>)
    result.push(<span key={`h${p.idx}`} className={p.cls}>{p.text}</span>)
    cursor = p.idx + p.text.length
  }
  if (cursor < code.length) result.push(<span key="tail" className="text-foreground/80">{code.slice(cursor)}</span>)

  return <code className={className ?? 'text-[11px] font-mono leading-relaxed'}>{result}</code>
}
