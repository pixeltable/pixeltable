import type React from 'react'

// Light-mode colors target -700 for AA contrast on white/near-white surfaces;
// dark-mode colors target -300/-400 for contrast on near-black surfaces.
const PY_TOKENS: [RegExp, string][] = [
  [/('[^']*'|"[^"]*")/g, 'text-emerald-700 dark:text-emerald-400'],
  [/\b(\d+\.?\d*)\b/g, 'text-blue-700 dark:text-blue-400'],
  [/\b(True|False|None)\b/g, 'text-amber-700 dark:text-amber-400'],
  [/([a-zA-Z_]\w*)(?=\s*\()/g, 'text-purple-700 dark:text-purple-300'],
  [/([[\](),=])/g, 'text-muted-foreground'],
]

const OPEN_TO_CLOSE: Record<string, string> = { '(': ')', '[': ']', '{': '}' }
const CLOSERS = new Set([')', ']', '}'])

function isWordChar(ch: string): boolean {
  return /[A-Za-z0-9_]/.test(ch)
}

/**
 * Lightweight pretty-printer for Pixeltable computed_with / Python-ish call strings.
 * Indent nested (), [], {}; respect string literals. Fail-soft → original on imbalance.
 */
export function formatPythonExpr(code: string): string {
  const src = code.trim()
  if (!src) return code

  let out = ''
  let depth = 0
  let i = 0
  let quote: '"' | "'" | null = null
  let escape = false
  /** True when the previous source characters we skipped were whitespace. */
  let skippedWs = false
  let lastNonWsChar = ''
  const stack: string[] = []

  const indent = (): string => '  '.repeat(depth)

  const peekNonWs = (from: number): string => {
    let j = from
    while (j < src.length && /\s/.test(src[j]!)) j++
    return src[j] ?? ''
  }

  const emit = (text: string) => {
    out += text
    skippedWs = false
    for (let k = text.length - 1; k >= 0; k--) {
      const c = text[k]!
      if (!/\s/.test(c)) {
        lastNonWsChar = c
        break
      }
    }
  }

  while (i < src.length) {
    const ch = src[i]!

    if (quote !== null) {
      emit(ch)
      if (escape) escape = false
      else if (ch === '\\') escape = true
      else if (ch === quote) quote = null
      i++
      continue
    }

    if (ch === '"' || ch === "'") {
      if (skippedWs && lastNonWsChar && isWordChar(lastNonWsChar)) emit(' ')
      quote = ch
      emit(ch)
      i++
      continue
    }

    if (/\s/.test(ch)) {
      skippedWs = true
      i++
      continue
    }

    if (ch in OPEN_TO_CLOSE) {
      const closer = OPEN_TO_CLOSE[ch]!
      stack.push(closer)
      emit(ch)
      depth++
      i++
      const next = peekNonWs(i)
      if (next && next !== closer) {
        emit('\n' + indent())
      }
      continue
    }

    if (CLOSERS.has(ch)) {
      if (stack.length === 0 || stack[stack.length - 1] !== ch) return code
      stack.pop()
      const prev = lastNonWsChar
      depth = Math.max(0, depth - 1)
      if (prev && !(prev in OPEN_TO_CLOSE)) {
        // Trim trailing indent whitespace, then break before closer.
        out = out.replace(/[ \t]+$/u, '')
        if (!out.endsWith('\n')) emit('\n')
        emit(indent() + ch)
      } else {
        emit(ch)
      }
      i++
      continue
    }

    if (ch === ',' && depth >= 1) {
      emit(',')
      i++
      while (i < src.length && /\s/.test(src[i]!)) i++
      if (i < src.length && !CLOSERS.has(src[i]!)) {
        emit('\n' + indent())
      }
      continue
    }

    if (ch === ':' && depth >= 1) {
      emit(': ')
      i++
      while (i < src.length && /\s/.test(src[i]!)) i++
      continue
    }

    if (ch === '=') {
      emit('=')
      i++
      while (i < src.length && /\s/.test(src[i]!)) i++
      continue
    }

    // Preserve a single space only when the source had whitespace between word tokens.
    if (skippedWs && lastNonWsChar && isWordChar(lastNonWsChar) && isWordChar(ch)) emit(' ')

    emit(ch)
    i++
  }

  if (quote !== null || stack.length > 0) return code
  return out
}

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
    if (p.idx > cursor) result.push(<span key={`t${cursor}`} className="text-foreground">{code.slice(cursor, p.idx)}</span>)
    result.push(<span key={`h${p.idx}`} className={p.cls}>{p.text}</span>)
    cursor = p.idx + p.text.length
  }
  if (cursor < code.length) result.push(<span key="tail" className="text-foreground">{code.slice(cursor)}</span>)

  return <code className={className ?? 'text-[11px] font-mono leading-relaxed'}>{result}</code>
}
