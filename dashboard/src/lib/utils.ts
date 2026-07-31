import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Router hrefs for a catalog path. The path is encoded into a single segment so a hosted uri's '//'
// (e.g. pxt://org:db/t) survives: react-router collapses an embedded '//' in a raw pathname, which would
// corrupt the path before it reaches the API. react-router decodes the splat param back to the raw path.
export function tableHref(path: string): string {
  return `/table/${encodeURIComponent(path)}`
}

export function dirHref(path: string): string {
  return `/dir/${encodeURIComponent(path)}`
}
