// src/lib/types.ts
export type SearchType = 'text' | 'image'

export interface SearchResult {
  frame: string
  confidence: number
}

export interface VideoProcessResponse {
  message: string
  success: boolean
}

export interface SearchResponse {
  frames: string[]
  success: boolean
}
