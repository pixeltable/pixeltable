// src/lib/api.ts
import { VideoProcessResponse, SearchResponse, SearchType } from './types'

export async function processVideo(file: File): Promise<VideoProcessResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch('http://localhost:8081/api/process-video', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Failed to process video: ${response.statusText}`)
  }

  return response.json()
}

export async function searchFrames(
  query: string | File,
  searchType: SearchType,
  numResults: number
): Promise<SearchResponse> {
  const formData = new FormData()
  
  if (searchType === 'text') {
    formData.append('query', query as string)
  } else {
    formData.append('image', query as File)
  }
  
  formData.append('search_type', searchType)
  formData.append('num_results', numResults.toString())

  const response = await fetch('http://localhost:8081/api/search', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`)
  }

  return response.json()
}