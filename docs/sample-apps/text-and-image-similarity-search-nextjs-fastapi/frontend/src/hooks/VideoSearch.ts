// src/hooks/useVideoSearch.ts
import { useState } from 'react'
import { SearchType } from '@/lib/types'
import { processVideo, searchFrames } from '@/lib/api'

export function useVideoSearch() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [searchType, setSearchType] = useState<SearchType>('text')
  const [textQuery, setTextQuery] = useState('')
  const [imageQuery, setImageQuery] = useState<File | null>(null)
  const [numResults, setNumResults] = useState(5)
  const [status, setStatus] = useState('')
  const [results, setResults] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const handleVideoUpload = async (file: File) => {
    setVideoFile(file)
  }

  const handleProcessVideo = async () => {
    if (!videoFile) return

    setIsLoading(true)
    setStatus('Processing video...')
    setProgress(0)
    setError(null)

    try {
      const response = await processVideo(videoFile)
      setStatus(response.message)
      setProgress(100)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process video')
      setStatus('Error processing video')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSearch = async () => {
    if (searchType === 'text' && !textQuery) return
    if (searchType === 'image' && !imageQuery) return

    setIsLoading(true)
    setStatus('Searching...')
    setError(null)

    try {
      const query = searchType === 'text' ? textQuery : imageQuery!
      const response = await searchFrames(query, searchType, numResults)
      setResults(response.frames)
      setStatus('Search complete')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
      setStatus('Error performing search')
    } finally {
      setIsLoading(false)
    }
  }

  return {
    videoFile,
    searchType,
    textQuery,
    imageQuery,
    numResults,
    status,
    results,
    isLoading,
    progress,
    error,
    setVideoFile,
    setSearchType,
    setTextQuery,
    setImageQuery,
    setNumResults,
    handleVideoUpload,
    handleProcessVideo,
    handleSearch,
  }
}