// src/app/page.tsx
'use client'

'use client'

import { useState } from 'react'

type SearchType = 'text' | 'image'
type AccordionState = {
  howItWorks: boolean
  whatItDoes: boolean
}

export default function VideoSearch() {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [searchType, setSearchType] = useState<SearchType>('text')
  const [textQuery, setTextQuery] = useState('')
  const [imageQuery, setImageQuery] = useState<File | null>(null)
  const [numResults, setNumResults] = useState(5)
  const [status, setStatus] = useState('')
  const [results, setResults] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [accordionState, setAccordionState] = useState<AccordionState>({
    howItWorks: false,
    whatItDoes: false,
  })

  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return
    setVideoFile(e.target.files[0])
  }

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

  // Validate file type
  if (!file.type.startsWith('image/')) {
      setStatus('Please select a valid image file')
      return
  }

  setImageQuery(file)
  setStatus('Image selected for search')
}

  const processVideo = async () => {
    if (!videoFile) return

    setIsLoading(true)
    setStatus('Processing video...')
    setProgress(0)

    const formData = new FormData()
    formData.append('file', videoFile)

    try {
      const response = await fetch('http://localhost:8000/api/process-video', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Failed to process video')

      const data = await response.json()
      setStatus(data.message)
      setProgress(100)
    } catch (error) {
      setStatus('Error processing video')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  const performSearch = async () => {
    if (searchType === 'text' && !textQuery) {
        setStatus('Please enter a text query')
        return
    }
    if (searchType === 'image' && !imageQuery) {
        setStatus('Please select an image')
        return
    }

    setIsLoading(true)
    setStatus('Searching...')

    try {
        const formData = new FormData()

        if (searchType === 'text') {
            // For text search, we'll send the query in the formData
            formData.append('query', new File([textQuery], textQuery, { type: 'text/plain' }))
        } else if (imageQuery) {
            // For image search, send the image file
            formData.append('query', imageQuery)
        }
        
        formData.append('search_type', searchType)
        formData.append('num_results', numResults.toString())

        const response = await fetch('http://localhost:8000/api/search', {
            method: 'POST',
            body: formData,
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Search failed' }))
            throw new Error(errorData.detail || 'Search failed')
        }

        const data = await response.json()
        
        if (!data.success) {
            throw new Error(data.detail || 'Search failed')
        }

        if (data.frames && Array.isArray(data.frames)) {
            setResults(data.frames)
            setStatus(`Found ${data.frames.length} matching frames`)
        } else {
            setResults([])
            setStatus('No matches found')
        }
    } catch (error) {
        console.error('Search error:', error)
        setStatus(error instanceof Error ? error.message : 'Search failed')
        setResults([])
    } finally {
        setIsLoading(false)
    }
}

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8 text-center">
          <img 
            src="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/source/data/pixeltable-logo-large.png"
            alt="Pixeltable"
            className="h-12 mb-4 mx-auto"
          />
          <h1 className="text-4xl font-bold mb-2 text-black-600">
          Text and Image similarity search with embedding indexes
          </h1>
          <p className="text-black-600 max-w-2xl mx-auto">
          Pixeltable is a declarative interface for working with text, images, embeddings, and even video, enabling you to store, transform, index, and iterate on data.
          </p>
        </div>

        {/* Info Sections */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          {/* How it Works */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <button 
              onClick={() => setAccordionState(prev => ({
                ...prev,
                howItWorks: !prev.howItWorks
              }))}
              className="w-full text-left px-6 py-4 flex justify-between items-center bg-gray-50 hover:bg-gray-100 transition-colors"
            >
              <h3 className="text-lg font-semibold text-gray-800">How It Works</h3>
              <svg 
                className={`w-5 h-5 transform transition-transform ${
                  accordionState.howItWorks ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div className={`px-6 py-4 transition-all duration-300 ease-in-out ${
              accordionState.howItWorks ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0 overflow-hidden'
            }`}>
              <ol className="space-y-3 text-gray-600">
                <li className="flex items-start">
                  <span className="flex items-center justify-center w-6 h-6 bg-blue-100 rounded-full text-blue-600 text-sm font-semibold mr-3 shrink-0 mt-0.5">1</span>
                  <span>Each video frame is analyzed and indexed for efficient searching</span>
                </li>
                <li className="flex items-start">
                  <span className="flex items-center justify-center w-6 h-6 bg-blue-100 rounded-full text-blue-600 text-sm font-semibold mr-3 shrink-0 mt-0.5">2</span>
                  <span>Embedding generation to match your queries with relevant frames</span>
                </li>
                <li className="flex items-start">
                  <span className="flex items-center justify-center w-6 h-6 bg-blue-100 rounded-full text-blue-600 text-sm font-semibold mr-3 shrink-0 mt-0.5">3</span>
                  <span>Results are ranked by relevance and displayed in a visual grid</span>
                </li>
              </ol>
            </div>
          </div>

          {/* What It Does */}
          <div className="bg-white rounded-xl shadow-md overflow-hidden">
            <button 
              onClick={() => setAccordionState(prev => ({
                ...prev,
                whatItDoes: !prev.whatItDoes
              }))}
              className="w-full text-left px-6 py-4 flex justify-between items-center bg-gray-50 hover:bg-gray-100 transition-colors"
            >
              <h3 className="text-lg font-semibold text-gray-800">What It Does</h3>
              <svg 
                className={`w-5 h-5 transform transition-transform ${
                  accordionState.whatItDoes ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div className={`px-6 py-4 transition-all duration-300 ease-in-out ${
              accordionState.whatItDoes ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0 overflow-hidden'
            }`}>
              <ul className="space-y-3 text-gray-600">
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-green-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Search video frames using natural language descriptions</span>
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-green-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Find visually similar frames using text/image-based search</span>
                </li>
                <li className="flex items-center">
                  <svg className="w-5 h-5 text-green-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Extract and analyze frames</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Control Panel */}
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Search Controls</h2>
              <p className="text-gray-600 text-sm">Upload a video and configure your search</p>
            </div>

            {/* Video Upload Section */}
            <div className="space-y-4 mb-8">
              <div 
                className="border-2 border-dashed border-gray-200 rounded-lg p-6 text-center
                  hover:border-blue-400 transition-colors cursor-pointer"
                onClick={() => document.getElementById('video-upload')?.click()}
              >
                <input
                  id="video-upload"
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  className="hidden"
                />
                <svg
                  className="mx-auto h-12 w-12 text-gray-400"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4-4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p className="mt-2 text-sm text-gray-600">
                  {videoFile ? videoFile.name : 'Drop your video here or click to browse'}
                </p>
              </div>

              <button
                onClick={processVideo}
                disabled={!videoFile || isLoading}
                className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg
                  hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                  transition-colors"
              >
                {isLoading ? 'Processing...' : 'Process Video'}
              </button>

              {progress > 0 && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
            </div>

            {/* Search Configuration */}
            <div className="space-y-4">
              <div className="flex rounded-lg overflow-hidden">
                <button
                  onClick={() => setSearchType('text')}
                  className={`flex-1 py-2 px-4 text-sm font-medium ${
                    searchType === 'text'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  } transition-colors`}
                >
                  Text Search
                </button>
                <button
                  onClick={() => setSearchType('image')}
                  className={`flex-1 py-2 px-4 text-sm font-medium ${
                    searchType === 'image'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  } transition-colors`}
                >
                  Image Search
                </button>
              </div>

              {searchType === 'text' ? (
                  <input
                      type="text"
                      value={textQuery}
                      onChange={(e) => setTextQuery(e.target.value)}
                      placeholder="Describe what you're looking for..."
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 
                          focus:ring-blue-500 focus:border-blue-500"
                  />
              ) : (
                <div className="space-y-4">
            <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center
                    hover:border-blue-500 transition-colors cursor-pointer"
                onClick={() => document.getElementById('image-upload')?.click()}
            >
                <input
                    id="image-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                />
                {imageQuery ? (
                    <div className="space-y-2">
                        <img 
                            src={URL.createObjectURL(imageQuery)} 
                            alt="Search reference"
                            className="max-h-32 mx-auto rounded-lg"
                        />
                        <p className="text-sm text-gray-600">
                            Click to change image
                        </p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        <svg
                            className="mx-auto h-12 w-12 text-gray-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path 
                                strokeLinecap="round" 
                                strokeLinejoin="round" 
                                strokeWidth={2} 
                                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" 
                            />
                        </svg>
                        <p className="text-sm text-gray-600">
                            Upload a reference image to search
                        </p>
                    </div>
                )}
            </div>
        </div>
    )}

              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-700">
                  Number of Results: {numResults}
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={numResults}
                  onChange={(e) => setNumResults(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <button
                onClick={performSearch}
                disabled={isLoading}
                className="w-full py-2 px-4 bg-gray-800 text-white rounded-lg
                  hover:bg-gray-900 disabled:bg-gray-300 disabled:cursor-not-allowed
                  transition-colors"
              >
                {isLoading ? 'Searching...' : 'Search Frames'}
              </button>
            </div>

            {/* Status Messages */}
            {status && (
              <div className={`mt-4 p-4 rounded-lg ${
                status.includes('Error')
                  ? 'bg-red-50 text-red-600'
                  : 'bg-blue-50 text-blue-600'
              }`}>
                {status}
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 bg-white rounded-xl shadow-md p-6">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Search Results</h2>
              <p className="text-gray-600 text-sm">
                {results.length 
                  ? `Found ${results.length} matching frames`
                  : 'Process a video and search to see results'}
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {results.map((frame, index) => (
                <div
                  key={index}
                  className="relative aspect-video group rounded-lg overflow-hidden
                    hover:shadow-lg transition-all duration-300"
                >
                  <img
                    src={frame}
                    alt={`Frame ${index + 1}`}
                    className="object-cover w-full h-full transition-transform duration-300
                      group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-20
                    transition-opacity duration-300" />
                </div>
              ))}
            </div>

            {!results.length && !isLoading && (
              <div className="text-center py-12">
                <svg
                  className="mx-auto h-12 w-12 text-gray-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
                <p className="mt-4 text-gray-600">No results to display yet</p>
                <p className="text-sm text-gray-500">
                  Upload a video and perform a search to see results
                </p>
              </div>
            )}

            {isLoading && (
                <div className="absolute inset-0 bg-white/80 flex items-center justify-center z-10">
                    <div className="flex flex-col items-center space-y-2">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                        <p className="text-sm text-gray-600">{status}</p>
                    </div>
                </div>
            )}
          </div>
        </div>
      </div>
      <footer className="bg-gray-50 border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-12 grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-2">
              <h3 className="text-lg font-semibold text-gray-900">Pixeltable</h3>
              <p className="mt-2 text-base text-gray-500">
              Unifying Data, Models, and Orchestration
              </p>
              <div className="mt-4 flex space-x-6">
                <a href="https://twitter.com/pixeltablehq" className="text-gray-400 hover:text-gray-500">
                  <span className="sr-only">Twitter</span>
                  <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                  </svg>
                </a>
                <a href="https://github.com/pixeltable/pixeltable" className="text-gray-400 hover:text-gray-500">
                  <span className="sr-only">GitHub</span>
                  <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                  </svg>
                </a>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-semibold text-gray-400 tracking-wider uppercase">Resources</h3>
              <ul role="list" className="mt-4 space-y-4">
                <li>
                  <a href="https://docs.pixeltable.com/" className="text-base text-gray-500 hover:text-gray-900">
                    Documentation
                  </a>
                </li>
                <li>
                  <a href="https://pixeltable.github.io/pixeltable/api/pixeltable/" className="text-base text-gray-500 hover:text-gray-900">
                    API Reference
                  </a>
                </li>
                <li>
                  <a href="https://docs.pixeltable.com/docs/pixeltable-basics" className="text-base text-gray-500 hover:text-gray-900">
                    10-minute tour of Pixeltable
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-400 tracking-wider uppercase">Tech Stack</h3>
              <ul role="list" className="mt-4 space-y-4">
                <li>
                  <a href="https://nextjs.org" className="text-base text-gray-500 hover:text-gray-900">
                    Next.js
                  </a>
                </li>
                <li>
                  <a href="https://fastapi.tiangolo.com" className="text-base text-gray-500 hover:text-gray-900">
                    FastAPI
                  </a>
                </li>
                <li>
                  <a href="https://www.pixeltable.com" className="text-base text-gray-500 hover:text-gray-900">
                    Pixeltable
                  </a>
                </li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-200 py-8">
            <p className="text-base text-gray-400 text-center">
              &copy; {new Date().getFullYear()} Pixeltable
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
