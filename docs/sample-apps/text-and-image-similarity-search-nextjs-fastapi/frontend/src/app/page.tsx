// src/app/page.tsx
'use client'

import React, { useState, useMemo } from 'react'

type SearchType = 'text' | 'image'
type ActiveTab = 'video' | 'image'
type AccordionState = {
  howItWorks: boolean
  whatItDoes: boolean
}

export default function VideoSearch() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('video')
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [searchType, setSearchType] = useState<SearchType>('text')
  const [textQuery, setTextQuery] = useState('')
  const [imageQuery, setImageQuery] = useState<File | null>(null)
  const [numResults, setNumResults] = useState(5)
  const [status, setStatus] = useState('')
  const [results, setResults] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [accordionState, setAccordionState] = useState<AccordionState>({
    howItWorks: false,
    whatItDoes: false,
  })

  // New state specific to Image Search tab
  const [imageUploadFile, setImageUploadFile] = useState<File | null>(null);
  const [imageCategory, setImageCategory] = useState<string>('');
  const [imageTagsInput, setImageTagsInput] = useState<string>('');
  const [imageSearchResults, setImageSearchResults] = useState<{
    image: string;
    tags: string[] | null;
  }[]>([]);

  const [similarityThreshold, setSimilarityThreshold] = useState(0.5); // Add state for threshold

  const [filterTagsInput, setFilterTagsInput] = useState<string>('');
  const [activeTagFilters, setActiveTagFilters] = useState<Set<string>>(new Set()); // State for active tag checkboxes
  const [isDraggingImage, setIsDraggingImage] = useState(false); // State for upload drag feedback
  const [isDraggingSearchImage, setIsDraggingSearchImage] = useState(false); // State for search drag feedback

  // Effect to update active filters when results change
  React.useEffect(() => {
    const allTagsFound = new Set<string>();
    let hasUntagged = false;
    imageSearchResults.forEach(result => {
      if (result.tags && result.tags.length > 0) {
        result.tags.forEach(tag => allTagsFound.add(tag));
      } else {
        hasUntagged = true;
      }
    });

    // Initialize/update filters to include all found tags + Untagged if present
    setActiveTagFilters(prevFilters => {
        // Create a new set based on found tags + Untagged
        const initialFilters = new Set(allTagsFound);
        if (hasUntagged) {
            initialFilters.add('Untagged');
        }
        // Optionally preserve existing selections if needed, but for now, let's reset to all found
        // Or merge: allTagsFound.forEach(tag => newFilters.add(tag));
        return initialFilters; // Reset filters to all found tags + Untagged
    });

  }, [imageSearchResults]); // Dependency array ensures this runs only when results change

  // Calculate grouped image results using useMemo at the top level
  const groupedImageResultsDisplay = useMemo((): React.JSX.Element | false | null => {
    const groupedResults: { [key: string]: typeof imageSearchResults } = {};
    const untaggedResults: typeof imageSearchResults = [];

    imageSearchResults.forEach(result => {
      if (result.tags && result.tags.length > 0) {
        result.tags.forEach(tag => {
          if (!groupedResults[tag]) {
            groupedResults[tag] = [];
          }
          if (!groupedResults[tag].some(r => r.image === result.image)) {
              groupedResults[tag].push(result);
          }
        });
      } else {
          if (!untaggedResults.some(r => r.image === result.image)) {
              untaggedResults.push(result);
          }
      }
    });

    if (imageSearchResults.length === 0) {
      return (
        !isLoading && <p className="text-center text-neutral-500 py-8">No images found or search not performed yet.</p>
      );
    }

    const sortedTags = Object.keys(groupedResults).sort();

    // Filter tags based on activeTagFilters state
    const tagsToDisplay = sortedTags.filter(tag => activeTagFilters.has(tag));
    const showUntagged = untaggedResults.length > 0 && activeTagFilters.has('Untagged');

    // If no tags are selected to be displayed, show a message
    if (tagsToDisplay.length === 0 && !showUntagged) {
        return <p className="text-center text-neutral-500 py-8">No tags selected for display. Check the filters.</p>;
    }

    return (
      <div className="space-y-6">
        {tagsToDisplay.map(tag => ( // Use filtered tags
          <div key={tag}>
            <h4 className="text-md font-semibold text-neutral-700 mb-3 capitalize border-b pb-1">Tag: {tag}</h4>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {groupedResults[tag].map((result, index) => (
                <div key={`${tag}-${index}`} className="group relative aspect-square bg-neutral-100 rounded-lg overflow-hidden shadow hover:shadow-md transition-shadow">
                  <img src={result.image} alt={`Search result ${index + 1}`} className="object-cover w-full h-full" />
                  <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-2 py-1 space-y-0.5 text-white text-xs">
                    {result.tags && result.tags.length > 0 && (
                      <p className="truncate font-medium" title={result.tags.join(', ')}>
                        Tags: {result.tags.join(', ')}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
        {/* Display untagged images if any AND if selected in filters */}
        {showUntagged && (
          <div>
              <h4 className="text-md font-semibold text-neutral-700 mb-3 capitalize border-b pb-1">Untagged</h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {untaggedResults.map((result, index) => (
                  <div key={`untagged-${index}`} className="group relative aspect-square bg-neutral-100 rounded-lg overflow-hidden shadow hover:shadow-md transition-shadow">
                  <img src={result.image} alt={`Search result ${index + 1}`} className="object-cover w-full h-full" />
                  <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-2 py-1 space-y-0.5 text-white text-xs">
                  </div>
                  </div>
              ))}
              </div>
          </div>
        )}
      </div>
    );
  }, [imageSearchResults, isLoading, activeTagFilters]); // Add activeTagFilters to dependency array

  // Handler function for tag filter checkboxes
  const handleTagFilterChange = (tag: string) => {
    setActiveTagFilters(prevFilters => {
        const newFilters = new Set(prevFilters);
        if (newFilters.has(tag)) {
            newFilters.delete(tag);
        } else {
            newFilters.add(tag);
        }
        return newFilters;
    });
  };

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

  // Handler for selecting image to UPLOAD
  const handleImageFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setImageUploadFile(file);
      setStatus('Image selected for upload.');
    } else {
      setImageUploadFile(null);
      setStatus('Please select a valid image file.');
    }
  };

  // Handler for submitting the UPLOAD image and category
  const handleImageSubmit = async () => {
    // Convert comma-separated tags string to JSON array string
    const tagsArray = imageTagsInput.split(',').map(tag => tag.trim()).filter(tag => tag !== '');
    const tagsJsonString = JSON.stringify(tagsArray);

    if (!imageUploadFile || tagsArray.length === 0) {
      setStatus('Please select an image and enter at least one tag.');
      return;
    }

    setIsLoading(true);
    setStatus('Uploading image...');
    const formData = new FormData();
    formData.append('file', imageUploadFile);
    formData.append('tags', tagsJsonString);

    try {
      const response = await fetch('http://localhost:8000/api/upload-image', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to upload image');
      }
      setStatus(data.message);
      // Clear inputs after successful upload
      setImageUploadFile(null);
      setImageTagsInput('');
    } catch (error) {
      console.error('Upload error:', error);
      setStatus(error instanceof Error ? error.message : 'Image upload failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Handler for selecting image for SEARCH query
  const handleImageQuerySelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setImageQuery(file); // Reusing imageQuery state for search input
      setStatus('Image selected for search.');
    } else {
      setImageQuery(null);
      setStatus('Please select a valid image file for search.');
    }
  };

  // Handler for performing the IMAGE search
  const performImageSearch = async () => {
    if (searchType === 'text' && !textQuery) {
      setStatus('Please enter a text query.');
      return;
    }
    if (searchType === 'image' && !imageQuery) {
      setStatus('Please select an image for search.');
      return;
    }

    setIsLoading(true);
    setStatus('Searching images...');
    setImageSearchResults([]); // Clear previous results

    try {
      const formData = new FormData();
      if (searchType === 'text') {
        formData.append('query', new File([textQuery], textQuery || 'query.txt', { type: 'text/plain' }));
      } else if (imageQuery) {
        formData.append('query', imageQuery);
      }

      formData.append('search_type', searchType);
      formData.append('num_results', numResults.toString());
      formData.append('similarity_threshold', similarityThreshold.toString()); // Add threshold to form data

      const response = await fetch('http://localhost:8000/api/search-images', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.detail || 'Image search failed');
      }

      if (data.images && Array.isArray(data.images)) {
        setImageSearchResults(data.images);
        setStatus(`Found ${data.images.length} matching images.`);
      } else {
        setStatus('No matching images found.');
      }
    } catch (error) {
      console.error('Image Search error:', error);
      setStatus(error instanceof Error ? error.message : 'Image search failed');
      setImageSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  // --- Drag and Drop Handlers for Image Upload ---
  const handleDragOverImage = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); // Necessary to allow dropping
    setIsDraggingImage(true);
  };

  const handleDragLeaveImage = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingImage(false);
  };

  const handleDropImage = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingImage(false);
    setStatus(''); // Clear previous status

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      // For simplicity, we'll process the first valid image dropped.
      // Could be extended to handle multiple files, maybe queueing uploads.
      const imageFile = Array.from(files).find(file => file.type.startsWith('image/'));
      if (imageFile) {
        setImageUploadFile(imageFile);
        setStatus(`Selected ${imageFile.name} for upload.`);
      } else {
        setStatus('No valid image files found in drop.');
      }
    } else {
        // Handle cases where non-files are dragged/dropped if necessary
    }
  };
  // --- End Drag and Drop Handlers ---

  // --- Drag and Drop Handlers for Image Search Input ---
  const handleDragOverSearch = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingSearchImage(true);
  };

  const handleDragLeaveSearch = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingSearchImage(false);
  };

  const handleDropSearch = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingSearchImage(false);
    setStatus('');

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const imageFile = Array.from(files).find(file => file.type.startsWith('image/'));
      if (imageFile) {
        setImageQuery(imageFile); // Set the imageQuery state
        setStatus(`Selected ${imageFile.name} for search.`);
      } else {
        setStatus('No valid image files found in drop.');
      }
    }
  };
  // --- End Search Drag and Drop Handlers ---

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Simplified Header - Logo Removed */}
        <div className="mb-8 text-center pt-4 pb-6 border-b border-neutral-200">
          {/* <img
            src="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/pixeltable-logo-large.png"
            alt="Pixeltable"
            className="h-10 mb-3 mx-auto"
          /> */}
          <h1 className="text-3xl font-bold mb-1 text-neutral-800">
            Text and Image similarity search with embedding indexes
          </h1>
          <p className="text-neutral-600 max-w-2xl mx-auto text-sm">
            Pixeltable is a declarative interface for working with text, images, embeddings, and even video, enabling you to store, transform, index, and iterate on data.
          </p>
        </div>

        {/* Tabs */}
        <div className="mb-8 border-b border-gray-200">
          <nav className="-mb-px flex space-x-8" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('video')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'video'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-neutral-500 hover:text-neutral-700 hover:border-neutral-300'
              }`}
            >
              Video Frame Search
            </button>
            <button
              onClick={() => setActiveTab('image')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'image'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-neutral-500 hover:text-neutral-700 hover:border-neutral-300'
              }`}
            >
              Image Search
            </button>
          </nav>
        </div>

        {/* Conditional Content based on activeTab */}
        {activeTab === 'video' && (
          <>
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
                    className={`w-5 h-5 transform transition-transform text-neutral-500 ${
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
                  <ol className="space-y-3 text-neutral-600">
                    <li className="flex items-start">
                      <span className="flex items-center justify-center w-6 h-6 bg-primary-100 rounded-full text-primary-600 text-sm font-semibold mr-3 shrink-0 mt-0.5">1</span>
                      <span>Each video frame is analyzed and indexed for efficient searching</span>
                    </li>
                    <li className="flex items-start">
                      <span className="flex items-center justify-center w-6 h-6 bg-primary-100 rounded-full text-primary-600 text-sm font-semibold mr-3 shrink-0 mt-0.5">2</span>
                      <span>Embedding generation to match your queries with relevant frames</span>
                    </li>
                    <li className="flex items-start">
                      <span className="flex items-center justify-center w-6 h-6 bg-primary-100 rounded-full text-primary-600 text-sm font-semibold mr-3 shrink-0 mt-0.5">3</span>
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
                    className={`w-5 h-5 transform transition-transform text-neutral-500 ${
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
                  <ul className="space-y-3 text-neutral-600">
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
                          ? 'bg-primary-600 text-white'
                          : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
                      } transition-colors`}
                    >
                      Text Search
                    </button>
                    <button
                      onClick={() => setSearchType('image')}
                      className={`flex-1 py-2 px-4 text-sm font-medium ${
                        searchType === 'image'
                          ? 'bg-primary-600 text-white'
                          : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200'
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
                          className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2
                              focus:ring-primary-500 focus:border-primary-500"
                      />
                  ) : (
                    // Use the correct image selection logic for search query
                    <div
                      className={`border-2 border-dashed rounded-md p-4 text-center transition-colors cursor-pointer bg-white ${isDraggingSearchImage ? 'border-primary-500 bg-primary-50' : 'border-neutral-300 hover:border-primary-400'}`}
                      onClick={() => document.getElementById('video-image-search-upload')?.click()}
                      onDragOver={handleDragOverSearch}
                      onDragLeave={handleDragLeaveSearch}
                      onDrop={handleDropSearch}
                    >
                      <input
                        id="video-image-search-upload" // Unique ID
                        type="file"
                        accept="image/*"
                        onChange={handleImageQuerySelect} // Use the query select handler
                        className="hidden"
                      />
                      {imageQuery ? ( // Check imageQuery state
                        <div className="space-y-2">
                          <img src={URL.createObjectURL(imageQuery)} alt="Search query" className="max-h-24 mx-auto rounded-lg" />
                          <p className="text-sm text-neutral-600">Click to change search image</p>
                        </div>
                      ) : (
                        <div className="space-y-1">
                          <svg className="mx-auto h-8 w-8 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                          <p className="text-sm text-neutral-600">Upload reference image</p>
                        </div>
                      )}
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
          </>
        )}

        {activeTab === 'image' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Image Upload Panel */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="mb-6">
                  <h2 className="text-xl font-semibold mb-2">Upload & Tag Image</h2>
                  <p className="text-neutral-600 text-sm">Add images to your searchable collection.</p>
                </div>
                <div className="space-y-4">
                  {/* Image Upload Input */}
                  <div
                    className={`border-2 border-dashed rounded-md p-5 text-center transition-colors cursor-pointer bg-neutral-50 ${isDraggingImage ? 'border-primary-500 bg-primary-50' : 'border-neutral-200 hover:border-primary-400'}`}
                    onClick={() => document.getElementById('image-upload-input')?.click()}
                    onDragOver={handleDragOverImage}
                    onDragLeave={handleDragLeaveImage}
                    onDrop={handleDropImage}
                  >
                    <input
                      id="image-upload-input"
                      type="file"
                      accept="image/*"
                      onChange={handleImageFileSelect}
                      className="hidden"
                    />
                    {imageUploadFile ? (
                      <div className="space-y-2">
                        <img
                          src={URL.createObjectURL(imageUploadFile)}
                          alt="Selected for upload"
                          className="max-h-32 mx-auto rounded-lg"
                        />
                        <p className="text-sm text-neutral-600">{imageUploadFile.name}</p>
                        <p className="text-xs text-neutral-500">Click to change image</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <svg className="mx-auto h-12 w-12 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        <p className="text-sm text-neutral-600">Click to select an image</p>
                      </div>
                    )}
                  </div>

                  {/* Category Input - Replaced with Tags Input */}
                  <input
                    type="text"
                    value={imageTagsInput}
                    onChange={(e) => setImageTagsInput(e.target.value)}
                    placeholder="Enter tags, comma-separated (e.g., shoes, summer)"
                    className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                  />

                  {/* Upload Button */}
                  <button
                    onClick={handleImageSubmit}
                    disabled={!imageUploadFile || !imageTagsInput || isLoading}
                    className="w-full py-2 px-4 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:bg-neutral-300 disabled:cursor-not-allowed transition-colors"
                  >
                    {isLoading ? 'Uploading...' : 'Upload Image'}
                  </button>
                </div>
              </div>

              {/* Image Search Panel & Results */}
              <div className="lg:col-span-2 bg-white rounded-xl shadow-md p-6 space-y-6">
                 {/* Search Controls - Reusing logic/state for now */}
                 <div>
                   <h2 className="text-xl font-semibold mb-4">Search Images</h2>
                   <div className="space-y-4 p-4 border rounded-lg bg-neutral-50">
                     <div className="flex rounded-lg overflow-hidden border border-neutral-200">
                       <button
                         onClick={() => setSearchType('text')}
                         className={`flex-1 py-2 px-4 text-sm font-medium ${searchType === 'text' ? 'bg-primary-600 text-white' : 'bg-white text-neutral-600 hover:bg-neutral-100'} transition-colors`}
                       >
                         Text Search
                       </button>
                       <button
                         onClick={() => setSearchType('image')}
                         className={`flex-1 py-2 px-4 text-sm font-medium ${searchType === 'image' ? 'bg-primary-600 text-white' : 'bg-white text-neutral-600 hover:bg-neutral-100'} transition-colors border-l border-neutral-200`}
                       >
                         Image Search
                       </button>
                     </div>

                     {searchType === 'text' ? (
                       <input
                         type="text"
                         value={textQuery}
                         onChange={(e) => setTextQuery(e.target.value)}
                         placeholder="Describe the image you want..."
                         className="w-full px-4 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                       />
                     ) : (
                       <div
                         className={`border-2 border-dashed rounded-md p-4 text-center transition-colors cursor-pointer bg-white ${isDraggingSearchImage ? 'border-primary-500 bg-primary-50' : 'border-neutral-300 hover:border-primary-400'}`}
                         onClick={() => document.getElementById('image-search-upload')?.click()}
                         onDragOver={handleDragOverSearch}
                         onDragLeave={handleDragLeaveSearch}
                         onDrop={handleDropSearch}
                       >
                         <input
                           id="image-search-upload"
                           type="file"
                           accept="image/*"
                           onChange={handleImageQuerySelect}
                           className="hidden"
                         />
                         {imageQuery ? (
                           <div className="space-y-2">
                             <img src={URL.createObjectURL(imageQuery)} alt="Search query" className="max-h-24 mx-auto rounded-lg" />
                             <p className="text-sm text-neutral-600">Click to change search image</p>
                           </div>
                         ) : (
                           <div className="space-y-1">
                             <svg className="mx-auto h-8 w-8 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                             <p className="text-sm text-neutral-600">Upload reference image</p>
                           </div>
                         )}
                       </div>
                     )}

                     <div className="space-y-1">
                       <label className="text-sm font-medium text-neutral-700">Number of Results: {numResults}</label>
                       <input
                         type="range" min="1" max="20" value={numResults}
                         onChange={(e) => setNumResults(Number(e.target.value))}
                         className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer range-thumb:bg-primary-600"
                       />
                     </div>

                     {/* Similarity Threshold Slider */}
                     <div className="space-y-1">
                       <label className="text-sm font-medium text-neutral-700">
                         Similarity Threshold: {similarityThreshold.toFixed(2)} (Higher = stricter matches)
                       </label>
                       <input
                         type="range" min="0.1" max="1.0" step="0.05" value={similarityThreshold}
                         onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
                         className="w-full h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer range-thumb:bg-primary-600"
                       />
                     </div>

                     <button
                       onClick={performImageSearch}
                       disabled={isLoading || (searchType === 'text' && !textQuery) || (searchType === 'image' && !imageQuery)}
                       className="w-full py-2 px-4 bg-neutral-800 text-white rounded-lg hover:bg-neutral-900 disabled:bg-neutral-400 disabled:cursor-not-allowed transition-colors"
                     >
                       {isLoading ? 'Searching Images...' : 'Search Images'}
                     </button>
                   </div>
                 </div>

                 {/* Search Results Display */}
                 <div>
                   <h3 className="text-lg font-semibold mb-2 text-neutral-800">Image Search Results</h3>
                   {status && <p className="text-sm text-neutral-600 mb-4">{status}</p>}
                   {isLoading && <p className="text-sm text-primary-600">Loading...</p>}

                   {/* Tag Filter Checkboxes - Render if there are results */}
                   {imageSearchResults.length > 0 && (
                        <div className="mb-4 p-3 border rounded-lg bg-neutral-50">
                            <h5 className="text-sm font-semibold mb-2 text-neutral-600">Filter by Tag:</h5>
                            <div className="flex flex-wrap gap-2">
                                {/* Get unique tags from results for checkbox creation */}
                                {[
                                    ...new Set(imageSearchResults.flatMap(r => r.tags || [])),
                                    ...(imageSearchResults.some(r => !r.tags || r.tags.length === 0) ? ['Untagged'] : [])
                                ]
                                .sort()
                                .map(tag => (
                                    <label key={tag} className="flex items-center space-x-1.5 text-sm cursor-pointer bg-white px-2 py-1 rounded border border-neutral-200 hover:bg-neutral-100">
                                        <input
                                            type="checkbox"
                                            checked={activeTagFilters.has(tag)}
                                            onChange={() => handleTagFilterChange(tag)}
                                            className="rounded text-primary-600 focus:ring-primary-500 h-4 w-4"
                                        />
                                        <span>{tag}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                    )}

                   {/* Group results by tag */}
                   {groupedImageResultsDisplay}
                 </div>
              </div>
            </div>
        )}

      </div>
    </div>
  )
}
