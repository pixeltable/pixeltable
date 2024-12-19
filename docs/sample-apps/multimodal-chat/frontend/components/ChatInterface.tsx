'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from "framer-motion";
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import {
  MessageCircle,
  UploadCloud,
  FileText,
  Loader2,
  Video,
  ChevronLeft,
  Send,
  Music
} from 'lucide-react';
import { cn } from '@/utils/utils';

interface Message {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  usedFiles?: string[];
}

interface AudioFile {
  id: string;
  name: string;
  size?: number;
  type: 'audio';
  duration?: number;
  status: 'success' | 'error' | 'uploading';
  error?: string;
  uploadProgress?: number;
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadProgress?: number;
  status: 'uploading' | 'success' | 'error';
  error?: string;
}

interface VideoFile {
  id: string;
  name: string;
  size?: number;
  type: 'video';
  status: 'success' | 'error' | 'uploading';
  error?: string;
  uploadProgress?: number;
}
interface MarkdownProps {
  node?: any;
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

const MAX_FILE_SIZE = 150 * 1024 * 1024;
const ALLOWED_DOCUMENT_TYPES = "application/pdf,.pdf,application/msword,.doc,.docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.txt,.py";
const ALLOWED_VIDEO_TYPES = "video/*";

const ChatMessage = ({ message }: { message: Message }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={cn(
        "flex flex-col mb-4",
        message.type === 'user' ? "items-end" : "items-start",
        message.type === 'system' && "items-center"
      )}
    >
      <div className={cn(
        "max-w-[80%] rounded-lg px-4 py-2",
        message.type === 'user' && "bg-black text-white",
        message.type === 'bot' && "bg-white text-black",
        message.type === 'system' && "bg-white text-black text-sm"
      )}>
        {message.type === 'bot' ? (
          <div className="prose max-w-none text-black">
            <ReactMarkdown
              components={{
                code: ({ node, inline, className, children, ...props }: MarkdownProps) => {
                  if (inline) {
                    return (
                      <code className="bg-gray-200 rounded px-1 py-0.5" {...props}>
                        {children}
                      </code>
                    );
                  }
                  return (
                    <pre className="bg-white text-black rounded p-4 overflow-x-auto">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        ) : (
          message.content
        )}
      </div>

      {message.usedFiles && message.usedFiles.length > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          <div className="flex items-center gap-1 mb-1">
            <FileText size={12} />
            <span>Sources:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {message.usedFiles.map((file, index) => (
              <span
                key={index}
                className="bg-gray-100 px-2 py-1 rounded-full text-gray-600 flex items-center gap-1"
              >
                <FileText size={10} />
                {file}
              </span>
            ))}
          </div>
        </div>
      )}

      <span className="text-xs text-gray-400 mt-1">
        {format(message.timestamp, 'HH:mm')}
      </span>
    </motion.div>
  );
};

const LoadingMessage = () => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    className="flex items-center gap-2 text-gray-500 p-4"
  >
    <Loader2 className="animate-spin" size={16} />
    <span>AI is thinking...</span>
  </motion.div>
);

const EmptyState = ({ type }: { type: 'documents' | 'videos' | 'audio' }) => (
  <div className="text-center py-8">
    <div className="w-12 h-12 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
      {type === 'documents' ? <FileText className="h-8 w-8 text-gray-400" /> :
       type === 'videos' ? <Video className="h-8 w-8 text-gray-400" /> :
       <Music className="h-8 w-8 text-gray-400" />}
    </div>
    <h4 className="text-sm font-medium text-gray-900">No {type} yet</h4>
    <p className="mt-1 text-sm text-gray-500">
      {type === 'documents'
        ? "Upload documents to to get started"
        : type === 'videos'
        ? "Upload videos to get started"
        : "Upload audio files to get started"}
    </p>
  </div>
);

const SuggestedQuestions = ({ onSelect }: { onSelect: (question: string) => void }) => {
  const suggestions = [
    {
      category: "Document Analysis",
      questions: [
        "Summarize the key points from all the documents",
        "Compare and contrast the main ideas across documents",
        "What are the common themes in these documents?",
        "Extract the most important findings from these materials"
      ]
    },
    {
      category: "Video Content",
      questions: [
        "What are the main topics discussed in the videos?",
        "Summarize the key points from the video transcripts",
        "Are there any contradictions between different videos?",
        "Extract the most important quotes from the videos"
      ]
    },
    {
      category: "Cross-Reference",
      questions: [
        "How do the documents and videos complement each other?",
        "Find similar topics between documents and videos",
        "What unique insights does each medium provide?",
        "Create a timeline of events from all sources"
      ]
    }
  ];

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="grid gap-6 mb-8">
        {suggestions.map((group) => (
          <div key={group.category} className="space-y-3">
            <h3 className="text-sm font-medium text-gray-500">{group.category}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {group.questions.map((question) => (
                <button
                  key={question}
                  onClick={() => onSelect(question)}
                  className="p-3 text-left text-sm bg-white hover:bg-gray-50
                           border rounded-lg transition-colors duration-200
                           hover:border-blue-500 hover:text-blue-600"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function EnhancedChatInterface() {
  // States
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [videoFiles, setVideoFiles] = useState<VideoFile[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [audioFiles, setAudioFiles] = useState<AudioFile[]>([]);

  // Refs
  const videoInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);

  // Effects
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    fetchFiles();
  }, []);


  // Utility functions
  const validateFile = (file: File): string | null => {
    if (!file.type.startsWith('application/pdf') &&
        !file.type.includes('word') &&
        !file.type.includes('document')) {
      return 'Invalid file type. Only PDF, MD, Text documents are allowed.';
    }
    if (file.size > MAX_FILE_SIZE) {
      return 'File size exceeds 10MB limit.';
    }
    return null;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // API Functions
  const fetchFiles = async () => {
    setIsLoadingFiles(true);
    try {
      const response = await fetch('http://localhost:8000/api/files');
      if (!response.ok) throw new Error('Failed to fetch files');
      const data = await response.json();
      const processedFiles = {
        documents: data.files.filter((f: any) => f.type === 'document').map((f: any) => ({
          id: f.id,
          name: f.name,
          size: f.size,
          type: 'document',
          status: 'success' as const,
          uploadProgress: 100,
          domain: f.domain
        })),
        videos: data.files.filter((f: any) => f.type === 'video').map((f: any) => ({
          id: f.id,
          name: f.name,
          size: f.size,
          type: 'video' as const,
          status: 'success' as const,
          uploadProgress: 100,
          domain: f.domain
        }))
      };

      setFiles(processedFiles.documents);
      setVideoFiles(processedFiles.videos);
    } catch (error) {
      console.error('Error fetching files:', error);
    } finally {
      setIsLoadingFiles(false);
    }
  };

  const handleFileUpload = async (uploadedFiles: FileList) => {
    const file = uploadedFiles[0];
    if (!file) return;

    // Log file info for debugging
    console.log("File:", file.name, file.type);

    const newFile: UploadedFile = {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      type: file.type || 'text/x-python', // Handle Python files that might not have type
      uploadProgress: 0,
      status: 'uploading'
    };

    setFiles(prev => [...prev, newFile]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      await fetchFiles();

      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        type: 'system',
        content: `Successfully uploaded ${file.name}`,
        timestamp: new Date()
      }]);

    } catch (error) {
      console.error('Upload error:', error);
      setFiles(prev => prev.map(f =>
        f.id === newFile.id
          ? { ...f, status: 'error', error: error instanceof Error ? error.message : 'Upload failed' }
          : f
      ));
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAudioUpload = async (uploadedFiles: FileList) => {
    const file = uploadedFiles[0];
    if (!file) return;

    const newAudio: AudioFile = {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      type: 'audio',
      status: 'uploading',
      uploadProgress: 0
    };

    setAudioFiles(prev => [...prev, newAudio]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/audio/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();

      // Update audio file with duration
      setAudioFiles(prev => prev.map(a =>
        a.id === newAudio.id
          ? { ...a, status: 'success', duration: data.duration }
          : a
      ));

      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        type: 'system',
        content: `Successfully uploaded ${file.name}`,
        timestamp: new Date()
      }]);
    } catch (error) {
      console.error('Audio upload error:', error);
      setAudioFiles(prev => prev.filter(a => a.id !== newAudio.id));
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        type: 'system',
        content: `Error uploading audio: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      }]);
    }

    if (audioInputRef.current) {
      audioInputRef.current.value = '';
    }
  };

  const handleVideoUpload = async (uploadedFiles: FileList) => {
    const file = uploadedFiles[0];
    if (!file) return;

    const newVideo: VideoFile = {
      id: crypto.randomUUID(),
      name: file.name,
      size: file.size,
      type: 'video',
      status: 'uploading',
      uploadProgress: 0
    };

    setVideoFiles(prev => [...prev, newVideo]);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/videos/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      await fetchFiles();

      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        type: 'system',
        content: `Successfully uploaded ${file.name}`,
        timestamp: new Date()
      }]);
    } catch (error) {
      console.error('Video upload error:', error);
      setVideoFiles(prev => prev.filter(v => v.id !== newVideo.id));
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        type: 'system',
        content: `Error uploading video: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      }]);
    }

    if (videoInputRef.current) {
      videoInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMessage: Message = {
      id: crypto.randomUUID(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    setInput('');
    setIsSending(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) throw new Error('Failed to get response');
      const data = await response.json();

      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        type: 'bot',
        content: data.response,
        timestamp: new Date(),
        usedFiles: data.used_files
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        type: 'system',
        content: 'Error getting response. Please try again.',
        timestamp: new Date()
      }]);
    } finally {
      setIsSending(false);
    }
  };

  // Event Handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      const file = droppedFiles[0];
      if (file.type.startsWith('video/')) {
        await handleVideoUpload(droppedFiles);
      } else if (file.type.startsWith('audio/')) {
      await handleAudioUpload(droppedFiles);
    } else if (file.type.startsWith('application/')) {
      await handleFileUpload(droppedFiles);
    } else {
        setMessages(prev => [...prev, {
          id: crypto.randomUUID(),
          type: 'system',
          content: 'Invalid file type. Only PDF, MD, Text and videos are allowed.',
          timestamp: new Date()
        }]);
      }
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50 text-black">
      <AnimatePresence>
        {showSidebar && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="h-screen flex flex-col bg-white border-r border-gray-200"
          >
          {/* Fixed Header */}
          <div className="p-4 border-b border-gray-200 shrink-0">
            <h1 className="text-lg font-semibold">Files</h1>
          </div>

          {/* Scrollable Content */}
          <div className="flex-1 overflow-y-auto">
            {/* Documents Section */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-800">Documents</h2>
                <span className="text-sm text-gray-500">{files.length} files</span>
              </div>
              {/* Document Upload Area */}
              <div
                className={cn(
                  "relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer",
                  "transition-colors duration-200",
                  isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept={ALLOWED_DOCUMENT_TYPES}
                  onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
                />
                <UploadCloud className="mx-auto h-12 w-12 text-gray-400 mb-3" />
                <p className="text-sm text-gray-600 mb-1">
                  Drop files here or click to upload
                </p>
                <p className="text-xs text-gray-500">
                  PDF, MD, and Text documents only
                </p>
              </div>

              {/* Document List */}
              <div className="flex-1 overflow-y-auto mt-4">
                {isLoadingFiles ? (
                  <div className="text-center py-8">Loading...</div>
                ) : files.length === 0 ? (
                  <EmptyState type="documents" />
                ) : (
                  <AnimatePresence>
                    {files.map((file) => (
                      <motion.div
                        key={file.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className={cn(
                          "mb-3 p-3 rounded-lg border bg-white hover:bg-gray-50",
                          file.status === 'error' ? "border-red-200 bg-red-50" : "border-gray-200"
                        )}
                      >
                        <div className="flex items-center gap-3">
                          <FileText className="h-8 w-8 text-gray-400 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate">
                              {file.name}
                            </p>
                            <p className="text-xs text-gray-500">
                              {formatFileSize(file.size)}
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                )}
              </div>
            </div>

            {/* Audio Section */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Audio Files</h3>
                <span className="text-sm text-gray-500">{audioFiles.length} files</span>
              </div>

              <div
                className={cn(
                  "relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer",
                  "transition-colors duration-200",
                  isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => audioInputRef.current?.click()}
              >
                <input
                  ref={audioInputRef}
                  type="file"
                  className="hidden"
                  accept="audio/*"
                  onChange={(e) => e.target.files && handleAudioUpload(e.target.files)}
                />
                <Music className="mx-auto h-12 w-12 text-gray-400 mb-3" />
                <p className="text-sm text-gray-600 mb-1">
                  Drop audio here or click to upload
                </p>
                <p className="text-xs text-gray-500">
                  MP3, WAV, OGG formats supported
                </p>
              </div>

              {/* Audio List */}
              <div className="mt-4 space-y-3">
                {isLoadingFiles ? (
                  <div className="text-center py-8">Loading...</div>
                ) : audioFiles.length === 0 ? (
                  <EmptyState type="audio" />
                ) : (
                  <AnimatePresence>
                    {audioFiles.map((file) => (
                      <motion.div
                        key={file.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="p-3 rounded-lg border border-gray-200 bg-white hover:bg-gray-50"
                      >
                        <div className="flex items-center gap-3">
                          <Music className="h-8 w-8 text-gray-400 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate">
                              {file.name}
                            </p>
                            <p className="text-xs text-gray-500">
                              {file.size ? formatFileSize(file.size) : 'Size unknown'} •
                              {file.duration ? ` ${Math.round(file.duration)}s` : ''} • Audio
                            </p>
                          </div>
                        </div>
                        {file.status === 'uploading' && (
                          <div className="mt-2">
                            <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500 transition-all duration-300"
                                style={{ width: `${file.uploadProgress}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </AnimatePresence>
                )}
              </div>
            </div>

            {/* Videos Section */}
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Videos</h3>
                <span className="text-sm text-gray-500">{videoFiles.length} videos</span>
              </div>

              <div
                className={cn(
                  "relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer",
                  "transition-colors duration-200",
                  isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => videoInputRef.current?.click()}
              >
                <input
                  ref={videoInputRef}
                  type="file"
                  className="hidden"
                  accept="video/*"
                  onChange={(e) => e.target.files && handleVideoUpload(e.target.files)}
                />
                <Video className="mx-auto h-12 w-12 text-gray-400 mb-3" />
                <p className="text-sm text-gray-600 mb-1">
                  Drop video here or click to upload
                </p>
                <p className="text-xs text-gray-500">
                  Support for MP4, WebM videos
                </p>
              </div>

              {/* Video List */}
              <div className="mt-4 space-y-3">
                {isLoadingFiles ? (
                  <div className="text-center py-8">Loading...</div>
                ) : videoFiles.length === 0 ? (
                  <EmptyState type="videos" />
                ) : (
                  <AnimatePresence>
                    {videoFiles.map((file) => (
                      <motion.div
                        key={file.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="p-3 rounded-lg border border-gray-200 bg-white hover:bg-gray-50"
                      >
                        <div className="flex items-center gap-3">
                          <Video className="h-8 w-8 text-gray-400 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate">
                              {file.name}
                            </p>
                            <p className="text-xs text-gray-500">
                              {file.size ? formatFileSize(file.size) : 'Size unknown'} • Video
                            </p>
                          </div>
                        </div>
                        {file.status === 'uploading' && (
                          <div className="mt-2">
                            <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500 transition-all duration-300"
                                style={{ width: `${file.uploadProgress}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </AnimatePresence>
                )}
              </div>
            </div>
          </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white overflow-hidden">
        <div className="h-14 border-b flex items-center px-4 justify-between sticky top-0 bg-white border-gray-200 z-10">
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="p-2 hover:bg-gray-100 rounded-lg"
          >
            {showSidebar ? <ChevronLeft /> : <MessageCircle />}
          </button>
          <div className="flex items-center gap-4">
            <img 
              src="/images/pixeltable-logo-large-768x147.png" 
              alt="Pixeltable Logo" 
              className="h-6 w-auto"
            />
          </div>
          <div className="w-8" />
        </div>

        {messages.length === 0 && showSuggestions && (
          <SuggestedQuestions
            onSelect={(question) => {
              setInput(question);
              setShowSuggestions(false);
            }}
          />
        )}

        {/* Messages Area */}
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto"
        >
          <div className="px-4 py-6">
            <div className="max-w-3xl mx-auto">
              <AnimatePresence>
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                {isSending && <LoadingMessage />}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>

        {/* Chat Input */}
        <div className="border-t p-4 bg-white border-gray-200">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSubmit} className="flex items-center gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 px-4 py-2 rounded-lg border bg-white text-black border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isSending}
              />
              <button
                type="submit"
                disabled={isSending || !input.trim()}
                className="px-4 py-2 rounded-lg bg-zinc-900 text-white hover:bg-zinc-950 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isSending ? (
                  <>
                    <Loader2 className="animate-spin" size={16} />
                    <span>Sending...</span>
                  </>
                ) : (
                  <>
                    <Send size={16} />
                    <span>Send</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}