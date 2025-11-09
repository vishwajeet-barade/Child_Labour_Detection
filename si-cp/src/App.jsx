import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_BASE_URL = 'http://localhost:5000/api'

function App() {
  const [videoFile, setVideoFile] = useState(null)
  const [videoPreview, setVideoPreview] = useState(null)
  const [processedVideoUrl, setProcessedVideoUrl] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [isReady, setIsReady] = useState(false)
  const [outputFilename, setOutputFilename] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  // Poll for processing status
  useEffect(() => {
    if (isProcessing && outputFilename) {
      const pollInterval = setInterval(async () => {
        try {
          const response = await fetch(`${API_BASE_URL}/status/${outputFilename}`)
          const data = await response.json()
          
          if (data.status === 'ready') {
            clearInterval(pollInterval)
            setIsProcessing(false)
            setIsReady(true)
            setProcessingProgress(100)
            // Load processed video directly from static outputs folder
            setProcessedVideoUrl(`http://localhost:5000/outputs/${outputFilename}`)
          } else {
            // Increment progress gradually while processing
            setProcessingProgress((prev) => {
              if (prev < 90) {
                return prev + 2
              }
              return prev
            })
          }
        } catch (err) {
          console.error('Error checking status:', err)
        }
      }, 2000) // Poll every 2 seconds

      return () => clearInterval(pollInterval)
    }
  }, [isProcessing, outputFilename])

  const handleFileSelect = async (file) => {
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file)
      setIsProcessing(true)
      setIsReady(false)
      setProcessingProgress(0)
      setError(null)
      setProcessedVideoUrl(null)
      setOutputFilename(null)
      
      // Show preview of original video
      const reader = new FileReader()
      reader.onload = (e) => {
        setVideoPreview(e.target.result)
      }
      reader.readAsDataURL(file)
      
      // Upload and process video
      try {
        const formData = new FormData()
        formData.append('video', file)
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData
        })
        
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Upload failed')
        }
        
        const data = await response.json()
        setOutputFilename(data.output_filename)
        setProcessingProgress(10) // Start at 10%
      } catch (err) {
        console.error('Upload error:', err)
        setError(err.message || 'Failed to upload video')
        setIsProcessing(false)
        alert(`Error: ${err.message || 'Failed to upload video'}`)
      }
    } else {
      alert('Please select a valid video file')
    }
  }

  const handleInputChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleRemove = () => {
    setVideoFile(null)
    setVideoPreview(null)
    setProcessedVideoUrl(null)
    setIsProcessing(false)
    setIsReady(false)
    setProcessingProgress(0)
    setOutputFilename(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="app-container">
      <h1>Video Upload</h1>
      
      <div className="upload-container">
        {!videoFile ? (
          <div
            className={`upload-area ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="upload-content">
              <svg
                className="upload-icon"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <h2>Drag & Drop your video here</h2>
              <p>or</p>
              <button className="upload-button" onClick={handleButtonClick}>
                Browse Files
              </button>
              <p className="file-types">Supports: MP4, AVI, MOV, WebM, and other video formats</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleInputChange}
              style={{ display: 'none' }}
            />
          </div>
        ) : isProcessing ? (
          <div className="processing-container">
            <div className="processing-content">
              <div className="spinner"></div>
              <h2>Processing Video...</h2>
              <p className="processing-text">Please wait while we prepare your video for playback</p>
              <div className="progress-bar-container">
                <div className="progress-bar">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${processingProgress}%` }}
                  ></div>
                </div>
                <span className="progress-text">{Math.round(processingProgress)}%</span>
              </div>
              <div className="processing-steps">
                <div className={`processing-step ${processingProgress > 15 ? 'completed' : ''}`}>
                  <span className="step-icon">✓</span>
                  <span>Uploading video...</span>
                </div>
                <div className={`processing-step ${processingProgress > 40 ? 'completed' : ''}`}>
                  <span className="step-icon">✓</span>
                  <span>Detecting faces and actions...</span>
                </div>
                <div className={`processing-step ${processingProgress > 80 ? 'completed' : ''}`}>
                  <span className="step-icon">✓</span>
                  <span>Processing video frames...</span>
                </div>
              </div>
            </div>
          </div>
        ) : isReady && processedVideoUrl ? (
          <div className="video-preview-container">
            <div className="video-preview">
              <h3 style={{ color: 'rgba(255, 255, 255, 0.87)', marginBottom: '1rem' }}>Processed Video (Child Labour Detection)</h3>
              <video
                src={processedVideoUrl}
                controls
                className="preview-video"
                onError={(e) => {
                  const video = e.target
                  const error = video.error
                  let errorMessage = 'Failed to load processed video. '
                  
                  if (error) {
                    switch (error.code) {
                      case error.MEDIA_ERR_ABORTED:
                        errorMessage += 'Video playback was aborted.'
                        break
                      case error.MEDIA_ERR_NETWORK:
                        errorMessage += 'Network error occurred while loading video.'
                        break
                      case error.MEDIA_ERR_DECODE:
                        errorMessage += 'Video codec not supported by browser. The video may need to be re-encoded.'
                        break
                      case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                        errorMessage += 'Video format not supported. Please try re-encoding the video.'
                        break
                      default:
                        errorMessage += `Error code: ${error.code}`
                    }
                  }
                  
                  console.error('Video playback error:', {
                    error: error,
                    code: error?.code,
                    message: errorMessage,
                    src: processedVideoUrl
                  })
                  setError(errorMessage)
                }}
                onLoadStart={() => {
                  console.log('Video loading started:', processedVideoUrl)
                  setError(null)
                }}
                onLoadedMetadata={() => {
                  console.log('Video metadata loaded successfully')
                  setError(null)
                }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="video-info">
              <h3>Video Details</h3>
              <div className="info-row">
                <span className="info-label">Original File Name:</span>
                <span className="info-value">{videoFile.name}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Original File Size:</span>
                <span className="info-value">{formatFileSize(videoFile.size)}</span>
              </div>
              <div className="info-row">
                <span className="info-label">File Type:</span>
                <span className="info-value">{videoFile.type}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Status:</span>
                <span className="info-value" style={{ color: '#4ade80' }}>✓ Processing Complete</span>
              </div>
              {error && (
                <div className="info-row">
                  <span className="info-label" style={{ color: '#ef4444' }}>Error:</span>
                  <span className="info-value" style={{ color: '#ef4444' }}>{error}</span>
                </div>
              )}
              <div className="button-group">
                <button className="remove-button" onClick={handleRemove}>
                  Remove Video
                </button>
                <button className="change-button" onClick={handleButtonClick}>
                  Change Video
                </button>
              </div>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleInputChange}
              style={{ display: 'none' }}
            />
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default App
