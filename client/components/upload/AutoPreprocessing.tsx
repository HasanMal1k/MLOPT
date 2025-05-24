// components/upload/AutoPreprocessing.tsx - IMPROVED VERSION
'use client'

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { CheckCircle2, ArrowLeft, Settings2, AlertCircle, RefreshCw } from "lucide-react"
import AutoPreprocessingReport from "@/components/AutoPreprocessingReport"

interface ProcessingInfo {
  filename: string;
  original_filename: string;
  processed_filename: string;
  status: {
    status: string;
    progress: number;
    message: string;
  };
}

interface AutoPreprocessingProps {
  files: File[]
  onBack: () => void
  onContinue: (preprocessedFiles: File[], results: Record<string, any>) => void
}

export default function AutoPreprocessing({
  files,
  onBack,
  onContinue
}: AutoPreprocessingProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStatus, setProcessingStatus] = useState<Record<string, any>>({})
  const [processedFiles, setProcessedFiles] = useState<File[]>([])
  const [processingResults, setProcessingResults] = useState<Record<string, any>>({})
  const [error, setError] = useState<string | null>(null)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [isComplete, setIsComplete] = useState(false)

  const startPreprocessing = async () => {
    setIsProcessing(true)
    setError(null)
    setProcessingProgress(0)
    
    try {
      console.log('Starting preprocessing for files:', files.map(f => f.name))
      
      // Step 1: Upload files to Python backend for preprocessing
      const pythonFormData = new FormData()
      files.forEach((file, index) => {
        pythonFormData.append('files', file)
        console.log(`Adding file ${index}: ${file.name} (${file.size} bytes)`)
      })
      
      setProcessingProgress(10)
      
      // Send to Python backend
      console.log('Sending files to Python backend...')
      const pythonResponse = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: pythonFormData,
      })

      if (!pythonResponse.ok) {
        const errorText = await pythonResponse.text()
        console.error('Python backend error:', errorText)
        throw new Error(`Backend returned status ${pythonResponse.status}: ${pythonResponse.statusText}`)
      }

      const preprocessingResult = await pythonResponse.json()
      setProcessingProgress(30)
      
      console.log('Initial preprocessing result:', preprocessingResult)
      
      // Initialize status tracking for preprocessing
      const processingInfo = (preprocessingResult.processing_info || []) as ProcessingInfo[]
      
      if (processingInfo.length === 0) {
        console.log('No processing info returned, using original files')
        setProcessedFiles(files)
        setProcessingResults({})
        setProcessingProgress(100)
        setIsComplete(true)
        return
      }

      // Set up initial status tracking
      const initialStatus: Record<string, any> = {}
      const fileMapping: Record<string, string> = {}
      
      processingInfo.forEach((info: ProcessingInfo) => {
        const safeFilename = info.filename
        initialStatus[safeFilename] = {
          status: info.status.status,
          progress: info.status.progress,
          message: info.status.message,
          original_filename: info.original_filename,
          processed_filename: info.processed_filename
        }
        fileMapping[info.original_filename] = safeFilename
      })

      setProcessingStatus(initialStatus)
      console.log('Initial status set:', initialStatus)
      
      // Step 2: Poll processing status
      const safeFilenames = processingInfo.map((info: ProcessingInfo) => info.filename)
      console.log('Tracking status for files:', safeFilenames)
      
      const finalResults = await trackProcessingStatus(safeFilenames, fileMapping)
      setProcessingProgress(70)
      
      // Step 3: Attempt to retrieve processed files
      await retrieveProcessedFiles(files, finalResults, fileMapping)
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Preprocessing failed'
      setError(errorMessage)
      console.error('Preprocessing error:', err)
      
      // Even on error, allow user to continue with original files
      setProcessedFiles(files)
      setProcessingResults({})
    } finally {
      setIsProcessing(false)
    }
  }

  const trackProcessingStatus = async (
    safeFilenames: string[], 
    fileMapping: Record<string, string>
  ): Promise<Record<string, any>> => {
    let allCompleted = false
    let attempts = 0
    const maxAttempts = 20
    const pollInterval = 3000 // 3 seconds
    let finalResults: Record<string, any> = {}
    
    console.log(`Starting status tracking for ${safeFilenames.length} files`)
    
    while (!allCompleted && attempts < maxAttempts) {
      attempts++
      let completedCount = 0
      const currentStatuses: Record<string, any> = {}
      
      console.log(`Status check attempt ${attempts}/${maxAttempts}`)
      
      // Check status for each file
      for (const safeFilename of safeFilenames) {
        try {
          const response = await fetch(`http://localhost:8000/processing-status/${safeFilename}`)
          
          if (response.ok) {
            const status = await response.json()
            console.log(`Status for ${safeFilename}:`, status)
            
            currentStatuses[safeFilename] = {
              status: status.status || 'processing',
              progress: status.progress || 0,
              message: status.message || 'Processing...',
              results: status.results || null,
              file_mapping: status.file_mapping || null
            }
            
            // Count as completed if progress is 100 or status is completed
            if (status.progress === 100 || status.status === 'completed') {
              completedCount++
              if (status.results) {
                finalResults[safeFilename] = { results: status.results }
              }
            } else if (status.progress === -1 || status.status === 'error') {
              // Also count errors as "completed" to avoid infinite loop
              completedCount++
              console.warn(`File ${safeFilename} failed processing:`, status.message)
            }
          } else {
            console.warn(`Failed to get status for ${safeFilename}: ${response.status}`)
            // Assume completed on HTTP errors
            currentStatuses[safeFilename] = {
              status: 'unknown',
              progress: 100,
              message: 'Status check failed',
              results: null
            }
            completedCount++
          }
        } catch (error) {
          console.error(`Network error checking status for ${safeFilename}:`, error)
          currentStatuses[safeFilename] = {
            status: 'error',
            progress: 100,
            message: 'Network error during status check',
            results: null
          }
          completedCount++
        }
      }
      
      // Update the processing status state
      setProcessingStatus(prev => ({ ...prev, ...currentStatuses }))
      
      // Update overall progress
      const overallProgress = 30 + (completedCount / safeFilenames.length) * 40
      setProcessingProgress(Math.min(overallProgress, 70))
      
      // Check if all files are completed
      if (completedCount >= safeFilenames.length) {
        allCompleted = true
        console.log('All files completed processing')
        break
      }
      
      // Wait before next poll
      if (!allCompleted && attempts < maxAttempts) {
        console.log(`Waiting ${pollInterval}ms before next status check...`)
        await new Promise(resolve => setTimeout(resolve, pollInterval))
      }
    }
    
    if (!allCompleted) {
      console.warn(`Processing status tracking timed out after ${attempts} attempts`)
    }
    
    return finalResults
  }

  const retrieveProcessedFiles = async (
    originalFiles: File[], 
    finalResults: Record<string, any>,
    fileMapping: Record<string, string>
  ) => {
    console.log('Starting file retrieval...')
    
    const retrievedFiles: File[] = []
    const results: Record<string, any> = {}
    
    for (const originalFile of originalFiles) {
      const safeFilename = fileMapping[originalFile.name]
      let processedFile: File | null = null
      
      console.log(`Processing file: ${originalFile.name} -> ${safeFilename}`)
      
      // Collect results if available
      if (safeFilename && finalResults[safeFilename]?.results) {
        results[originalFile.name] = finalResults[safeFilename].results
        console.log(`Results collected for ${originalFile.name}`)
      }
      
      // Try multiple endpoints to retrieve the processed file
      const possibleEndpoints = [
        `http://localhost:8000/processed-files/${safeFilename}`,
        `http://localhost:8000/processed-files/processed_${safeFilename}`,
        `http://localhost:8000/files/processed/${safeFilename}`,
        `http://localhost:8000/files/processed/processed_${safeFilename}`,
        `http://localhost:8000/download/${safeFilename}`,
        `http://localhost:8000/download/processed_${safeFilename}`,
      ]
      
      for (const endpoint of possibleEndpoints) {
        try {
          console.log(`Trying to fetch from: ${endpoint}`)
          const fileResponse = await fetch(endpoint)
          
          if (fileResponse.ok) {
            const blob = await fileResponse.blob()
            if (blob.size > 0) { // Make sure we got actual content
              processedFile = new File([blob], originalFile.name, { type: 'text/csv' })
              console.log(`✓ Successfully retrieved processed file from: ${endpoint}`)
              console.log(`File size: ${blob.size} bytes`)
              break
            } else {
              console.log(`Empty response from: ${endpoint}`)
            }
          } else {
            console.log(`Failed to fetch from ${endpoint}: ${fileResponse.status}`)
          }
        } catch (error) {
          console.log(`Error fetching from ${endpoint}:`, error)
        }
      }
      
      // Use processed file if found, otherwise use original
      if (processedFile) {
        retrievedFiles.push(processedFile)
        console.log(`✓ Using processed version of ${originalFile.name}`)
      } else {
        retrievedFiles.push(originalFile)
        console.log(`⚠ Using original version of ${originalFile.name} (processed file not found)`)
      }
    }
    
    console.log(`File retrieval complete: ${retrievedFiles.length} files total`)
    console.log(`Results collected for ${Object.keys(results).length} files`)
    
    setProcessedFiles(retrievedFiles)
    setProcessingResults(results)
    setProcessingProgress(100)
    setIsComplete(true)
  }

  const handleContinue = () => {
    onContinue(processedFiles, processingResults)
  }

  const handleSkipPreprocessing = () => {
    onContinue(files, {})
  }

  if (isComplete) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-green-500" />
            <span>Auto Preprocessing Complete</span>
          </CardTitle>
          <CardDescription>
            Your files have been automatically preprocessed and are ready for custom cleaning
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-medium text-green-800 mb-2">Preprocessing Summary</h3>
            <p className="text-green-700 mb-4">
              Successfully processed {processedFiles.length} of {files.length} files
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-4">
              {files.map((file, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <span className="text-green-700">{file.name}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Review
          </Button>
          <Button onClick={handleContinue} className="gap-2">
            <span>Continue to Custom Cleaning</span>
            <Settings2 className="h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    )
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Auto Preprocessing</CardTitle>
        <CardDescription>
          Automatically clean and preprocess your data files
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!isProcessing ? (
          <div className="text-center p-8">
            <RefreshCw className="h-12 w-12 mx-auto mb-4 text-blue-500" />
            <h3 className="text-lg font-medium mb-2">Ready for Auto Preprocessing</h3>
            <p className="text-sm text-muted-foreground mb-6 max-w-2xl mx-auto">
              Our system will automatically clean your data by handling missing values, 
              detecting and converting date columns, removing unnecessary columns, and 
              performing basic feature engineering.
            </p>
            
            <div className="bg-blue-50 rounded-lg p-4 mb-6 text-left max-w-md mx-auto">
              <h4 className="font-medium text-blue-800 mb-2">What happens during auto preprocessing:</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Remove columns with excessive missing values</li>
                <li>• Handle missing values with appropriate imputation</li>
                <li>• Detect and convert date/time columns</li>
                <li>• Remove duplicate rows and columns</li>
                <li>• Basic feature engineering for dates</li>
              </ul>
            </div>
            
            <div className="flex gap-4 justify-center">
              <Button onClick={startPreprocessing} size="lg" className="gap-2">
                <RefreshCw className="h-4 w-4" />
                Start Auto Preprocessing
              </Button>
              
              <Button 
                variant="outline" 
                onClick={handleSkipPreprocessing}
                size="lg"
              >
                Skip Preprocessing
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-lg font-medium mb-2">Processing Your Files</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Please wait while we automatically preprocess your {files.length} files
              </p>
            </div>
            
            <div className="mb-8">
              <p className="text-sm font-medium mb-2">Overall Progress</p>
              <Progress value={processingProgress} className="h-3 w-full" />
              <p className="text-xs text-muted-foreground mt-2">
                {processingProgress < 100 
                  ? `Processing ${files.length} files... (${Math.round(processingProgress)}%)` 
                  : `Completed processing ${files.length} files`}
              </p>
            </div>
            
            {Object.keys(processingStatus).length > 0 && (
              <div>
                <p className="text-sm font-medium mb-3">File Processing Status:</p>
                {Object.entries(processingStatus).map(([filename, status]) => (
                  <div key={filename} className="mb-3">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium">
                        {status.original_filename || filename}
                      </span>
                      <span className={`
                        ${status.status === 'completed' ? 'text-green-600' : ''}
                        ${status.status === 'error' ? 'text-red-600' : ''}
                      `}>
                        {status.message}
                      </span>
                    </div>
                    <Progress 
                      value={status.progress < 0 ? 100 : status.progress} 
                      className={`h-2 w-full ${
                        status.progress < 0 ? 'bg-red-300' : 
                        status.status === 'completed' ? 'bg-green-300' : ''
                      }`} 
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Processing Issues</AlertTitle>
            <AlertDescription>
              {error}
              <br />
              <span className="text-sm mt-2 block">
                You can still continue with the original files or retry preprocessing.
              </span>
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Review
        </Button>
        <div className="flex gap-2">
          {error && (
            <Button variant="outline" onClick={startPreprocessing}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Retry
            </Button>
          )}
          <Button onClick={handleSkipPreprocessing} variant="outline">
            Skip & Continue
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}