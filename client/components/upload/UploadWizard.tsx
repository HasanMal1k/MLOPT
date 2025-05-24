
'use client'

import { useState, useCallback } from "react"
import { useDropzone, type FileRejection } from "react-dropzone"
import { FilePlus2 } from "lucide-react"
import { Stepper, Step, StepDescription, StepTitle } from "@/components/ui/steppar"
import { toast } from "@/hooks/use-toast"
import FilePreview from "@/components/FilePreview"
import EdaReportViewer from "@/components/EdaReportViewer"
import FileSelection from "./FileSelection"
import FileReview from "./FileReview"
import AutoPreprocessing from "./AutoPreprocessing"
import CustomCleaning from "./CustomCleaning"
import FinalUpload from "./FinalUpload"
import UploadComplete from "./UploadComplete"

interface ProcessingInfo {
  filename: string;
  status: {
    status: string;
    progress: number;
    message: string;
  };
}

interface UploadResult {
  name: string;
  success: boolean;
}

interface UploadWizardProps {
  isDragActive: boolean
}

export default function UploadWizard({ isDragActive }: UploadWizardProps) {
  const [files, setFiles] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>("upload")
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedFileForPreview, setSelectedFileForPreview] = useState<File | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [isEdaReportOpen, setIsEdaReportOpen] = useState(false)
  const [activeFileIndex, setActiveFileIndex] = useState(0)
  const [reviewedFiles, setReviewedFiles] = useState<Set<string>>(new Set())
  
  // Auto preprocessing state
  const [preprocessedFiles, setPreprocessedFiles] = useState<File[]>([])
  const [preprocessingResults, setPreprocessingResults] = useState<Record<string, any>>({})
  
  // Custom cleaning state
  const [customCleanedFiles, setCustomCleanedFiles] = useState<File[]>([])
  const [customCleaningResults, setCustomCleaningResults] = useState<any[]>([])
  
  // Final state
  const [uploadComplete, setUploadComplete] = useState(false)
  const [uploadSummary, setUploadSummary] = useState<{
    totalFiles: number;
    successCount: number;
    filesProcessed: UploadResult[];
  }>({
    totalFiles: 0,
    successCount: 0,
    filesProcessed: []
  })

  const transformPreprocessingResults = (serverResponse: any) => {
    if (!serverResponse) return null
    
    console.log("Transforming server response:", serverResponse)

    if (serverResponse.preprocessing_info) {
      return {
        success: serverResponse.success !== false,
        original_shape: serverResponse.original_shape || [0, 0],
        processed_shape: serverResponse.processed_shape || [0, 0],
        columns_dropped: serverResponse.preprocessing_info.columns_dropped || [],
        date_columns_detected: serverResponse.preprocessing_info.date_columns_detected || [],
        columns_cleaned: serverResponse.preprocessing_info.columns_cleaned || [],
        missing_value_stats: serverResponse.preprocessing_info.missing_value_stats || {},
        engineered_features: serverResponse.preprocessing_info.engineered_features || [],
        transformation_details: serverResponse.preprocessing_info.transformation_details || {}
      }
    }
    
    if (serverResponse.results && typeof serverResponse.results === 'object') {
      return transformPreprocessingResults(serverResponse.results)
    }
    
    if (serverResponse.report && typeof serverResponse.report === 'object') {
      const report = serverResponse.report
      return {
        success: serverResponse.success !== false,
        original_shape: report.original_shape || serverResponse.original_shape || [0, 0],
        processed_shape: report.processed_shape || serverResponse.processed_shape || [0, 0],
        columns_dropped: report.columns_dropped || [],
        date_columns_detected: report.date_columns_detected || [],
        columns_cleaned: report.columns_cleaned || [],
        missing_value_stats: report.missing_value_stats || {},
        engineered_features: report.engineered_features || [],
        transformation_details: report.transformation_details || {}
      }
    }
    
    return {
      success: serverResponse.success !== false,
      original_shape: serverResponse.original_shape || [0, 0],
      processed_shape: serverResponse.processed_shape || [0, 0],
      columns_dropped: serverResponse.columns_dropped || [],
      date_columns_detected: serverResponse.date_columns_detected || [],
      columns_cleaned: serverResponse.columns_cleaned || [],
      missing_value_stats: serverResponse.missing_value_stats || {},
      engineered_features: serverResponse.engineered_features || [],
      transformation_details: serverResponse.transformation_details || {}
    }
  }

  const uploadData = async () => {
    if (files.length === 0) {
      setError("Please select at least one file to upload")
      return
    }

    setIsUploading(true)
    setError(null)
    setUploadProgress(0)
    
    try {
      // Step 1: Upload to Python for preprocessing
      const pythonFormData = new FormData()
      files.forEach(file => {
        pythonFormData.append('files', file)
      })
      
      setUploadProgress(10)
      
      // Send to Python backend
      const pythonResponse = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: pythonFormData,
      })

      if (!pythonResponse.ok) {
        const errorData = await pythonResponse.json()
        throw new Error(errorData.detail || 'Preprocessing failed')
      }

      const preprocessingResult = await pythonResponse.json()
      setUploadProgress(50)
      
      // Initialize status tracking for preprocessing
      const processingInfo = (preprocessingResult.processing_info || []) as ProcessingInfo[]
      const initialStatus: Record<string, any> = {}
      processingInfo.forEach((info: ProcessingInfo) => {
        initialStatus[info.filename] = {
          status: info.status.status,
          progress: info.status.progress,
          message: info.status.message
        }
      })

      setProcessingStatus(initialStatus)
      
      // Step 2: Poll processing status
      const fileNames = processingInfo.map((info: ProcessingInfo) => info.filename)
      if (fileNames.length > 0) {
        await trackProcessingStatus(fileNames)
      }

      setUploadProgress(60)
      if (fileNames.length > 0 && Object.values(processingStatus).every(status => status.progress === 100)) {
        setUploadProgress(70)
        
        try {
          const preprocessingData = {}
          for (const fileName of fileNames) {
            const statusInfo = processingStatus[fileName]
            if (statusInfo && statusInfo.results) {
              console.log(`Preprocessing results for ${fileName}:`, statusInfo.results)
              preprocessingData[fileName] = statusInfo.results
            }
          }
          
          setFilePreprocessingResults(preprocessingData)
          setUploadProgress(80)
        } catch (err) {
          console.error('Error processing transformations:', err)
        }
      }
      
      // Step 3: Upload preprocessed files to database
      setUploadProgress(70)
      const uploadResults: UploadResult[] = []
      
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('preprocessed', 'true')

        if (filePreprocessingResults[file.name]) {
          formData.append('preprocessing_results', JSON.stringify(filePreprocessingResults[file.name]))
        }
        
        try {
          const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
          })

          const result = await response.json()
          
          if (!response.ok) {
            uploadResults.push({
              name: file.name,
              success: false
            })
            console.error(`Upload failed for ${file.name}:`, result.error || result.details)
          } else {
            uploadResults.push({
              name: file.name,
              success: true
            })
          }
        } catch (err) {
          uploadResults.push({
            name: file.name,
            success: false
          })
          console.error(`Exception during upload for ${file.name}:`, err)
        }
      }
      
      const successCount = uploadResults.filter(r => r.success).length
      
      setUploadSummary({
        totalFiles: files.length,
        successCount: successCount,
        filesProcessed: uploadResults
      })
      
      setUploadProgress(100)
      setUploadComplete(true)
      
      if (successCount === files.length) {
        toast({
          title: "Success",
          description: `All ${files.length} files have been uploaded and preprocessed successfully`,
        })
      } else {
        toast({
          variant: "destructive",
          title: "Partial success",
          description: `Uploaded ${successCount} of ${files.length} files successfully`,
        })
      }
      
      setCurrentStep(4)
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed'
      setError(errorMessage)
      console.error('Upload error:', err)
    } finally {
      setIsUploading(false)
    }
  }

  const trackProcessingStatus = async (fileNames: string[]) => {
    let allCompleted = false
    let attempts = 0
    const maxAttempts = 30
    
    while (!allCompleted && attempts < maxAttempts) {
      attempts++
      let completedCount = 0
      
      for (const fileName of fileNames) {
        try {
          const response = await fetch(`http://localhost:8000/processing-status/${fileName}`)
          if (response.ok) {
            const status = await response.json()
            
            if (status.results) {
              console.log(`[DEBUG] Full results structure for ${fileName}:`, 
                JSON.stringify(status.results, null, 2))
            }
            
            setProcessingStatus(prev => ({
              ...prev,
              [fileName]: {
                status: status.status,
                progress: status.progress,
                message: status.message,
                results: status.results
              }
            }))
            
            if (status.progress === 100 || status.progress === -1) {
              completedCount++
            }
          }
        } catch (error) {
          console.error(`Error checking status for ${fileName}:`, error)
        }
      }
      
      if (completedCount === fileNames.length) {
        allCompleted = true
        
        const preprocessingData = {}
        for (const fileName of fileNames) {
          const statusInfo = processingStatus[fileName]
          if (statusInfo && statusInfo.results) {
            console.log(`Final preprocessing results for ${fileName}:`, statusInfo.results)
            preprocessingData[fileName] = statusInfo.results
          }
        }
        
        setFilePreprocessingResults(preprocessingData)
      } else {
        await new Promise(resolve => setTimeout(resolve, 10000))
      }
    }
    
    return allCompleted
  }

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prevFiles) => [...prevFiles, ...acceptedFiles])
    setError(null)
  }, [])

  const onDropRejected = useCallback((fileRejections: FileRejection[]) => {
    if (fileRejections.length > 0) {
      const rejection = fileRejections[0]
      if (rejection.errors[0]?.code === "file-invalid-type") {
        setError("Only CSV and XLSX files are allowed.")
      } else {
        setError(`Error: ${rejection.errors[0]?.message || "There was an error uploading your file."}`)
      }
    }
  }, [])

  const { getRootProps, getInputProps, open } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      "text/csv": ['.csv'],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    maxSize: 10485760, // 10MB
    noClick: true
  })
  
  const handlePreviewFile = (file: File) => {
    setSelectedFileForPreview(file)
    setIsPreviewOpen(true)
  }
  
  const handleViewEDA = (fileIndex: number) => {
    setActiveFileIndex(fileIndex)
    setIsEdaReportOpen(true)
  }
  
  const handleEdaReportClosed = () => {
    setIsEdaReportOpen(false)
    
    if (activeFileIndex >= 0 && activeFileIndex < files.length) {
      const fileName = files[activeFileIndex].name
      setReviewedFiles(prev => new Set([...prev, fileName]))
    }
    
    if (files.length > 0 && reviewedFiles.size === files.length) {
      setTimeout(() => {
        toast({
          title: "All files reviewed",
          description: "Ready to proceed with upload and preprocessing",
        })
      }, 500)
    }
  }
  
  const handleContinueToReview = () => {
    if (files.length === 0) {
      setError("Please select at least one file to upload")
      return
    }
    setCurrentStep(1)
  }
  
  const handleStartUpload = () => {
    if (files.length === 0) {
      setError("Please select at least one file to upload")
      return
    }
    
    setCurrentStep(2) // Go to auto preprocessing
  }
  
  const handleAutoPreprocessingComplete = (processedFiles: File[], results: Record<string, any>) => {
    setPreprocessedFiles(processedFiles)
    setPreprocessingResults(results)
    setCurrentStep(3) // Go to custom cleaning
  }
  
  const handleCustomCleaningComplete = (cleanedResults: any[]) => {
    setCustomCleaningResults(cleanedResults)
    
    // Download the custom cleaned files if available
    if (cleanedResults.length > 0) {
      const downloadCustomCleanedFiles = async () => {
        const cleanedFiles: File[] = []
        
        for (const result of cleanedResults) {
          try {
            const response = await fetch(`http://localhost:8000/preprocessing_results/${result.transformed_file}`)
            if (response.ok) {
              const blob = await response.blob()
              const cleanedFile = new File([blob], result.transformed_file, { type: 'text/csv' })
              cleanedFiles.push(cleanedFile)
            }
          } catch (err) {
            console.error(`Failed to download cleaned file ${result.transformed_file}:`, err)
          }
        }
        
        setCustomCleanedFiles(cleanedFiles)
        setCurrentStep(4) // Go to final upload
      }
      
      downloadCustomCleanedFiles()
    } else {
      // No custom cleaning, use preprocessed files
      setCustomCleanedFiles(preprocessedFiles)
      setCurrentStep(4)
    }
  }
  
  const handleFinalUploadComplete = (summary: any) => {
    setUploadSummary(summary)
    setUploadComplete(true)
    setCurrentStep(5) // Go to complete
    
    if (summary.successCount === summary.totalFiles) {
      toast({
        title: "Success",
        description: `All ${summary.totalFiles} files have been uploaded successfully`,
      })
    } else {
      toast({
        variant: "destructive",
        title: "Partial success",
        description: `Uploaded ${summary.successCount} of ${summary.totalFiles} files successfully`,
      })
    }
  }
  
  const areAllFilesReviewed = files.length > 0 && 
    files.every(file => reviewedFiles.has(file.name))
  
  const handleFinish = (destination: 'dashboard' | 'feature-engineering') => {
    window.location.href = destination === 'dashboard' 
      ? '/dashboard' 
      : '/dashboard/feature-engineering'
  }

  return (
    <div className="max-w-5xl mx-auto">
      <Stepper currentStep={currentStep}>
        <Step>
          <StepTitle>Select Files</StepTitle>
          <StepDescription>Choose data files to upload</StepDescription>
        </Step>
        <Step>
          <StepTitle>Review Files</StepTitle>
          <StepDescription>Preview and analyze data</StepDescription>
        </Step>
        <Step>
          <StepTitle>Auto Preprocessing</StepTitle>
          <StepDescription>Automatic data cleaning</StepDescription>
        </Step>
        <Step>
          <StepTitle>Custom Cleaning</StepTitle>
          <StepDescription>Customize data cleaning</StepDescription>
        </Step>
        <Step>
          <StepTitle>Upload to Database</StepTitle>
          <StepDescription>Save processed data</StepDescription>
        </Step>
        <Step>
          <StepTitle>Complete</StepTitle>
          <StepDescription>Ready for analysis</StepDescription>
        </Step>
      </Stepper>
      
      <div className="mt-8">
        {currentStep === 0 && (
          <FileSelection
            files={files}
            onFilesChange={setFiles}
            onPreviewFile={handlePreviewFile}
            onContinue={handleContinueToReview}
            error={error}
            isDragActive={isDragActive}
            getRootProps={getRootProps}
            getInputProps={getInputProps}
            open={open}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        )}
        
        {currentStep === 1 && (
          <FileReview
            files={files}
            reviewedFiles={reviewedFiles}
            onPreviewFile={handlePreviewFile}
            onViewEDA={handleViewEDA}
            onBack={() => setCurrentStep(0)}
            onStartUpload={handleStartUpload}
            areAllFilesReviewed={areAllFilesReviewed}
          />
        )}
        
        {currentStep === 2 && (
          <AutoPreprocessing
            files={files}
            onBack={() => setCurrentStep(1)}
            onContinue={handleAutoPreprocessingComplete}
          />
        )}
        
        {currentStep === 3 && (
          <CustomCleaning
            files={preprocessedFiles.length > 0 ? preprocessedFiles : files}
            onBack={() => setCurrentStep(2)}
            onContinue={handleCustomCleaningComplete}
            onPreviewFile={handlePreviewFile}
          />
        )}
        
        {currentStep === 4 && (
          <FinalUpload
            originalFiles={files}
            processedFiles={customCleanedFiles.length > 0 ? customCleanedFiles : preprocessedFiles}
            preprocessingResults={preprocessingResults}
            customCleaningResults={customCleaningResults}
            onBack={() => setCurrentStep(3)}
            onComplete={handleFinalUploadComplete}
          />
        )}
        
        {currentStep === 5 && (
          <UploadComplete
            uploadSummary={uploadSummary}
            filePreprocessingResults={preprocessingResults}
            customCleaningResults={customCleaningResults}
            onFinish={handleFinish}
            transformPreprocessingResults={transformPreprocessingResults}
          />
        )}
      </div>
      
      {/* File Preview Dialog */}
      {selectedFileForPreview && (
        <FilePreview 
          fileMetadata={{
            id: Date.now().toString(),
            user_id: "temporary",
            filename: selectedFileForPreview.name,
            original_filename: selectedFileForPreview.name,
            file_size: selectedFileForPreview.size,
            mime_type: selectedFileForPreview.type,
            upload_date: new Date().toISOString(),
            column_names: [],
            row_count: 0,
            file_preview: [],
            statistics: {}
          }}
          isOpen={isPreviewOpen} 
          onClose={() => setIsPreviewOpen(false)} 
        />
      )}
      
      {/* EDA Report Dialog */}
      {files[activeFileIndex] && (
        <EdaReportViewer 
          fileMetadata={{
            id: Date.now().toString(),
            user_id: "temporary",
            filename: files[activeFileIndex].name,
            original_filename: files[activeFileIndex].name,
            file_size: files[activeFileIndex].size,
            mime_type: files[activeFileIndex].type,
            upload_date: new Date().toISOString(),
            column_names: [],
            row_count: 0,
            file_preview: [],
            statistics: {}
          }}
          originalFile={files[activeFileIndex]}
          isOpen={isEdaReportOpen} 
          onClose={handleEdaReportClosed} 
        />
      )}
    </div>
  )
}