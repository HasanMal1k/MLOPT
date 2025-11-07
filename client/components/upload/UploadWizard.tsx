// components/upload/UploadWizard.tsx - Fixed Version
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
import DataTypeDetection from "./DataTypeDetection"
import TimeSeriesProcessing from "./TimeSeriesProcessing"
import CustomCleaning from "./CustomCleaning"
import FinalUpload from "./FinalUpload"
import UploadComplete from "./UploadComplete"
import extractMetadata from "@/utils/extractMetaData"

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
  
  // Kaggle import tracking
  const [kaggleImportCount, setKaggleImportCount] = useState(0)
  
  // Auto preprocessing state
  const [preprocessedFiles, setPreprocessedFiles] = useState<File[]>([])
  const [preprocessingResults, setPreprocessingResults] = useState<Record<string, any>>({})
  
  // Data type detection state
  const [detectedDataTypes, setDetectedDataTypes] = useState<Record<string, 'normal' | 'time_series'>>({})
  
  // Time series processing state
  const [timeSeriesProcessedFiles, setTimeSeriesProcessedFiles] = useState<File[]>([])
  const [timeSeriesResults, setTimeSeriesResults] = useState<Record<string, any>>({})
  
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

  // Cache for file metadata to avoid recomputing
  const [fileMetadataCache, setFileMetadataCache] = useState<Map<string, any>>(new Map())

  // Fixed: Kaggle file import handler that prevents duplication
  const handleKaggleFileImported = (file: File) => {
    // Check if file already exists (by name and size to avoid exact duplicates)
    const exists = files.some(existingFile => 
      existingFile.name === file.name && existingFile.size === file.size
    )
    
    if (!exists) {
      setFiles(prev => [...prev, file])
      setKaggleImportCount(prev => prev + 1)
      
      // Switch to local files tab to show the imported file
      setActiveTab("upload")
      
      // Show success toast
      toast({
        title: "Kaggle import successful",
        description: `${file.name} has been imported from Kaggle and added to your upload queue`
      })
    } else {
      toast({
        title: "File already exists",
        description: `${file.name} is already in your upload queue`
      })
    }
  }

  // Create file metadata with caching to avoid recomputation
  const createFileMetadataWithPreview = async (file: File) => {
    const cacheKey = `${file.name}_${file.size}_${file.lastModified}`
    
    if (fileMetadataCache.has(cacheKey)) {
      return fileMetadataCache.get(cacheKey)
    }

    try {
      // Extract actual metadata from the file
      const metadata = await extractMetadata(file)
      
      const fileMetadata = {
        id: Date.now().toString(),
        user_id: "temporary",
        filename: file.name,
        original_filename: file.name,
        file_size: file.size,
        mime_type: file.type,
        upload_date: new Date().toISOString(),
        column_names: metadata.columns,
        row_count: metadata.rowCount,
        file_preview: metadata.preview,
        statistics: metadata.statistics
      }
      
      // Cache the result
      setFileMetadataCache(prev => new Map(prev).set(cacheKey, fileMetadata))
      
      return fileMetadata
    } catch (error) {
      console.error('Error extracting metadata:', error)
      
      // Fallback to basic metadata
      const basicMetadata = {
        id: Date.now().toString(),
        user_id: "temporary",
        filename: file.name,
        original_filename: file.name,
        file_size: file.size,
        mime_type: file.type,
        upload_date: new Date().toISOString(),
        column_names: [],
        row_count: 0,
        file_preview: [],
        statistics: {}
      }
      
      setFileMetadataCache(prev => new Map(prev).set(cacheKey, basicMetadata))
      return basicMetadata
    }
  }

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

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Filter out duplicates
    const newFiles = acceptedFiles.filter(newFile => 
      !files.some(existingFile => 
        existingFile.name === newFile.name && existingFile.size === newFile.size
      )
    )
    
    if (newFiles.length < acceptedFiles.length) {
      toast({
        title: "Duplicate files ignored",
        description: `${acceptedFiles.length - newFiles.length} file(s) were already in your queue`
      })
    }
    
    setFiles((prevFiles) => [...prevFiles, ...newFiles])
    setError(null)
  }, [files])

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
  
  const handlePreviewFile = async (file: File) => {
    try {
      const metadata = await createFileMetadataWithPreview(file)
      setSelectedFileForPreview(metadata)
      setIsPreviewOpen(true)
    } catch (error) {
      console.error('Error creating preview:', error)
      toast({
        variant: "destructive",
        title: "Preview Error",
        description: "Could not generate preview for this file"
      })
    }
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
    setCurrentStep(3) // Go to data type detection
  }
  
  // Handle data type detection complete
  const handleDataTypeDetectionNormal = (normalFiles: File[], results: Record<string, any>) => {
    setPreprocessedFiles(normalFiles)
    setPreprocessingResults(results)
    setCurrentStep(4) // Go to custom cleaning
  }
  
  const handleDataTypeDetectionTimeSeries = (timeSeriesFiles: File[], results: Record<string, any>) => {
    setPreprocessedFiles(timeSeriesFiles)
    setPreprocessingResults(results)
    setCurrentStep(5) // Go to time series processing
  }
  
  // Handle time series processing complete
  const handleTimeSeriesProcessingComplete = (processedFiles: File[], results: Record<string, any>) => {
    setTimeSeriesProcessedFiles(processedFiles)
    setTimeSeriesResults(results)
    setCurrentStep(6) // Go directly to final upload (skip custom cleaning for time series)
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
        setCurrentStep(6) // Go to final upload
      }
      
      downloadCustomCleanedFiles()
    } else {
      // No custom cleaning, use preprocessed files
      setCustomCleanedFiles(preprocessedFiles)
      setCurrentStep(6)
    }
  }
  
  const handleFinalUploadComplete = (summary: any) => {
    setUploadSummary(summary)
    setUploadComplete(true)
    setCurrentStep(7) // Go to complete
    
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
  
  const handleFinish = (destination: 'dashboard' | 'feature-engineering' | 'transformations' | 'blueprints') => {
    // Get the latest uploaded file for context
    const latestFile = uploadSummary.filesProcessed[uploadSummary.filesProcessed.length - 1]?.name
    
    switch (destination) {
      case 'transformations':
        window.location.href = latestFile 
          ? `/dashboard/transformations?file=${encodeURIComponent(latestFile)}`
          : '/dashboard/transformations'
        break
      case 'blueprints':
        window.location.href = latestFile
          ? `/dashboard/blueprints/train?file=${encodeURIComponent(latestFile)}`
          : '/dashboard/blueprints/train'
        break
      case 'feature-engineering':
        window.location.href = '/dashboard/feature-engineering'
        break
      case 'dashboard':
      default:
        window.location.href = '/dashboard'
        break
    }
  }

  // Determine which files to use for final upload and what results to include
  const getFinalUploadData = () => {
    // If we have time series processed files, use those
    if (timeSeriesProcessedFiles.length > 0) {
      return {
        originalFiles: files,
        processedFiles: timeSeriesProcessedFiles,
        preprocessingResults: { ...preprocessingResults, ...timeSeriesResults },
        customCleaningResults: []
      }
    }
    
    // Otherwise use custom cleaned files or preprocessed files
    return {
      originalFiles: files,
      processedFiles: customCleanedFiles.length > 0 ? customCleanedFiles : preprocessedFiles,
      preprocessingResults,
      customCleaningResults
    }
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
          <StepTitle>Data Type Detection</StepTitle>
          <StepDescription>Identify normal vs time series</StepDescription>
        </Step>
        <Step>
          <StepTitle>Custom Cleaning</StepTitle>
          <StepDescription>Customize data cleaning</StepDescription>
        </Step>
        <Step>
          <StepTitle>Time Series Processing</StepTitle>
          <StepDescription>Time series specific processing</StepDescription>
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
            onKaggleFileImported={handleKaggleFileImported}
            kaggleImportCount={kaggleImportCount}
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
          <DataTypeDetection
            files={preprocessedFiles.length > 0 ? preprocessedFiles : files}
            preprocessingResults={preprocessingResults}
            onBack={() => setCurrentStep(2)}
            onContinueNormal={handleDataTypeDetectionNormal}
            onContinueTimeSeries={handleDataTypeDetectionTimeSeries}
          />
        )}
        
        {currentStep === 4 && (
          <CustomCleaning
            files={preprocessedFiles.length > 0 ? preprocessedFiles : files}
            onBack={() => setCurrentStep(3)}
            onContinue={handleCustomCleaningComplete}
            onPreviewFile={handlePreviewFile}
          />
        )}
        
        {currentStep === 5 && (
          <TimeSeriesProcessing
            files={preprocessedFiles.length > 0 ? preprocessedFiles : files}
            preprocessingResults={preprocessingResults}
            onBack={() => setCurrentStep(3)}
            onComplete={handleTimeSeriesProcessingComplete}
          />
        )}
        
        {currentStep === 6 && (
          <FinalUpload
            {...getFinalUploadData()}
            onBack={() => {
              // Go back to appropriate step based on what processing was done
              if (timeSeriesProcessedFiles.length > 0) {
                setCurrentStep(5) // Back to time series processing
              } else {
                setCurrentStep(4) // Back to custom cleaning
              }
            }}
            onComplete={handleFinalUploadComplete}
          />
        )}
        
        {currentStep === 7 && (
          <UploadComplete
            uploadSummary={uploadSummary}
            filePreprocessingResults={{ ...preprocessingResults, ...timeSeriesResults }}
            customCleaningResults={customCleaningResults}
            onFinish={handleFinish}
            transformPreprocessingResults={transformPreprocessingResults}
          />
        )}
      </div>
      
      {/* File Preview Dialog */}
      {selectedFileForPreview && (
        <FilePreview 
          fileMetadata={selectedFileForPreview}
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