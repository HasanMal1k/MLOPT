// components/upload/FinalUpload.tsx - Updated with Time Series Support
'use client'

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { ArrowLeft, Upload, AlertCircle, Database, Clock, Settings2 } from "lucide-react"
import { toast } from "@/hooks/use-toast"
import { Badge } from "@/components/ui/badge"

interface UploadResult {
  name: string;
  success: boolean;
}

interface FinalUploadProps {
  originalFiles: File[]
  processedFiles: File[]
  preprocessingResults: Record<string, any>
  customCleaningResults: any[]
  onBack: () => void
  onComplete: (uploadSummary: any) => void
}

export default function FinalUpload({
  originalFiles,
  processedFiles,
  preprocessingResults,
  customCleaningResults,
  onBack,
  onComplete
}: FinalUploadProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  // Determine if we have time series data
  const hasTimeSeriesData = Object.values(preprocessingResults).some(
    result => result.dataset_type === 'time_series'
  )

  const hasCustomCleaning = customCleaningResults.length > 0

  const startUpload = async () => {
    setIsUploading(true)
    setError(null)
    setUploadProgress(0)
    
    try {
      const uploadResults: UploadResult[] = []
      const totalFiles = originalFiles.length
      
      // Upload each processed file to the database
      for (let i = 0; i < totalFiles; i++) {
        const originalFile = originalFiles[i]
        const processedFile = processedFiles[i] || originalFile
        
        setUploadProgress((i / totalFiles) * 90) // Leave 10% for final processing
        
        const formData = new FormData()
        formData.append('file', processedFile)
        formData.append('original_filename', originalFile.name)
        formData.append('preprocessed', 'true')
        
        // Determine dataset type and add appropriate metadata
        const fileResults = preprocessingResults[originalFile.name]
        const isTimeSeries = fileResults?.dataset_type === 'time_series'
        
        if (isTimeSeries) {
          // Add time series specific metadata
          formData.append('dataset_type', 'time_series')
          
          if (fileResults.time_series_config) {
            formData.append('time_series_config', JSON.stringify(fileResults.time_series_config))
          }
          
          if (fileResults.processing_statistics) {
            formData.append('time_series_statistics', JSON.stringify(fileResults.processing_statistics))
          }
        } else {
          formData.append('dataset_type', 'normal')
        }
        
        // Add preprocessing results if available
        if (fileResults && !isTimeSeries) {
          formData.append('preprocessing_results', JSON.stringify(fileResults))
        } else if (fileResults && isTimeSeries) {
          // For time series, we might still have some preprocessing info from the initial auto-preprocessing
          const preprocessingInfo = {
            ...fileResults,
            is_time_series_processed: true
          }
          formData.append('preprocessing_results', JSON.stringify(preprocessingInfo))
        }
        
        // Add custom cleaning info if available
        if (customCleaningResults.length > 0 && customCleaningResults[i]) {
          const customResult = customCleaningResults[i]
          formData.append('custom_cleaned', 'true')
          formData.append('custom_cleaning_report', JSON.stringify(customResult.report))
          formData.append('custom_cleaning_config', JSON.stringify({
            applied: true,
            transformations: customResult.report?.transformations_applied || [],
            columns_dropped: customResult.report?.columns_dropped || [],
            data_types_changed: customResult.report?.data_types || {}
          }))
        }
        
        try {
          const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
          })

          const result = await response.json()
          
          if (!response.ok) {
            uploadResults.push({
              name: originalFile.name,
              success: false
            })
            console.error(`Upload failed for ${originalFile.name}:`, result.error || result.details)
          } else {
            uploadResults.push({
              name: originalFile.name,
              success: true
            })
          }
        } catch (err) {
          uploadResults.push({
            name: originalFile.name,
            success: false
          })
          console.error(`Exception during upload for ${originalFile.name}:`, err)
        }
      }
      
      setUploadProgress(100)
      
      const successCount = uploadResults.filter(r => r.success).length
      
      const uploadSummary = {
        totalFiles: totalFiles,
        successCount: successCount,
        filesProcessed: uploadResults,
        hasTimeSeriesData,
        hasCustomCleaning
      }
      
      if (successCount === totalFiles) {
        toast({
          title: "Success",
          description: `All ${totalFiles} files have been uploaded successfully`,
        })
      } else {
        toast({
          variant: "destructive",
          title: "Partial success",
          description: `Uploaded ${successCount} of ${totalFiles} files successfully`,
        })
      }
      
      onComplete(uploadSummary)
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed'
      setError(errorMessage)
      console.error('Upload error:', err)
    } finally {
      setIsUploading(false)
    }
  }

  const getProcessingTypeIcon = () => {
    if (hasTimeSeriesData) {
      return <Clock className="h-12 w-12 mx-auto mb-4 text-blue-500" />
    } else if (hasCustomCleaning) {
      return <Settings2 className="h-12 w-12 mx-auto mb-4 text-purple-500" />
    } else {
      return <Database className="h-12 w-12 mx-auto mb-4 text-green-500" />
    }
  }

  const getProcessingSummary = () => {
    const items = []
    
    if (hasTimeSeriesData) {
      const timeSeriesCount = Object.values(preprocessingResults).filter(
        result => result.dataset_type === 'time_series'
      ).length
      items.push(`Time series processing: ${timeSeriesCount} files`)
    }
    
    if (hasCustomCleaning) {
      items.push(`Custom cleaning: ${customCleaningResults.length} files`)
    }
    
    const normalFiles = originalFiles.length - (hasTimeSeriesData ? 
      Object.values(preprocessingResults).filter(r => r.dataset_type === 'time_series').length : 0)
    
    if (normalFiles > 0) {
      items.push(`Standard processing: ${normalFiles} files`)
    }
    
    return items
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Final Upload to Database</CardTitle>
        <CardDescription>
          Upload your processed files to the database with appropriate metadata
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!isUploading ? (
          <div className="text-center p-8">
            {getProcessingTypeIcon()}
            <h3 className="text-lg font-medium mb-2">Ready to Upload</h3>
            <p className="text-sm text-muted-foreground mb-6 max-w-2xl mx-auto">
              Your files have been processed and are ready to be uploaded to the database. 
              Each file will be stored with its appropriate processing metadata.
            </p>
            
            <div className="bg-muted/50 rounded-lg p-4 mb-6 text-left max-w-lg mx-auto">
              <h4 className="font-medium mb-3">Upload Summary:</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Total files:</span>
                  <Badge variant="outline">{originalFiles.length}</Badge>
                </div>
                {getProcessingSummary().map((item, index) => (
                  <div key={index} className="flex justify-between text-sm">
                    <span>{item.split(':')[0]}:</span>
                    <Badge variant="outline" className={
                      item.includes('Time series') ? 'bg-blue-50 text-blue-700 border-blue-200' :
                      item.includes('Custom') ? 'bg-purple-50 text-purple-700 border-purple-200' :
                      'bg-green-50 text-green-700 border-green-200'
                    }>
                      {item.split(':')[1]?.trim()}
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
            
            {hasTimeSeriesData && (
              <Alert className="mb-4 border-blue-200 bg-blue-50">
                <Clock className="h-4 w-4 text-blue-600" />
                <AlertTitle className="text-blue-800">Time Series Data Detected</AlertTitle>
                <AlertDescription className="text-blue-700">
                  Your time series files have been processed with temporal-aware cleaning and will be 
                  stored with specialized time series metadata for optimal analysis.
                </AlertDescription>
              </Alert>
            )}
            
            <Button onClick={startUpload} size="lg" className="gap-2">
              <Upload className="h-4 w-4" />
              Upload to Database
            </Button>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-lg font-medium mb-2">Uploading Files</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Please wait while we upload your {originalFiles.length} processed files to the database
              </p>
            </div>
            
            <div className="mb-8">
              <p className="text-sm font-medium mb-2">Upload Progress</p>
              <Progress value={uploadProgress} className="h-3 w-full" />
              <p className="text-xs text-muted-foreground mt-2">
                {uploadProgress < 100 
                  ? `Uploading ${originalFiles.length} files with metadata...` 
                  : `Completed uploading ${originalFiles.length} files`}
              </p>
            </div>

            <div className="bg-muted/30 rounded-lg p-4">
              <h4 className="text-sm font-medium mb-2">What's being uploaded:</h4>
              <ul className="text-xs text-muted-foreground space-y-1">
                <li>• Processed data files with optimized formats</li>
                <li>• Processing metadata and transformation history</li>
                {hasTimeSeriesData && <li>• Time series configuration and frequency settings</li>}
                {hasCustomCleaning && <li>• Custom cleaning configurations and reports</li>}
                <li>• Data quality statistics and column information</li>
              </ul>
            </div>
          </div>
        )}
        
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error During Upload</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={onBack} disabled={isUploading}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <Button disabled={true}>
          Continue
        </Button>
      </CardFooter>
    </Card>
  )
}