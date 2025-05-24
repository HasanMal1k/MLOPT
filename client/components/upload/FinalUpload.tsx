// components/upload/FinalUpload.tsx
'use client'

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { ArrowLeft, Upload, AlertCircle, Database } from "lucide-react"
import { toast } from "@/hooks/use-toast"

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
        
        // Add preprocessing results if available
        if (preprocessingResults[originalFile.name]) {
          formData.append('preprocessing_results', JSON.stringify(preprocessingResults[originalFile.name]))
        }
        
        // Add custom cleaning info if available
        if (customCleaningResults.length > 0 && customCleaningResults[i]) {
          const customResult = customCleaningResults[i]
          formData.append('custom_cleaned', 'true')
          formData.append('custom_cleaning_report', JSON.stringify(customResult.report))
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
        filesProcessed: uploadResults
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

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Final Upload to Database</CardTitle>
        <CardDescription>
          Upload your processed and cleaned files to the database
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!isUploading ? (
          <div className="text-center p-8">
            <Database className="h-12 w-12 mx-auto mb-4 text-green-500" />
            <h3 className="text-lg font-medium mb-2">Ready to Upload</h3>
            <p className="text-sm text-muted-foreground mb-6 max-w-2xl mx-auto">
              Your files have been preprocessed and custom cleaned. They are now ready to be 
              uploaded to the database for analysis and machine learning.
            </p>
            
            <div className="bg-green-50 rounded-lg p-4 mb-6 text-left max-w-md mx-auto">
              <h4 className="font-medium text-green-800 mb-2">Upload Summary:</h4>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• Files to upload: {originalFiles.length}</li>
                <li>• Auto preprocessing: ✓ Complete</li>
                <li>• Custom cleaning: {customCleaningResults.length > 0 ? '✓ Applied' : '○ Skipped'}</li>
                <li>• Ready for machine learning: ✓</li>
              </ul>
            </div>
            
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
                  ? `Uploading ${originalFiles.length} files...` 
                  : `Completed uploading ${originalFiles.length} files`}
              </p>
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