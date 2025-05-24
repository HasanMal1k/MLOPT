// components/upload/UploadProgress.tsx
'use client'

import { Progress } from "@/components/ui/progress"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ProcessingInfo {
  filename: string;
  status: {
    status: string;
    progress: number;
    message: string;
  };
}

interface UploadProgressProps {
  files: File[]
  uploadProgress: number
  processingStatus: Record<string, any>
  error: string | null
}

export default function UploadProgress({
  files,
  uploadProgress,
  processingStatus,
  error
}: UploadProgressProps) {
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Uploading & Processing Your Data</CardTitle>
        <CardDescription>
          Please wait while we upload and preprocess your files
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mt-2 mb-8">
          <p className="text-sm font-medium mb-2">Overall Progress</p>
          <Progress value={uploadProgress} className="h-2 w-full" />
          <p className="text-xs text-muted-foreground mt-2">
            {uploadProgress < 100 
              ? `Processing ${files.length} files...` 
              : `Completed processing ${files.length} files`}
          </p>
        </div>
        
        {Object.keys(processingStatus).length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-medium mb-3">File Status:</p>
            {Object.entries(processingStatus).map(([filename, status]) => (
              <div key={filename} className="mb-3">
                <div className="flex justify-between text-xs">
                  <span className="font-medium">{filename}</span>
                  <span>{status.message}</span>
                </div>
                <Progress 
                  value={status.progress < 0 ? 100 : status.progress} 
                  className={`h-2 w-full ${status.progress < 0 ? 'bg-red-300' : ''}`} 
                />
              </div>
            ))}
          </div>
        )}
        
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error During Processing</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="justify-between">
        <Button variant="outline" disabled={true}>
          Back
        </Button>
        <Button disabled={true}>
          Continue
        </Button>
      </CardFooter>
    </Card>
  )
}