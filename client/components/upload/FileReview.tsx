// components/upload/FileReview.tsx
'use client'

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { BarChart2, Eye, CheckCircle2, ArrowLeft, Upload } from "lucide-react"

interface FileReviewProps {
  files: File[]
  reviewedFiles: Set<string>
  onPreviewFile: (file: File) => void
  onViewEDA: (fileIndex: number) => void
  onBack: () => void
  onStartUpload: () => void
  areAllFilesReviewed: boolean
}

export default function FileReview({
  files,
  reviewedFiles,
  onPreviewFile,
  onViewEDA,
  onBack,
  onStartUpload,
  areAllFilesReviewed
}: FileReviewProps) {
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Review Your Data</CardTitle>
        <CardDescription>
          Preview and generate reports for your data before uploading
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Alert className="border-blue-200 bg-blue-50 text-blue-800 mb-6">
          <BarChart2 className="h-4 w-4" />
          <AlertTitle>Review Required</AlertTitle>
          <AlertDescription>
            Please review the EDA report for each file before proceeding with upload and preprocessing.
            This helps ensure your data is suitable for analysis.
          </AlertDescription>
        </Alert>
        
        <Table>
          <TableCaption>Review all files before continuing</TableCaption>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Format</TableHead>
              <TableHead className="text-center">Status</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {files.map((file, index) => (
              <TableRow key={index}>
                <TableCell className="font-medium">{file.name}</TableCell>
                <TableCell>{(file.size / 1048576).toFixed(2)} MB</TableCell>
                <TableCell>{file.type || file.name.split(".").pop()?.toUpperCase()}</TableCell>
                <TableCell className="text-center">
                  {reviewedFiles.has(file.name) ? (
                    <Badge variant="outline" className="bg-green-50 text-green-700">
                      <CheckCircle2 className="h-3 w-3 mr-1" />
                      <span>Reviewed</span>
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="bg-amber-50 text-amber-700">
                      <span>Needs Review</span>
                    </Badge>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex justify-end gap-2">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => onPreviewFile(file)}
                      className="gap-1"
                    >
                      <Eye className="h-3 w-3" />
                      <span>Preview</span>
                    </Button>
                    <Button 
                      variant={reviewedFiles.has(file.name) ? "outline" : "default"}
                      size="sm" 
                      onClick={() => onViewEDA(index)}
                      className="gap-1"
                    >
                      <BarChart2 className="h-3 w-3" />
                      <span>{reviewedFiles.has(file.name) ? "View Report Again" : "Generate Report"}</span>
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
      <CardFooter className="justify-between">
        <Button variant="outline" onClick={onBack}>
          Back to Files
        </Button>
        <Button 
          onClick={onStartUpload} 
          disabled={!areAllFilesReviewed}
          className="gap-2"
        >
          <Upload className="h-4 w-4" />
          <span>Upload & Process Data</span>
        </Button>
      </CardFooter>
    </Card>
  )
}