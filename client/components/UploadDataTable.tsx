"use client"

import type React from "react"
import { useState } from "react"
import { Eye, X, BarChart2 } from "lucide-react"
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import FilePreview from "./FilePreview"
import EdaReportViewer from "./EdaReportViewer"

type data = {
  files: File[]
  setFiles: React.Dispatch<React.SetStateAction<File[]>>
  onUpload: (files: File[]) => Promise<void>
}

// Create a mock FileMetadata from a File object
const createMockFileMetadata = (file: File) => {
  return {
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
  };
};

function UploadedDataTable({ files, setFiles, onUpload }: data) {
  const [previewFile, setPreviewFile] = useState<any | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [isEdaReportOpen, setIsEdaReportOpen] = useState(false)
  const [currentFileIndex, setCurrentFileIndex] = useState<number>(-1)
  const [currentFileObj, setCurrentFileObj] = useState<File | null>(null)
  const [reviewedFiles, setReviewedFiles] = useState<Set<string>>(new Set())
  const [canProceed, setCanProceed] = useState<boolean>(false)

  const removeItem = (fileName: string) => {
    const updatedFiles = files.filter((item) => item.name !== fileName)
    setFiles(updatedFiles)
    
    // Remove from reviewed files if it exists
    if (reviewedFiles.has(fileName)) {
      const updatedReviewed = new Set(reviewedFiles)
      updatedReviewed.delete(fileName)
      setReviewedFiles(updatedReviewed)
    }
    
    // Update canProceed status
    updateCanProceedStatus(updatedFiles, reviewedFiles)
  }

  const handlePreview = (file: File) => {
    // Create a mock file metadata for preview
    const mockFileMetadata = createMockFileMetadata(file);
    setPreviewFile(mockFileMetadata)
    setIsPreviewOpen(true)
  }

  const closePreview = () => {
    setIsPreviewOpen(false)
    setPreviewFile(null)
  }
  
  const handleShowEDA = (file: File, index: number) => {
    const mockFileMetadata = createMockFileMetadata(file);
    setPreviewFile(mockFileMetadata)
    setCurrentFileIndex(index)
    setCurrentFileObj(file)  // Store the actual File object
    setIsEdaReportOpen(true)
  }
  
  const closeEdaReport = () => {
    setIsEdaReportOpen(false)
    
    // Mark the file as reviewed
    if (currentFileIndex >= 0 && currentFileIndex < files.length) {
      const fileName = files[currentFileIndex].name
      const updatedReviewed = new Set(reviewedFiles)
      updatedReviewed.add(fileName)
      setReviewedFiles(updatedReviewed)
      
      // Update canProceed status
      updateCanProceedStatus(files, updatedReviewed)
    }
    
    setCurrentFileIndex(-1)
    setCurrentFileObj(null)
  }
  
  const updateCanProceedStatus = (filesList: File[], reviewed: Set<string>) => {
    // Can proceed if all files have been reviewed
    setCanProceed(filesList.length > 0 && filesList.every(file => reviewed.has(file.name)))
  }
  
  const handleUpload = async () => {
    if (canProceed) {
      await onUpload(files)
    }
  }

  return (
    <>
      <Table className="mt-10">
        <TableCaption>Review data before preprocessing</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>Format</TableHead>
            <TableHead className="text-center">Preview</TableHead>
            <TableHead className="text-center">EDA Report</TableHead>
            <TableHead className="text-center">Status</TableHead>
            <TableHead className="text-right">Remove</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {files.map((item, index) => (
            <TableRow key={index}>
              <TableCell className="font-semibold">{item.name}</TableCell>
              <TableCell className="font-semibold">{(item.size / 1048576).toFixed(1)} MB</TableCell>
              <TableCell className="font-semibold">{item.type || item.name.split(".").pop()?.toUpperCase()}</TableCell>
              <TableCell className="text-center">
                <button
                  onClick={() => handlePreview(item)}
                  className="hover:bg-gray-100 p-2 rounded-full transition-colors"
                  title="Preview file"
                >
                  <Eye size={18} color="gray" />
                </button>
              </TableCell>
              <TableCell className="text-center">
                <button
                  onClick={() => handleShowEDA(item, index)}
                  className="hover:bg-gray-100 p-2 rounded-full transition-colors"
                  title="View EDA Report"
                >
                  <BarChart2 size={18} color={reviewedFiles.has(item.name) ? "green" : "gray"} />
                </button>
              </TableCell>
              <TableCell className="text-center">
                {reviewedFiles.has(item.name) ? (
                  <span className="text-green-600 text-sm">Reviewed</span>
                ) : (
                  <span className="text-amber-600 text-sm">Pending review</span>
                )}
              </TableCell>
              <TableCell className="text-right">
                <button
                  onClick={() => removeItem(item.name)}
                  className="hover:bg-gray-100 p-2 rounded-full transition-colors"
                  title="Remove file"
                >
                  <X size={18} color="gray" />
                </button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {files.length > 0 && (
        <div className="mt-6">
          {canProceed ? (
            <Alert className="bg-green-50 border-green-200 mb-4">
              <AlertTitle className="text-green-800">All files have been reviewed</AlertTitle>
              <AlertDescription className="text-green-700">
                You've reviewed all your files. Click "Process & Upload Data" below to continue with preprocessing and uploading.
              </AlertDescription>
            </Alert>
          ) : (
            <Alert className="bg-amber-50 border-amber-200 mb-4">
              <AlertTitle className="text-amber-800">Review required</AlertTitle>
              <AlertDescription className="text-amber-700">
                Please review the EDA report for each file before proceeding with preprocessing and upload.
              </AlertDescription>
            </Alert>
          )}
          
          <div className="flex justify-center">
            <Button 
              onClick={handleUpload}
              disabled={!canProceed}
              className="px-6"
            >
              Process & Upload Data
            </Button>
          </div>
        </div>
      )}

      <FilePreview fileMetadata={previewFile} isOpen={isPreviewOpen} onClose={closePreview} />
      
      {/* Pass the original File object to the EdaReportViewer */}
      {previewFile && (
        <EdaReportViewer
          fileMetadata={previewFile}
          originalFile={currentFileObj}
          isOpen={isEdaReportOpen}
          onClose={closeEdaReport}
        />
      )}
    </>
  )
}

export default UploadedDataTable