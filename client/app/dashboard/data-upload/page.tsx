// app/dashboard/data-upload/page.tsx
'use client'

import { useState, useCallback } from "react"
import { useDropzone, type FileRejection } from "react-dropzone"
import { FilePlus2 } from "lucide-react"
import UploadWizard from "@/components/upload/UploadWizard"

export default function DataUpload() {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    // This will be handled by the UploadWizard component
  }, [])

  const onDropRejected = useCallback((fileRejections: FileRejection[]) => {
    // This will be handled by the UploadWizard component
  }, [])

  const { getRootProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      "text/csv": ['.csv'],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    maxSize: 10485760, // 10MB
    noClick: true
  })

  return (
    <div {...getRootProps()} className="h-screen w-full px-6 md:px-10 py-10">
      {isDragActive && (
        <div className="absolute h-full w-full inset-0 bg-primary/10 flex items-center justify-center text-4xl backdrop-blur-sm z-50">
          <div className="bg-white dark:bg-gray-900 shadow-lg rounded-xl p-8 border border-primary">
            <FilePlus2 className="mx-auto h-16 w-16 text-primary mb-4" />
            <p className="text-center text-2xl font-medium">Drop Your Files Here</p>
          </div>
        </div>
      )}
      
      <div className="text-4xl font-bold mb-6">
        Upload Data
      </div>
      
      <UploadWizard isDragActive={isDragActive} />
    </div>
  )
}