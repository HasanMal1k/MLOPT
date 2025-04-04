"use client"

import type React from "react"

import { useState } from "react"
import { Eye, X } from "lucide-react"
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import FilePreview from "./FilePreview"

type data = {
  files: File[]
  setFiles: React.Dispatch<React.SetStateAction<File[]>>
}

function UploadedDataTable({ files, setFiles }: data) {
  const [previewFile, setPreviewFile] = useState<File | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)

  const removeItem = (fileName: string) => {
    const updatedFiles = files.filter((item) => item.name !== fileName)
    setFiles(updatedFiles)
  }

  const handlePreview = (file: File) => {
    setPreviewFile(file)
    setIsPreviewOpen(true)
  }

  const closePreview = () => {
    setIsPreviewOpen(false)
    setPreviewFile(null)
  }

  return (
    <>
      <Table className="mt-10">
        <TableCaption>Uploaded Data</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>Format</TableHead>
            <TableHead className="text-center">Preview</TableHead>
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

      <FilePreview file={previewFile} isOpen={isPreviewOpen} onClose={closePreview} />
    </>
  )
}

export default UploadedDataTable

