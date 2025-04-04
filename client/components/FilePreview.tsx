"use client"

import { useState, useEffect } from "react"
import { X } from "lucide-react"
import Papa from "papaparse"
import * as XLSX from "xlsx"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface FilePreviewProps {
  file: File | null
  isOpen: boolean
  onClose: () => void
}

export default function FilePreview({ file, isOpen, onClose }: FilePreviewProps) {
  const [previewData, setPreviewData] = useState<any[]>([])
  const [headers, setHeaders] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (file && isOpen) {
      setIsLoading(true)
      setError(null)

      const fileType = file.name.split(".").pop()?.toLowerCase()

      if (fileType === "csv") {
        Papa.parse(file, {
          header: true,
          preview: 100, // Limit to first 100 rows for performance
          complete: (results) => {
            if (results.data && results.data.length > 0) {
              setPreviewData(results.data)
              setHeaders(results.meta.fields || [])
            } else {
              setError("No data found in the CSV file")
            }
            setIsLoading(false)
          },
          error: (error) => {
            setError(`Error parsing CSV: ${error.message}`)
            setIsLoading(false)
          },
        })
      } else if (fileType === "xlsx") {
        const reader = new FileReader()
        reader.onload = (e) => {
          try {
            const data = e.target?.result
            const workbook = XLSX.read(data, { type: "array" })
            const sheetName = workbook.SheetNames[0]
            const worksheet = workbook.Sheets[sheetName]
            const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 })

            if (jsonData && jsonData.length > 0) {
              const headers = jsonData[0] as string[]
              setHeaders(headers)

              // Convert the data to the format we need
              const rows = jsonData.slice(1, 101).map((row: any) => {
                const rowData: Record<string, any> = {}
                headers.forEach((header, index) => {
                  rowData[header] = row[index]
                })
                return rowData
              })

              setPreviewData(rows)
            } else {
              setError("No data found in the Excel file")
            }
          } catch (err) {
            setError(`Error parsing Excel file: ${err instanceof Error ? err.message : "Unknown error"}`)
          }
          setIsLoading(false)
        }
        reader.onerror = () => {
          setError("Error reading the file")
          setIsLoading(false)
        }
        reader.readAsArrayBuffer(file)
      } else {
        setError("Unsupported file format")
        setIsLoading(false)
      }
    }
  }, [file, isOpen])

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle>{file ? `Preview: ${file.name}` : "File Preview"}</DialogTitle>
          <DialogClose className="hover:bg-gray-100 p-2 rounded-full transition-colors">
            <X className="h-4 w-4" />
          </DialogClose>
        </DialogHeader>

        <div className="flex-1 overflow-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <p className="text-gray-400">Loading preview...</p>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-64">
              <p className="text-red-500">{error}</p>
            </div>
          ) : previewData.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  {headers.map((header, index) => (
                    <TableHead key={index}>{header}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {previewData.map((row, rowIndex) => (
                  <TableRow key={rowIndex}>
                    {headers.map((header, cellIndex) => (
                      <TableCell key={cellIndex}>{row[header] !== undefined ? String(row[header]) : ""}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="flex items-center justify-center h-64">
              <p className="text-gray-400">No data to preview</p>
            </div>
          )}
        </div>

        <div className="text-xs text-gray-400 mt-2">{previewData.length > 0 && "Showing first 100 rows"}</div>
      </DialogContent>
    </Dialog>
  )
}

