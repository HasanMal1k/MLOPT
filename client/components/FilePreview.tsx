"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose, DialogDescription } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { X, BarChart, ArrowLeft, ArrowRight, Maximize2, Minimize2, Download } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import TransformationsView from "./TransformationsView"
import { format, parseISO, isValid } from "date-fns"
import EdaReportViewer from "./EdaReportViewer"

interface FilePreviewProps {
  fileMetadata: FileMetadata | null
  isOpen: boolean
  onClose: () => void
}

export interface FileMetadata {
  id: string
  user_id: string
  name?: string
  filename: string
  original_filename: string
  file_size: number
  mime_type: string
  upload_date: string
  column_names: string[]
  row_count: number
  file_preview: Record<string, any>[]
  statistics: Record<string, any>
  preprocessing_info?: {
    is_preprocessed: boolean
    preprocessing_date?: string
    columns_cleaned?: string[]
    auto_detected_dates?: string[]
    dropped_columns?: string[]
    missing_value_stats?: Record<string, any>
    engineered_features?: string[]
    transformation_details?: Record<string, any>
  }
}

export default function FilePreview({ fileMetadata, isOpen, onClose }: FilePreviewProps) {
  const [isEdaReportOpen, setIsEdaReportOpen] = useState(false)
  const [isTableExpanded, setIsTableExpanded] = useState(false)

  if (!fileMetadata) return null

  const headers = fileMetadata.column_names || []
  const previewData = fileMetadata.file_preview || []
  const isPreprocessed = fileMetadata.preprocessing_info?.is_preprocessed || false

  const formatSafeDate = (dateValue: any): string => {
    if (!dateValue) return ""

    try {
      let date: Date

      if (typeof dateValue === "string") {
        date = parseISO(dateValue)
      } else if (dateValue instanceof Date) {
        date = dateValue
      } else {
        return String(dateValue)
      }

      if (!isValid(date)) {
        return String(dateValue)
      }

      return format(date, "yyyy-MM-dd HH:mm:ss")
    } catch (error) {
      return String(dateValue)
    }
  }

  const handleOpenEdaReport = () => {
    setIsEdaReportOpen(true)
  }

  const handleDownload = async () => {
    try {
      const response = await fetch(`http://localhost:8000/download/${fileMetadata.filename}`)
      if (!response.ok) throw new Error('Download failed')
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = fileMetadata.filename
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Error downloading file:', error)
    }
  }

  // Format cell content with proper truncation
  const formatCellContent = (value: any, maxLength = 30): string => {
    if (value === null || value === undefined) return ""
    const stringValue = String(value)
    if (stringValue.length <= maxLength) return stringValue
    return stringValue.substring(0, maxLength) + "..."
  }

  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
        <DialogContent
          className={`${isTableExpanded ? "max-w-[95vw] w-[95vw]" : "max-w-6xl w-[90vw]"} max-h-[90vh] h-[90vh] flex flex-col bg-black border-white/20`}
        >
          <DialogHeader className="flex flex-row items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-3">
              <DialogTitle className="flex items-center gap-2 text-white">
                {fileMetadata ? `Preview: ${fileMetadata.original_filename}` : "File Preview"}
                {isPreprocessed && (
                  <Badge variant="outline" className="bg-black text-white border-white/20">
                    Preprocessed
                  </Badge>
                )}
              </DialogTitle>
              <DialogDescription className="sr-only">
                Preview and analyze your data file with statistics and visualizations
              </DialogDescription>
            </div>
            <div className="flex items-center gap-2">
              {isPreprocessed && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDownload}
                  className="h-8 px-3 text-white hover:bg-white/10 hover:text-white"
                  title="Download preprocessed data"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsTableExpanded(!isTableExpanded)}
                className="h-8 w-8 p-0 text-white hover:bg-white/10 hover:text-white"
                title={isTableExpanded ? "Minimize" : "Expand for better table view"}
              >
                {isTableExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </Button>
              <DialogClose className="hover:bg-white/10 hover:text-white p-2 rounded-full transition-colors text-white">
                <X className="h-4 w-4" />
              </DialogClose>
            </div>
          </DialogHeader>

          <Tabs defaultValue="data" className="flex-1 flex flex-col min-h-0">
            <TabsList className="flex-shrink-0 bg-black">
              <TabsTrigger
                value="data"
                className="text-white data-[state=active]:bg-white/10 data-[state=active]:text-white"
              >
                Data Preview
              </TabsTrigger>
              {isPreprocessed && (
                <TabsTrigger
                  value="preprocessing"
                  className="text-white data-[state=active]:bg-white/10 data-[state=active]:text-white"
                >
                  Preprocessing Info
                </TabsTrigger>
              )}
              {fileMetadata.preprocessing_info?.engineered_features &&
                fileMetadata.preprocessing_info.engineered_features.length > 0 && (
                  <TabsTrigger
                    value="transformations"
                    className="text-white data-[state=active]:bg-white/10 data-[state=active]:text-white"
                  >
                    Transformations
                  </TabsTrigger>
                )}
            </TabsList>

            <TabsContent value="data" className="flex-1 flex flex-col min-h-0">
              {previewData.length > 0 ? (
                <div className="flex-1 flex flex-col min-h-0">
                  {/* Info bar */}
                  <div className="flex items-center justify-between mb-3 text-sm text-white flex-shrink-0">
                    <span>
                      {headers.length} columns × {previewData.length} preview rows
                    </span>
                    <div className="flex items-center gap-2">
                      <ArrowLeft className="h-3 w-3" />
                      <span>Scroll horizontally to see all columns</span>
                      <ArrowRight className="h-3 w-3" />
                    </div>
                  </div>

                  {/* Table container with minimal styling */}
                  <div
                    className="flex-1 rounded-lg bg-black relative"
                    style={{
                      minHeight: "400px",
                      maxHeight: "100%",
                    }}
                  >
                    <div
                      className="w-full h-full overflow-x-auto overflow-y-auto"
                      style={{
                        scrollbarWidth: "thin",
                        scrollbarColor: "#ffffff #000000",
                      }}
                    >
                      <div
                        style={{
                          minWidth: `${Math.max(headers.length * 180, 1200)}px`,
                          width: "max-content",
                        }}
                      >
                        <table
                          className="border-collapse"
                          style={{
                            width: "100%",
                            userSelect: "text",
                          }}
                        >
                          <thead className="sticky top-0 bg-black z-10">
                            <tr className="border-b border-white/20">
                              {headers.map((header, index) => (
                                <th
                                  key={index}
                                  className="text-left p-4 font-medium text-sm"
                                  style={{
                                    minWidth: "180px",
                                    width: "200px",
                                    userSelect: "text",
                                  }}
                                >
                                  <div className="flex flex-col gap-1">
                                    <span
                                      className="font-medium text-white truncate"
                                      title={header}
                                      style={{ userSelect: "text" }}
                                    >
                                      {header}
                                    </span>
                                    {fileMetadata.statistics[header] && (
                                      <span
                                        className="text-xs text-white/60 font-normal"
                                        style={{ userSelect: "text" }}
                                      >
                                        {fileMetadata.statistics[header].type === "numeric"
                                          ? "Numeric"
                                          : fileMetadata.statistics[header].type === "categorical"
                                            ? "Text"
                                            : "Mixed"}
                                      </span>
                                    )}
                                  </div>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {previewData.map((row, rowIndex) => (
                              <tr key={rowIndex} className="group" style={{ userSelect: "text" }}>
                                {headers.map((header, cellIndex) => (
                                  <td
                                    key={cellIndex}
                                    className="p-4 text-sm"
                                    style={{
                                      minWidth: "180px",
                                      width: "200px",
                                      userSelect: "text",
                                    }}
                                    title={String(row[header] || "")}
                                  >
                                    <div className="truncate" style={{ userSelect: "text" }}>
                                      {row[header] === null || row[header] === undefined ? (
                                        <span className="text-white/40 italic" style={{ userSelect: "text" }}>
                                          null
                                        </span>
                                      ) : (
                                        <span className="text-white" style={{ userSelect: "text" }}>
                                          {formatCellContent(row[header])}
                                        </span>
                                      )}
                                    </div>
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-64">
                  <p className="text-white">No preview data available</p>
                </div>
              )}

              
            </TabsContent>

            {isPreprocessed && (
              <TabsContent value="preprocessing" className="flex-1 overflow-auto">
                <div className="h-full overflow-auto p-4">
                  <h3 className="font-semibold text-lg mb-4 text-white">Preprocessing Summary</h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {fileMetadata.preprocessing_info?.preprocessing_date && (
                      <div>
                        <h4 className="font-medium text-sm text-white opacity-70 mb-1">Preprocessing Date</h4>
                        <p className="text-white">
                          {new Date(fileMetadata.preprocessing_info.preprocessing_date).toLocaleString()}
                        </p>
                      </div>
                    )}

                    {fileMetadata.preprocessing_info?.auto_detected_dates &&
                      fileMetadata.preprocessing_info.auto_detected_dates.length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm text-white opacity-70 mb-1">Auto-Detected Date Columns</h4>
                          <div className="flex flex-wrap gap-1">
                            {fileMetadata.preprocessing_info.auto_detected_dates.map((col) => (
                              <Badge key={col} variant="outline" className="bg-black text-white border-white/20">
                                {col}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                    {fileMetadata.preprocessing_info?.dropped_columns &&
                      fileMetadata.preprocessing_info.dropped_columns.length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm text-white opacity-70 mb-1">Dropped Columns</h4>
                          <div className="flex flex-wrap gap-1">
                            {fileMetadata.preprocessing_info.dropped_columns.map((col) => (
                              <Badge key={col} variant="outline" className="bg-black text-white border-white/20">
                                {col}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                    {fileMetadata.preprocessing_info?.columns_cleaned &&
                      fileMetadata.preprocessing_info.columns_cleaned.length > 0 && (
                        <div>
                          <h4 className="font-medium text-sm text-white opacity-70 mb-1">
                            Columns with Cleaned Values
                          </h4>
                          <div className="flex flex-wrap gap-1">
                            {fileMetadata.preprocessing_info.columns_cleaned.map((col) => (
                              <Badge key={col} variant="outline" className="bg-black text-white border-white/20">
                                {col}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                  </div>

                  {fileMetadata.preprocessing_info?.missing_value_stats && (
                    <div className="mt-6">
                      <h4 className="font-medium text-white mb-2">Missing Value Statistics</h4>
                      <div className="rounded-lg overflow-auto max-h-64">
                        <table className="w-full">
                          <thead className="bg-black sticky top-0">
                            <tr>
                              <th className="text-left p-3 font-medium text-white border-b border-white/20">Column</th>
                              <th className="text-left p-3 font-medium text-white border-b border-white/20">
                                Original Missing
                              </th>
                              <th className="text-left p-3 font-medium text-white border-b border-white/20">
                                Imputation Method
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(fileMetadata.preprocessing_info.missing_value_stats).map(
                              ([column, stats]) => (
                                <tr key={column}>
                                  <td className="p-3 font-medium text-white">{column}</td>
                                  <td className="p-3 text-white">
                                    {stats.missing_count} ({stats.missing_percentage}%)
                                  </td>
                                  <td className="p-3 text-white">{stats.imputation_method}</td>
                                </tr>
                              ),
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </TabsContent>
            )}

            {fileMetadata.preprocessing_info?.engineered_features && (
              <TabsContent value="transformations" className="flex-1 overflow-auto">
                <div className="h-full overflow-auto">
                  <TransformationsView
                    engineeredFeatures={fileMetadata.preprocessing_info.engineered_features}
                    transformationDetails={fileMetadata.preprocessing_info.transformation_details}
                  />
                </div>
              </TabsContent>
            )}
          </Tabs>

          <div className="text-xs text-white mt-2 flex justify-between items-center flex-shrink-0">
            <span>{previewData.length > 0 && `Showing ${previewData.length} of ${fileMetadata.row_count} rows`}</span>
            <span>File size: {(fileMetadata.file_size / (1024 * 1024)).toFixed(2)} MB</span>
          </div>
        </DialogContent>
      </Dialog>

      {/* EDA Report Dialog */}
      <EdaReportViewer fileMetadata={fileMetadata} isOpen={isEdaReportOpen} onClose={() => setIsEdaReportOpen(false)} />
    </>
  )
}
