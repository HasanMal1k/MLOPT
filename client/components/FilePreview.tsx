"use client"

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { X } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface FilePreviewProps {
  fileMetadata: FileMetadata | null;
  isOpen: boolean;
  onClose: () => void;
}

export interface FileMetadata {
  id: string;
  user_id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  mime_type: string;
  upload_date: string;
  column_names: string[];
  row_count: number;
  file_preview: Record<string, any>[];
  statistics: Record<string, any>;
  preprocessing_info?: {
    is_preprocessed: boolean;
    preprocessing_date?: string;
    columns_cleaned?: string[];
    auto_detected_dates?: string[];
    dropped_columns?: string[];
    missing_value_stats?: Record<string, any>;
  };
}

export default function FilePreview({ fileMetadata, isOpen, onClose }: FilePreviewProps) {
  if (!fileMetadata) return null
  
  const headers = fileMetadata.column_names || []
  const previewData = fileMetadata.file_preview || []
  const isPreprocessed = fileMetadata.preprocessing_info?.is_preprocessed || false
  
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <div>
            <DialogTitle className="flex items-center gap-2">
              {fileMetadata ? `Preview: ${fileMetadata.original_filename}` : "File Preview"}
              {isPreprocessed && (
                <Badge variant="outline" className="ml-2 bg-green-50 text-green-700 border-green-200">
                  Preprocessed
                </Badge>
              )}
            </DialogTitle>
          </div>
          <DialogClose className="hover:bg-gray-100 p-2 rounded-full transition-colors">
            <X className="h-4 w-4" />
          </DialogClose>
        </DialogHeader>

        <Tabs defaultValue="data" className="w-full">
          <TabsList>
            <TabsTrigger value="data">Data Preview</TabsTrigger>
            <TabsTrigger value="stats">Statistics</TabsTrigger>
            {isPreprocessed && (
              <TabsTrigger value="preprocessing">Preprocessing Info</TabsTrigger>
            )}
          </TabsList>
          
          <TabsContent value="data" className="flex-1 overflow-auto">
            {previewData.length > 0 ? (
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
                <p className="text-gray-400">No preview data available</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="stats">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {headers.map(header => {
                const stats = fileMetadata.statistics[header];
                if (!stats) return null;
                
                return (
                  <div key={header} className="border rounded-lg p-4">
                    <h3 className="font-bold text-lg mb-2">{header}</h3>
                    {stats.type === 'categorical' ? (
                      <p>Unique values: {stats.uniqueCount}</p>
                    ) : (
                      <>
                        <p>Min: {stats.min}</p>
                        <p>Max: {stats.max}</p>
                        <p>Mean: {stats.mean.toFixed(2)}</p>
                        <p>Count: {stats.count}</p>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </TabsContent>
          
          {isPreprocessed && (
            <TabsContent value="preprocessing">
              <div className="p-4">
                <h3 className="font-semibold text-lg mb-4">Preprocessing Summary</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {fileMetadata.preprocessing_info?.preprocessing_date && (
                    <div>
                      <h4 className="font-medium text-sm text-gray-500 mb-1">Preprocessing Date</h4>
                      <p>{new Date(fileMetadata.preprocessing_info.preprocessing_date).toLocaleString()}</p>
                    </div>
                  )}
                  
                  {fileMetadata.preprocessing_info?.auto_detected_dates && fileMetadata.preprocessing_info.auto_detected_dates.length > 0 && (
                    <div>
                      <h4 className="font-medium text-sm text-gray-500 mb-1">Auto-Detected Date Columns</h4>
                      <div className="flex flex-wrap gap-1">
                        {fileMetadata.preprocessing_info.auto_detected_dates.map(col => (
                          <Badge key={col} variant="secondary" className="bg-blue-50 text-blue-700 border-blue-200">
                            {col}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {fileMetadata.preprocessing_info?.dropped_columns && fileMetadata.preprocessing_info.dropped_columns.length > 0 && (
                    <div>
                      <h4 className="font-medium text-sm text-gray-500 mb-1">Dropped Columns</h4>
                      <div className="flex flex-wrap gap-1">
                        {fileMetadata.preprocessing_info.dropped_columns.map(col => (
                          <Badge key={col} variant="secondary" className="bg-gray-100 text-gray-700">
                            {col}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {fileMetadata.preprocessing_info?.columns_cleaned && fileMetadata.preprocessing_info.columns_cleaned.length > 0 && (
                    <div>
                      <h4 className="font-medium text-sm text-gray-500 mb-1">Columns with Cleaned Values</h4>
                      <div className="flex flex-wrap gap-1">
                        {fileMetadata.preprocessing_info.columns_cleaned.map(col => (
                          <Badge key={col} variant="secondary" className="bg-green-50 text-green-700">
                            {col}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                
                {fileMetadata.preprocessing_info?.missing_value_stats && (
                  <div className="mt-6">
                    <h4 className="font-medium text-gray-700 mb-2">Missing Value Statistics</h4>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Column</TableHead>
                          <TableHead>Original Missing</TableHead>
                          <TableHead>Imputation Method</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Object.entries(fileMetadata.preprocessing_info.missing_value_stats).map(([column, stats]) => (
                          <TableRow key={column}>
                            <TableCell>{column}</TableCell>
                            <TableCell>{stats.missing_count} ({stats.missing_percentage}%)</TableCell>
                            <TableCell>{stats.imputation_method}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </div>
            </TabsContent>
          )}
        </Tabs>

        <div className="text-xs text-gray-400 mt-2">
          {previewData.length > 0 && `Showing ${previewData.length} of ${fileMetadata.row_count} rows`}
        </div>
      </DialogContent>
    </Dialog>
  )
}