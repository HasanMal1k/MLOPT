"use client"

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { X } from "lucide-react"

interface FilePreviewProps {
  fileMetadata: File | null;
  isOpen: boolean;
  onClose: () => void;
}

interface FileMetadata {
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
}


export default function FilePreview({ fileMetadata, isOpen, onClose }: FilePreviewProps) {
  if (!fileMetadata) return null
  
  const headers = fileMetadata.column_names || []
  const previewData = fileMetadata.file_preview || []
  
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle>{fileMetadata ? `Preview: ${fileMetadata.original_filename}` : "File Preview"}</DialogTitle>
          <DialogClose className="hover:bg-gray-100 p-2 rounded-full transition-colors">
            <X className="h-4 w-4" />
          </DialogClose>
        </DialogHeader>

        <Tabs defaultValue="data" className="w-full">
          <TabsList>
            <TabsTrigger value="data">Data Preview</TabsTrigger>
            <TabsTrigger value="stats">Statistics</TabsTrigger>
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
        </Tabs>

        <div className="text-xs text-gray-400 mt-2">
          {previewData.length > 0 && `Showing ${previewData.length} of ${fileMetadata.row_count} rows`}
        </div>
      </DialogContent>
    </Dialog>
  )
}