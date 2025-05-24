// components/upload/FileSelection.tsx
'use client'

import { useCallback } from "react"
import { useDropzone, type FileRejection } from "react-dropzone"
import { FilePlus2, X, Eye, Link as LinkIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, FileText, ChevronRight } from "lucide-react"
import KaggleUpload from "@/components/KaggleUpload"

interface FileSelectionProps {
  files: File[]
  onFilesChange: (files: File[]) => void
  onPreviewFile: (file: File) => void
  onContinue: () => void
  error: string | null
  isDragActive: boolean
  getRootProps: () => any
  getInputProps: () => any
  open: () => void
  activeTab: string
  onTabChange: (tab: string) => void
}

export default function FileSelection({
  files,
  onFilesChange,
  onPreviewFile,
  onContinue,
  error,
  isDragActive,
  getRootProps,
  getInputProps,
  open,
  activeTab,
  onTabChange
}: FileSelectionProps) {
  const removeFile = (index: number) => {
    const newFiles = [...files]
    newFiles.splice(index, 1)
    onFilesChange(newFiles)
  }

  const handleKaggleFileImported = (file: File) => {
    onFilesChange([...files, file])
    onTabChange("upload")
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Select Your Data Files</CardTitle>
        <CardDescription>
          Upload CSV or Excel files for data processing and analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={onTabChange} className="mt-2">
          <TabsList className="mb-4">
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <FilePlus2 className="h-4 w-4" />
              File Upload
            </TabsTrigger>
            <TabsTrigger value="kaggle" className="flex items-center gap-2">
              <LinkIcon className="h-4 w-4" />
              Import from Kaggle
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <input {...getInputProps()} />
            <div 
              className="cursor-pointer bg-muted/30 mt-4 h-40 rounded-lg border-2 border-dashed border-primary/20 flex items-center justify-center flex-col gap-3 hover:bg-muted/40 transition-all" 
              onClick={open}
            >
              <FilePlus2 className="h-10 w-10 text-primary/60" />
              <div className="text-center">
                <p className="text-muted-foreground font-medium mb-1">Click Here Or Drag And Drop Your Files</p>
                <p className="text-xs text-muted-foreground">Accepts CSV and Excel (XLSX) files up to 10MB</p>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="kaggle">
            <KaggleUpload onFileImported={handleKaggleFileImported} />
          </TabsContent>
        </Tabs>
        
        {files.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Selected Files ({files.length})</h3>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Format</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {files.map((file, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{file.name}</TableCell>
                    <TableCell>{(file.size / 1048576).toFixed(2)} MB</TableCell>
                    <TableCell>{file.type || file.name.split(".").pop()?.toUpperCase()}</TableCell>
                    <TableCell className="text-right">
                      <Button variant="ghost" size="icon" onClick={() => onPreviewFile(file)} title="Preview file">
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" onClick={() => removeFile(index)} title="Remove file">
                        <X className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
        
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        {files.length === 0 && !error && (
          <Alert className="mt-4 border-purple-200 bg-purple-50 text-purple-800">
            <FileText className="h-4 w-4" />
            <AlertTitle>No files selected</AlertTitle>
            <AlertDescription>
              Select one or more data files to begin the upload and preprocessing workflow.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="justify-between">
        <Button variant="outline" disabled>
          Back
        </Button>
        <Button 
          onClick={onContinue} 
          disabled={files.length === 0}
          className="gap-2"
        >
          <span>Continue to Review</span>
          <ChevronRight className="h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  )
}