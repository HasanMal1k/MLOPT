// components/upload/FileSelection.tsx - Corrected with proper imports
'use client'

import { useState } from "react"
import { FilePlus2, Upload, Database, Cloud, ExternalLink, Eye, X, BarChart } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
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
  onKaggleFileImported: (file: File) => void
  kaggleImportCount: number
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
  onTabChange,
  onKaggleFileImported,
  kaggleImportCount
}: FileSelectionProps) {
  const [localKaggleCount, setLocalKaggleCount] = useState(0)

  const handleKaggleFileImported = (file: File) => {
    // Add file to the main files array
    onFilesChange([...files, file])
    
    // Track Kaggle imports locally
    setLocalKaggleCount(prev => prev + 1)
    
    // Call parent handler
    onKaggleFileImported(file)
    
    // Switch to local files tab to show the imported file
    onTabChange("upload")
  }

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index)
    onFilesChange(newFiles)
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getFileIcon = (file: File) => {
    const extension = file.name.split('.').pop()?.toLowerCase()
    switch (extension) {
      case 'csv':
        return <Database className="h-4 w-4 text-green-600" />
      case 'xlsx':
      case 'xls':
        return <BarChart className="h-4 w-4 text-blue-600" />
      default:
        return <Database className="h-4 w-4 text-gray-600" />
    }
  }

  const displayKaggleCount = kaggleImportCount || localKaggleCount

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Upload Your Data Files</h2>
        <p className="text-muted-foreground">
          Choose files from your computer, import from Kaggle, or connect cloud storage
        </p>
      </div>

      {/* Upload Methods Tabs */}
      <Tabs value={activeTab} onValueChange={onTabChange} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="upload" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Local Files
            {files.length > 0 && (
              <Badge variant="secondary" className="ml-1">
                {files.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="kaggle" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            Kaggle Import
            {displayKaggleCount > 0 && (
              <Badge variant="secondary" className="ml-1 bg-green-100 text-green-700">
                {displayKaggleCount}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="cloud" className="flex items-center gap-2" disabled>
            <Cloud className="h-4 w-4" />
            Cloud Storage
            <Badge variant="outline" className="ml-1 text-xs">
              Soon
            </Badge>
          </TabsTrigger>
        </TabsList>

        {/* Local File Upload Tab */}
        <TabsContent value="upload" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Local Files
              </CardTitle>
              <CardDescription>
                Upload CSV or Excel files from your computer for preprocessing
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`
                  border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
                  ${isDragActive 
                    ? 'border-primary bg-primary/5' 
                    : 'border-gray-300 hover:border-gray-400'
                  }
                `}
              >
                <input {...getInputProps()} />
                <div className="space-y-4">
                  <FilePlus2 className="mx-auto h-12 w-12 text-gray-400" />
                  <div>
                    <p className="text-lg font-medium">
                      {isDragActive ? "Drop files here..." : "Drag & drop files here"}
                    </p>
                    <p className="text-sm text-gray-500 mt-1">
                      or{" "}
                      <button 
                        onClick={open} 
                        className="text-primary font-medium hover:underline"
                      >
                        click to browse
                      </button>
                    </p>
                  </div>
                  <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
                    <span>• Supports CSV, Excel (.xlsx)</span>
                    <span>• Max 10MB per file</span>
                    <span>• Multiple files allowed</span>
                  </div>
                </div>
              </div>

              {/* Error Display */}
              {error && (
                <Alert variant="destructive">
                  <AlertTitle>Upload Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {/* Selected Files List */}
              {files.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">Selected Files ({files.length})</h4>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onFilesChange([])}
                    >
                      Clear All
                    </Button>
                  </div>
                  
                  <div className="border rounded-lg divide-y">
                    {files.map((file, index) => (
                      <div key={index} className="p-4 flex items-center justify-between hover:bg-gray-50/50">
                        <div className="flex items-center gap-3">
                          <div className="h-10 w-10 rounded-lg bg-blue-100 flex items-center justify-center">
                            {getFileIcon(file)}
                          </div>
                          <div>
                            <p className="font-medium text-sm">{file.name}</p>
                            <div className="flex items-center gap-3 text-xs text-muted-foreground">
                              <span>{formatFileSize(file.size)}</span>
                              <span>•</span>
                              <span>{file.type || 'Unknown type'}</span>
                              <span>•</span>
                              <span>{file.lastModified ? new Date(file.lastModified).toLocaleDateString() : 'Unknown date'}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => onPreviewFile(file)}
                            className="text-blue-600 hover:text-blue-700"
                          >
                            <Eye className="h-4 w-4 mr-1" />
                            Preview
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeFile(index)}
                            className="text-red-600 hover:text-red-700"
                          >
                            <X className="h-4 w-4 mr-1" />
                            Remove
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {/* File Summary */}
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-blue-800 font-medium">
                        Ready to process {files.length} file{files.length === 1 ? '' : 's'}
                      </span>
                      <span className="text-blue-600">
                        Total size: {formatFileSize(files.reduce((total, file) => total + file.size, 0))}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Quick Upload Button for convenience */}
              {files.length === 0 && (
                <div className="flex justify-center pt-2">
                  <Button variant="outline" onClick={open}>
                    <Upload className="h-4 w-4 mr-2" />
                    Browse Files
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Kaggle Import Tab */}
        <TabsContent value="kaggle" className="space-y-4">
          <KaggleUpload onFileImported={handleKaggleFileImported} />
          
          {/* Show imported files count */}
          {displayKaggleCount > 0 && (
            <Alert className="bg-green-50 border-green-200">
              <Database className="h-4 w-4 text-green-600" />
              <AlertTitle className="text-green-800">
                Kaggle Import Summary
              </AlertTitle>
              <AlertDescription className="text-green-700">
                Successfully imported {displayKaggleCount} file{displayKaggleCount === 1 ? '' : 's'} from Kaggle. 
                Switch to the "Local Files" tab to see all your selected files.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>

        {/* Cloud Storage Tab (Placeholder) */}
        <TabsContent value="cloud" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="h-5 w-5" />
                Cloud Storage Import
              </CardTitle>
              <CardDescription>
                Import files from Google Drive, Dropbox, OneDrive, and other cloud services
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12">
                <div className="flex justify-center mb-6">
                  <div className="relative">
                    <Cloud className="h-16 w-16 text-gray-300" />
                    <div className="absolute -top-1 -right-1 h-6 w-6 bg-orange-100 rounded-full flex items-center justify-center">
                      <span className="text-xs font-medium text-orange-600">Soon</span>
                    </div>
                  </div>
                </div>
                <h3 className="text-lg font-medium text-gray-700 mb-2">
                  Cloud Storage Integration Coming Soon
                </h3>
                <p className="text-sm text-gray-500 max-w-md mx-auto mb-6">
                  We're building integrations with popular cloud storage services. 
                  You'll be able to import files directly from:
                </p>
                <div className="grid grid-cols-2 gap-3 max-w-xs mx-auto mb-6">
                  <div className="p-2 border rounded text-sm text-gray-600">Google Drive</div>
                  <div className="p-2 border rounded text-sm text-gray-600">Dropbox</div>
                  <div className="p-2 border rounded text-sm text-gray-600">OneDrive</div>
                  <div className="p-2 border rounded text-sm text-gray-600">AWS S3</div>
                </div>
                <div className="flex gap-3 justify-center">
                  <Button variant="outline" size="sm" asChild>
                    <a href="https://docs.example.com/cloud-storage" target="_blank" rel="noopener noreferrer">
                      <ExternalLink className="h-4 w-4 mr-2" />
                      Learn More
                    </a>
                  </Button>
                  <Button variant="outline" size="sm" disabled>
                    Request Early Access
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Continue Button */}
      <div className="flex justify-between items-center pt-6 border-t">
        <div className="text-sm text-muted-foreground">
          {files.length === 0 ? (
            <span>Select files from any source to continue</span>
          ) : (
            <div className="flex items-center gap-4">
              <span>{files.length} file{files.length === 1 ? '' : 's'} selected</span>
              {displayKaggleCount > 0 && (
                <Badge variant="secondary" className="bg-green-100 text-green-700">
                  {displayKaggleCount} from Kaggle
                </Badge>
              )}
            </div>
          )}
        </div>
        <Button 
          onClick={onContinue} 
          disabled={files.length === 0}
          size="lg"
          className="min-w-[160px]"
        >
          Continue to Review
          {files.length > 0 && (
            <Badge variant="secondary" className="ml-2 bg-white/20">
              {files.length}
            </Badge>
          )}
        </Button>
      </div>
    </div>
  )
}