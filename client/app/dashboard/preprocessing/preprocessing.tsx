'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { createClient } from '@/utils/supabase/client'
import { type FileMetadata } from '@/components/FilePreview'
import { useSearchParams } from 'next/navigation'
import FilePreview from '@/components/FilePreview'

import { 
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { 
  ChevronRight,
  ArrowLeft,
  Download,
  Eye,
  CheckCircle2,
  FileText,
  Trash2,
  Loader2
} from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"

// Define interfaces for column analysis from backend
interface ColumnAnalysis {
  name: string;
  current_type: string;
  suggested_type: string;
  sample_values: string[];
}

interface FileAnalysis {
  success: boolean;
  filename: string;
  row_count: number;
  column_count: number;
  columns_info: ColumnAnalysis[];
}

interface TransformationResult {
  success: boolean;
  message: string;
  transformed_file: string;
  report_file: string;
  report: {
    data_types: Record<string, {
      original: string;
      converted_to: string;
    }>;
    columns_dropped: string[];
    transformations_applied: string[];
  };
}

interface PreviewData {
  original: Record<string, any>[];
  transformed: Record<string, any>[];
  columns: {
    original: string[];
    transformed: string[];
  };
}

export default function CustomPreprocessing() {
  // State
  const [activeTab, setActiveTab] = useState<string>("file_selection")
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [fileAnalysis, setFileAnalysis] = useState<FileAnalysis | null>(null)
  const [columnEdits, setColumnEdits] = useState<Record<string, {
    newType?: string;
    drop?: boolean;
  }>>({})
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [previewErrorMessage, setPreviewErrorMessage] = useState<string | null>(null)
  const [isApplyingTransformations, setIsApplyingTransformations] = useState(false)
  const [transformationResult, setTransformationResult] = useState<TransformationResult | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<PreviewData | null>(null)
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  
  // Track if column edits have been changed since the last preview update
  const [previewNeedsUpdate, setPreviewNeedsUpdate] = useState(false)
  
  const searchParams = useSearchParams()
  const fileId = searchParams.get('file')
  const supabase = createClient()
  
  // Use refs for debouncing and tracking the latest edits
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const latestColumnEditsRef = useRef<Record<string, any>>({})
  
  // Fetch user's files
  useEffect(() => {
    async function fetchFiles() {
      try {
        setIsLoading(true)
        const { data: { user } } = await supabase.auth.getUser()
        
        if (user) {
          const { data, error } = await supabase
            .from('files')
            .select('*')
            .eq('user_id', user.id)
            .order('upload_date', { ascending: false })
          
          if (data) {
            setFiles(data as FileMetadata[])
            
            // If a file ID is provided in the URL, select that file
            if (fileId) {
              const selected = data.find(file => file.id === fileId)
              if (selected) {
                setSelectedFile(selected as FileMetadata)
              }
            }
          }
          
          if (error) {
            console.error('Error fetching files:', error.message)
            setErrorMessage(`Error fetching files: ${error.message}`)
          }
        }
      } catch (error) {
        console.error('Error in fetchFiles:', error)
        setErrorMessage(`Error loading files: ${error}`)
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchFiles()
  }, [fileId, supabase])
  
  // Handle file selection
  const handleSelectFile = (file: FileMetadata) => {
    setSelectedFile(file)
    // Reset analysis state
    setFileAnalysis(null)
    setColumnEdits({})
    setTransformationResult(null)
    setPreviewData(null)
    setActiveTab("file_analysis")
  }
  
  // Analyze file for custom preprocessing
  const analyzeFile = async () => {
    if (!selectedFile) return
    
    setIsAnalyzing(true)
    setErrorMessage(null)
    
    try {
      // Download the file content
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      // Fetch the file
      const response = await fetch(publicUrl)
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`)
      }
      
      const fileBlob = await response.blob()
      const fileObj = new File([fileBlob], selectedFile.original_filename, { 
        type: selectedFile.mime_type 
      })
      
      // Create form data
      const formData = new FormData()
      formData.append('file', fileObj)
      
      // Send to backend for analysis
      const analysisResponse = await fetch('http://localhost:8000/custom-preprocessing/analyze-file/', {
        method: 'POST',
        body: formData
      })
      
      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json()
        throw new Error(errorData.detail || 'Analysis failed')
      }
      
      const analysisResult = await analysisResponse.json()
      setFileAnalysis(analysisResult)
      
      // Initialize column edits with default values
      const initialEdits: Record<string, any> = {}
      analysisResult.columns_info.forEach((col: ColumnAnalysis) => {
        initialEdits[col.name] = {
          newType: col.suggested_type,
          drop: false
        }
      })
      setColumnEdits(initialEdits)
      // Update the ref with initial values
      latestColumnEditsRef.current = initialEdits
      
      setActiveTab("column_types")
    } catch (error) {
      console.error('Error analyzing file:', error)
      setErrorMessage(`Error analyzing file: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Memoize updatePreview to avoid recreating it on every render
  const updatePreview = useCallback(async () => {
    if (!selectedFile || !fileAnalysis) return
    
    // Check if we're already loading to prevent race conditions
    if (isLoadingPreview) {
      // Mark that we need another update after this one
      setPreviewNeedsUpdate(true)
      return
    }
    
    setIsLoadingPreview(true)
    setPreviewNeedsUpdate(false)
    setPreviewErrorMessage(null)
    
    try {
      // Download the file content
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      // Fetch the file
      const response = await fetch(publicUrl)
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`)
      }
      
      const fileBlob = await response.blob()
      const fileObj = new File([fileBlob], selectedFile.original_filename, { 
        type: selectedFile.mime_type 
      })
      
      // Prepare transformation config using the latest column edits from ref
      const transformationConfig = {
        data_types: {} as Record<string, string>,
        columns_to_drop: [] as string[]
      }
      
      // Use the current state instead of ref to ensure we have the latest data
      const currentEdits = latestColumnEditsRef.current
      
      // Add data type transformations and collect columns to drop
      Object.entries(currentEdits).forEach(([colName, edit]) => {
        if (edit.newType) {
          transformationConfig.data_types[colName] = edit.newType
        }
        
        if (edit.drop) {
          transformationConfig.columns_to_drop.push(colName)
        }
      })
      
      // Create form data
      const formData = new FormData()
      formData.append('file', fileObj)
      formData.append('transformations', JSON.stringify(transformationConfig))
      
      // Send to backend for preview
      const previewResponse = await fetch('http://localhost:8000/custom-preprocessing/preview-transformation/', {
        method: 'POST',
        body: formData
      })
      
      if (!previewResponse.ok) {
        const errorData = await previewResponse.json()
        throw new Error(errorData.detail || 'Preview failed')
      }
      
      const result = await previewResponse.json()
      if (result.success && result.preview) {
        setPreviewData(result.preview)
      } else {
        setPreviewErrorMessage("Couldn't generate preview with the current settings")
      }
    } catch (error) {
      console.error('Error generating preview:', error)
      setPreviewErrorMessage(`Preview error: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsLoadingPreview(false)
      
      // If another update was requested while this one was running, run it now
      if (previewNeedsUpdate) {
        setTimeout(() => {
          updatePreview()
        }, 100)
      }
    }
  }, [selectedFile, fileAnalysis, isLoadingPreview])
  
  // Update column edit with immediate preview update
  const updateColumnEdit = (columnName: string, field: string, value: any) => {
    // Synchronously update state
    setColumnEdits(prev => {
      const newEdits = {
        ...prev,
        [columnName]: {
          ...prev[columnName],
          [field]: value
        }
      }
      
      // Update the ref immediately so updatePreview has access to latest values
      latestColumnEditsRef.current = newEdits
      
      return newEdits
    })
    
    // Clear any existing timeout
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current)
    }
    
    // For drop column operations, update immediately
    if (field === 'drop') {
      // Update immediately for drop operations
      setTimeout(() => {
        updatePreview()
      }, 10)
    } else {
      // For other operations, debounce to avoid too many API calls
      previewTimeoutRef.current = setTimeout(() => {
        updatePreview()
      }, 300)
    }
  }
  
  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current)
      }
    }
  }, [])
  
  // Initialize preview when opening the column types tab
  useEffect(() => {
    if (activeTab === "column_types" && fileAnalysis && (!previewData || Object.keys(previewData).length === 0) && !isLoadingPreview) {
      updatePreview()
    }
  }, [activeTab, fileAnalysis, previewData, isLoadingPreview, updatePreview])
  
  // Apply transformations
  const applyTransformations = async () => {
    if (!selectedFile || !fileAnalysis) return
    
    setIsApplyingTransformations(true)
    setErrorMessage(null)
    
    try {
      // Download the file content
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      // Fetch the file
      const response = await fetch(publicUrl)
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`)
      }
      
      const fileBlob = await response.blob()
      const fileObj = new File([fileBlob], selectedFile.original_filename, { 
        type: selectedFile.mime_type 
      })
      
      // Prepare transformation config
      const transformationConfig = {
        data_types: {} as Record<string, string>,
        columns_to_drop: [] as string[]
      }
      
      // Add data type transformations and collect columns to drop
      Object.entries(columnEdits).forEach(([colName, edit]) => {
        if (edit.newType) {
          transformationConfig.data_types[colName] = edit.newType
        }
        
        if (edit.drop) {
          transformationConfig.columns_to_drop.push(colName)
        }
      })
      
      // Create form data
      const formData = new FormData()
      formData.append('file', fileObj)
      formData.append('transformations', JSON.stringify(transformationConfig))
      
      // Send to backend for transformations
      const transformResponse = await fetch('http://localhost:8000/custom-preprocessing/apply-transformations/', {
        method: 'POST',
        body: formData
      })
      
      if (!transformResponse.ok) {
        const errorData = await transformResponse.json()
        throw new Error(errorData.detail || 'Transformation failed')
      }
      
      const result = await transformResponse.json()
      setTransformationResult(result)
      
      // Create a download URL
      setDownloadUrl(`http://localhost:8000/preprocessing_results/${result.transformed_file}`)
      
      setActiveTab("results")
    } catch (error) {
      console.error('Error applying transformations:', error)
      setErrorMessage(`Error applying transformations: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsApplyingTransformations(false)
    }
  }
  
  // Handle file preview
  const handlePreview = () => {
    if (selectedFile) {
      setIsPreviewOpen(true)
    }
  }
  
  // Format data type for display
  const formatDataType = (type: string) => {
    switch (type) {
      case 'int':
      case 'int64':
      case 'int32':
        return 'Integer'
      case 'float':
      case 'float64':
      case 'float32':
        return 'Decimal'
      case 'object':
      case 'string':
        return 'Text'
      case 'datetime64[ns]':
      case 'datetime':
        return 'Date & Time'
      default:
        return type
    }
  }
  
  const handleDownload = () => {
    if (downloadUrl) {
      window.open(downloadUrl, '_blank')
    }
  }

  // Check if any columns are marked as dropped
  const hasDroppedColumns = () => {
    return Object.values(columnEdits).some(edit => edit.drop)
  }

  // Check if any column types have been changed
  const hasChangedTypes = () => {
    if (!fileAnalysis) return false
    
    return fileAnalysis.columns_info.some(col => {
      const edit = columnEdits[col.name]
      return edit && edit.newType && edit.newType !== col.current_type
    })
  }
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-8">
        CSV Data Transformation Tool
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full max-w-md mb-6">
          <TabsTrigger value="file_selection" className="flex-1">Select File</TabsTrigger>
          <TabsTrigger value="file_analysis" className="flex-1" disabled={!selectedFile}>
            File Analysis
          </TabsTrigger>
          <TabsTrigger value="column_types" className="flex-1" disabled={!fileAnalysis}>
            Data Transformation
          </TabsTrigger>
          <TabsTrigger value="results" className="flex-1" disabled={!transformationResult}>
            Results
          </TabsTrigger>
        </TabsList>
        
        {/* File Selection Tab */}
        <TabsContent value="file_selection" className="border-none p-0">
          {isLoading ? (
            <div className="flex justify-center items-center h-48">
              <p>Loading files...</p>
            </div>
          ) : files.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 gap-4">
              <p className="text-gray-500">No files found. Upload files to get started.</p>
              <Button variant="outline" asChild>
                <a href="/dashboard/data-upload">Upload Files</a>
              </Button>
            </div>
          ) : (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Select a CSV File</CardTitle>
                  <CardDescription>
                    Choose a file to transform
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableCaption>Your available files</TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Filename</TableHead>
                        <TableHead>Size</TableHead>
                        <TableHead>Rows</TableHead>
                        <TableHead>Columns</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {files.map((file) => (
                        <TableRow key={file.id} className={selectedFile?.id === file.id ? "bg-muted/50" : ""}>
                          <TableCell className="font-medium">{file.original_filename}</TableCell>
                          <TableCell>{(file.file_size / 1048576).toFixed(2)} MB</TableCell>
                          <TableCell>{file.row_count}</TableCell>
                          <TableCell>{file.column_names.length}</TableCell>
                          <TableCell className="text-right">
                            <Button
                              variant="outline"
                              className="ml-2"
                              onClick={() => handleSelectFile(file)}
                            >
                              Select
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
        
        {/* File Analysis Tab */}
        <TabsContent value="file_analysis" className="border-none p-0">
          {selectedFile && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Preview of Data</CardTitle>
                      <CardDescription>
                        Preview {selectedFile.original_filename} before transformation
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("file_selection")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-3 gap-4 mb-6">
                    <div className="flex flex-col border rounded-md p-4">
                      <span className="text-sm font-medium text-muted-foreground">File Size</span>
                      <span className="font-bold text-xl">{(selectedFile.file_size / 1048576).toFixed(2)} MB</span>
                    </div>
                    <div className="flex flex-col border rounded-md p-4">
                      <span className="text-sm font-medium text-muted-foreground">Rows</span>
                      <span className="font-bold text-xl">{selectedFile.row_count}</span>
                    </div>
                    <div className="flex flex-col border rounded-md p-4">
                      <span className="text-sm font-medium text-muted-foreground">Columns</span>
                      <span className="font-bold text-xl">{selectedFile.column_names.length}</span>
                    </div>
                  </div>
                  
                  {fileAnalysis ? (
                    <div>
                      <h3 className="text-lg font-medium mb-2">File Analysis Complete</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        The file has been analyzed. You can now proceed to the next step to customize data types and transformations.
                      </p>
                      <Button onClick={() => setActiveTab("column_types")}>
                        <ChevronRight className="mr-2 h-4 w-4" />
                        Continue to Data Transformation
                      </Button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-4 py-8">
                      {isAnalyzing ? (
                        <>
                          <Progress value={50} className="w-1/2 mb-2" />
                          <p>Analyzing file, please wait...</p>
                        </>
                      ) : (
                        <>
                          <p className="text-center text-gray-500 mb-4">
                            Click the button below to analyze the file and proceed with transformation.
                          </p>
                          <Button onClick={analyzeFile} disabled={isAnalyzing}>
                            {isAnalyzing ? "Analyzing..." : "Analyze File"}
                          </Button>
                        </>
                      )}
                    </div>
                  )}
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={() => handlePreview()}>
                    <Eye className="mr-2 h-4 w-4" />
                    Preview Data
                  </Button>
                </CardFooter>
              </Card>
              
              {errorMessage && (
                <Alert variant="destructive">
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{errorMessage}</AlertDescription>
                </Alert>
              )}
            </div>
          )}
        </TabsContent>
        
        {/* Column Types and Transformations Tab */}
        <TabsContent value="column_types" className="border-none p-0">
          {fileAnalysis && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Data Transformation</CardTitle>
                      <CardDescription>
                        Select data types and choose columns to drop
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("file_analysis")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="mb-8">
                    <h3 className="text-lg font-medium mb-2">Data Type Selection</h3>
                    <div className="rounded-md border">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Column Name</TableHead>
                            <TableHead>Current Type</TableHead>
                            <TableHead>New Type</TableHead>
                            <TableHead>Drop Column</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {fileAnalysis.columns_info.map((column) => (
                          <TableRow 
                            key={column.name}
                            className={`group ${columnEdits[column.name]?.drop ? "bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50" : "hover:bg-muted"}`}
                          >
                            <TableCell className={`font-medium ${columnEdits[column.name]?.drop ? "text-red-600 dark:text-red-400 line-through" : ""}`}>
                              {column.name}
                            </TableCell>
                            <TableCell className={columnEdits[column.name]?.drop ? "text-red-600 dark:text-red-400" : ""}>
                              {formatDataType(column.current_type)}
                            </TableCell>
                            <TableCell>
                              <Select
                                value={columnEdits[column.name]?.newType || column.suggested_type}
                                onValueChange={(value) => updateColumnEdit(column.name, 'newType', value)}
                                disabled={columnEdits[column.name]?.drop}
                              >
                                <SelectTrigger className={`w-full ${
                                  !columnEdits[column.name]?.drop && 
                                  columnEdits[column.name]?.newType !== column.current_type && 
                                  columnEdits[column.name]?.newType !== column.suggested_type 
                                    ? 'border-amber-300 bg-amber-50 hover:bg-amber-100 dark:bg-amber-900/30 dark:border-amber-700' : ''
                                }`}>
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="string">Text (String)</SelectItem>
                                  <SelectItem value="int">Integer</SelectItem>
                                  <SelectItem value="float">Decimal (Float)</SelectItem>
                                  <SelectItem value="datetime">Date & Time</SelectItem>
                                </SelectContent>
                              </Select>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center space-x-2">
                                <Checkbox 
                                  checked={!!columnEdits[column.name]?.drop}
                                  onCheckedChange={(checked) => updateColumnEdit(column.name, 'drop', checked)}
                                  id={`drop-${column.name}`}
                                  className="data-[state=checked]:bg-red-600 data-[state=checked]:border-red-600 focus:ring-red-200"
                                />
                                <label 
                                  htmlFor={`drop-${column.name}`}
                                  className={`text-sm cursor-pointer select-none ${columnEdits[column.name]?.drop ? "text-red-600 dark:text-red-400 font-medium" : ""}`}
                                >
                                  Drop
                                </label>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                        
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                  
                  {/* Live Preview */}
                  <div className="mt-8">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-lg font-medium">Live Preview</h3>
                      {isLoadingPreview && (
                        <div className="flex items-center text-sm text-amber-600">
                          <Loader2 className="animate-spin mr-2 h-4 w-4 text-amber-600" />
                          Updating preview...
                        </div>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      This shows how your data will look after transformation (first 5 rows).
                      <span className="italic"> Updates automatically as you make changes.</span>
                    </p>
                    
                    <div className={`border rounded-md overflow-x-auto transition-opacity duration-200 ${isLoadingPreview ? 'opacity-50' : ''}`}>
                      {previewData ? (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              {previewData.columns.transformed.map((col, idx) => (
                                <TableHead key={idx}>{col}</TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {previewData.transformed.map((row, rowIdx) => (
                              <TableRow key={rowIdx}>
                                {previewData.columns.transformed.map((col, colIdx) => (
                                  <TableCell key={colIdx}>
                                    {row[col] !== null && row[col] !== undefined ? String(row[col]) : "null"}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      ) : (
                        <div className="p-6 text-center text-muted-foreground">
                          {isLoadingPreview ? 
                            "Loading preview data..." : 
                            "No preview available. Make changes to see a live preview."}
                        </div>
                      )}
                    </div>
                    
                    {previewErrorMessage && (
                      <Alert variant="destructive" className="mt-4">
                        <AlertTitle>Preview Error</AlertTitle>
                        <AlertDescription>{previewErrorMessage}</AlertDescription>
                      </Alert>
                    )}
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={() => setActiveTab("file_analysis")}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                  <Button 
                    onClick={applyTransformations} 
                    disabled={isApplyingTransformations}
                    variant={(hasDroppedColumns() || hasChangedTypes()) ? "default" : "outline"}
                    className={(hasDroppedColumns() || hasChangedTypes()) ? "bg-green-600 hover:bg-green-700" : ""}
                  >
                    {isApplyingTransformations ? (
                      <>
                        <Loader2 className="animate-spin mr-2 h-4 w-4" />
                        Processing...
                      </>
                    ) : hasDroppedColumns() || hasChangedTypes() ? (
                      "Apply Transformations"
                    ) : (
                      "No Changes to Apply"
                    )}
                  </Button>
                </CardFooter>
              </Card>
              
              {errorMessage && (
                <Alert variant="destructive">
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{errorMessage}</AlertDescription>
                </Alert>
              )}
            </div>
          )}
        </TabsContent>
        
        {/* Results Tab */}
        <TabsContent value="results" className="border-none p-0">
          {transformationResult && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Transformed Data</CardTitle>
                      <CardDescription>
                        Your data has been transformed successfully
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("column_types")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center p-6 bg-green-50 rounded-md mb-6">
                    <CheckCircle2 className="h-8 w-8 text-green-500 mr-3" />
                    <div>
                      <h3 className="font-medium">Transformation Completed Successfully</h3>
                      <p className="text-sm text-gray-500">
                        Your transformations have been applied to the data.
                      </p>
                    </div>
                  </div>
                  
                  <div className="mt-6">
                    <h3 className="text-lg font-medium mb-2">Applied Transformations</h3>
                    {transformationResult.report.transformations_applied.length > 0 ? (
                      <ul className="list-disc pl-6 space-y-1">
                        {transformationResult.report.transformations_applied.map((transformation, index) => (
                          <li key={index}>{transformation}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-muted-foreground">No transformations were applied</p>
                    )}
                  </div>
                  
                  {transformationResult.report.columns_dropped && transformationResult.report.columns_dropped.length > 0 && (
                    <div className="mt-6">
                      <h3 className="text-lg font-medium mb-2">Dropped Columns</h3>
                      <div className="flex flex-wrap gap-2">
                        {transformationResult.report.columns_dropped.map((column, idx) => (
                          <div key={idx} className="px-3 py-1 bg-red-50 text-red-700 rounded-md flex items-center gap-1">
                            <Trash2 className="h-3 w-3" />
                            {column}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
                <CardFooter className="flex justify-end gap-3">
                  <Button 
                    onClick={handleDownload}
                    disabled={!downloadUrl}
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Download Transformed CSV
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
      
      {selectedFile && (
        <FilePreview 
          fileMetadata={selectedFile} 
          isOpen={isPreviewOpen} 
          onClose={() => setIsPreviewOpen(false)} 
        />
      )}
    </section>
  )
}