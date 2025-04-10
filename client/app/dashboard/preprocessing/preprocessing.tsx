'use client'

import { useState, useEffect, Suspense } from 'react'
import { createClient } from '@/utils/supabase/client'
import { type FileMetadata } from '@/components/FilePreview'
import { useSearchParams } from 'next/navigation'
import FilePreview from '@/components/FilePreview'
import PreprocessingReport from '@/components/PreprocessingReport'
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
  Calendar,
  SlidersHorizontal,
  Check,
  X,
  ArrowUpDown,
  Filter,
  FileText,
  Trash2
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { 
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

// Define interfaces for column analysis from backend
interface ColumnAnalysis {
  name: string;
  current_type: string;
  suggested_type: string;
  possible_types: string[];
  sample_values: string[];
  missing_percentage: number;
  unique_count: number;
  unique_percentage: number;
  value_distribution?: Record<string, number>;
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
    column_changes: Record<string, any>;
    row_count: {
      before: number;
      after: number;
    };
    data_types: Record<string, {
      original: string;
      converted_to: string;
    }>;
    missing_values: Record<string, {
      before: number;
      after: number;
      method: string;
    }>;
    transformations_applied: string[];
  };
}

interface ProcessedResult {
  success: boolean;
  operations_applied: string[];
  message: string;
  result_file: string;
  statistics?: {
    rows_affected: number;
    outliers_removed?: number;
    features_encoded?: string[];
  };
}

const operationNames: Record<string, string> = {
  missing_values: "Handle Missing Values",
  outliers: "Detect & Remove Outliers",
  feature_encoding: "Categorical Encoding",
  feature_scaling: "Feature Scaling",
  date_features: "Extract Date Features"
};

export default function CustomPreprocessing() {
  // State
  const [activeTab, setActiveTab] = useState<string>("file_selection")
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [processedResult, setProcessedResult] = useState<ProcessedResult | null>(null);
  const [isLoading, setIsLoading] = useState(true)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [fileAnalysis, setFileAnalysis] = useState<FileAnalysis | null>(null)
  const [columnEdits, setColumnEdits] = useState<Record<string, {
    newType?: string;
    missingValueAction?: string;
    missingValueFillValue?: string;
    drop?: boolean;
  }>>({})
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isApplyingTransformations, setIsApplyingTransformations] = useState(false)
  const [transformationResult, setTransformationResult] = useState<TransformationResult | null>(null)
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  
  const searchParams = useSearchParams()
  const fileId = searchParams.get('file')
  const supabase = createClient()
  
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
              const selectedFile = data.find(file => file.id === fileId)
              if (selectedFile) {
                setSelectedFile(selectedFile as FileMetadata)
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
  }, [fileId])
  
  // Handle file selection
  const handleSelectFile = (file: FileMetadata) => {
    setSelectedFile(file)
    // Reset analysis state
    setFileAnalysis(null)
    setColumnEdits({})
    setTransformationResult(null)
    setSelectedColumns([])
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
          missingValueAction: col.missing_percentage > 0 ? "none" : undefined,
          drop: false
        }
      })
      setColumnEdits(initialEdits)
      
      // Select all columns by default
      setSelectedColumns(analysisResult.columns_info.map((col: ColumnAnalysis) => col.name))
      
      setActiveTab("column_types")
    } catch (error) {
      console.error('Error analyzing file:', error)
      setErrorMessage(`Error analyzing file: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Toggle column selection
  const toggleColumnSelection = (columnName: string) => {
    setSelectedColumns(prev => 
      prev.includes(columnName)
        ? prev.filter(name => name !== columnName)
        : [...prev, columnName]
    )
  }
  
  // Update column edit
  const updateColumnEdit = (columnName: string, field: string, value: any) => {
    setColumnEdits(prev => ({
      ...prev,
      [columnName]: {
        ...prev[columnName],
        [field]: value
      }
    }))
  }
  
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
        missing_values: {} as Record<string, any>,
        column_operations: [] as any[]
      }
      
      // Only process selected columns
      selectedColumns.forEach(colName => {
        const edit = columnEdits[colName]
        
        // Data type transformation
        if (edit.newType) {
          transformationConfig.data_types[colName] = edit.newType
        }
        
        // Missing value handling
        if (edit.missingValueAction && edit.missingValueAction !== "none") {
          transformationConfig.missing_values[colName] = {
            method: edit.missingValueAction,
            value: edit.missingValueFillValue || ""
          }
        }
        
        // Column drop
        if (edit.drop) {
          transformationConfig.column_operations.push({
            type: "drop",
            column: colName
          })
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
  
  // Format percentage with 2 decimal places
  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`
  }
  
  // Format data type for display
  const formatDataType = (type: string) => {
    switch (type) {
      case 'int64':
      case 'int32':
      case 'integer':
        return 'Integer'
      case 'float64':
      case 'float32':
      case 'float':
        return 'Decimal'
      case 'object':
      case 'string':
        return 'Text'
      case 'datetime64[ns]':
      case 'datetime':
        return 'Date & Time'
      case 'category':
        return 'Category'
      default:
        return type
    }
  }
  
  // Get icon for data type
  const getDataTypeIcon = (type: string) => {
    switch (type) {
      case 'int64':
      case 'int32':
      case 'integer':
      case 'float64':
      case 'float32':
      case 'float':
        return <ArrowUpDown className="h-4 w-4" />
      case 'object':
      case 'string':
        return <FileText className="h-4 w-4" />
      case 'datetime64[ns]':
      case 'datetime':
        return <Calendar className="h-4 w-4" />
      case 'category':
        return <Filter className="h-4 w-4" />
      default:
        return <SlidersHorizontal className="h-4 w-4" />
    }
  }
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-8">
        Custom Preprocessing
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full max-w-md mb-6">
          <TabsTrigger value="file_selection" className="flex-1">Select File</TabsTrigger>
          <TabsTrigger value="file_analysis" className="flex-1" disabled={!selectedFile}>
            File Analysis
          </TabsTrigger>
          <TabsTrigger value="column_types" className="flex-1" disabled={!fileAnalysis}>
            Column Types
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
                  <CardTitle>Select a File for Custom Preprocessing</CardTitle>
                  <CardDescription>
                    Choose a file to analyze and customize preprocessing steps for your data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableCaption>Your available files for preprocessing</TableCaption>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Filename</TableHead>
                        <TableHead>Size</TableHead>
                        <TableHead>Rows</TableHead>
                        <TableHead>Columns</TableHead>
                        <TableHead>Status</TableHead>
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
                          <TableCell>
                            {file.preprocessing_info?.is_preprocessed ? (
                              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                                Auto Preprocessed
                              </Badge>
                            ) : (
                              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                                Raw
                              </Badge>
                            )}
                          </TableCell>
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
                      <CardTitle>File Analysis</CardTitle>
                      <CardDescription>
                        Analyze {selectedFile.original_filename} for custom preprocessing
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
                      <h3 className="text-lg font-medium mb-2">Column Analysis</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        The file has been analyzed. You can now proceed to the next step to customize column types and preprocessing steps.
                      </p>
                      <Button onClick={() => setActiveTab("column_types")}>
                        <ChevronRight className="mr-2 h-4 w-4" />
                        Configure Column Types
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
                            Click the button below to analyze the file and get column information for custom preprocessing.
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
        
        {/* Column Types Tab */}
        <TabsContent value="column_types" className="border-none p-0">
          {fileAnalysis && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Configure Column Types</CardTitle>
                      <CardDescription>
                        Customize data types and preprocessing for each column
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("file_analysis")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex justify-end mb-4">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => setSelectedColumns(fileAnalysis.columns_info.map(col => col.name))}
                      className="mr-2"
                    >
                      Select All
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => setSelectedColumns([])}
                    >
                      Deselect All
                    </Button>
                  </div>
                  
                  <div className="border rounded-md">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[50px]">Select</TableHead>
                          <TableHead>Column Name</TableHead>
                          <TableHead>Original Type</TableHead>
                          <TableHead>New Type</TableHead>
                          <TableHead>Missing Values</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {fileAnalysis.columns_info.map((column) => (
                          <TableRow key={column.name}>
                            <TableCell>
                              <Checkbox 
                                checked={selectedColumns.includes(column.name)}
                                onCheckedChange={() => toggleColumnSelection(column.name)}
                              />
                            </TableCell>
                            <TableCell className="font-medium">
                              <div className="flex flex-col">
                                <span>{column.name}</span>
                                <span className="text-xs text-muted-foreground">
                                  {column.unique_count} unique values ({formatPercentage(column.unique_percentage)})
                                </span>
                              </div>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center gap-1">
                                {getDataTypeIcon(column.current_type)}
                                <span>{formatDataType(column.current_type)}</span>
                              </div>
                            </TableCell>
                            <TableCell>
                              <Select
                                value={columnEdits[column.name]?.newType || column.suggested_type}
                                onValueChange={(value) => updateColumnEdit(column.name, 'newType', value)}
                                disabled={!selectedColumns.includes(column.name)}
                              >
                                <SelectTrigger className="w-full">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="string">Text (String)</SelectItem>
                                  <SelectItem value="integer">Integer</SelectItem>
                                  <SelectItem value="float">Decimal (Float)</SelectItem>
                                  <SelectItem value="datetime">Date & Time</SelectItem>
                                  <SelectItem value="category">Category</SelectItem>
                                </SelectContent>
                              </Select>
                            </TableCell>
                            <TableCell>
                              {column.missing_percentage > 0 ? (
                                <div className="flex flex-col gap-2">
                                  <div className="flex items-center gap-1">
                                    <span className="text-sm text-muted-foreground">
                                      {formatPercentage(column.missing_percentage)} missing
                                    </span>
                                  </div>
                                  <Select
                                    value={columnEdits[column.name]?.missingValueAction || "none"}
                                    onValueChange={(value) => updateColumnEdit(column.name, 'missingValueAction', value)}
                                    disabled={!selectedColumns.includes(column.name)}
                                  >
                                    <SelectTrigger className="w-full h-8 text-xs">
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="none">No Action</SelectItem>
                                      <SelectItem value="drop">Drop Rows</SelectItem>
                                      <SelectItem value="fill_value">Fill with Value</SelectItem>
                                      <SelectItem value="mean">Fill with Mean</SelectItem>
                                      <SelectItem value="median">Fill with Median</SelectItem>
                                    </SelectContent>
                                  </Select>
                                  
                                  {columnEdits[column.name]?.missingValueAction === "fill_value" && (
                                    <Input
                                      className="h-8 text-xs"
                                      placeholder="Fill value"
                                      value={columnEdits[column.name]?.missingValueFillValue || ""}
                                      onChange={(e) => updateColumnEdit(column.name, 'missingValueFillValue', e.target.value)}
                                      disabled={!selectedColumns.includes(column.name)}
                                    />
                                  )}
                                </div>
                              ) : (
                                <span className="text-xs text-green-600">No missing values</span>
                              )}
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center">
                                <TooltipProvider>
                                  <Tooltip>
                                    <TooltipTrigger asChild>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => updateColumnEdit(column.name, 'drop', !columnEdits[column.name]?.drop)}
                                        disabled={!selectedColumns.includes(column.name)}
                                        className={columnEdits[column.name]?.drop ? "text-red-500" : ""}
                                      >
                                        {columnEdits[column.name]?.drop ? <X className="h-4 w-4" /> : <Trash2 className="h-4 w-4" />}
                                      </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                      {columnEdits[column.name]?.drop ? "Cancel Drop" : "Drop Column"}
                                    </TooltipContent>
                                  </Tooltip>
                                </TooltipProvider>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={() => setActiveTab("file_analysis")}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                  <Button 
                    onClick={applyTransformations} 
                    disabled={selectedColumns.length === 0 || isApplyingTransformations}
                  >
                    {isApplyingTransformations ? "Processing..." : "Apply Transformations"}
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
          {selectedFile && processedResult && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Processing Results</CardTitle>
                      <CardDescription>
                        Results for {selectedFile.original_filename}
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("operations")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center p-6 bg-green-50 rounded-md mb-6">
                    <CheckCircle2 className="h-8 w-8 text-green-500 mr-3" />
                    <div>
                      <h3 className="font-medium">Processing Completed Successfully</h3>
                      <p className="text-sm text-gray-500">
                        Your custom preprocessing operations have been applied.
                      </p>
                    </div>
                  </div>

                  <div className="grid gap-6">
                    <div>
                      <h3 className="text-lg font-medium mb-3">Operations Applied</h3>
                      <div className="flex flex-wrap gap-2">
                        {processedResult.operations_applied.map(op => (
                          <Badge key={op} className="bg-blue-50 text-blue-700 border-blue-200">
                            {operationNames[op] || op}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    
                    {processedResult.statistics && (
                      <div>
                        <h3 className="text-lg font-medium mb-3">Statistics</h3>
                        <table className="w-full text-sm">
                          <tbody>
                            <tr className="border-b">
                              <td className="py-2 font-medium">Rows Affected</td>
                              <td className="py-2 text-right">{processedResult.statistics.rows_affected}</td>
                            </tr>
                            
                            {processedResult.statistics.outliers_removed !== undefined && processedResult.statistics.outliers_removed > 0 && (
                              <tr className="border-b">
                                <td className="py-2 font-medium">Outliers Removed</td>
                                <td className="py-2 text-right">{processedResult.statistics.outliers_removed}</td>
                              </tr>
                            )}
                            
                            {processedResult.statistics.features_encoded && processedResult.statistics.features_encoded.length > 0 && (
                              <tr className="border-b">
                                <td className="py-2 font-medium">Categorical Features Encoded</td>
                                <td className="py-2 text-right">
                                  {processedResult.statistics.features_encoded.join(", ")}
                                </td>
                              </tr>
                            )}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </CardContent>
                <CardFooter className="flex justify-end gap-3">
                <Button variant="outline" onClick={() => handlePreview()}>
                    <Eye className="mr-2 h-4 w-4" />
                    Preview Data
                </Button>
                  <Button>
                    <Download className="mr-2 h-4 w-4" />
                    Download Processed File
                  </Button>
                </CardFooter>
              </Card>
              
              {/* Add comprehensive preprocessing report if available */}
              {selectedFile.preprocessing_info?.is_preprocessed && (
                <Card>
                  <CardHeader>
                    <CardTitle>Automatic Preprocessing Report</CardTitle>
                    <CardDescription>
                      Detailed analysis of the automatic preprocessing performed on this file
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-6">
                      <div className="bg-amber-50 border border-amber-200 rounded-md p-4">
                        <p className="text-sm text-amber-800">
                          <strong>Note:</strong> This report shows the automatic preprocessing that was previously applied 
                          to this file during upload. This is separate from the custom operations you just applied.
                        </p>
                      </div>
                      
                      {/* Import and use the PreprocessingReport component */}
                      {selectedFile.preprocessing_info && (
                        <PreprocessingReport 
                          report={{
                            columns_dropped: selectedFile.preprocessing_info.dropped_columns || [],
                            columns_cleaned: selectedFile.preprocessing_info.columns_cleaned || [],
                            date_columns_detected: selectedFile.preprocessing_info.auto_detected_dates || [],
                            missing_value_stats: selectedFile.preprocessing_info.missing_value_stats || {},
                            original_shape: [selectedFile.row_count, selectedFile.column_names.length],
                            processed_shape: [selectedFile.row_count, selectedFile.column_names.length]
                          }} 
                          onDownload={() => {}} 
                        />
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
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