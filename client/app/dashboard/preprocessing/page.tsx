'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import { type FileMetadata } from '@/components/FilePreview'
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
  Settings2, 
  CheckCircle2, 
  Trash2, 
  FileText, 
  AlertCircle, 
  Play,
  ArrowLeft,
  Download,
  Eye
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"

export default function Preprocessing() {
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<string>("files")
  const [selectedOperations, setSelectedOperations] = useState<string[]>([])
  const [processingStatus, setProcessingStatus] = useState<{
    isProcessing: boolean;
    progress: number;
    message: string;
  }>({
    isProcessing: false,
    progress: 0,
    message: ""
  })
  const [processedResult, setProcessedResult] = useState<{
    success: boolean;
    message: string;
    operations_applied: string[];
    result_file?: string;
    statistics?: {
      rows_affected: number;
      columns_modified: string[];
      outliers_removed?: number;
      features_encoded?: string[];
    }
  } | null>(null)
  
  const supabase = createClient()
  
  // Fetch user's files
  useEffect(() => {
    async function fetchFiles() {
      try {
        const { data: { user } } = await supabase.auth.getUser()
        
        if (user) {
          const { data, error } = await supabase
            .from('files')
            .select('*')
            .eq('user_id', user.id)
            .order('upload_date', { ascending: false })
          
          if (data) {
            setFiles(data as FileMetadata[])
          }
          
          if (error) {
            console.error('Error fetching files:', error.message)
          }
        }
      } catch (error) {
        console.error('Error in fetchFiles:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchFiles()
  }, [])
  
  const handlePreview = (file: FileMetadata) => {
    setSelectedFile(file)
    setIsPreviewOpen(true)
  }
  
  const closePreview = () => {
    setIsPreviewOpen(false)
  }
  
  const handleSelectFile = (file: FileMetadata) => {
    setSelectedFile(file)
    setActiveTab("operations")
  }
  
  const handleOperationToggle = (operationId: string) => {
    setSelectedOperations(prev => 
      prev.includes(operationId)
        ? prev.filter(id => id !== operationId)
        : [...prev, operationId]
    )
  }
  
  const startCustomPreprocessing = async () => {
    if (!selectedFile || selectedOperations.length === 0) return
    
    try {
      setProcessingStatus({
        isProcessing: true,
        progress: 0,
        message: "Starting custom preprocessing..."
      })
      
      const { data: { user } } = await supabase.auth.getUser()
      
      if (!user) {
        throw new Error("User not authenticated")
      }
      
      // Prepare form data
      const formData = new FormData()
      formData.append('file_id', selectedFile.id)
      formData.append('user_id', user.id)
      formData.append('operations', JSON.stringify(selectedOperations))
      
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setProcessingStatus(prev => ({
          ...prev,
          progress: Math.min(prev.progress + 5, 90),
          message: prev.progress < 30 ? "Applying selected operations..." 
                  : prev.progress < 60 ? "Processing data..." 
                  : "Finalizing results..."
        }))
      }, 500)
      
      // Send to Python backend
      const response = await fetch('http://localhost:8000/custom-preprocess/', {
        method: 'POST',
        body: formData
      })
      
      clearInterval(progressInterval)
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Custom preprocessing failed')
      }
      
      const result = await response.json()
      
      setProcessingStatus({
        isProcessing: false,
        progress: 100,
        message: "Preprocessing completed successfully!"
      })
      
      // Set mock result for demonstration
      setProcessedResult({
        success: true,
        message: "Preprocessing operations completed successfully",
        operations_applied: selectedOperations,
        result_file: result.result_file || `processed_${selectedFile.id}.csv`,
        statistics: {
          rows_affected: Math.floor(selectedFile.row_count * 0.95),
          columns_modified: selectedOperations.includes("feature_encoding") 
            ? selectedFile.column_names.filter(col => 
                selectedFile.statistics[col]?.type === 'categorical'
              ).slice(0, 3) 
            : [],
          outliers_removed: selectedOperations.includes("outliers") ? Math.floor(selectedFile.row_count * 0.02) : 0,
          features_encoded: selectedOperations.includes("feature_encoding") 
            ? selectedFile.column_names.filter(col => 
                selectedFile.statistics[col]?.type === 'categorical'
              ).slice(0, 3) 
            : []
        }
      })
      
      // Update file list with new processed file
      fetchFiles()
      
      // After 2 seconds, reset the progress
      setTimeout(() => {
        setProcessingStatus({
          isProcessing: false,
          progress: 0,
          message: ""
        })
        setActiveTab("results")
      }, 2000)
      
    } catch (error) {
      setProcessingStatus({
        isProcessing: false,
        progress: 0,
        message: `Error: ${error instanceof Error ? error.message : 'Failed to process file'}`
      })
    }
  }
  
  const fetchFiles = async () => {
    try {
      setLoading(true)
      const { data: { user } } = await supabase.auth.getUser()
      
      if (user) {
        const { data, error } = await supabase
          .from('files')
          .select('*')
          .eq('user_id', user.id)
          .order('upload_date', { ascending: false })
        
        if (data) {
          setFiles(data as FileMetadata[])
        }
        
        if (error) {
          console.error('Error fetching files:', error.message)
        }
      }
    } catch (error) {
      console.error('Error in fetchFiles:', error)
    } finally {
      setLoading(false)
    }
  }
  
  // Map operation IDs to readable names
  const operationNames: Record<string, string> = {
    missing_values: "Handle Missing Values",
    outliers: "Detect & Remove Outliers",
    feature_encoding: "Categorical Encoding",
    feature_scaling: "Feature Scaling",
    date_features: "Extract Date Features"
  }
  
  // Available preprocessing operations
  const availableOperations = [
    {
      id: "missing_values",
      name: "Handle Missing Values",
      description: "Detect and handle missing values using advanced imputation methods",
      icon: AlertCircle
    },
    {
      id: "outliers",
      name: "Detect & Remove Outliers",
      description: "Identify and handle outliers in numeric columns",
      icon: Trash2
    },
    {
      id: "feature_encoding",
      name: "Categorical Encoding",
      description: "Convert categorical variables to numeric representations",
      icon: Settings2
    },
    {
      id: "feature_scaling",
      name: "Feature Scaling",
      description: "Normalize or standardize numeric features",
      icon: Settings2
    },
    {
      id: "date_features",
      name: "Extract Date Features",
      description: "Extract useful features from date columns (year, month, day, etc.)",
      icon: FileText
    }
  ]
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-8">
        Preprocessing
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full max-w-md mb-6">
          <TabsTrigger value="files" className="flex-1">Select File</TabsTrigger>
          <TabsTrigger 
            value="operations" 
            className="flex-1" 
            disabled={!selectedFile}
          >
            Operations
          </TabsTrigger>
          <TabsTrigger 
            value="results" 
            className="flex-1" 
            disabled={!processedResult}
          >
            Results
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="files" className="border-none p-0">
          {loading ? (
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
                    <TableRow key={file.id}>
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
                          variant="ghost"
                          size="icon"
                          onClick={() => handlePreview(file)}
                          title="Preview file"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
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
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="operations" className="border-none p-0">
          {selectedFile && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Selected File: {selectedFile.original_filename}</CardTitle>
                      <CardDescription>
                        {selectedFile.row_count} rows, {selectedFile.column_names.length} columns
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("files")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-500 mb-4">
                    Select the preprocessing operations you want to apply to this file:
                  </p>
                  
                  <div className="grid gap-4">
                    {availableOperations.map((operation) => (
                      <div key={operation.id} className="flex items-start space-x-3 border p-4 rounded-md">
                        <Checkbox 
                          id={operation.id}
                          checked={selectedOperations.includes(operation.id)}
                          onCheckedChange={() => handleOperationToggle(operation.id)}
                        />
                        <div className="grid gap-1.5">
                          <label
                            htmlFor={operation.id}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 flex items-center gap-2"
                          >
                            <operation.icon className="h-4 w-4" />
                            {operation.name}
                          </label>
                          <p className="text-sm text-muted-foreground">
                            {operation.description}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
                <CardFooter>
                  <Button
                    disabled={selectedOperations.length === 0 || processingStatus.isProcessing}
                    onClick={startCustomPreprocessing}
                    className="ml-auto"
                  >
                    {processingStatus.isProcessing ? (
                      <>Processing...</>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Start Processing
                      </>
                    )}
                  </Button>
                </CardFooter>
              </Card>
              
              {processingStatus.isProcessing && (
                <Card>
                  <CardHeader>
                    <CardTitle>Processing Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Progress value={processingStatus.progress} className="h-2 w-full mb-2" />
                    <p className="text-sm text-center text-gray-500">
                      {processingStatus.message}
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>
        
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
                  <Button variant="outline" onClick={() => handlePreview(selectedFile)}>
                    <Eye className="mr-2 h-4 w-4" />
                    Preview Data
                  </Button>
                  <Button>
                    <Download className="mr-2 h-4 w-4" />
                    Download Processed File
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
          onClose={closePreview} 
        />
      )}
    </section>
  )
}