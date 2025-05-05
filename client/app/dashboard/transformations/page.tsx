// client/app/dashboard/transformations/page.tsx
"use client"

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { createClient } from '@/utils/supabase/client'
import { type FileMetadata } from '@/components/FilePreview'
import FilePreview from '@/components/FilePreview'
import TransformationsView from '@/components/TransformationsView'
import ManualTransformations from '@/components/ManualTransformations'
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
  ArrowLeft,
  Download,
  Eye,
  CheckCircle2,
  FileX,
  GitBranch,
  RefreshCw,
  Sparkles,
  Sliders
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"

// Define column info interface
interface ColumnInfo {
  name: string;
  type: string;
  sample_values: string[];
}

// Define transformation config interface
interface TransformationConfig {
  log_transform: string[];
  sqrt_transform: string[];
  squared_transform: string[];
  reciprocal_transform: string[];
  binning: {column: string; bins: number; labels?: string[]}[];
  one_hot_encoding: string[];
  datetime_features: {column: string; features: string[]}[];
}

export default function TransformationsPage() {
  const [activeTab, setActiveTab] = useState<string>("file_selection")
  const [transformMode, setTransformMode] = useState<"auto" | "manual">("auto")
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [columnInfo, setColumnInfo] = useState<ColumnInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [isApplyingTransformations, setIsApplyingTransformations] = useState(false)
  const [transformationResult, setTransformationResult] = useState<any | null>(null)
  const [transformProgress, setTransformProgress] = useState(0)
  
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
                setActiveTab("transformations")
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
  
  // Handle file selection and analyze columns
  const handleSelectFile = async (file: FileMetadata) => {
    setSelectedFile(file)
    setTransformationResult(null)
    setIsAnalyzing(true)
    setErrorMessage(null)
    
    try {
      // Download the file content
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${file.user_id}/${file.filename}`)
      
      // Fetch the file
      const response = await fetch(publicUrl)
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`)
      }
      
      const fileBlob = await response.blob()
      const fileObj = new File([fileBlob], file.original_filename, { 
        type: file.mime_type 
      })
      
      // Create form data
      const formData = new FormData()
      formData.append('file', fileObj)
      
      // Send to backend for analysis
      const analysisResponse = await fetch('http://localhost:8000/transformations/analyze-columns/', {
        method: 'POST',
        body: formData
      })
      
      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json()
        throw new Error(errorData.detail || 'Analysis failed')
      }
      
      const analysisResult = await analysisResponse.json()
      
      // Extract column information
      const columns: ColumnInfo[] = analysisResult.columns.map((col: any) => ({
        name: col.name,
        type: col.type,
        sample_values: col.sample_values || []
      }))
      
      setColumnInfo(columns)
      setActiveTab("transformations")
    } catch (error) {
      console.error('Error analyzing file:', error)
      setErrorMessage(`Error analyzing file: ${error instanceof Error ? error.message : String(error)}`)
      // Still move to transformations tab with empty column info
      setColumnInfo([])
      setActiveTab("transformations")
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Apply auto transformations
  const applyAutoTransformations = async () => {
    if (!selectedFile) return
    
    setIsApplyingTransformations(true)
    setErrorMessage(null)
    setTransformProgress(0)
    
    try {
      // Download the file content
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      setTransformProgress(20)
      
      // Fetch the file
      const response = await fetch(publicUrl)
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`)
      }
      
      const fileBlob = await response.blob()
      const fileObj = new File([fileBlob], selectedFile.original_filename, { 
        type: selectedFile.mime_type 
      })
      
      setTransformProgress(40)
      
      // Create form data
      const formData = new FormData()
      formData.append('file', fileObj)
      
      setTransformProgress(60)
      
      // Send to backend for auto transformations
      const transformResponse = await fetch('http://localhost:8000/transformations/auto-transform/', {
        method: 'POST',
        body: formData
      })
      
      setTransformProgress(80)
      
      if (!transformResponse.ok) {
        const errorData = await transformResponse.json()
        throw new Error(errorData.detail || 'Transformation failed')
      }
      
      const result = await transformResponse.json()
      
      // Add validation and default values
      if (!result || !result.success) {
        throw new Error(result?.error || 'Transformation failed')
      }
      
      // Ensure the result has the expected structure
      const transformResult = {
        success: result.success,
        original_shape: result.original_shape || [0, 0],
        transformed_shape: result.transformed_shape || [0, 0],
        transformations: result.transformations || {},
        original_filename: result.original_filename || selectedFile.original_filename,
        transformed_filename: result.transformed_filename || 'transformed_file'
      }
      
      setTransformationResult(transformResult)
      setTransformProgress(100)
      setActiveTab("results")
    } catch (error) {
      console.error('Error applying transformations:', error)
      setErrorMessage(`Error applying transformations: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsApplyingTransformations(false)
    }
  }
  
  // Apply manual transformations
  const applyManualTransformations = async (config: TransformationConfig) => {
    if (!selectedFile) return
    
    setIsApplyingTransformations(true)
    setErrorMessage(null)
    setTransformProgress(0)
    
    try {
      // Download the file content
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      setTransformProgress(20)
      
      // Fetch the file
      const response = await fetch(publicUrl)
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`)
      }
      
      const fileBlob = await response.blob()
      const fileObj = new File([fileBlob], selectedFile.original_filename, { 
        type: selectedFile.mime_type 
      })
      
      setTransformProgress(40)
      
      // Create form data
      const formData = new FormData()
      formData.append('file', fileObj)
      formData.append('transformations', JSON.stringify(config))
      
      setTransformProgress(60)
      
      // Send to backend for specific transformations
      const transformResponse = await fetch('http://localhost:8000/transformations/specific-transform/', {
        method: 'POST',
        body: formData
      })
      
      setTransformProgress(80)
      
      if (!transformResponse.ok) {
        const errorData = await transformResponse.json()
        throw new Error(errorData.detail || 'Transformation failed')
      }
      
      const result = await transformResponse.json()
      
      // Add validation and default values
      if (!result || !result.success) {
        throw new Error(result?.error || 'Transformation failed')
      }
      
      // Ensure the result has the expected structure
      const transformResult = {
        success: result.success,
        original_filename: result.original_filename || selectedFile.original_filename,
        transformed_filename: result.transformed_filename || 'transformed_file',
        transformations_applied: result.transformations_applied || []
      }
      
      setTransformationResult(transformResult)
      setTransformProgress(100)
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
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-8">
        Feature Transformations
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full max-w-md mb-6">
          <TabsTrigger value="file_selection" className="flex-1">Select File</TabsTrigger>
          <TabsTrigger value="transformations" className="flex-1" disabled={!selectedFile}>
            Transformations
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
                  <CardTitle>Select a File for Feature Transformations</CardTitle>
                  <CardDescription>
                    Choose a file to apply mathematical transformations and feature engineering
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableCaption>Your available files for transformations</TableCaption>
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
                                Preprocessed
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
                              disabled={isAnalyzing}
                            >
                              {isAnalyzing && selectedFile?.id === file.id ? "Analyzing..." : "Select"}
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
        
        {/* Transformations Tab */}
        <TabsContent value="transformations" className="border-none p-0">
          {selectedFile && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Feature Transformations</CardTitle>
                      <CardDescription>
                        Apply transformations to {selectedFile.original_filename}
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
                  
                  {selectedFile.preprocessing_info?.engineered_features && 
                   selectedFile.preprocessing_info.engineered_features.length > 0 && (
                    <Alert className="mb-6">
                      <AlertTitle>Existing Transformations Detected</AlertTitle>
                      <AlertDescription>
                        This file already has {selectedFile.preprocessing_info.engineered_features.length} engineered 
                        features from previous transformations. You can apply additional transformations below.
                      </AlertDescription>
                    </Alert>
                  )}
                  
                  <div className="border-b mb-8">
                    <div className="flex space-x-2 mb-4">
                      <Button
                        variant={transformMode === "auto" ? "default" : "outline"}
                        onClick={() => setTransformMode("auto")}
                        className="flex items-center gap-2"
                      >
                        <Sparkles className="h-4 w-4" />
                        Auto Transformations
                      </Button>
                      <Button
                        variant={transformMode === "manual" ? "default" : "outline"}
                        onClick={() => setTransformMode("manual")}
                        className="flex items-center gap-2"
                      >
                        <Sliders className="h-4 w-4" />
                        Manual Transformations
                      </Button>
                    </div>
                  </div>
                  
                  {transformMode === "auto" ? (
                    <div className="text-center p-8 border rounded-lg mb-8">
                      <GitBranch className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                      <h3 className="text-lg font-medium mb-2">Automatic Feature Engineering</h3>
                      <p className="mb-6 max-w-2xl mx-auto">
                        Let the system automatically detect the best transformations for your data. 
                        This will create features like date components, mathematical transformations, 
                        and categorical encodings based on your data types.
                      </p>
                      <Button 
                        onClick={applyAutoTransformations} 
                        disabled={isApplyingTransformations}
                        size="lg"
                      >
                        {isApplyingTransformations ? (
                          <span className="flex items-center gap-2">
                            <RefreshCw className="h-4 w-4 animate-spin" />
                            Processing...
                          </span>
                        ) : (
                          <span className="flex items-center gap-2">
                            <Sparkles className="h-4 w-4" />
                            Apply Auto Transformations
                          </span>
                        )}
                      </Button>
                    </div>
                  ) : (
                    <ManualTransformations 
                      columns={columnInfo} 
                      onApplyTransformations={applyManualTransformations}
                      isLoading={isApplyingTransformations}
                    />
                  )}
                  
                  {isApplyingTransformations && (
                    <div className="mt-6">
                      <p className="text-sm mb-2">Transformation Progress</p>
                      <Progress value={transformProgress} className="h-2 w-full" />
                    </div>
                  )}
                </CardContent>
                <CardFooter className="flex justify-between">
                  <div className="flex gap-2">
                    <Button variant="outline" onClick={() => setActiveTab("file_selection")}>
                      <ArrowLeft className="mr-2 h-4 w-4" />
                      Back
                    </Button>
                    <Button variant="outline" onClick={handlePreview}>
                      <Eye className="mr-2 h-4 w-4" />
                      Preview File
                    </Button>
                  </div>
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
                    <CardTitle>Transformation Results</CardTitle>
                    <CardDescription>
                      Results of applying transformations to {selectedFile?.original_filename}
                    </CardDescription>
                  </div>
                  <Button variant="ghost" size="icon" onClick={() => setActiveTab("transformations")}>
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              
              <CardContent>
                {/* Different displays based on transformation type */}
                {transformationResult.original_shape ? (
                  // Auto transformation results
                  <>
                    <div className="flex items-center justify-center p-6 bg-green-50 rounded-md mb-6">
                      <CheckCircle2 className="h-8 w-8 text-green-500 mr-3" />
                      <div>
                        <h3 className="font-medium">Transformations Completed Successfully</h3>
                        <p className="text-sm text-gray-500">
                          Transformed file created with engineered features.
                        </p>
                      </div>
                    </div>
                    
                    <div className="grid md:grid-cols-3 gap-4 mb-6">
                      <div className="flex flex-col border rounded-md p-4">
                        <span className="text-sm font-medium text-muted-foreground">Original Shape</span>
                        <span className="font-bold text-xl">
                          {transformationResult.original_shape[0]} rows × {transformationResult.original_shape[1]} columns
                        </span>
                      </div>
                      <div className="flex flex-col border rounded-md p-4">
                        <span className="text-sm font-medium text-muted-foreground">Transformed Shape</span>
                        <span className="font-bold text-xl">
                          {transformationResult.transformed_shape[0]} rows × {transformationResult.transformed_shape[1]} columns
                        </span>
                      </div>
                      <div className="flex flex-col border rounded-md p-4">
                        <span className="text-sm font-medium text-muted-foreground">New Features</span>
                        <span className="font-bold text-xl">
                          {transformationResult.transformed_shape[1] - transformationResult.original_shape[1]}
                        </span>
                      </div>
                    </div>
                  </>
                ) : (
                  // Manual transformation results
                  <div className="flex items-center justify-center p-6 bg-green-50 rounded-md mb-6">
                    <CheckCircle2 className="h-8 w-8 text-green-500 mr-3" />
                    <div>
                      <h3 className="font-medium">Specific Transformations Applied Successfully</h3>
                      <p className="text-sm text-gray-500">
                        Your custom transformations have been applied to the file.
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Add this check for the transformations object */}
                {transformationResult.transformations && (
                  <div className="bg-muted rounded-lg p-6">
                    <h3 className="text-lg font-medium mb-3">Transformations Applied</h3>
                    <div className="grid gap-4">
                      {Object.entries(transformationResult.transformations).map(([key, value]: [string, any]) => {
                        // Skip if there are no transformations of this type
                        if (!value || (Array.isArray(value) && value.length === 0)) {
                          return null;
                        }
                        
                        return (
                          <div key={key} className="border rounded-lg p-4 bg-background">
                            <h4 className="font-medium mb-2">
                              {key === 'datetime_features' ? 'DateTime Features' :
                              key === 'categorical_encodings' ? 'Categorical Encodings' :
                              key === 'numeric_transformations' ? 'Numeric Transformations' :
                              key === 'binned_features' ? 'Binned Features' : key}
                              {Array.isArray(value) && ` (${value.length})`}
                            </h4>
                            
                            {Array.isArray(value) && value.length > 0 && (
                              <div className="grid grid-cols-2 gap-2 mt-1">
                                {key === 'datetime_features' && value.map((item, i) => (
                                  <div key={i} className="text-sm">
                                    <span className="font-medium">{item.source_column}</span>
                                    <span className="text-muted-foreground"> → </span>
                                    <span>
                                      {item.derived_features.length} features
                                    </span>
                                  </div>
                                ))}
                                
                                {key === 'categorical_encodings' && value.map((item, i) => (
                                  <div key={i} className="text-sm">
                                    <span className="font-medium">{item.source_column}</span>
                                    <span className="text-muted-foreground"> → </span>
                                    <span>
                                      {item.derived_features.length} features
                                    </span>
                                  </div>
                                ))}
                                
                                {key === 'numeric_transformations' && value.map((item, i) => (
                                  <div key={i} className="text-sm">
                                    <span className="font-medium">{item.source_column}</span>
                                    <span className="text-muted-foreground"> → </span>
                                    <span>
                                      {item.derived_features.join(', ')}
                                    </span>
                                  </div>
                                ))}
                                
                                {key === 'binned_features' && value.map((item, i) => (
                                  <div key={i} className="text-sm">
                                    <span className="font-medium">{item.source_column}</span>
                                    <span className="text-muted-foreground"> → </span>
                                    <span>
                                      {item.derived_feature} ({item.bins} bins)
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline" onClick={() => setActiveTab("transformations")}>
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Transformations
                </Button>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={handlePreview}>
                    <Eye className="mr-2 h-4 w-4" />
                    Preview Original File
                  </Button>
                  <Button>
                    <Download className="mr-2 h-4 w-4" />
                    Download Transformed File
                  </Button>
                </div>
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