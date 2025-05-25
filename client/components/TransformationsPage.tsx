// components/TransformationsPage.tsx
'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import { useSearchParams } from 'next/navigation'
import { 
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { 
  Download,
  Play,
  Eye,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ArrowUpDown,
  Binary,
  Calendar
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import type { FileMetadata } from '@/components/FilePreview'

interface ColumnInfo {
  name: string;
  type: string;
  sample_values: string[];
}

interface TransformationConfig {
  log_transform: string[];
  sqrt_transform: string[];
  squared_transform: string[];
  reciprocal_transform: string[];
  binning: Array<{column: string; bins: number}>;
  one_hot_encoding: string[];
  datetime_features: Array<{column: string; features: string[]}>;
}

interface PreviewData {
  original: Record<string, any>[];
  transformed: Record<string, any>[];
  columns: {
    original: string[];
    transformed: string[];
  };
}

export default function TransformationsPage() {
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [columns, setColumns] = useState<ColumnInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const [previewData, setPreviewData] = useState<PreviewData | null>(null)
  const [transformationConfig, setTransformationConfig] = useState<TransformationConfig>({
    log_transform: [],
    sqrt_transform: [],
    squared_transform: [],
    reciprocal_transform: [],
    binning: [],
    one_hot_encoding: [],
    datetime_features: []
  })
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null)
  
  const { toast } = useToast()
  const supabase = createClient()
  const searchParams = useSearchParams()
  const fileId = searchParams.get('file')

  useEffect(() => {
    fetchFiles()
  }, [])

  useEffect(() => {
    if (fileId && files.length > 0) {
      const file = files.find(f => f.id === fileId)
      if (file) {
        handleSelectFile(file)
      }
    }
  }, [fileId, files])

  const fetchFiles = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser()
      
      if (user) {
        const { data, error } = await supabase
          .from('files')
          .select('*')
          .eq('user_id', user.id)
          .order('upload_date', { ascending: false })
        
        if (error) throw error
        setFiles(data as FileMetadata[])
      }
    } catch (error) {
      console.error('Error fetching files:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch files"
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectFile = async (file: FileMetadata) => {
    setSelectedFile(file)
    setIsAnalyzing(true)
    setColumns([])
    setPreviewData(null)
    setDownloadUrl(null)
    
    try {
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${file.user_id}/${file.filename}`)
      
      const response = await fetch(publicUrl)
      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`)
      }
      
      const fileBlob = await response.blob()
      const formData = new FormData()
      formData.append('file', fileBlob, file.original_filename)
      
      const analysisResponse = await fetch('http://localhost:8000/transformations/analyze-columns/', {
        method: 'POST',
        body: formData
      })
      
      if (!analysisResponse.ok) {
        throw new Error('Failed to analyze file')
      }
      
      const analysisResult = await analysisResponse.json()
      setColumns(analysisResult.columns)
      
      toast({
        title: "File loaded",
        description: `Successfully loaded ${file.original_filename}`
      })
      
    } catch (error) {
      console.error('Error analyzing file:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to analyze the selected file"
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const generatePreview = async () => {
    if (!selectedFile) return
    
    setIsPreviewLoading(true)
    
    try {
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      const response = await fetch(publicUrl)
      const fileBlob = await response.blob()
      
      const formData = new FormData()
      formData.append('file', fileBlob, selectedFile.original_filename)
      formData.append('transformations', JSON.stringify(transformationConfig))
      
      // Debug: Log the transformation config
      console.log('Transformation config:', transformationConfig)
      
      // Use the custom preprocessing preview endpoint (it should handle transformations)
      const previewResponse = await fetch('http://localhost:8000/custom-preprocessing/preview-transformation/', {
        method: 'POST',
        body: formData
      })
      
      if (!previewResponse.ok) {
        const errorText = await previewResponse.text()
        console.error('Preview error:', errorText)
        throw new Error(`Failed to generate preview: ${previewResponse.status}`)
      }
      
      const previewResult = await previewResponse.json()
      console.log('Preview result:', previewResult)
      
      // Handle the response format from custom preprocessing
      if (previewResult.success && previewResult.preview) {
        setPreviewData({
          original: previewResult.preview.original,
          transformed: previewResult.preview.transformed,
          columns: previewResult.preview.columns
        })
      } else {
        console.error('Unexpected preview response format:', previewResult)
        throw new Error(previewResult.error || 'Invalid preview response format')
      }
      
    } catch (error) {
      console.error('Error generating preview:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: `Preview failed: ${error.message}`
      })
    } finally {
      setIsPreviewLoading(false)
    }
  }

  const applyTransformations = async () => {
    if (!selectedFile) return
    
    setIsProcessing(true)
    
    try {
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      const response = await fetch(publicUrl)
      const fileBlob = await response.blob()
      
      const formData = new FormData()
      formData.append('file', fileBlob, selectedFile.original_filename)
      formData.append('transformations', JSON.stringify(transformationConfig))
      
      const transformResponse = await fetch('http://localhost:8000/transformations/specific-transform/', {
        method: 'POST',
        body: formData
      })
      
      if (!transformResponse.ok) {
        throw new Error('Failed to apply transformations')
      }
      
      const transformResult = await transformResponse.json()
      setDownloadUrl(`http://localhost:8000/transformed-files/${transformResult.transformed_filename}`)
      
      toast({
        title: "Success",
        description: "Transformations applied successfully!"
      })
      
    } catch (error) {
      console.error('Error applying transformations:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to apply transformations"
      })
    } finally {
      setIsProcessing(false)
    }
  }

  const toggleTransformation = (type: keyof TransformationConfig, column: string) => {
    setTransformationConfig(prev => {
      if (type === 'binning') {
        const exists = prev.binning.some(item => item.column === column)
        if (exists) {
          return {
            ...prev,
            binning: prev.binning.filter(item => item.column !== column)
          }
        } else {
          return {
            ...prev,
            binning: [...prev.binning, { column, bins: 4 }]
          }
        }
      } else if (type === 'datetime_features') {
        const exists = prev.datetime_features.some(item => item.column === column)
        if (exists) {
          return {
            ...prev,
            datetime_features: prev.datetime_features.filter(item => item.column !== column)
          }
        } else {
          return {
            ...prev,
            datetime_features: [...prev.datetime_features, { 
              column, 
              features: ['year', 'month', 'day', 'dayofweek'] 
            }]
          }
        }
      } else {
        const array = prev[type] as string[]
        if (array.includes(column)) {
          return {
            ...prev,
            [type]: array.filter(col => col !== column)
          }
        } else {
          return {
            ...prev,
            [type]: [...array, column]
          }
        }
      }
    })
  }

  const isTransformationSelected = (type: keyof TransformationConfig, column: string): boolean => {
    if (type === 'binning') {
      return transformationConfig.binning.some(item => item.column === column)
    } else if (type === 'datetime_features') {
      return transformationConfig.datetime_features.some(item => item.column === column)
    } else {
      const array = transformationConfig[type] as string[]
      return array.includes(column)
    }
  }

  const getTotalTransformations = (): number => {
    return (
      transformationConfig.log_transform.length +
      transformationConfig.sqrt_transform.length +
      transformationConfig.squared_transform.length +
      transformationConfig.reciprocal_transform.length +
      transformationConfig.binning.length +
      transformationConfig.one_hot_encoding.length +
      transformationConfig.datetime_features.length
    )
  }

  const numericColumns = columns.filter(col => 
    col.type === 'integer' || col.type === 'float'
  )
  
  const categoricalColumns = columns.filter(col => 
    col.type === 'string' || col.type === 'category'
  )
  
  const dateColumns = columns.filter(col => 
    col.type === 'datetime' || col.type === 'datetime64[ns]'
  )

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      </div>
    )
  }

  return (
    <section className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-6">
        Data Transformations
      </div>

      {/* File Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select File</CardTitle>
          <CardDescription>Choose a dataset to transform</CardDescription>
        </CardHeader>
        <CardContent>
          {files.length === 0 ? (
            <p className="text-center text-muted-foreground">No files found</p>
          ) : (
            <div className="space-y-2">
              {files.map((file) => (
                <div
                  key={file.id}
                  className={`p-3 border rounded cursor-pointer ${
                    selectedFile?.id === file.id ? 'border-primary bg-primary/5' : 'hover:border-primary/50'
                  }`}
                  onClick={() => handleSelectFile(file)}
                >
                  <div className="font-medium">{file.original_filename}</div>
                  <div className="text-sm text-muted-foreground">
                    {file.row_count.toLocaleString()} rows × {file.column_names.length} columns
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Loading State */}
      {isAnalyzing && (
        <Card>
          <CardContent className="flex items-center justify-center p-8">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
              <p>Analyzing file structure...</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Transformation Configuration */}
      {selectedFile && !isAnalyzing && columns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Configure Transformations</CardTitle>
            <CardDescription>
              Select which transformations to apply to your columns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="numeric" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="numeric">
                  <ArrowUpDown className="w-4 h-4 mr-2" />
                  Numeric ({numericColumns.length})
                </TabsTrigger>
                <TabsTrigger value="categorical">
                  <Binary className="w-4 h-4 mr-2" />
                  Categorical ({categoricalColumns.length})
                </TabsTrigger>
                <TabsTrigger value="datetime">
                  <Calendar className="w-4 h-4 mr-2" />
                  DateTime ({dateColumns.length})
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="numeric" className="space-y-4">
                {/* Log Transform */}
                <div>
                  <h3 className="font-medium mb-2">Log Transform</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Apply log(x+1) to reduce positive skewness
                  </p>
                  <div className="space-y-2">
                    {numericColumns.map(column => (
                      <div key={column.name} className="flex items-center space-x-2">
                        <Checkbox
                          checked={isTransformationSelected('log_transform', column.name)}
                          onCheckedChange={() => toggleTransformation('log_transform', column.name)}
                        />
                        <span className="text-sm">{column.name}</span>
                        <span className="text-xs text-muted-foreground">
                          ({column.sample_values.slice(0, 3).join(', ')})
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Square Root Transform */}
                <div>
                  <h3 className="font-medium mb-2">Square Root Transform</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Apply sqrt(x) for moderate skewness reduction
                  </p>
                  <div className="space-y-2">
                    {numericColumns.map(column => (
                      <div key={column.name} className="flex items-center space-x-2">
                        <Checkbox
                          checked={isTransformationSelected('sqrt_transform', column.name)}
                          onCheckedChange={() => toggleTransformation('sqrt_transform', column.name)}
                        />
                        <span className="text-sm">{column.name}</span>
                        <span className="text-xs text-muted-foreground">
                          ({column.sample_values.slice(0, 3).join(', ')})
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Binning */}
                <div>
                  <h3 className="font-medium mb-2">Binning</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Convert continuous variables to categories
                  </p>
                  <div className="space-y-3">
                    {numericColumns.map(column => (
                      <div key={column.name} className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <Checkbox
                            checked={isTransformationSelected('binning', column.name)}
                            onCheckedChange={() => toggleTransformation('binning', column.name)}
                          />
                          <span className="text-sm">{column.name}</span>
                        </div>
                        {isTransformationSelected('binning', column.name) && (
                          <div className="ml-6">
                            <Select
                              value={transformationConfig.binning.find(b => b.column === column.name)?.bins.toString() || '4'}
                              onValueChange={(value) => {
                                setTransformationConfig(prev => ({
                                  ...prev,
                                  binning: prev.binning.map(b => 
                                    b.column === column.name 
                                      ? { ...b, bins: parseInt(value) }
                                      : b
                                  )
                                }))
                              }}
                            >
                              <SelectTrigger className="w-32">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="2">2 bins</SelectItem>
                                <SelectItem value="3">3 bins</SelectItem>
                                <SelectItem value="4">4 bins</SelectItem>
                                <SelectItem value="5">5 bins</SelectItem>
                                <SelectItem value="10">10 bins</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="categorical" className="space-y-4">
                <div>
                  <h3 className="font-medium mb-2">One-Hot Encoding</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Convert categorical variables to binary columns
                  </p>
                  {categoricalColumns.length > 0 ? (
                    <div className="space-y-2">
                      {categoricalColumns.map(column => (
                        <div key={column.name} className="flex items-center space-x-2">
                          <Checkbox
                            checked={isTransformationSelected('one_hot_encoding', column.name)}
                            onCheckedChange={() => toggleTransformation('one_hot_encoding', column.name)}
                          />
                          <span className="text-sm">{column.name}</span>
                          <span className="text-xs text-muted-foreground">
                            ({column.sample_values.slice(0, 3).join(', ')})
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No categorical columns found</p>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="datetime" className="space-y-4">
                <div>
                  <h3 className="font-medium mb-2">DateTime Feature Extraction</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    Extract year, month, day components from date columns
                  </p>
                  {dateColumns.length > 0 ? (
                    <div className="space-y-2">
                      {dateColumns.map(column => (
                        <div key={column.name} className="flex items-center space-x-2">
                          <Checkbox
                            checked={isTransformationSelected('datetime_features', column.name)}
                            onCheckedChange={() => toggleTransformation('datetime_features', column.name)}
                          />
                          <span className="text-sm">{column.name}</span>
                          <span className="text-xs text-muted-foreground">
                            (extracts: year, month, day, day_of_week)
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No datetime columns found</p>
                  )}
                </div>
              </TabsContent>
            </Tabs>

            {/* Action Buttons */}
            <div className="flex gap-4 mt-6">
              <Button 
                variant="outline" 
                onClick={generatePreview} 
                disabled={isPreviewLoading || getTotalTransformations() === 0}
              >
                {isPreviewLoading ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Eye className="w-4 h-4 mr-2" />
                )}
                Preview
              </Button>
              <Button 
                onClick={applyTransformations} 
                disabled={isProcessing || getTotalTransformations() === 0}
              >
                {isProcessing ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                Apply Transformations
              </Button>
            </div>

            {getTotalTransformations() > 0 && (
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
                <p className="text-sm text-blue-800">
                  <strong>{getTotalTransformations()}</strong> transformations selected
                </p>
                <details className="mt-2">
                  <summary className="text-xs cursor-pointer">Debug: Show config</summary>
                  <pre className="text-xs mt-1 bg-white p-2 rounded border overflow-x-auto">
                    {JSON.stringify(transformationConfig, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Preview */}
      {previewData && (
        <Card>
          <CardHeader>
            <CardTitle>Transformation Preview</CardTitle>
            <CardDescription>
              Preview of transformed data (first 5 rows)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <h3 className="font-medium mb-3 flex items-center gap-2">
                Transformed Data 
                <Badge variant="outline">
                  {previewData.columns.transformed.length} columns
                </Badge>
                {previewData.columns.transformed.length > previewData.columns.original.length && (
                  <Badge variant="secondary" className="bg-green-50 text-green-700">
                    +{previewData.columns.transformed.length - previewData.columns.original.length} new columns
                  </Badge>
                )}
              </h3>
              <div className="overflow-x-auto border rounded">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {previewData.columns.transformed.map((col, index) => (
                        <TableHead key={index}>
                          {col}
                          {!previewData.columns.original.includes(col) && (
                            <Badge variant="secondary" className="ml-1 text-xs bg-green-50 text-green-700">
                              new
                            </Badge>
                          )}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {previewData.transformed.map((row, rowIndex) => (
                      <TableRow key={rowIndex}>
                        {previewData.columns.transformed.map((col, colIndex) => (
                          <TableCell key={colIndex}>
                            {row[col] !== null && row[col] !== undefined ? String(row[col]) : 'null'}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
            
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Preview Notice</AlertTitle>
              <AlertDescription>
                This preview shows only the first 5 rows. The full transformation will be applied to your entire dataset.
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      )}

      {/* Download */}
      {downloadUrl && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-500" />
              Transformation Complete
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded">
              <div>
                <p className="font-medium">Your transformed file is ready!</p>
                <p className="text-sm text-muted-foreground">
                  Applied {getTotalTransformations()} transformations to {selectedFile?.original_filename}
                </p>
              </div>
              <Button asChild>
                <a href={downloadUrl} download>
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </a>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processing Overlay */}
      {(isProcessing || isPreviewLoading) && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg max-w-sm w-full mx-4">
            <div className="text-center">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
              <h3 className="font-medium mb-2">
                {isProcessing ? 'Applying Transformations' : 'Generating Preview'}
              </h3>
              <Progress value={isProcessing ? 70 : 50} className="mb-2" />
              <p className="text-sm text-muted-foreground">
                Please wait...
              </p>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}