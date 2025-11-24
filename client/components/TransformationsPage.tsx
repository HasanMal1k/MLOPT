// components/TransformationsPage.tsx
'use client'

import { useState, useEffect, useCallback } from 'react'
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
  Calendar,
  Database,
  BarChart3,
  FileText,
  TrendingUp,
  Info,
  Sparkles,
  Settings,
  Youtube
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import type { FileMetadata } from '@/components/FilePreview'
import { ScrollArea } from "@/components/ui/scroll-area"

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

interface DataPreview {
  data: Record<string, any>[];
  columns: string[];
  total_rows: number;
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
  const [dataPreview, setDataPreview] = useState<DataPreview | null>(null)
  const [livePreview, setLivePreview] = useState<PreviewData | null>(null)
  const [isLivePreviewLoading, setIsLivePreviewLoading] = useState(false)
  const [currentStep, setCurrentStep] = useState<'selection' | 'configuration'>('selection')
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
    setCurrentStep('configuration')
    setIsAnalyzing(true)
    setColumns([])
    setPreviewData(null)
    setDataPreview(null)
    setLivePreview(null)
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
      
      // Analyze columns
      const analysisResponse = await fetch('http://localhost:8000/transformations/analyze-columns/', {
        method: 'POST',
        body: formData
      })
      
      if (!analysisResponse.ok) {
        throw new Error('Failed to analyze file')
      }
      
      const analysisResult = await analysisResponse.json()
      setColumns(analysisResult.columns)
      
      // Get data preview (first 10 rows)
      const previewFormData = new FormData()
      previewFormData.append('file', fileBlob, file.original_filename)
      previewFormData.append('rows', '10')
      
      const previewResponse = await fetch('http://localhost:8000/files/preview/', {
        method: 'POST',
        body: previewFormData
      })
      
      if (previewResponse.ok) {
        const previewResult = await previewResponse.json()
        setDataPreview({
          data: previewResult.preview || [],
          columns: previewResult.columns || [],
          total_rows: file.row_count
        })
      }
      
      toast({
        title: "File loaded",
        description: `Successfully loaded ${file.name || file.original_filename}`
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
        description: `Preview failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      })
    } finally {
      setIsPreviewLoading(false)
    }
  }

  // Generate live preview automatically when transformations change
  const generateLivePreview = useCallback(async () => {
    console.log('=== generateLivePreview called ===')
    console.log('Selected file:', selectedFile?.name || selectedFile?.original_filename)
    console.log('Total transformations:', getTotalTransformations())
    console.log('Transformation config:', transformationConfig)
    
    if (!selectedFile || getTotalTransformations() === 0) {
      console.log('Skipping preview - no file or no transformations')
      setLivePreview(null)
      setIsLivePreviewLoading(false)
      return
    }
    
    console.log('Fetching file and generating preview...')
    setIsLivePreviewLoading(true)
    
    try {
      const { data: { publicUrl } } = supabase.storage
        .from('data-files')
        .getPublicUrl(`${selectedFile.user_id}/${selectedFile.filename}`)
      
      console.log('File URL:', publicUrl)
      
      const response = await fetch(publicUrl)
      const fileBlob = await response.blob()
      
      const formData = new FormData()
      formData.append('file', fileBlob, selectedFile.original_filename)
      formData.append('transformations', JSON.stringify(transformationConfig))
      
      console.log('Sending request to preview-transformation endpoint...')
      
      const previewResponse = await fetch('http://localhost:8000/custom-preprocessing/preview-transformation/', {
        method: 'POST',
        body: formData
      })
      
      console.log('Preview response status:', previewResponse.status)
      
      if (!previewResponse.ok) {
        const errorText = await previewResponse.text()
        console.error('Preview error response:', errorText)
        throw new Error('Failed to generate live preview')
      }
      
      const previewResult = await previewResponse.json()
      console.log('Preview result:', previewResult)
      
      if (previewResult.success && previewResult.preview) {
        console.log('Setting live preview with data:', {
          originalRows: previewResult.preview.original.length,
          transformedRows: previewResult.preview.transformed.length,
          originalCols: previewResult.preview.columns.original.length,
          transformedCols: previewResult.preview.columns.transformed.length
        })
        setLivePreview({
          original: previewResult.preview.original,
          transformed: previewResult.preview.transformed,
          columns: previewResult.preview.columns
        })
        console.log('Live preview state updated successfully!')
      } else {
        console.error('Invalid preview result format:', previewResult)
      }
      
    } catch (error) {
      console.error('Error generating live preview:', error)
      toast({
        variant: "destructive",
        title: "Preview Error",
        description: "Failed to generate live preview. Check console for details."
      })
    } finally {
      setIsLivePreviewLoading(false)
    }
  }, [selectedFile, transformationConfig])

  // Debounce live preview generation
  useEffect(() => {
    console.log('useEffect triggered - transformationConfig changed:', transformationConfig)
    console.log('Current step:', currentStep, 'Selected file:', selectedFile?.name || selectedFile?.original_filename)
    
    // Reduce delay to 300ms for faster feedback
    const timeoutId = setTimeout(() => {
      if (currentStep === 'configuration' && selectedFile) {
        console.log('Generating live preview after 300ms delay...')
        generateLivePreview()
      }
    }, 300) // Wait only 300ms for faster response

    return () => clearTimeout(timeoutId)
  }, [transformationConfig, selectedFile, currentStep, generateLivePreview])

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
          const newConfig = {
            ...prev,
            binning: prev.binning.filter(item => item.column !== column)
          }
          console.log('Transformation updated (binning removed):', newConfig)
          return newConfig
        } else {
          const newConfig = {
            ...prev,
            binning: [...prev.binning, { column, bins: 4 }]
          }
          console.log('Transformation updated (binning added):', newConfig)
          return newConfig
        }
      } else if (type === 'datetime_features') {
        const exists = prev.datetime_features.some(item => item.column === column)
        if (exists) {
          const newConfig = {
            ...prev,
            datetime_features: prev.datetime_features.filter(item => item.column !== column)
          }
          console.log('Transformation updated (datetime removed):', newConfig)
          return newConfig
        } else {
          const newConfig = {
            ...prev,
            datetime_features: [...prev.datetime_features, { 
              column, 
              features: ['year', 'month', 'day', 'dayofweek'] 
            }]
          }
          console.log('Transformation updated (datetime added):', newConfig)
          return newConfig
        }
      } else {
        const array = prev[type] as string[]
        if (array.includes(column)) {
          const newConfig = {
            ...prev,
            [type]: array.filter(col => col !== column)
          }
          console.log(`Transformation updated (${type} removed):`, newConfig)
          return newConfig
        } else {
          const newConfig = {
            ...prev,
            [type]: [...array, column]
          }
          console.log(`Transformation updated (${type} added):`, newConfig)
          return newConfig
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

  // STEP 1: File Selection View
  if (currentStep === 'selection') {
    return (
      <section className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Data Transformations</h1>
            <p className="text-muted-foreground">
              Select a dataset to apply mathematical and encoding transformations
            </p>
          </div>

          {/* Tutorial Alert */}
          <Alert className="mb-6">
            <Youtube className="h-4 w-4" />
            <AlertTitle>Watch Tutorial</AlertTitle>
            <AlertDescription>
              Learn how to transform your data with our{' '}
              <a 
                href="https://www.youtube.com/watch?v=znlODwYKlrI" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary hover:underline font-medium"
              >
                transformations activity video guide
              </a>
            </AlertDescription>
          </Alert>

          {/* File Selection List */}
          {files.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center p-12">
                <Database className="h-16 w-16 text-gray-400 mb-4" />
                <h3 className="text-xl font-semibold mb-2">No Files Available</h3>
                <p className="text-muted-foreground text-center mb-4">
                  Upload a dataset first to start transforming data
                </p>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Available Datasets</CardTitle>
                <CardDescription>Select a file to configure transformations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {files.map((file) => (
                    <div
                      key={file.id}
                      className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer transition-colors hover:border-gray-400 dark:hover:border-gray-600"
                      onClick={() => handleSelectFile(file)}
                    >
                      <div className="flex items-center gap-4 flex-1">
                        <FileText className="h-5 w-5 text-gray-600" />
                        <div className="flex-1 min-w-0">
                          <div className="font-medium truncate">{file.name || file.original_filename}</div>
                          <div className="text-sm text-muted-foreground">
                            Uploaded {new Date(file.upload_date).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-6 text-sm text-muted-foreground">
                        <div className="flex items-center gap-2">
                          <BarChart3 className="h-4 w-4" />
                          <span>{file.row_count.toLocaleString()} rows</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Database className="h-4 w-4" />
                          <span>{file.column_names.length} columns</span>
                        </div>
                        <div className="text-gray-400">
                          {file.file_size ? `${(file.file_size / 1024 / 1024).toFixed(2)} MB` : 'N/A'}
                        </div>
                        <Button variant="outline" size="sm">
                          Configure
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </section>
    )
  }

  // STEP 2: Configuration View with Live Preview
  return (
    <section className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="max-w-7xl mx-auto">
        {/* Header with Back Button */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button 
              variant="outline" 
              onClick={() => {
                setCurrentStep('selection')
                setSelectedFile(null)
                setTransformationConfig({
                  log_transform: [],
                  sqrt_transform: [],
                  squared_transform: [],
                  reciprocal_transform: [],
                  binning: [],
                  one_hot_encoding: [],
                  datetime_features: []
                })
                setLivePreview(null)
              }}
            >
              <ArrowUpDown className="h-4 w-4 mr-2 rotate-90" />
              Back to Files
            </Button>
            <div>
              <h1 className="text-2xl font-bold">Transform: {selectedFile?.name || selectedFile?.original_filename}</h1>
              <p className="text-sm text-muted-foreground mt-1">
                Select transformations to apply to your dataset
              </p>
            </div>
          </div>

          {getTotalTransformations() > 0 && (
            <Button 
              onClick={applyTransformations} 
              disabled={isProcessing}
              size="lg"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Applying...
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  Apply {getTotalTransformations()} Transformation{getTotalTransformations() !== 1 ? 's' : ''}
                </>
              )}
            </Button>
          )}
        </div>

        {/* Loading State */}
        {isAnalyzing && (
          <Card className="mb-6">
            <CardContent className="flex items-center justify-center p-12">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
                <p className="font-medium">Loading file data...</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main Content: Transformation Configuration */}
        {!isAnalyzing && selectedFile && (
          <div className="space-y-6">
            {/* Transformation Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Configure Transformations
                </CardTitle>
                <CardDescription>
                  Select transformations to apply to your dataset
                  {getTotalTransformations() > 0 && (
                    <Badge className="ml-2" variant="secondary">
                      {getTotalTransformations()} selected
                    </Badge>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="numeric" className="w-full">
                  <TabsList className="grid w-full grid-cols-3 mb-6">
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
              
              <TabsContent value="numeric" className="space-y-6">
                {numericColumns.length === 0 ? (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>No numeric columns found in this dataset</AlertDescription>
                  </Alert>
                ) : (
                  <>
                    {/* Log Transform */}
                    <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-900">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="h-8 w-8 rounded-full bg-gray-700 dark:bg-gray-600 text-white flex items-center justify-center text-sm font-bold">
                          log
                        </div>
                        <div>
                          <h3 className="font-semibold">Log Transform</h3>
                          <p className="text-sm text-muted-foreground">
                            Apply log(x+1) to reduce positive skewness
                          </p>
                        </div>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                        {numericColumns.map(column => (
                          <div 
                            key={column.name} 
                            className={`flex items-center space-x-3 p-3 rounded-md border transition-colors ${
                              isTransformationSelected('log_transform', column.name)
                                ? 'bg-gray-100 dark:bg-gray-800 border-gray-400 dark:border-gray-600'
                                : 'bg-white dark:bg-gray-950 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                            }`}
                          >
                            <Checkbox
                              checked={isTransformationSelected('log_transform', column.name)}
                              onCheckedChange={() => toggleTransformation('log_transform', column.name)}
                            />
                            <div className="flex-1">
                              <span className="text-sm font-medium">{column.name}</span>
                              <div className="text-xs text-muted-foreground">
                                Sample: {column.sample_values.slice(0, 2).join(', ')}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Square Root Transform */}
                    <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-900">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="h-8 w-8 rounded-full bg-gray-700 dark:bg-gray-600 text-white flex items-center justify-center text-sm font-bold">
                          √
                        </div>
                        <div>
                          <h3 className="font-semibold">Square Root Transform</h3>
                          <p className="text-sm text-muted-foreground">
                            Apply sqrt(x) for moderate skewness reduction
                          </p>
                        </div>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                        {numericColumns.map(column => (
                          <div 
                            key={column.name}
                            className={`flex items-center space-x-3 p-3 rounded-md border transition-colors ${
                              isTransformationSelected('sqrt_transform', column.name)
                                ? 'bg-gray-100 dark:bg-gray-800 border-gray-400 dark:border-gray-600'
                                : 'bg-white dark:bg-gray-950 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                            }`}
                          >
                            <Checkbox
                              checked={isTransformationSelected('sqrt_transform', column.name)}
                              onCheckedChange={() => toggleTransformation('sqrt_transform', column.name)}
                            />
                            <div className="flex-1">
                              <span className="text-sm font-medium">{column.name}</span>
                              <div className="text-xs text-muted-foreground">
                                Sample: {column.sample_values.slice(0, 2).join(', ')}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Binning */}
                    <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-900">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="h-8 w-8 rounded-full bg-gray-700 dark:bg-gray-600 text-white flex items-center justify-center">
                          <BarChart3 className="h-4 w-4" />
                        </div>
                        <div>
                          <h3 className="font-semibold">Binning</h3>
                          <p className="text-sm text-muted-foreground">
                            Convert continuous variables to categories
                          </p>
                        </div>
                      </div>
                      <div className="space-y-3 mt-3">
                        {numericColumns.map(column => (
                          <div key={column.name} className="space-y-2">
                            <div 
                              className={`flex items-center space-x-3 p-3 rounded-md border transition-colors ${
                                isTransformationSelected('binning', column.name)
                                  ? 'bg-gray-100 dark:bg-gray-800 border-gray-400 dark:border-gray-600'
                                  : 'bg-white dark:bg-gray-950 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                              }`}
                            >
                              <Checkbox
                                checked={isTransformationSelected('binning', column.name)}
                                onCheckedChange={() => toggleTransformation('binning', column.name)}
                              />
                              <div className="flex-1">
                                <span className="text-sm font-medium">{column.name}</span>
                              </div>
                              {isTransformationSelected('binning', column.name) && (
                                <Select
                                  value={(transformationConfig.binning.find(b => b.column === column.name)?.bins || 4).toString()}
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
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </TabsContent>

              <TabsContent value="categorical" className="space-y-4">
                {categoricalColumns.length === 0 ? (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>No categorical columns found in this dataset</AlertDescription>
                  </Alert>
                ) : (
                  <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-900">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-8 w-8 rounded-full bg-gray-700 dark:bg-gray-600 text-white flex items-center justify-center">
                        <Binary className="h-4 w-4" />
                      </div>
                      <div>
                        <h3 className="font-semibold">One-Hot Encoding</h3>
                        <p className="text-sm text-muted-foreground">
                          Convert categorical variables to binary columns
                        </p>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                      {categoricalColumns.map(column => (
                        <div 
                          key={column.name}
                          className={`flex items-center space-x-3 p-3 rounded-md border transition-colors ${
                            isTransformationSelected('one_hot_encoding', column.name)
                              ? 'bg-gray-100 dark:bg-gray-800 border-gray-400 dark:border-gray-600'
                              : 'bg-white dark:bg-gray-950 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                          }`}
                        >
                          <Checkbox
                            checked={isTransformationSelected('one_hot_encoding', column.name)}
                            onCheckedChange={() => toggleTransformation('one_hot_encoding', column.name)}
                          />
                          <div className="flex-1">
                            <span className="text-sm font-medium">{column.name}</span>
                            <div className="text-xs text-muted-foreground">
                              Sample: {column.sample_values.slice(0, 2).join(', ')}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="datetime" className="space-y-4">
                {dateColumns.length === 0 ? (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>No datetime columns found in this dataset</AlertDescription>
                  </Alert>
                ) : (
                  <div className="p-4 border rounded-lg bg-gray-50 dark:bg-gray-900">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-8 w-8 rounded-full bg-gray-700 dark:bg-gray-600 text-white flex items-center justify-center">
                        <Calendar className="h-4 w-4" />
                      </div>
                      <div>
                        <h3 className="font-semibold">DateTime Feature Extraction</h3>
                        <p className="text-sm text-muted-foreground">
                          Extract year, month, day, and day of week from date columns
                        </p>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                      {dateColumns.map(column => (
                        <div 
                          key={column.name}
                          className={`flex items-center space-x-3 p-3 rounded-md border transition-colors ${
                            isTransformationSelected('datetime_features', column.name)
                              ? 'bg-gray-100 dark:bg-gray-800 border-gray-400 dark:border-gray-600'
                              : 'bg-white dark:bg-gray-950 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                          }`}
                        >
                          <Checkbox
                            checked={isTransformationSelected('datetime_features', column.name)}
                            onCheckedChange={() => toggleTransformation('datetime_features', column.name)}
                          />
                          <div className="flex-1">
                            <span className="text-sm font-medium">{column.name}</span>
                            <div className="text-xs text-muted-foreground">
                              Extracts: year, month, day, day_of_week
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </TabsContent>
            </Tabs>
        </CardContent>
      </Card>

      {/* Live Data Preview at Bottom - Always Visible */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Live Preview
            {getTotalTransformations() > 0 && (
              <Badge variant="secondary" className="ml-2">
                {getTotalTransformations()} transformation{getTotalTransformations() !== 1 ? 's' : ''} applied
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            {getTotalTransformations() === 0 
              ? 'Select transformations above to see a live preview'
              : isLivePreviewLoading
                ? 'Generating preview...'
                : livePreview 
                  ? `Preview updates automatically as you select transformations`
                  : 'Generating preview...'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {getTotalTransformations() === 0 ? (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription>
                Check the boxes above to select transformations and see a live preview of your transformed data
              </AlertDescription>
            </Alert>
          ) : isLivePreviewLoading ? (
            <div className="flex items-center justify-center p-12">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                <p className="text-sm text-muted-foreground">Generating preview...</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Processing {getTotalTransformations()} transformation{getTotalTransformations() !== 1 ? 's' : ''}...
                </p>
              </div>
            </div>
          ) : livePreview ? (
              <div className="space-y-4">
                {/* Stats */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border">
                    <div className="text-sm text-muted-foreground">Original Columns</div>
                    <div className="text-2xl font-bold">
                      {livePreview.columns.original.length}
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg border">
                    <div className="text-sm text-muted-foreground">After Transform</div>
                    <div className="text-2xl font-bold">
                      {livePreview.columns.transformed.length}
                      {livePreview.columns.transformed.length > livePreview.columns.original.length && (
                        <span className="text-sm ml-1">
                          (+{livePreview.columns.transformed.length - livePreview.columns.original.length})
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Transformed Data Table with Scroll */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-semibold">Transformed Data</h4>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        {livePreview.columns.transformed.length} columns • Scroll to view all →
                      </span>
                      <Badge variant="outline">
                        {livePreview.transformed.length} rows
                      </Badge>
                    </div>
                  </div>
                  <div className="border rounded-lg overflow-auto max-h-[400px]">
                    <Table>
                      <TableHeader className="bg-gray-50 dark:bg-gray-900 sticky top-0">
                        <TableRow>
                          {livePreview.columns.transformed.map((col, index) => (
                            <TableHead key={index} className="font-semibold text-xs whitespace-nowrap">
                              {col}
                              {!livePreview.columns.original.includes(col) && (
                                <Badge variant="secondary" className="ml-1 text-xs">
                                  NEW
                                </Badge>
                              )}
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {livePreview.transformed.map((row, rowIndex) => (
                          <TableRow key={rowIndex}>
                            {livePreview.columns.transformed.map((col, colIndex) => (
                              <TableCell key={colIndex} className="text-xs whitespace-nowrap">
                                {row[col] !== null && row[col] !== undefined 
                                  ? String(row[col]).length > 50 
                                    ? String(row[col]).substring(0, 50) + '...'
                                    : String(row[col])
                                  : <span className="text-gray-400">null</span>
                                }
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>

                <Alert className="bg-gray-50 dark:bg-gray-900">
                  <CheckCircle2 className="h-4 w-4" />
                  <AlertDescription>
                    Preview updates automatically. Click "Apply" button at the top to save these transformations.
                  </AlertDescription>
                </Alert>
              </div>
            ) : (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Failed to generate preview. Please check your transformations and try again.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
    </div>
        )}

        {/* Download Success */}
        {downloadUrl && (
          <Card className="mt-6 border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle2 className="w-6 h-6" />
                Transformation Complete!
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Your transformed file is ready!</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Applied {getTotalTransformations()} transformations to {selectedFile?.name || selectedFile?.original_filename}
                  </p>
                </div>
                <div className="flex gap-3">
                  <Button asChild>
                    <a href={downloadUrl} download>
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </a>
                  </Button>
                  <Button 
                    variant="outline"
                    onClick={() => {
                      setDownloadUrl(null)
                      setLivePreview(null)
                      setTransformationConfig({
                        log_transform: [],
                        sqrt_transform: [],
                        squared_transform: [],
                        reciprocal_transform: [],
                        binning: [],
                        one_hot_encoding: [],
                        datetime_features: []
                      })
                    }}
                  >
                    New Transformation
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Processing Overlay */}
      {isProcessing && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-900 p-8 rounded-xl shadow-2xl max-w-md w-full mx-4">
            <div className="text-center">
              <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">Applying Transformations</h3>
              <p className="text-sm text-muted-foreground">
                Processing {getTotalTransformations()} transformation{getTotalTransformations() !== 1 ? 's' : ''}...
              </p>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}