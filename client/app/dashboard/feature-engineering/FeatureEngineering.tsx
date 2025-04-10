'use client'

import { useState, useEffect, Suspense } from "react"
import { useSearchParams } from "next/navigation"
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
  ArrowLeft,
  Download,
  Eye,
  CheckCircle2,
  Calendar,
  SlidersHorizontal,
  Code,
  ArrowUpDown,
  Filter,
  Clock,
  Move,
  Plus,
  ChevronRight
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
import { 
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

// Define types
interface FeatureOperation {
  id: string;
  type: string;
  column: string;
  features?: string[];
  bins?: number;
  labels?: string[];
  drop_first?: boolean;
}

interface EngineeringResult {
  success: boolean;
  message: string;
  transformed_file: string;
  original_column_count: number;
  new_column_count: number;
  total_column_count: number;
  report: {
    original_columns: string[];
    new_columns: string[];
    operations_applied: string[];
  };
}

export default function FeatureEngineering() {
  // State
  const [activeTab, setActiveTab] = useState<string>("file_selection")
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [featureOperations, setFeatureOperations] = useState<FeatureOperation[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [engineeringResult, setEngineeringResult] = useState<EngineeringResult | null>(null)
  
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
                setActiveTab("feature_operations")
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
    setFeatureOperations([])
    setEngineeringResult(null)
    setActiveTab("feature_operations")
  }
  
  // Add new feature operation
  const addFeatureOperation = (type: string) => {
    if (!selectedFile) return
    
    const newOperation: FeatureOperation = {
      id: `op-${Date.now()}`,
      type,
      column: selectedFile.column_names[0] || "",
    }
    
    if (type === "datetime_features") {
      newOperation.features = ["year", "month"]
    } else if (type === "binning") {
      newOperation.bins = 3
      newOperation.labels = ["low", "medium", "high"]
    } else if (type === "one_hot_encoding") {
      newOperation.drop_first = false
    }
    
    setFeatureOperations([...featureOperations, newOperation])
  }
  
  // Update feature operation
  const updateFeatureOperation = (id: string, field: string, value: any) => {
    setFeatureOperations(prevOps => prevOps.map(op => 
      op.id === id ? { ...op, [field]: value } : op
    ))
  }
  
  // Remove feature operation
  const removeFeatureOperation = (id: string) => {
    setFeatureOperations(prevOps => prevOps.filter(op => op.id !== id))
  }
  
  // Toggle datetime feature
  const toggleDatetimeFeature = (opId: string, feature: string) => {
    setFeatureOperations(prevOps => prevOps.map(op => {
      if (op.id === opId) {
        const features = op.features || []
        return {
          ...op,
          features: features.includes(feature)
            ? features.filter(f => f !== feature)
            : [...features, feature]
        }
      }
      return op
    }))
  }
  
  // Apply feature engineering
  const applyFeatureEngineering = async () => {
    if (!selectedFile || featureOperations.length === 0) return
    
    setIsProcessing(true)
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
      formData.append('operations', JSON.stringify(featureOperations))
      
      // Send to backend for processing
      const engineeringResponse = await fetch('http://localhost:8000/custom-preprocessing/feature-engineering/', {
        method: 'POST',
        body: formData
      })
      
      if (!engineeringResponse.ok) {
        const errorData = await engineeringResponse.json()
        throw new Error(errorData.detail || 'Feature engineering failed')
      }
      
      const result = await engineeringResponse.json()
      setEngineeringResult(result)
      setActiveTab("results")
    } catch (error) {
      console.error('Error applying feature engineering:', error)
      setErrorMessage(`Error applying feature engineering: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsProcessing(false)
    }
  }
  
  // Handle file preview
  const handlePreview = () => {
    if (selectedFile) {
      setIsPreviewOpen(true)
    }
  }
  
  // Get operation icon
  const getOperationIcon = (type: string) => {
    switch (type) {
      case 'datetime_features':
        return <Calendar className="h-5 w-5" />
      case 'one_hot_encoding':
        return <Code className="h-5 w-5" />
      case 'binning':
        return <SlidersHorizontal className="h-5 w-5" />
      default:
        return <Plus className="h-5 w-5" />
    }
  }
  
  // Get operation title
  const getOperationTitle = (type: string) => {
    switch (type) {
      case 'datetime_features':
        return 'Extract Date Features'
      case 'one_hot_encoding':
        return 'One-Hot Encoding'
      case 'binning':
        return 'Numeric Binning'
      default:
        return type
    }
  }
  
  // Get column type
  const getColumnType = (column: string) => {
    if (!selectedFile) return 'unknown'
    
    const stats = selectedFile.statistics[column]
    if (!stats) return 'unknown'
    
    return stats.type || 'unknown'
  }
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-8">
        Feature Engineering
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full max-w-md mb-6">
          <TabsTrigger value="file_selection" className="flex-1">Select File</TabsTrigger>
          <TabsTrigger value="feature_operations" className="flex-1" disabled={!selectedFile}>
            Feature Operations
          </TabsTrigger>
          <TabsTrigger value="results" className="flex-1" disabled={!engineeringResult}>
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
                  <CardTitle>Select a File for Feature Engineering</CardTitle>
                  <CardDescription>
                    Choose a file to create new features for machine learning
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableCaption>Your available files for feature engineering</TableCaption>
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
        
        {/* Feature Operations Tab */}
        <TabsContent value="feature_operations" className="border-none p-0">
          {selectedFile && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Feature Engineering Operations</CardTitle>
                      <CardDescription>
                        Create new features from {selectedFile.original_filename}
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("file_selection")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-col gap-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <Card className="bg-muted/30 border-dashed cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => addFeatureOperation("datetime_features")}>
                        <CardContent className="flex flex-col items-center justify-center p-6">
                          <Calendar className="h-8 w-8 mb-2 text-blue-500" />
                          <h3 className="font-medium text-center">Extract Date Features</h3>
                          <p className="text-sm text-center text-muted-foreground mt-2">
                            Create year, month, day features from date columns
                          </p>
                        </CardContent>
                      </Card>
                      
                      <Card className="bg-muted/30 border-dashed cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => addFeatureOperation("one_hot_encoding")}>
                        <CardContent className="flex flex-col items-center justify-center p-6">
                          <Code className="h-8 w-8 mb-2 text-purple-500" />
                          <h3 className="font-medium text-center">One-Hot Encoding</h3>
                          <p className="text-sm text-center text-muted-foreground mt-2">
                            Convert categorical variables to binary columns
                          </p>
                        </CardContent>
                      </Card>
                      
                      <Card className="bg-muted/30 border-dashed cursor-pointer hover:bg-muted/50 transition-colors" onClick={() => addFeatureOperation("binning")}>
                        <CardContent className="flex flex-col items-center justify-center p-6">
                          <SlidersHorizontal className="h-8 w-8 mb-2 text-orange-500" />
                          <h3 className="font-medium text-center">Numeric Binning</h3>
                          <p className="text-sm text-center text-muted-foreground mt-2">
                            Convert numeric data into categorical bins
                          </p>
                        </CardContent>
                      </Card>
                    </div>
                    
                    <div className="mt-4">
                      <h3 className="text-lg font-medium mb-4">
                        {featureOperations.length > 0 
                          ? 'Selected Operations'
                          : 'No operations selected yet. Click on an operation type above to add it.'}
                      </h3>
                      
                      {featureOperations.map((operation, index) => (
                        <Card key={operation.id} className="mb-4">
                          <CardHeader className="pb-2">
                            <div className="flex justify-between items-center">
                              <div className="flex items-center gap-2">
                                {getOperationIcon(operation.type)}
                                <CardTitle className="text-base">{getOperationTitle(operation.type)}</CardTitle>
                              </div>
                              <Button 
                                variant="ghost" 
                                size="sm" 
                                className="text-muted-foreground"
                                onClick={() => removeFeatureOperation(operation.id)}
                              >
                                Remove
                              </Button>
                            </div>
                          </CardHeader>
                          <CardContent>
                            <div className="grid gap-4">
                              <div className="grid gap-2">
                                <Label htmlFor={`column-${operation.id}`}>Select Column</Label>
                                <Select
                                  value={operation.column}
                                  onValueChange={(value) => updateFeatureOperation(operation.id, 'column', value)}
                                >
                                  <SelectTrigger id={`column-${operation.id}`}>
                                    <SelectValue placeholder="Select column" />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {selectedFile.column_names.filter(col => {
                                      // Filter columns based on operation type
                                      const colType = getColumnType(col)
                                      if (operation.type === 'datetime_features') {
                                        return true  // All columns for now, ideally only date columns
                                      } else if (operation.type === 'one_hot_encoding') {
                                        return colType === 'categorical' // Only categorical columns
                                      } else if (operation.type === 'binning') {
                                        return colType === 'numeric' // Only numeric columns
                                      }
                                      return true
                                    }).map(col => (
                                      <SelectItem key={col} value={col}>{col}</SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>
                              
                              {operation.type === 'datetime_features' && (
                                <div className="grid gap-2">
                                  <Label>Select Features to Extract</Label>
                                  <div className="flex flex-wrap gap-3 mt-1">
                                    <div className="flex items-center space-x-2">
                                      <Checkbox 
                                        id={`year-${operation.id}`}
                                        checked={(operation.features || []).includes('year')}
                                        onCheckedChange={() => toggleDatetimeFeature(operation.id, 'year')}
                                      />
                                      <label htmlFor={`year-${operation.id}`} className="text-sm">Year</label>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                      <Checkbox 
                                        id={`month-${operation.id}`}
                                        checked={(operation.features || []).includes('month')}
                                        onCheckedChange={() => toggleDatetimeFeature(operation.id, 'month')}
                                      />
                                      <label htmlFor={`month-${operation.id}`} className="text-sm">Month</label>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                      <Checkbox 
                                        id={`day-${operation.id}`}
                                        checked={(operation.features || []).includes('day')}
                                        onCheckedChange={() => toggleDatetimeFeature(operation.id, 'day')}
                                      />
                                      <label htmlFor={`day-${operation.id}`} className="text-sm">Day</label>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                      <Checkbox 
                                        id={`dayofweek-${operation.id}`}
                                        checked={(operation.features || []).includes('dayofweek')}
                                        onCheckedChange={() => toggleDatetimeFeature(operation.id, 'dayofweek')}
                                      />
                                      <label htmlFor={`dayofweek-${operation.id}`} className="text-sm">Day of Week</label>
                                    </div>
                                  </div>
                                </div>
                              )}
                              
                              {operation.type === 'one_hot_encoding' && (
                                <div className="grid gap-2">
                                  <div className="flex items-center space-x-2">
                                    <Checkbox 
                                      id={`drop-first-${operation.id}`}
                                      checked={operation.drop_first}
                                      onCheckedChange={(checked) => 
                                        updateFeatureOperation(operation.id, 'drop_first', checked === true)
                                      }
                                    />
                                    <label htmlFor={`drop-first-${operation.id}`} className="text-sm">
                                      Drop first category (avoid dummy variable trap)
                                    </label>
                                  </div>
                                </div>
                              )}
                              
                              {operation.type === 'binning' && (
                                <div className="grid gap-4">
                                  <div className="grid gap-2">
                                    <Label htmlFor={`bins-${operation.id}`}>Number of Bins</Label>
                                    <Input
                                      id={`bins-${operation.id}`}
                                      type="number"
                                      min="2"
                                      max="10"
                                      value={operation.bins || 3}
                                      onChange={(e) => updateFeatureOperation(operation.id, 'bins', parseInt(e.target.value))}
                                      className="w-full"
                                    />
                                  </div>
                                  
                                  <div className="grid gap-2">
                                    <Label>Bin Labels (comma-separated)</Label>
                                    <Input
                                      placeholder="low, medium, high"
                                      value={(operation.labels || []).join(', ')}
                                      onChange={(e) => {
                                        const labels = e.target.value.split(',').map(l => l.trim()).filter(l => l)
                                        updateFeatureOperation(operation.id, 'labels', labels)
                                      }}
                                      className="w-full"
                                    />
                                    <p className="text-sm text-muted-foreground">
                                      Leave empty to use numeric bin labels (0, 1, 2, ...)
                                    </p>
                                  </div>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
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
                  <Button 
                    onClick={applyFeatureEngineering} 
                    disabled={featureOperations.length === 0 || isProcessing}
                  >
                    {isProcessing ? "Processing..." : "Apply Feature Engineering"}
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
          {engineeringResult && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Feature Engineering Results</CardTitle>
                      <CardDescription>
                        Results of applying feature engineering operations
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => setActiveTab("feature_operations")}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center p-6 bg-green-50 rounded-md mb-6">
                    <CheckCircle2 className="h-8 w-8 text-green-500 mr-3" />
                    <div>
                      <h3 className="font-medium">Feature Engineering Completed Successfully</h3>
                      <p className="text-sm text-gray-500">
                        {engineeringResult.new_column_count} new columns were created.
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid md:grid-cols-3 gap-4 mb-6">
                    <div className="flex flex-col border rounded-md p-4">
                      <span className="text-sm font-medium text-muted-foreground">Original Columns</span>
                      <span className="font-bold text-xl">{engineeringResult.original_column_count}</span>
                    </div>
                    <div className="flex flex-col border rounded-md p-4">
                      <span className="text-sm font-medium text-muted-foreground">New Columns</span>
                      <span className="font-bold text-xl">{engineeringResult.new_column_count}</span>
                    </div>
                    <div className="flex flex-col border rounded-md p-4">
                      <span className="text-sm font-medium text-muted-foreground">Total Columns</span>
                      <span className="font-bold text-xl">{engineeringResult.total_column_count}</span>
                    </div>
                  </div>
                  
                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value="operations">
                      <AccordionTrigger>Applied Operations</AccordionTrigger>
                      <AccordionContent>
                        <ScrollArea className="h-[200px] rounded-md border p-4">
                          <ul className="space-y-2">
                            {engineeringResult.report.operations_applied.map((operation, index) => (
                              <li key={index} className="flex items-center gap-2">
                                <CheckCircle2 className="h-4 w-4 text-green-600" />
                                <span>{operation}</span>
                              </li>
                            ))}
                          </ul>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                    
                    <AccordionItem value="columns">
                      <AccordionTrigger>New Columns Created</AccordionTrigger>
                      <AccordionContent>
                        <div className="rounded-md border">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Column Name</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {engineeringResult.report.new_columns.map((column) => (
                                <TableRow key={column}>
                                  <TableCell className="font-medium">{column}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={() => setActiveTab("feature_operations")}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back to Feature Operations
                  </Button>
                  <div className="flex gap-2">
                    <Button variant="outline">
                      <Eye className="mr-2 h-4 w-4" />
                      Preview Results
                    </Button>
                    <Button>
                      <Download className="mr-2 h-4 w-4" />
                      Download Engineered File
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