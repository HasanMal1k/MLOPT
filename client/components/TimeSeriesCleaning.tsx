'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import { type FileMetadata } from '@/components/FilePreview'
import { useSearchParams } from 'next/navigation'
import FilePreview from "@/components/FilePreview"

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
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { 
  ChevronRight,
  ArrowLeft,
  Clock,
  Calendar,
  AlertTriangle,
  CheckCircle2,
  Eye,
  FileDown,
  Loader2,
  Clock3,
  BarChart2,
  AlertCircle
} from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Stepper, Step, StepTitle, StepDescription } from "@/components/ui/steppar"

export default function TimeSeriesCleaning() {
  const [activeTab, setActiveTab] = useState<string>("file_selection")
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [fileAnalysis, setFileAnalysis] = useState<any>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  
  // Time series specific states
  const [dateColumns, setDateColumns] = useState<string[]>([])
  const [selectedDateColumn, setSelectedDateColumn] = useState<string>("")
  const [timeSeriesFrequency, setTimeSeriesFrequency] = useState<string>("D")
  const [columnsToExclude, setColumnsToExclude] = useState<string[]>([])
  const [imputationMethod, setImputationMethod] = useState<string>("auto")
  const [processingResult, setProcessingResult] = useState<any>(null)
  const [processingSteps, setProcessingSteps] = useState<string[]>([])
  const [availableFrequencies, setAvailableFrequencies] = useState<any[]>([
    { code: "D", description: "Calendar day frequency" },
    { code: "B", description: "Business day frequency" },
    { code: "W", description: "Weekly frequency" },
    { code: "M", description: "Month end frequency" },
    { code: "Q", description: "Quarter end frequency" },
    { code: "A", description: "Year end frequency" },
    { code: "H", description: "Hourly frequency" },
    { code: "T", description: "Minute frequency" },
    { code: "S", description: "Second frequency" }
  ])
  
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
    // Reset all states
    setFileAnalysis(null)
    setDateColumns([])
    setSelectedDateColumn("")
    setTimeSeriesFrequency("D")
    setColumnsToExclude([])
    setImputationMethod("auto")
    setProcessingResult(null)
    setProcessingSteps([])
    setCurrentStep(1)
    setActiveTab("file_analysis")
  }
  
  // Analyze file for time series data
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
      const analysisResponse = await fetch('http://localhost:8000/time-series/analyze-time-series/', {
        method: 'POST',
        body: formData
      })
      
      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json()
        throw new Error(errorData.detail || 'Analysis failed')
      }
      
      const analysisResult = await analysisResponse.json()
      setFileAnalysis(analysisResult)
      
      // Set detected date columns
      if (analysisResult.date_columns) {
        setDateColumns(analysisResult.date_columns)
        // Set the recommended date column from analysis
        if (analysisResult.recommendations?.recommended_date_column) {
          setSelectedDateColumn(analysisResult.recommendations.recommended_date_column)
        } else if (analysisResult.date_columns.length > 0) {
          setSelectedDateColumn(analysisResult.date_columns[0])
        }
      }
      
      setCurrentStep(2)
      setActiveTab("date_column")
    } catch (error) {
      console.error('Error analyzing file:', error)
      setErrorMessage(`Error analyzing file: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Handle frequency selection
  const handleFrequencyChange = (value: string) => {
    setTimeSeriesFrequency(value)
  }
  
  // Toggle column exclusion
  const toggleColumnExclusion = (column: string) => {
    setColumnsToExclude(prev => {
      if (prev.includes(column)) {
        return prev.filter(col => col !== column)
      } else {
        return [...prev, column]
      }
    })
  }
  
  // Process time series data
  const processTimeSeries = async () => {
    if (!selectedFile || !selectedDateColumn) return
    
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
      formData.append('date_column', selectedDateColumn)
      formData.append('frequency', timeSeriesFrequency)
      
      // Add columns to drop if any
      if (columnsToExclude.length > 0) {
        formData.append('drop_columns', JSON.stringify(columnsToExclude))
      }
      
      // Set imputation method
      formData.append('imputation_method', imputationMethod)
      
      // Send to backend for processing
      const processResponse = await fetch('http://localhost:8000/time-series/process-time-series/', {
        method: 'POST',
        body: formData
      })
      
      if (!processResponse.ok) {
        const errorData = await processResponse.json()
        throw new Error(errorData.error || 'Time series processing failed')
      }
      
      const processResult = await processResponse.json()
      setProcessingResult(processResult)
      
      // Extract processing steps
      if (processResult.processing_steps) {
        setProcessingSteps(processResult.processing_steps)
      }
      
      setCurrentStep(5)
      setActiveTab("results")
    } catch (error) {
      console.error('Error processing time series:', error)
      setErrorMessage(`Error processing time series: ${error instanceof Error ? error.message : String(error)}`)
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
  
  // Download processed file
  const handleDownload = () => {
    if (processingResult && processingResult.processed_filename) {
      window.open(`http://localhost:8000/time_series_results/${processingResult.processed_filename}`, '_blank')
    }
  }

  // Handle next step in wizard
  const handleNextStep = () => {
    if (currentStep === 1) {
      analyzeFile()
    } else if (currentStep === 2) {
      setCurrentStep(3)
      setActiveTab("frequency")
    } else if (currentStep === 3) {
      setCurrentStep(4)
      setActiveTab("columns")
    } else if (currentStep === 4) {
      processTimeSeries()
    }
  }
  
  // Handle previous step in wizard
  const handlePreviousStep = () => {
    if (currentStep === 2) {
      setCurrentStep(1)
      setActiveTab("file_analysis")
    } else if (currentStep === 3) {
      setCurrentStep(2)
      setActiveTab("date_column")
    } else if (currentStep === 4) {
      setCurrentStep(3)
      setActiveTab("frequency")
    } else if (currentStep === 5) {
      setCurrentStep(4)
      setActiveTab("columns")
    }
  }
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-8">
        Time Series Data Cleaning
      </div>
      
      {/* Stepper component */}
      <div className="mb-8">
        <Stepper currentStep={currentStep}>
          <Step>
            <StepTitle>Select File</StepTitle>
            <StepDescription>Choose a time series file</StepDescription>
          </Step>
          <Step>
            <StepTitle>Analyze</StepTitle>
            <StepDescription>Analyze time series data</StepDescription>
          </Step>
          <Step>
            <StepTitle>Date Column</StepTitle>
            <StepDescription>Select time column</StepDescription>
          </Step>
          <Step>
            <StepTitle>Frequency</StepTitle>
            <StepDescription>Set time frequency</StepDescription>
          </Step>
          <Step>
            <StepTitle>Columns</StepTitle>
            <StepDescription>Configure columns</StepDescription>
          </Step>
          <Step>
            <StepTitle>Results</StepTitle>
            <StepDescription>Processing results</StepDescription>
          </Step>
        </Stepper>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="w-full max-w-4xl mb-6">
          <TabsTrigger value="file_selection" className="flex-1">Select File</TabsTrigger>
          <TabsTrigger value="file_analysis" className="flex-1" disabled={!selectedFile}>
            File Analysis
          </TabsTrigger>
          <TabsTrigger value="date_column" className="flex-1" disabled={!fileAnalysis}>
            Date Column
          </TabsTrigger>
          <TabsTrigger value="frequency" className="flex-1" disabled={!selectedDateColumn}>
            Frequency
          </TabsTrigger>
          <TabsTrigger value="columns" className="flex-1" disabled={!timeSeriesFrequency}>
            Columns
          </TabsTrigger>
          <TabsTrigger value="results" className="flex-1" disabled={!processingResult}>
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
                  <CardTitle>Select a Time Series File</CardTitle>
                  <CardDescription>
                    Choose a file that contains time-based data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Alert className="mb-6">
                    <Calendar className="h-4 w-4" />
                    <AlertTitle>Time Series Data</AlertTitle>
                    <AlertDescription>
                      Time series files should contain at least one column with dates or timestamps.
                      The system will automatically detect date columns, but you'll be able to manually select them as well.
                    </AlertDescription>
                  </Alert>
                  
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
                      <CardTitle>Analyze Time Series Data</CardTitle>
                      <CardDescription>
                        Analyze {selectedFile.original_filename} for time series data
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
                        The file has been analyzed. We detected {fileAnalysis.date_columns?.length || 0} potential date/time columns.
                      </p>
                      <Button onClick={() => {
                        setActiveTab("date_column")
                        setCurrentStep(2)
                      }}>
                        <ChevronRight className="mr-2 h-4 w-4" />
                        Continue to Date Column Selection
                      </Button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-4 py-8">
                      {isAnalyzing ? (
                        <>
                          <Progress value={50} className="w-1/2 mb-2" />
                          <p>Analyzing file for time series data, please wait...</p>
                        </>
                      ) : (
                        <>
                          <p className="text-center text-gray-500 mb-4">
                            Click the button below to analyze the file for time series data.
                          </p>
                          <Button onClick={analyzeFile} disabled={isAnalyzing}>
                            {isAnalyzing ? "Analyzing..." : "Analyze Time Series Data"}
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
                  {fileAnalysis && (
                    <Button onClick={handleNextStep}>
                      Continue
                      <ChevronRight className="ml-2 h-4 w-4" />
                    </Button>
                  )}
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
        
        {/* Date Column Selection Tab */}
        <TabsContent value="date_column" className="border-none p-0">
          {fileAnalysis && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Select Date/Time Column</CardTitle>
                      <CardDescription>
                        Choose the column that contains date or time information
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={handlePreviousStep}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {dateColumns.length > 0 ? (
                    <div className="space-y-6">
                      <Alert className="bg-blue-50 border-blue-200">
                        <Clock className="h-4 w-4 text-blue-500" />
                        <AlertTitle className="text-blue-700">Date Columns Detected</AlertTitle>
                        <AlertDescription className="text-blue-600">
                          We've automatically detected {dateColumns.length} column(s) that appear to contain date information.
                          Select the primary date/time column to use as the time series index.
                        </AlertDescription>
                      </Alert>
                      
                      <div className="grid gap-4">
                        <Label htmlFor="date-column">Date/Time Column</Label>
                        <Select 
                          value={selectedDateColumn} 
                          onValueChange={setSelectedDateColumn}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select date column" />
                          </SelectTrigger>
                          <SelectContent>
                            {dateColumns.map(column => (
                              <SelectItem key={column} value={column}>
                                {column}
                              </SelectItem>
                            ))}
                            
                            {/* Allow selection of other columns if needed */}
                            {fileAnalysis.column_count > dateColumns.length && (
                              <>
                                <div className="px-2 py-1.5 text-sm font-semibold">Other Columns</div>
                                {selectedFile?.column_names.filter(col => !dateColumns.includes(col)).map(column => (
                                  <SelectItem key={column} value={column}>
                                    {column}
                                  </SelectItem>
                                ))}
                              </>
                            )}
                          </SelectContent>
                        </Select>
                      </div>
                      
                      {selectedDateColumn && (
                        <div className="border rounded-md p-4 bg-blue-50/50">
                          <h3 className="font-medium mb-2 flex items-center gap-2">
                            <Calendar className="h-4 w-4" />
                            Selected Column: <span className="font-bold">{selectedDateColumn}</span>
                          </h3>
                          <p className="text-sm text-muted-foreground mb-4">
                            This column will be converted to a datetime format and set as the index for your time series data.
                          </p>
                          
                          {fileAnalysis.column_stats && fileAnalysis.column_stats[selectedDateColumn] && (
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="font-medium">Data Type:</span>{' '}
                                {fileAnalysis.column_types[selectedDateColumn]}
                              </div>
                              <div>
                                <span className="font-medium">Missing Values:</span>{' '}
                                {fileAnalysis.column_stats[selectedDateColumn].missing || 0}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>No Date Columns Detected</AlertTitle>
                      <AlertDescription>
                        We couldn't detect any date or time columns in your data. Please select a column
                        that contains date/time information from the list below.
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={handlePreviousStep}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                  <Button 
                    onClick={handleNextStep}
                    disabled={!selectedDateColumn}
                  >
                    Continue
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}
        </TabsContent>
        
        {/* Frequency Selection Tab */}
        <TabsContent value="frequency" className="border-none p-0">
          {fileAnalysis && selectedDateColumn && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Set Time Series Frequency</CardTitle>
                      <CardDescription>
                        Specify the frequency for time series analysis
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={handlePreviousStep}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <Alert className="bg-purple-50 border-purple-200">
                      <Clock3 className="h-4 w-4 text-purple-500" />
                      <AlertTitle className="text-purple-700">Time Series Frequency</AlertTitle>
                      <AlertDescription className="text-purple-600">
                        Selecting a frequency will ensure your time series data is properly indexed with regular intervals.
                        This is essential for time series analysis and forecasting.
                      </AlertDescription>
                    </Alert>
                    
                    <div className="grid gap-4">
                      <Label htmlFor="frequency">Frequency</Label>
                      <Select 
                        value={timeSeriesFrequency} 
                        onValueChange={handleFrequencyChange}
                      >
                        <SelectTrigger className="w-full">
                          <SelectValue placeholder="Select frequency" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableFrequencies.map(freq => (
                            <SelectItem key={freq.code} value={freq.code}>
                              {freq.code} - {freq.description}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    
                    <div className="border rounded-md p-4 bg-purple-50/50">
                      <h3 className="font-medium mb-2">Selected Frequency: {timeSeriesFrequency}</h3>
                      <p className="text-sm text-muted-foreground">
                        {availableFrequencies.find(f => f.code === timeSeriesFrequency)?.description}
                      </p>
                      <div className="mt-4 text-sm">
                        <p className="font-medium mb-1">What this means:</p>
                        <ul className="list-disc pl-5 space-y-1 text-muted-foreground">
                          <li>Your data will be resampled to this frequency</li>
                          <li>Missing time points will be created and filled with imputed values</li>
                          <li>Irregular time intervals will be standardized</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={handlePreviousStep}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                  <Button 
                    onClick={handleNextStep}
                    disabled={!timeSeriesFrequency}
                  >
                    Continue
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}
        </TabsContent>
        
        {/* Column Configuration Tab */}
        <TabsContent value="columns" className="border-none p-0">
          {fileAnalysis && selectedDateColumn && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Configure Columns</CardTitle>
                      <CardDescription>
                        Select columns to exclude and set imputation method
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={handlePreviousStep}>
                      <ArrowLeft className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <Alert className="bg-green-50 border-green-200">
                      <BarChart2 className="h-4 w-4 text-green-500" />
                      <AlertTitle className="text-green-700">Column Configuration</AlertTitle>
                      <AlertDescription className="text-green-600">
                        Select columns to exclude from processing. All remaining columns will be converted to float 
                        type when possible, and any missing values will be automatically imputed.
                      </AlertDescription>
                    </Alert>
                    
                    <div className="border rounded-md p-4">
                      <h3 className="font-medium mb-3">Exclude Columns (Optional)</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        Select any columns you want to exclude from the time series processing.
                        The date column ({selectedDateColumn}) cannot be excluded as it's required for indexing.
                      </p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-60 overflow-y-auto">
                        {selectedFile?.column_names
                          .filter(col => col !== selectedDateColumn)
                          .map(column => (
                            <div key={column} className="flex items-center space-x-2">
                              <Checkbox 
                                id={`exclude-${column}`} 
                                checked={columnsToExclude.includes(column)}
                                onCheckedChange={() => toggleColumnExclusion(column)}
                              />
                              <label 
                                htmlFor={`exclude-${column}`}
                                className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                              >
                                {column}
                              </label>
                            </div>
                          ))
                        }
                      </div>
                    </div>
                    
                    <div className="border rounded-md p-4">
                      <h3 className="font-medium mb-3">Missing Value Handling</h3>
                      <p className="text-sm text-muted-foreground mb-4">
                        Select how missing values should be handled in your time series data.
                      </p>
                      
                      <div className="grid gap-4">
                        <Label htmlFor="imputation-method">Imputation Method</Label>
                        <Select 
                          value={imputationMethod} 
                          onValueChange={setImputationMethod}
                        >
                          <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select imputation method" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="auto">Auto (Smart Imputation)</SelectItem>
                            <SelectItem value="interpolate">Linear Interpolation</SelectItem>
                            <SelectItem value="mean">Mean Value</SelectItem>
                            <SelectItem value="median">Median Value</SelectItem>
                            <SelectItem value="ffill">Forward Fill</SelectItem>
                            <SelectItem value="bfill">Backward Fill</SelectItem>
                          </SelectContent>
                        </Select>
                        
                        <div className="bg-muted/40 p-3 rounded-md text-sm">
                          {imputationMethod === "auto" && (
                            <p>Smart imputation uses a combination of methods to handle missing values based on the data pattern.</p>
                          )}
                          {imputationMethod === "interpolate" && (
                            <p>Linear interpolation estimates missing values by drawing a straight line between adjacent known values.</p>
                          )}
                          {imputationMethod === "mean" && (
                            <p>Mean imputation replaces missing values with the average of all known values in the column.</p>
                          )}
                          {imputationMethod === "median" && (
                            <p>Median imputation replaces missing values with the median of all known values in the column.</p>
                          )}
                          {imputationMethod === "ffill" && (
                            <p>Forward fill propagates the last known value forward to fill missing values.</p>
                          )}
                          {imputationMethod === "bfill" && (
                            <p>Backward fill uses the next known value to fill missing values.</p>
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <div className="border rounded-md p-4 bg-amber-50">
                      <h3 className="font-medium mb-2 flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-amber-600" />
                        Important Notes
                      </h3>
                      <ul className="space-y-2 text-sm text-muted-foreground">
                        <li>All remaining columns will be converted to float type when possible</li>
                        <li>Columns that cannot be converted to float will be excluded</li>
                        <li>Missing values will be imputed using the selected method</li>
                        <li>Data will be resampled to the selected frequency ({timeSeriesFrequency})</li>
                        <li>The date column ({selectedDateColumn}) will be set as the index</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={handlePreviousStep}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                  <Button 
                    onClick={handleNextStep}
                  >
                    Process Time Series Data
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}
        </TabsContent>
        
        {/* Results Tab */}
        <TabsContent value="results" className="border-none p-0">
          {processingResult && (
            <div className="grid gap-6">
              <Card>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle>Time Series Processing Results</CardTitle>
                      <CardDescription>
                        Your time series data has been processed successfully
                      </CardDescription>
                    </div>
                    <Button variant="ghost" size="icon" onClick={handlePreviousStep}>
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
                        Your time series data has been processed according to your configuration.
                      </p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-xl font-bold">{processingResult.statistics?.original_rows || "N/A"}</div>
                        <p className="text-xs text-muted-foreground">Original Rows</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-xl font-bold">{processingResult.statistics?.processed_rows || "N/A"}</div>
                        <p className="text-xs text-muted-foreground">Processed Rows</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-xl font-bold">{processingResult.statistics?.missing_values_handled || "N/A"}</div>
                        <p className="text-xs text-muted-foreground">Missing Values Handled</p>
                      </CardContent>
                    </Card>
                  </div>
                  
                  <div className="border rounded-md p-4 mb-6">
                    <h3 className="font-medium mb-3">Processing Steps</h3>
                    <div className="bg-muted/20 p-4 rounded-md max-h-60 overflow-y-auto">
                      <ol className="list-decimal space-y-2 pl-5">
                        {processingSteps.map((step, index) => (
                          <li key={index} className="text-sm">{step}</li>
                        ))}
                      </ol>
                    </div>
                  </div>
                  
                  {processingResult.statistics?.columns_with_issues && 
                   processingResult.statistics.columns_with_issues.length > 0 && (
                    <Alert variant="destructive" className="mb-6">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertTitle>Issues Detected</AlertTitle>
                      <AlertDescription>
                        <p>The following columns could not be converted to float type:</p>
                        <ul className="list-disc pl-5 mt-2">
                          {processingResult.statistics.columns_with_issues.map((col: string) => (
                            <li key={col}>{col}</li>
                          ))}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
                <CardFooter className="flex justify-between">
                  <Button variant="outline" onClick={handlePreviousStep}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                  <Button onClick={handleDownload}>
                    <FileDown className="mr-2 h-4 w-4" />
                    Download Processed Data
                  </Button>
                </CardFooter>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
      
      {errorMessage && (
        <Alert variant="destructive" className="mt-6">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      )}
      
      {/* Loading Overlay */}
      {(isProcessing) && (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <h3 className="text-lg font-medium">Processing Time Series Data</h3>
              <Progress value={70} className="w-full h-2" />
              <p className="text-sm text-muted-foreground text-center">
                Please wait while we process your time series data...
              </p>
            </div>
          </div>
        </div>
      )}
      
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