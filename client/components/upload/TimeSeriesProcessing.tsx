// components/upload/TimeSeriesProcessing.tsx
'use client'

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { ArrowLeft, Clock, CheckCircle2, AlertCircle, TrendingUp, Calendar } from "lucide-react"


interface TimeSeriesProcessingProps {
  files: File[]
  preprocessingResults: Record<string, any>
  onBack: () => void
  onComplete: (processedFiles: File[], results: Record<string, any>) => void
}

interface FileTimeSeriesConfig {
  filename: string
  dateColumn: string
  frequency: string
  columnsToExclude: string[]
  imputationMethod: string
  availableDateColumns: string[]
  allColumns: string[]
}

interface ProcessingResult {
  filename: string
  success: boolean
  processedFilename?: string
  error?: string
  statistics?: any
  processingSteps?: string[]
}

const FREQUENCY_OPTIONS = [
  { code: "D", description: "Daily (Calendar day frequency)" },
  { code: "B", description: "Business Days (Weekdays only)" },
  { code: "W", description: "Weekly" },
  { code: "M", description: "Monthly (Month end)" },
  { code: "Q", description: "Quarterly (Quarter end)" },
  { code: "A", description: "Annual (Year end)" },
  { code: "H", description: "Hourly" },
  { code: "T", description: "Minute-level" },
  { code: "S", description: "Second-level" }
]

const IMPUTATION_OPTIONS = [
  { value: "auto", label: "Auto (Smart Imputation)", description: "Intelligent combination of methods" },
  { value: "interpolate", label: "Linear Interpolation", description: "Fill gaps with linear interpolation" },
  { value: "mean", label: "Mean Value", description: "Replace with column mean" },
  { value: "median", label: "Median Value", description: "Replace with column median" },
  { value: "ffill", label: "Forward Fill", description: "Propagate last valid value forward" },
  { value: "bfill", label: "Backward Fill", description: "Use next valid value" }
]

export default function TimeSeriesProcessing({
  files,
  preprocessingResults,
  onBack,
  onComplete
}: TimeSeriesProcessingProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [fileConfigs, setFileConfigs] = useState<FileTimeSeriesConfig[]>([])
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [processingResults, setProcessingResults] = useState<ProcessingResult[]>([])
  const [currentStep, setCurrentStep] = useState(0)
  const [processingProgress, setProcessingProgress] = useState(0)

  useEffect(() => {
    if (files.length > 0) {
      analyzeTimeSeriesFiles()
    }
  }, [files])

  const analyzeTimeSeriesFiles = async () => {
    setIsAnalyzing(true)
    const configs: FileTimeSeriesConfig[] = []

    try {
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)

        try {
          const response = await fetch('http://localhost:8000/time-series/analyze-time-series/', {
            method: 'POST',
            body: formData
          })

          if (response.ok) {
            const analysis = await response.json()
            
            // Get detected date columns
            const dateColumns = analysis.date_columns || []
            
            // Also check preprocessing results for additional date columns
            const preprocessingInfo = preprocessingResults[file.name]
            const preprocessedDateColumns = preprocessingInfo?.date_columns_detected || []
            
            // Combine and deduplicate date columns
            const allDateColumns = Array.from(new Set([...dateColumns, ...preprocessedDateColumns]))
            
            // Get all columns (we'll need this for exclusion options)
            const allColumns = Object.keys(analysis.column_types || {})
            
            configs.push({
              filename: file.name,
              dateColumn: allDateColumns[0] || '', // Default to first detected date column
              frequency: 'D', // Default to daily
              columnsToExclude: [],
              imputationMethod: 'auto',
              availableDateColumns: allDateColumns,
              allColumns: allColumns.filter(col => !allDateColumns.includes(col)) // Exclude date columns from exclusion list
            })
          } else {
            throw new Error('Analysis failed')
          }
        } catch (error) {
          console.error(`Error analyzing ${file.name}:`, error)
          // Fallback configuration
          const preprocessingInfo = preprocessingResults[file.name]
          const dateColumns = preprocessingInfo?.date_columns_detected || []
          
          configs.push({
            filename: file.name,
            dateColumn: dateColumns[0] || '',
            frequency: 'D',
            columnsToExclude: [],
            imputationMethod: 'auto',
            availableDateColumns: dateColumns,
            allColumns: []
          })
        }
      }

      setFileConfigs(configs)
      setAnalysisComplete(true)
    } catch (error) {
      console.error('Error in time series analysis:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const updateFileConfig = <K extends keyof FileTimeSeriesConfig>(
    filename: string, 
    key: K, 
    value: FileTimeSeriesConfig[K]
  ) => {
    setFileConfigs(prev => prev.map(config => 
      config.filename === filename ? { ...config, [key]: value } : config
    ))
  }

  const toggleColumnExclusion = (filename: string, column: string) => {
    setFileConfigs(prev => prev.map(config => {
      if (config.filename === filename) {
        const newExcluded = config.columnsToExclude.includes(column)
          ? config.columnsToExclude.filter(col => col !== column)
          : [...config.columnsToExclude, column]
        return { ...config, columnsToExclude: newExcluded }
      }
      return config
    }))
  }

  const processTimeSeriesFiles = async () => {
    setIsProcessing(true)
    setCurrentStep(0)
    setProcessingProgress(0)
    const results: ProcessingResult[] = []

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const config = fileConfigs.find(c => c.filename === file.name)
        
        if (!config) {
          results.push({
            filename: file.name,
            success: false,
            error: 'Configuration not found'
          })
          continue
        }

        setCurrentStep(i + 1)
        setProcessingProgress((i / files.length) * 90)

        const formData = new FormData()
        formData.append('file', file)
        formData.append('date_column', config.dateColumn)
        formData.append('frequency', config.frequency)
        formData.append('imputation_method', config.imputationMethod)
        
        if (config.columnsToExclude.length > 0) {
          formData.append('drop_columns', JSON.stringify(config.columnsToExclude))
        }

        try {
          const response = await fetch('http://localhost:8000/time-series/process-time-series/', {
            method: 'POST',
            body: formData
          })

          if (response.ok) {
            const result = await response.json()
            results.push({
              filename: file.name,
              success: true,
              processedFilename: result.processed_filename,
              statistics: result.statistics,
              processingSteps: result.processing_steps
            })
          } else {
            const errorData = await response.json()
            results.push({
              filename: file.name,
              success: false,
              error: errorData.error || 'Processing failed'
            })
          }
        } catch (error) {
          results.push({
            filename: file.name,
            success: false,
            error: error instanceof Error ? error.message : 'Processing failed'
          })
        }
      }

      setProcessingProgress(100)
      setProcessingResults(results)

      // Download processed files and create File objects
      const processedFiles: File[] = []
      const timeSeriesResults: Record<string, any> = {}

      for (const result of results) {
        if (result.success && result.processedFilename) {
          try {
            const response = await fetch(`http://localhost:8000/time_series_results/${result.processedFilename}`)
            if (response.ok) {
              const blob = await response.blob()
              const processedFile = new File([blob], result.filename, { type: 'text/csv' })
              processedFiles.push(processedFile)
              
              // Store the time series processing results
              timeSeriesResults[result.filename] = {
                dataset_type: 'time_series',
                time_series_config: fileConfigs.find(c => c.filename === result.filename),
                processing_statistics: result.statistics,
                processing_steps: result.processingSteps
              }
            }
          } catch (error) {
            console.error(`Error downloading processed file for ${result.filename}:`, error)
          }
        }
      }

      // Complete the processing
      onComplete(processedFiles.length > 0 ? processedFiles : files, timeSeriesResults)

    } catch (error) {
      console.error('Error processing time series files:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  if (isAnalyzing) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Analyzing Time Series Data</CardTitle>
          <CardDescription>
            Preparing time series configuration for your files
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center p-6">
          <Progress value={60} className="w-full mb-4" />
          <p className="text-center text-muted-foreground">
            Analyzing {files.length} files for time series patterns...
          </p>
        </CardContent>
      </Card>
    )
  }

  if (isProcessing) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Processing Time Series Data</CardTitle>
          <CardDescription>
            Applying time series transformations to your files
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-lg font-medium mb-2">Processing Files</h3>
              <p className="text-sm text-muted-foreground mb-4">
                File {currentStep} of {files.length}: {currentStep > 0 ? files[currentStep - 1]?.name : ''}
              </p>
            </div>
            
            <div className="mb-8">
              <Progress value={processingProgress} className="h-3 w-full" />
              <p className="text-xs text-muted-foreground mt-2">
                {processingProgress < 100 
                  ? `Processing time series data...` 
                  : `Time series processing complete`}
              </p>
            </div>

            {processingResults.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-3">Processing Status:</h4>
                {processingResults.map((result, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm mb-2">
                    {result.success ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red-500" />
                    )}
                    <span className={result.success ? "text-green-700" : "text-red-700"}>
                      {result.filename}: {result.success ? "Processed successfully" : result.error}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-5 w-5 text-blue-500" />
          Time Series Configuration
        </CardTitle>
        <CardDescription>
          Configure time series processing settings for your data
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Alert className="mb-6">
          <TrendingUp className="h-4 w-4" />
          <AlertTitle>Time Series Processing</AlertTitle>
          <AlertDescription>
            Configure how your time series data should be processed. We'll handle date parsing, 
            frequency setting, missing value imputation, and data type conversion automatically.
          </AlertDescription>
        </Alert>

        <div className="space-y-8">
          {fileConfigs.map((config, index) => (
            <Card key={index} className="border-2">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-blue-500" />
                  {config.filename}
                </CardTitle>
                <CardDescription>
                  Configure time series processing for this file
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Date Column Selection */}
                <div>
                  <Label className="text-sm font-medium">Date/Time Column *</Label>
                  <Select
                    value={config.dateColumn}
                    onValueChange={(value) => updateFileConfig(config.filename, 'dateColumn', value)}
                  >
                    <SelectTrigger className="w-full mt-2">
                      <SelectValue placeholder="Select the date/time column" />
                    </SelectTrigger>
                    <SelectContent>
                      {config.availableDateColumns.length > 0 ? (
                        config.availableDateColumns.map(column => (
                          <SelectItem key={column} value={column}>
                            {column}
                          </SelectItem>
                        ))
                      ) : (
                        <SelectItem value="no-columns" disabled>
                          No date columns detected
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground mt-1">
                    This column will be used as the time index for your time series data
                  </p>
                </div>

                {/* Frequency Selection */}
                <div>
                  <Label className="text-sm font-medium">Time Series Frequency</Label>
                  <Select
                    value={config.frequency}
                    onValueChange={(value) => updateFileConfig(config.filename, 'frequency', value)}
                  >
                    <SelectTrigger className="w-full mt-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {FREQUENCY_OPTIONS.map(option => (
                        <SelectItem key={option.code} value={option.code}>
                          {option.code} - {option.description}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground mt-1">
                    The frequency will determine how your time series data is indexed and resampled
                  </p>
                </div>

                {/* Imputation Method */}
                <div>
                  <Label className="text-sm font-medium">Missing Value Handling</Label>
                  <Select
                    value={config.imputationMethod}
                    onValueChange={(value) => updateFileConfig(config.filename, 'imputationMethod', value)}
                  >
                    <SelectTrigger className="w-full mt-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {IMPUTATION_OPTIONS.map(option => (
                        <SelectItem key={option.value} value={option.value}>
                          <div>
                            <div className="font-medium">{option.label}</div>
                            <div className="text-xs text-muted-foreground">{option.description}</div>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground mt-1">
                    How missing values in your time series should be filled
                  </p>
                </div>

                {/* Column Exclusion */}
                {config.allColumns.length > 0 && (
                  <div>
                    <Label className="text-sm font-medium mb-3 block">
                      Columns to Exclude (Optional)
                    </Label>
                    <p className="text-xs text-muted-foreground mb-3">
                      Select columns that should be excluded from time series processing
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-40 overflow-y-auto border rounded-md p-3">
                      {config.allColumns.map(column => (
                        <div key={column} className="flex items-center space-x-2">
                          <Checkbox
                            id={`${config.filename}-${column}`}
                            checked={config.columnsToExclude.includes(column)}
                            onCheckedChange={() => toggleColumnExclusion(config.filename, column)}
                          />
                          <Label
                            htmlFor={`${config.filename}-${column}`}
                            className="text-sm truncate cursor-pointer"
                            title={column}
                          >
                            {column}
                          </Label>
                        </div>
                      ))}
                    </div>
                    {config.columnsToExclude.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        <span className="text-xs text-muted-foreground">Excluded:</span>
                        {config.columnsToExclude.map(col => (
                          <Badge key={col} variant="secondary" className="text-xs">
                            {col}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Configuration Summary */}
                <div className="bg-blue-50 rounded-lg p-3">
                  <h4 className="text-sm font-medium text-blue-800 mb-2">Processing Summary:</h4>
                  <ul className="text-xs text-blue-700 space-y-1">
                    <li>• Date column: <strong>{config.dateColumn || 'None selected'}</strong></li>
                    <li>• Frequency: <strong>{FREQUENCY_OPTIONS.find(f => f.code === config.frequency)?.description}</strong></li>
                    <li>• Missing values: <strong>{IMPUTATION_OPTIONS.find(i => i.value === config.imputationMethod)?.label}</strong></li>
                    <li>• Columns to exclude: <strong>{config.columnsToExclude.length || 'None'}</strong></li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Global Processing Information */}
        <Alert className="mt-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>What happens during processing?</AlertTitle>
          <AlertDescription>
            <ul className="mt-2 space-y-1 text-sm">
              <li>1. Convert selected date column to datetime format</li>
              <li>2. Set the date column as the time series index</li>
              <li>3. Apply the specified frequency to create regular time intervals</li>
              <li>4. Handle missing values using the selected imputation method</li>
              <li>5. Convert all remaining columns to numeric format where possible</li>
              <li>6. Remove or exclude specified columns</li>
            </ul>
          </AlertDescription>
        </Alert>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <Button 
          onClick={processTimeSeriesFiles}
          disabled={fileConfigs.some(config => !config.dateColumn)}
        >
          Process Time Series Data
          <Clock className="ml-2 h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  )
}