// components/upload/DataTypeDetection.tsx
'use client'

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Database, Clock, AlertCircle, CheckCircle2, TrendingUp } from "lucide-react"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"

interface DataTypeDetectionProps {
  files: File[]
  preprocessingResults: Record<string, any>
  onBack: () => void
  onContinueNormal: (files: File[], results: Record<string, any>) => void
  onContinueTimeSeries: (files: File[], results: Record<string, any>) => void
}

interface FileAnalysis {
  filename: string
  detectedType: 'normal' | 'time_series' | 'uncertain'
  confidence: number
  dateColumns: string[]
  totalColumns: number
  totalRows: number
  reasons: string[]
}

export default function DataTypeDetection({
  files,
  preprocessingResults,
  onBack,
  onContinueNormal,
  onContinueTimeSeries
}: DataTypeDetectionProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<FileAnalysis[]>([])
  const [userSelections, setUserSelections] = useState<Record<string, 'normal' | 'time_series'>>({})
  const [analysisComplete, setAnalysisComplete] = useState(false)

  const analyzeDataTypes = async () => {
    setIsAnalyzing(true)
    const results: FileAnalysis[] = []

    try {
      for (const file of files) {
        // Send file to time series analysis endpoint
        const formData = new FormData()
        formData.append('file', file)

        try {
          const response = await fetch('http://localhost:8000/time-series/analyze-time-series/', {
            method: 'POST',
            body: formData
          })

          if (response.ok) {
            const analysisResult = await response.json()
            
            const dateColumns = analysisResult.date_columns || []
            const isTimeSeries = analysisResult.recommendations?.is_time_series || false
            
            // Calculate confidence based on multiple factors
            let confidence = 0
            const reasons: string[] = []

            if (dateColumns.length > 0) {
              confidence += 40
              reasons.push(`Found ${dateColumns.length} date column(s): ${dateColumns.join(', ')}`)
            }

            // Check if preprocessing detected date columns
            const preprocessingInfo = preprocessingResults[file.name]
            if (preprocessingInfo?.date_columns_detected?.length > 0) {
              confidence += 30
              reasons.push(`Preprocessing detected ${preprocessingInfo.date_columns_detected.length} date columns`)
            }

            // Check for time-related column names
            const timeRelatedColumns = analysisResult.recommendations?.date_columns || []
            if (timeRelatedColumns.length > 0) {
              confidence += 20
              reasons.push('Column names suggest temporal data')
            }

            // Check data shape (time series often have many rows)
            if (analysisResult.row_count > 100) {
              confidence += 10
              reasons.push('Dataset size suitable for time series analysis')
            }

            let detectedType: 'normal' | 'time_series' | 'uncertain'
            if (confidence >= 70) {
              detectedType = 'time_series'
            } else if (confidence >= 30) {
              detectedType = 'uncertain'
            } else {
              detectedType = 'normal'
              if (reasons.length === 0) {
                reasons.push('No clear temporal patterns detected')
              }
            }

            results.push({
              filename: file.name,
              detectedType,
              confidence,
              dateColumns,
              totalColumns: analysisResult.column_count,
              totalRows: analysisResult.row_count,
              reasons
            })
          } else {
            throw new Error('Analysis failed')
          }
        } catch (error) {
          console.error(`Error analyzing ${file.name}:`, error)
          // Fallback analysis based on preprocessing results
          const preprocessingInfo = preprocessingResults[file.name]
          const dateColumns = preprocessingInfo?.date_columns_detected || []
          
          results.push({
            filename: file.name,
            detectedType: dateColumns.length > 0 ? 'time_series' : 'normal',
            confidence: dateColumns.length > 0 ? 60 : 80,
            dateColumns,
            totalColumns: 0,
            totalRows: 0,
            reasons: dateColumns.length > 0 
              ? ['Date columns detected during preprocessing'] 
              : ['No temporal patterns detected']
          })
        }
      }

      setAnalysisResults(results)
      
      // Initialize user selections with detected types
      const initialSelections: Record<string, 'normal' | 'time_series'> = {}
      results.forEach(result => {
        initialSelections[result.filename] = result.detectedType === 'uncertain' 
          ? 'normal' 
          : result.detectedType
      })
      setUserSelections(initialSelections)
      
      setAnalysisComplete(true)
    } catch (error) {
      console.error('Error in data type analysis:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleSelectionChange = (filename: string, type: 'normal' | 'time_series') => {
    setUserSelections(prev => ({
      ...prev,
      [filename]: type
    }))
  }

  const handleContinue = () => {
    // Group files by selected type
    const normalFiles: File[] = []
    const timeSeriesFiles: File[] = []
    const normalResults: Record<string, any> = {}
    const timeSeriesResults: Record<string, any> = {}

    files.forEach(file => {
      const selection = userSelections[file.name]
      if (selection === 'time_series') {
        timeSeriesFiles.push(file)
        if (preprocessingResults[file.name]) {
          timeSeriesResults[file.name] = preprocessingResults[file.name]
        }
      } else {
        normalFiles.push(file)
        if (preprocessingResults[file.name]) {
          normalResults[file.name] = preprocessingResults[file.name]
        }
      }
    })

    // Process time series files first if any
    if (timeSeriesFiles.length > 0) {
      onContinueTimeSeries(timeSeriesFiles, timeSeriesResults)
    } else if (normalFiles.length > 0) {
      onContinueNormal(normalFiles, normalResults)
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 70) return "text-green-600"
    if (confidence >= 40) return "text-yellow-600"
    return "text-red-600"
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'time_series':
        return <Clock className="h-4 w-4 text-blue-500" />
      case 'normal':
        return <Database className="h-4 w-4 text-green-500" />
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
    }
  }

  if (!analysisComplete) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Analyzing Data Types</CardTitle>
          <CardDescription>
            Determining whether your data is suitable for time series analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!isAnalyzing ? (
            <div className="text-center p-8">
              <TrendingUp className="h-12 w-12 mx-auto mb-4 text-blue-500" />
              <h3 className="text-lg font-medium mb-2">Data Type Detection</h3>
              <p className="text-sm text-muted-foreground mb-6 max-w-2xl mx-auto">
                We'll analyze your preprocessed data to determine if it contains time series patterns.
                This helps us apply the most appropriate processing techniques.
              </p>
              
              <div className="bg-blue-50 rounded-lg p-4 mb-6 text-left max-w-md mx-auto">
                <h4 className="font-medium text-blue-800 mb-2">What we're looking for:</h4>
                <ul className="text-sm text-blue-700 space-y-1">
                  <li>• Date or timestamp columns</li>
                  <li>• Temporal patterns in data</li>
                  <li>• Time-based column names</li>
                  <li>• Sequential data structure</li>
                </ul>
              </div>
              
              <Button onClick={analyzeDataTypes} size="lg" className="gap-2">
                <TrendingUp className="h-4 w-4" />
                Analyze Data Types
              </Button>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="text-center">
                <h3 className="text-lg font-medium mb-2">Analyzing Your Data</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Please wait while we analyze {files.length} files for time series patterns
                </p>
              </div>
              
              <div className="mb-8">
                <Progress value={70} className="h-3 w-full" />
                <p className="text-xs text-muted-foreground mt-2">
                  Detecting temporal patterns and date columns...
                </p>
              </div>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <Button disabled>
            Continue
          </Button>
        </CardFooter>
      </Card>
    )
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-green-500" />
          Data Type Analysis Complete
        </CardTitle>
        <CardDescription>
          Review and confirm the detected data types for your files
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Review Required</AlertTitle>
          <AlertDescription>
            Please review the detected data types below. You can override our suggestions if needed.
            Time series data will receive specialized temporal processing.
          </AlertDescription>
        </Alert>

        <div className="space-y-6">
          {analysisResults.map((result, index) => (
            <Card key={index} className="border-2">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">{result.filename}</CardTitle>
                  <div className="flex items-center gap-2">
                    {getTypeIcon(result.detectedType)}
                    <Badge 
                      variant="outline" 
                      className={`${getConfidenceColor(result.confidence)} border-current`}
                    >
                      {result.confidence}% confidence
                    </Badge>
                  </div>
                </div>
                <CardDescription>
                  {result.totalRows > 0 && (
                    <span>{result.totalRows.toLocaleString()} rows, {result.totalColumns} columns</span>
                  )}
                  {result.dateColumns.length > 0 && (
                    <span className="ml-2">• Date columns: {result.dateColumns.join(', ')}</span>
                  )}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-4">
                  {/* Detection Reasons */}
                  <div>
                    <h4 className="text-sm font-medium mb-2">Detection Reasoning:</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      {result.reasons.map((reason, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="text-blue-500 mt-1">•</span>
                          <span>{reason}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* User Selection */}
                  <div>
                    <h4 className="text-sm font-medium mb-3">Data Type Selection:</h4>
                    <RadioGroup
                      value={userSelections[result.filename]}
                      onValueChange={(value: 'normal' | 'time_series') => 
                        handleSelectionChange(result.filename, value)
                      }
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="normal" id={`normal-${index}`} />
                        <Label htmlFor={`normal-${index}`} className="flex items-center gap-2">
                          <Database className="h-4 w-4 text-green-500" />
                          <div>
                            <span className="font-medium">Normal Dataset</span>
                            <p className="text-xs text-muted-foreground">
                              Standard data processing and feature engineering
                            </p>
                          </div>
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="time_series" id={`timeseries-${index}`} />
                        <Label htmlFor={`timeseries-${index}`} className="flex items-center gap-2">
                          <Clock className="h-4 w-4 text-blue-500" />
                          <div>
                            <span className="font-medium">Time Series Dataset</span>
                            <p className="text-xs text-muted-foreground">
                              Temporal data processing with time-aware cleaning
                            </p>
                          </div>
                        </Label>
                      </div>
                    </RadioGroup>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Summary */}
        <div className="mt-6 p-4 bg-muted/30 rounded-lg">
          <h4 className="font-medium mb-2">Processing Summary:</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">Normal datasets:</span>{' '}
              {Object.values(userSelections).filter(type => type === 'normal').length}
            </div>
            <div>
              <span className="font-medium">Time series datasets:</span>{' '}
              {Object.values(userSelections).filter(type => type === 'time_series').length}
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <Button onClick={handleContinue} className="gap-2">
          Continue Processing
          <Database className="h-4 w-4" />
        </Button>
      </CardFooter>
    </Card>
  )
}