// components/upload/DataTypeDetection.tsx
'use client'

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Database, Clock, AlertCircle, FileText } from "lucide-react"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"

interface DataTypeDetectionProps {
  files: File[]
  preprocessingResults: Record<string, any>
  onBack: () => void
  onContinueNormal: (files: File[], results: Record<string, any>) => void
  onContinueTimeSeries: (files: File[], results: Record<string, any>) => void
}

export default function DataTypeDetection({
  files,
  preprocessingResults,
  onBack,
  onContinueNormal,
  onContinueTimeSeries
}: DataTypeDetectionProps) {
  // Initialize with 'normal' as default for all files
  const [userSelections, setUserSelections] = useState<Record<string, 'normal' | 'time_series'>>(() => {
    const initialSelections: Record<string, 'normal' | 'time_series'> = {}
    files.forEach(file => {
      initialSelections[file.name] = 'normal'
    })
    return initialSelections
  })

  // Get file info for display
  const getFileInfo = (filename: string) => {
    const preprocessingInfo = preprocessingResults[filename]
    return {
      dateColumns: preprocessingInfo?.date_columns_detected || [],
      rowCount: preprocessingInfo?.processed_shape?.[0] || 0,
      columnCount: preprocessingInfo?.processed_shape?.[1] || 0
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

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5" />
          Select Data Type
        </CardTitle>
        <CardDescription>
          Choose the appropriate data type for each file to apply the correct processing
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Manual Selection Required</AlertTitle>
          <AlertDescription>
            Please select the data type for each file. Choose "Time Series" if your data has temporal patterns
            (dates, timestamps, sequential time-based data). Otherwise, select "Normal Dataset".
          </AlertDescription>
        </Alert>

        <div className="space-y-6">
          {files.map((file, index) => {
            const fileInfo = getFileInfo(file.name)
            
            return (
              <Card key={index} className="border-2">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">{file.name}</CardTitle>
                    <Badge variant="outline">
                      {userSelections[file.name] === 'time_series' ? 'Time Series' : 'Normal'}
                    </Badge>
                  </div>
                  <CardDescription>
                    {fileInfo.rowCount > 0 && (
                      <span>{fileInfo.rowCount.toLocaleString()} rows, {fileInfo.columnCount} columns</span>
                    )}
                    {fileInfo.dateColumns.length > 0 && (
                      <span className="ml-2 text-blue-600">
                        • Found date columns: {fileInfo.dateColumns.join(', ')}
                      </span>
                    )}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="space-y-4">
                    {/* Hints based on preprocessing */}
                    {fileInfo.dateColumns.length > 0 && (
                      <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                        <p className="text-sm text-blue-800 dark:text-blue-300">
                          💡 This file contains {fileInfo.dateColumns.length} date column(s). 
                          Consider selecting "Time Series" if your analysis involves temporal patterns.
                        </p>
                      </div>
                    )}

                    {/* User Selection */}
                    <div>
                      <h4 className="text-sm font-medium mb-3">Select Data Type:</h4>
                      <RadioGroup
                        value={userSelections[file.name]}
                        onValueChange={(value: 'normal' | 'time_series') => 
                          handleSelectionChange(file.name, value)
                        }
                      >
                        <div className="space-y-3">
                          <div className="flex items-start space-x-3 p-3 border-2 rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
                               onClick={() => handleSelectionChange(file.name, 'normal')}>
                            <RadioGroupItem value="normal" id={`normal-${index}`} className="mt-1" />
                            <Label htmlFor={`normal-${index}`} className="flex-1 cursor-pointer">
                              <div className="flex items-center gap-2 mb-1">
                                <Database className="h-4 w-4 text-green-500" />
                                <span className="font-medium">Normal Dataset</span>
                              </div>
                              <p className="text-xs text-muted-foreground">
                                Standard tabular data for classification, regression, or clustering. 
                                Includes general feature engineering and data cleaning.
                              </p>
                            </Label>
                          </div>

                          <div className="flex items-start space-x-3 p-3 border-2 rounded-lg hover:bg-accent/50 transition-colors cursor-pointer"
                               onClick={() => handleSelectionChange(file.name, 'time_series')}>
                            <RadioGroupItem value="time_series" id={`timeseries-${index}`} className="mt-1" />
                            <Label htmlFor={`timeseries-${index}`} className="flex-1 cursor-pointer">
                              <div className="flex items-center gap-2 mb-1">
                                <Clock className="h-4 w-4 text-blue-500" />
                                <span className="font-medium">Time Series Dataset</span>
                              </div>
                              <p className="text-xs text-muted-foreground">
                                Sequential temporal data for forecasting, trend analysis, or anomaly detection.
                                Includes time-aware processing, resampling, and lag features.
                              </p>
                            </Label>
                          </div>
                        </div>
                      </RadioGroup>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Summary */}
        <div className="mt-6 p-4 bg-muted/30 rounded-lg border">
          <h4 className="font-medium mb-2">Selection Summary:</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-green-500" />
              <span>
                <span className="font-medium">Normal datasets:</span>{' '}
                {Object.values(userSelections).filter(type => type === 'normal').length}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-blue-500" />
              <span>
                <span className="font-medium">Time series datasets:</span>{' '}
                {Object.values(userSelections).filter(type => type === 'time_series').length}
              </span>
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
        </Button>
      </CardFooter>
    </Card>
  )
}