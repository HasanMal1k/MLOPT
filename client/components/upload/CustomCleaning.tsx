// components/upload/CustomCleaning.tsx
'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { ArrowLeft, Loader2, Eye } from "lucide-react"

interface ColumnAnalysis {
  name: string;
  current_type: string;
  suggested_type: string;
  sample_values: string[];
}

interface FileAnalysis {
  success: boolean;
  filename: string;
  row_count: number;
  column_count: number;
  columns_info: ColumnAnalysis[];
  column_types: Record<string, string>;
  column_stats: Record<string, any>;
}

interface TransformationResult {
  success: boolean;
  message: string;
  transformed_file: string;
  report_file: string;
  report: {
    data_types: Record<string, {
      original: string;
      converted_to: string;
    }>;
    columns_dropped: string[];
    transformations_applied: string[];
  };
}

interface PreviewData {
  original: Record<string, any>[];
  transformed: Record<string, any>[];
  columns: {
    original: string[];
    transformed: string[];
  };
}

interface CustomCleaningProps {
  files: File[]
  onBack: () => void
  onContinue: (cleanedFiles: any[]) => void
  onPreviewFile: (file: File) => void
}

export default function CustomCleaning({
  files,
  onBack,
  onContinue,
  onPreviewFile
}: CustomCleaningProps) {
  const [selectedFileIndex, setSelectedFileIndex] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [fileAnalyses, setFileAnalyses] = useState<Record<string, FileAnalysis>>({})
  const [columnEdits, setColumnEdits] = useState<Record<string, Record<string, {
    newType?: string;
    drop?: boolean;
  }>>>({})
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [previewErrorMessage, setPreviewErrorMessage] = useState<string | null>(null)
  const [isApplyingTransformations, setIsApplyingTransformations] = useState(false)
  const [transformationResults, setTransformationResults] = useState<Record<string, TransformationResult>>({})
  const [previewData, setPreviewData] = useState<Record<string, PreviewData>>({})
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  const [previewNeedsUpdate, setPreviewNeedsUpdate] = useState(false)
  const [allFilesProcessed, setAllFilesProcessed] = useState(false)
  
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const latestColumnEditsRef = useRef<Record<string, any>>({})

  const currentFile = files[selectedFileIndex]
  const currentAnalysis = fileAnalyses[currentFile?.name]

  // Analyze all files when component mounts
  useEffect(() => {
    if (files.length > 0) {
      analyzeAllFiles()
    }
  }, [files])

  const analyzeAllFiles = async () => {
    setIsAnalyzing(true)
    setErrorMessage(null)
    
    const analyses: Record<string, FileAnalysis> = {}
    const edits: Record<string, Record<string, any>> = {}
    
    try {
      for (const file of files) {
        const formData = new FormData()
        formData.append('file', file)
        
        const analysisResponse = await fetch('http://localhost:8000/custom-preprocessing/analyze-file/', {
          method: 'POST',
          body: formData
        })
        
        if (!analysisResponse.ok) {
          const errorData = await analysisResponse.json()
          throw new Error(errorData.detail || 'Analysis failed')
        }
        
        const analysisResult = await analysisResponse.json()
        analyses[file.name] = analysisResult
        
        // Initialize column edits for this file
        const initialEdits: Record<string, any> = {}
        analysisResult.columns_info.forEach((col: ColumnAnalysis) => {
          initialEdits[col.name] = {
            newType: col.suggested_type,
            drop: false
          }
        })
        edits[file.name] = initialEdits
      }
      
      setFileAnalyses(analyses)
      setColumnEdits(edits)
      latestColumnEditsRef.current = edits
    } catch (error) {
      console.error('Error analyzing files:', error)
      setErrorMessage(`Error analyzing files: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const updatePreview = useCallback(async (fileName: string) => {
    if (!fileAnalyses[fileName] || isLoadingPreview) return
    
    setIsLoadingPreview(true)
    setPreviewNeedsUpdate(false)
    setPreviewErrorMessage(null)
    
    try {
      const file = files.find(f => f.name === fileName)
      if (!file) return
      
      const transformationConfig = {
        data_types: {} as Record<string, string>,
        columns_to_drop: [] as string[]
      }
      
      const currentEdits = latestColumnEditsRef.current[fileName] || {}
      
      Object.entries(currentEdits).forEach(([colName, edit]) => {
        if (edit.newType) {
          transformationConfig.data_types[colName] = edit.newType
        }
        if (edit.drop) {
          transformationConfig.columns_to_drop.push(colName)
        }
      })
      
      const formData = new FormData()
      formData.append('file', file)
      formData.append('transformations', JSON.stringify(transformationConfig))
      
      const previewResponse = await fetch('http://localhost:8000/custom-preprocessing/preview-transformation/', {
        method: 'POST',
        body: formData
      })
      
      if (!previewResponse.ok) {
        const errorData = await previewResponse.json()
        throw new Error(errorData.detail || 'Preview failed')
      }
      
      const result = await previewResponse.json()
      if (result.success && result.preview) {
        setPreviewData(prev => ({
          ...prev,
          [fileName]: result.preview
        }))
      } else {
        setPreviewErrorMessage("Couldn't generate preview with the current settings")
      }
    } catch (error) {
      console.error('Error generating preview:', error)
      setPreviewErrorMessage(`Preview error: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsLoadingPreview(false)
      
      if (previewNeedsUpdate) {
        setTimeout(() => {
          updatePreview(fileName)
        }, 100)
      }
    }
  }, [fileAnalyses, isLoadingPreview, files])

  const updateColumnEdit = (fileName: string, columnName: string, field: string, value: any) => {
    setColumnEdits(prev => {
      const newEdits = {
        ...prev,
        [fileName]: {
          ...prev[fileName],
          [columnName]: {
            ...prev[fileName]?.[columnName],
            [field]: value
          }
        }
      }
      
      latestColumnEditsRef.current = newEdits
      return newEdits
    })
    
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current)
    }
    
    if (field === 'drop') {
      setTimeout(() => {
        updatePreview(fileName)
      }, 10)
    } else {
      previewTimeoutRef.current = setTimeout(() => {
        updatePreview(fileName)
      }, 300)
    }
  }

  const applyTransformationsToFile = async (fileName: string) => {
    const file = files.find(f => f.name === fileName)
    if (!file || !fileAnalyses[fileName]) return null
    
    try {
      const transformationConfig = {
        data_types: {} as Record<string, string>,
        columns_to_drop: [] as string[]
      }
      
      const fileEdits = columnEdits[fileName] || {}
      Object.entries(fileEdits).forEach(([colName, edit]) => {
        if (edit.newType) {
          transformationConfig.data_types[colName] = edit.newType
        }
        if (edit.drop) {
          transformationConfig.columns_to_drop.push(colName)
        }
      })
      
      const formData = new FormData()
      formData.append('file', file)
      formData.append('transformations', JSON.stringify(transformationConfig))
      
      const transformResponse = await fetch('http://localhost:8000/custom-preprocessing/apply-transformations/', {
        method: 'POST',
        body: formData
      })
      
      if (!transformResponse.ok) {
        const errorData = await transformResponse.json()
        throw new Error(errorData.detail || 'Transformation failed')
      }
      
      const result = await transformResponse.json()
      return result
    } catch (error) {
      console.error(`Error applying transformations to ${fileName}:`, error)
      throw error
    }
  }

  const applyAllTransformations = async () => {
    setIsApplyingTransformations(true)
    setErrorMessage(null)
    
    try {
      const results: Record<string, TransformationResult> = {}
      
      for (const file of files) {
        const result = await applyTransformationsToFile(file.name)
        if (result) {
          results[file.name] = result
        }
      }
      
      setTransformationResults(results)
      setAllFilesProcessed(true)
      
      // Continue with the cleaned files
      onContinue(Object.values(results))
    } catch (error) {
      console.error('Error applying transformations:', error)
      setErrorMessage(`Error applying transformations: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      setIsApplyingTransformations(false)
    }
  }

  const formatDataType = (type: string) => {
    switch (type) {
      case 'int':
      case 'int64':
      case 'int32':
        return 'Integer'
      case 'float':
      case 'float64':
      case 'float32':
        return 'Decimal'
      case 'object':
      case 'string':
        return 'Text'
      case 'datetime64[ns]':
      case 'datetime':
        return 'Date & Time'
      default:
        return type
    }
  }

  const hasDroppedColumns = (fileName: string) => {
    const edits = columnEdits[fileName] || {}
    return Object.values(edits).some(edit => edit.drop)
  }

  const hasChangedTypes = (fileName: string) => {
    const analysis = fileAnalyses[fileName]
    if (!analysis) return false
    
    const edits = columnEdits[fileName] || {}
    return analysis.columns_info.some(col => {
      const edit = edits[col.name]
      return edit && edit.newType && edit.newType !== col.current_type
    })
  }

  if (isAnalyzing) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Analyzing Files for Custom Cleaning</CardTitle>
          <CardDescription>
            Please wait while we analyze your files for data type optimization
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center p-6">
          <Progress value={70} className="w-full mb-4" />
          <p className="text-center text-muted-foreground">
            Analyzing {files.length} files for custom preprocessing options...
          </p>
        </CardContent>
      </Card>
    )
  }

  if (allFilesProcessed) {
    return (
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle>Custom Cleaning Complete</CardTitle>
          <CardDescription>
            All files have been processed with your custom settings
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Object.entries(transformationResults).map(([fileName, result]) => (
              <div key={fileName} className="border rounded-lg p-4">
                <h4 className="font-medium mb-2">{fileName}</h4>
                <p className="text-sm text-muted-foreground">{result.message}</p>
                <div className="flex gap-4 mt-2 text-sm">
                  <span>Columns Dropped: {result.report.columns_dropped.length}</span>
                  <span>Types Changed: {Object.keys(result.report.data_types).length}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Custom Data Cleaning</CardTitle>
        <CardDescription>
          Customize data types and choose columns to drop before final upload
        </CardDescription>
      </CardHeader>
      <CardContent>
        {files.length > 1 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium mb-2">Select File to Configure</h3>
            <div className="flex flex-wrap gap-2">
              {files.map((file, index) => (
                <Button
                  key={index}
                  variant={selectedFileIndex === index ? "default" : "outline"}
                  onClick={() => setSelectedFileIndex(index)}
                  className="text-sm"
                >
                  {file.name}
                </Button>
              ))}
            </div>
          </div>
        )}

        {currentAnalysis && (
          <div className="space-y-6">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="flex flex-col border rounded-md p-4">
                <span className="text-sm font-medium text-muted-foreground">Rows</span>
                <span className="font-bold text-xl">{currentAnalysis.row_count}</span>
              </div>
              <div className="flex flex-col border rounded-md p-4">
                <span className="text-sm font-medium text-muted-foreground">Columns</span>
                <span className="font-bold text-xl">{currentAnalysis.column_count}</span>
              </div>
              <div className="flex flex-col border rounded-md p-4">
                <span className="text-sm font-medium text-muted-foreground">File</span>
                <span className="font-bold text-sm truncate">{currentFile.name}</span>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium mb-2">Data Type Configuration</h3>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Column Name</TableHead>
                      <TableHead>Current Type</TableHead>
                      <TableHead>New Type</TableHead>
                      <TableHead>Drop Column</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {currentAnalysis.columns_info.map((column) => (
                      <TableRow 
                        key={column.name}
                        className={`group ${columnEdits[currentFile.name]?.[column.name]?.drop ? "bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50" : "hover:bg-muted"}`}
                      >
                        <TableCell className={`font-medium ${columnEdits[currentFile.name]?.[column.name]?.drop ? "text-red-600 dark:text-red-400 line-through" : ""}`}>
                          {column.name}
                        </TableCell>
                        <TableCell className={columnEdits[currentFile.name]?.[column.name]?.drop ? "text-red-600 dark:text-red-400" : ""}>
                          {formatDataType(column.current_type)}
                        </TableCell>
                        <TableCell>
                          <Select
                            value={columnEdits[currentFile.name]?.[column.name]?.newType || column.suggested_type}
                            onValueChange={(value) => updateColumnEdit(currentFile.name, column.name, 'newType', value)}
                            disabled={columnEdits[currentFile.name]?.[column.name]?.drop}
                          >
                            <SelectTrigger className="w-full">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="string">Text (String)</SelectItem>
                              <SelectItem value="int">Integer</SelectItem>
                              <SelectItem value="float">Decimal (Float)</SelectItem>
                              <SelectItem value="datetime">Date & Time</SelectItem>
                            </SelectContent>
                          </Select>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <Checkbox 
                              checked={!!columnEdits[currentFile.name]?.[column.name]?.drop}
                              onCheckedChange={(checked) => updateColumnEdit(currentFile.name, column.name, 'drop', checked)}
                              id={`drop-${column.name}`}
                              className="data-[state=checked]:bg-red-600 data-[state=checked]:border-red-600 focus:ring-red-200"
                            />
                            <label 
                              htmlFor={`drop-${column.name}`}
                              className={`text-sm cursor-pointer select-none ${columnEdits[currentFile.name]?.[column.name]?.drop ? "text-red-600 dark:text-red-400 font-medium" : ""}`}
                            >
                              Drop
                            </label>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>

            {/* Live Preview */}
            <div className="mt-8">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-medium">Live Preview</h3>
                {isLoadingPreview && (
                  <div className="flex items-center text-sm text-amber-600">
                    <Loader2 className="animate-spin mr-2 h-4 w-4 text-amber-600" />
                    Updating preview...
                  </div>
                )}
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Preview of how your data will look after transformation (first 5 rows).
              </p>
              
              <div className={`border rounded-md overflow-x-auto transition-opacity duration-200 ${isLoadingPreview ? 'opacity-50' : ''}`}>
                {previewData[currentFile.name] ? (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {previewData[currentFile.name].columns.transformed.map((col, idx) => (
                          <TableHead key={idx}>{col}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {previewData[currentFile.name].transformed.map((row, rowIdx) => (
                        <TableRow key={rowIdx}>
                          {previewData[currentFile.name].columns.transformed.map((col, colIdx) => (
                            <TableCell key={colIdx}>
                              {row[col] !== null && row[col] !== undefined ? String(row[col]) : "null"}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="p-6 text-center text-muted-foreground">
                    {isLoadingPreview ? 
                      "Loading preview data..." : 
                      "Make changes to see a live preview."}
                  </div>
                )}
              </div>
              
              {previewErrorMessage && (
                <Alert variant="destructive" className="mt-4">
                  <AlertTitle>Preview Error</AlertTitle>
                  <AlertDescription>{previewErrorMessage}</AlertDescription>
                </Alert>
              )}
            </div>
          </div>
        )}

        {errorMessage && (
          <Alert variant="destructive" className="mt-4">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        <div className="flex gap-2">
          <Button variant="outline" onClick={onBack}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <Button variant="outline" onClick={() => onPreviewFile(currentFile)}>
            <Eye className="mr-2 h-4 w-4" />
            Preview Original
          </Button>
        </div>
        <Button 
          onClick={applyAllTransformations} 
          disabled={isApplyingTransformations}
          variant="default"
          className="bg-gray-50 "
        >
          {isApplyingTransformations ? (
            <>
              <Loader2 className="animate-spin mr-2 h-4 w-4" />
              Processing All Files...
            </>
          ) : (
            `Apply Custom Cleaning to All Files (${files.length})`
          )}
        </Button>
      </CardFooter>
    </Card>
  )
}