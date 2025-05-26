'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle,
  CardFooter 
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { BarChart3, LineChart, ScatterChart, PieChart, Activity, AlertCircle, Download, RefreshCw, X } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Bar, Scatter, Line, Pie } from 'react-chartjs-2'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
)

interface ChartViewerProps {
  fileMetadata?: any
  originalFile?: File
  isOpen: boolean
  onClose: () => void
}

interface FileAnalysis {
  success: boolean
  filename: string
  shape: { rows: number; columns: number }
  columns: Record<string, any>
  column_categories: {
    numeric: string[]
    categorical: string[]
    datetime: string[]
  }
  chart_suggestions: Array<{
    type: string
    x_axis?: string
    y_axis?: string
    category?: string
    value?: string
    description: string
  }>
}

interface ChartData {
  success: boolean
  chart_data: {
    labels?: string[]
    data: any[]
    type: string
  }
  data_points: number
}

export default function ChartViewer({ fileMetadata, originalFile, isOpen, onClose }: ChartViewerProps) {
  const [analysis, setAnalysis] = useState<FileAnalysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  
  const [chartData, setChartData] = useState<ChartData | null>(null)
  const [isGeneratingChart, setIsGeneratingChart] = useState(false)
  const [chartError, setChartError] = useState<string | null>(null)
  
  const [selectedChartType, setSelectedChartType] = useState('bar')
  const [selectedXAxis, setSelectedXAxis] = useState('')
  const [selectedYAxis, setSelectedYAxis] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('')
  const [selectedValue, setSelectedValue] = useState('')

  // Reset everything when dialog opens
  useEffect(() => {
    if (isOpen && (originalFile || fileMetadata)) {
      console.log('Dialog opened, starting analysis')
      setAnalysis(null)
      setChartData(null)
      setAnalysisError(null)
      setChartError(null)
      setSelectedChartType('bar')
      setSelectedXAxis('')
      setSelectedYAxis('')
      setSelectedCategory('')
      setSelectedValue('')
      analyzeFile()
    }
  }, [isOpen])

  const analyzeFile = async () => {
    console.log('Starting file analysis...')
    setIsAnalyzing(true)
    setAnalysisError(null)

    try {
      const formData = new FormData()
      
      if (originalFile) {
        console.log('Using original file:', originalFile.name)
        formData.append('file', originalFile)
      } else if (fileMetadata) {
        console.log('Using file metadata:', fileMetadata.original_filename)
        
        try {
          const supabase = createClient()
          const filePath = `${fileMetadata.user_id}/${fileMetadata.filename}`
          
          const { data, error } = await supabase.storage
            .from('data-files')
            .download(filePath)
          
          if (error || !data) {
            throw new Error('Failed to download file from storage')
          }
          
          const file = new File([data], fileMetadata.original_filename, { 
            type: fileMetadata.mime_type 
          })
          formData.append('file', file)
          console.log('File created from storage')
          
        } catch (storageError) {
          console.error('Storage error:', storageError)
          throw new Error(`Storage error: ${storageError.message}`)
        }
      } else {
        throw new Error('No file available')
      }

      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
      const response = await fetch(`${backendUrl}/visualization/analyze-for-charts/`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Analysis failed: ${response.status} - ${errorText}`)
      }

      const result = await response.json()
      console.log('Analysis result:', result)
      
      if (result.success) {
        setAnalysis(result)
        
        // Set default chart type
        setSelectedChartType('bar')
      } else {
        throw new Error('Analysis not successful')
      }
    } catch (error) {
      console.error('Analysis error:', error)
      setAnalysisError(error.message || 'Failed to analyze file')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const generateChart = async () => {
    console.log('Generating chart...')
    setIsGeneratingChart(true)
    setChartError(null)

    try {
      const config = {
        type: selectedChartType,
        x_axis: selectedXAxis,
        y_axis: selectedYAxis,
        category: selectedCategory,
        value: selectedValue,
        title: `${selectedChartType} Chart`
      }

      console.log('Chart config:', config)

      // Enhanced validation with helpful messages
      if (selectedChartType === 'pie') {
        if (!selectedCategory || !selectedValue) {
          throw new Error('Pie charts require both a category column (for labels) and a value column (for sizes)')
        }
        // Check if category column has reasonable number of unique values
        if (analysis.column_categories.categorical.length === 0) {
          throw new Error('No categorical columns available for pie chart. Try converting a text column to categorical first.')
        }
      } else if (selectedChartType === 'scatter') {
        if (!selectedXAxis || !selectedYAxis) {
          throw new Error('Scatter plots require both X and Y axis columns with numeric data')
        }
        // Additional validation for scatter plots - ensure numeric columns
        if (!analysis.column_categories.numeric.includes(selectedXAxis)) {
          throw new Error(`X-axis column "${selectedXAxis}" must be numeric for scatter plots. Please select a different column.`)
        }
        if (!analysis.column_categories.numeric.includes(selectedYAxis)) {
          throw new Error(`Y-axis column "${selectedYAxis}" must be numeric for scatter plots. Please select a different column.`)
        }
        if (selectedXAxis === selectedYAxis) {
          throw new Error('X and Y axis must be different columns for meaningful scatter plots')
        }
      } else if (selectedChartType === 'line') {
        if (!selectedXAxis || !selectedYAxis) {
          throw new Error('Line charts require both X and Y axis columns')
        }
        if (!analysis.column_categories.numeric.includes(selectedYAxis)) {
          throw new Error(`Y-axis column "${selectedYAxis}" should be numeric for meaningful line charts`)
        }
      } else { // bar chart
        if (!selectedXAxis || !selectedYAxis) {
          throw new Error('Bar charts require both X and Y axis columns')
        }
        if (!analysis.column_categories.numeric.includes(selectedYAxis)) {
          throw new Error(`Y-axis column "${selectedYAxis}" should be numeric for bar charts. The bars represent numeric values.`)
        }
      }

      const formData = new FormData()
      
      if (originalFile) {
        formData.append('file', originalFile)
      } else if (fileMetadata) {
        const supabase = createClient()
        const filePath = `${fileMetadata.user_id}/${fileMetadata.filename}`
        
        const { data, error } = await supabase.storage
          .from('data-files')
          .download(filePath)
        
        if (error || !data) {
          throw new Error('Failed to download file from storage. Please try refreshing the page.')
        }
        
        const file = new File([data], fileMetadata.original_filename, { 
          type: fileMetadata.mime_type 
        })
        formData.append('file', file)
      }

      formData.append('chart_config', JSON.stringify(config))

      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
      const response = await fetch(`${backendUrl}/visualization/generate-chart-data/`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Backend error response:', errorText)
        
        // Try to parse error message from backend
        try {
          const errorJson = JSON.parse(errorText)
          if (errorJson.detail) {
            throw new Error(`Chart generation failed: ${errorJson.detail}`)
          }
        } catch (parseError) {
          // If we can't parse the error, show a generic message with status
          throw new Error(`Chart generation failed (${response.status}). This might be due to incompatible data types or insufficient data for the selected chart type.`)
        }
        
        throw new Error(`Chart generation failed: ${errorText}`)
      }

      const result = await response.json()
      console.log('Chart result:', result)
      
      if (result.success && result.chart_data) {
        // Additional validation of returned data
        if (selectedChartType === 'scatter') {
          if (!Array.isArray(result.chart_data.data) || result.chart_data.data.length === 0) {
            throw new Error('No scatter plot data returned. The selected columns might not contain compatible numeric data.')
          }
          const validPoints = result.chart_data.data.filter(point => 
            point && typeof point.x !== 'undefined' && typeof point.y !== 'undefined'
          )
          if (validPoints.length === 0) {
            throw new Error('No valid data points found for scatter plot. Please check that both columns contain numeric values.')
          }
        } else if (['bar', 'line', 'pie'].includes(selectedChartType)) {
          if (!result.chart_data.labels || !result.chart_data.data || 
              result.chart_data.labels.length === 0 || result.chart_data.data.length === 0) {
            throw new Error(`No data available for ${selectedChartType} chart. The selected columns might be empty or contain incompatible data types.`)
          }
        }
        
        setChartData(result)
      } else {
        throw new Error(result.message || 'Chart generation was not successful. Please try different column combinations.')
      }
    } catch (error) {
      console.error('Chart generation error:', error)
      const errorMessage = error.message || 'Failed to generate chart'
      
      // Provide helpful suggestions based on error type
      let helpfulMessage = errorMessage
      if (errorMessage.includes('numeric')) {
        helpfulMessage += '\n\n💡 Tip: Numeric columns contain numbers and are suitable for measurements, counts, or calculations.'
      } else if (errorMessage.includes('categorical')) {
        helpfulMessage += '\n\n💡 Tip: Categorical columns contain text or categories and are suitable for grouping data.'
      } else if (errorMessage.includes('scatter')) {
        helpfulMessage += '\n\n💡 Tip: Scatter plots work best with two numeric columns that might have a relationship.'
      } else if (errorMessage.includes('pie')) {
        helpfulMessage += '\n\n💡 Tip: Pie charts need a category column (for slices) and a numeric column (for sizes).'
      }
      
      setChartError(helpfulMessage)
    } finally {
      setIsGeneratingChart(false)
    }
  }

  const applySuggestion = (suggestion) => {
    console.log('Applying suggestion:', suggestion)
    setSelectedChartType(suggestion.type)
    setSelectedXAxis(suggestion.x_axis || '')
    setSelectedYAxis(suggestion.y_axis || '')
    setSelectedCategory(suggestion.category || '')
    setSelectedValue(suggestion.value || '')
    setChartData(null)
    setChartError(null)
  }

  const handleClose = () => {
    console.log('Closing chart viewer')
    setAnalysis(null)
    setChartData(null)
    setAnalysisError(null)
    setChartError(null)
    setIsAnalyzing(false)
    setIsGeneratingChart(false)
    onClose()
  }

  const renderChart = () => {
    if (!chartData || !chartData.chart_data) return null

    const { chart_data } = chartData
    console.log('Rendering chart:', chart_data.type, chart_data)

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' as const },
        title: { display: true, text: `${selectedChartType} Chart` }
      }
    }

    const colors = [
      'rgba(255, 99, 132, 0.6)',
      'rgba(54, 162, 235, 0.6)',
      'rgba(255, 205, 86, 0.6)',
      'rgba(75, 192, 192, 0.6)',
      'rgba(153, 102, 255, 0.6)',
      'rgba(255, 159, 64, 0.6)'
    ]

    try {
      switch (chart_data.type) {
        case 'bar':
          const barData = {
            labels: chart_data.labels || [],
            datasets: [{
              label: selectedYAxis || 'Values',
              data: chart_data.data || [],
              backgroundColor: colors,
              borderColor: colors.map(c => c.replace('0.6', '1')),
              borderWidth: 1
            }]
          }
          return <Bar data={barData} options={chartOptions} />

        case 'line':
          const lineData = {
            labels: chart_data.labels || [],
            datasets: [{
              label: selectedYAxis || 'Values',
              data: chart_data.data || [],
              borderColor: 'rgb(75, 192, 192)',
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              tension: 0.1
            }]
          }
          return <Line data={lineData} options={chartOptions} />

        case 'scatter':
          // Ensure we have valid scatter data with x,y coordinates
          if (!chart_data.data || !Array.isArray(chart_data.data)) {
            return <div className="text-red-500">Invalid scatter plot data format</div>
          }
          
          // Filter and validate scatter data points
          const validScatterData = chart_data.data.filter(point => 
            point && 
            typeof point === 'object' &&
            typeof point.x !== 'undefined' && 
            typeof point.y !== 'undefined' &&
            point.x !== null && 
            point.y !== null && 
            !isNaN(Number(point.x)) && 
            !isNaN(Number(point.y))
          )
          
          console.log('Original scatter data:', chart_data.data.slice(0, 5))
          console.log('Valid scatter data points:', validScatterData.length)
          console.log('Sample valid points:', validScatterData.slice(0, 3))
          
          if (validScatterData.length === 0) {
            return <div className="text-red-500">No valid data points for scatter plot. Expected format: [{"{"}"x": value, "y": value{"}"}]</div>
          }
          
          const scatterData = {
            datasets: [{
              label: `${selectedYAxis || 'Y'} vs ${selectedXAxis || 'X'}`,
              data: validScatterData,
              backgroundColor: 'rgba(255, 99, 132, 0.6)',
              borderColor: 'rgba(255, 99, 132, 1)',
              pointRadius: 5,
              pointHoverRadius: 7
            }]
          }
          
          const scatterOptions = {
            ...chartOptions,
            scales: {
              x: {
                type: 'linear',
                position: 'bottom',
                title: {
                  display: true,
                  text: selectedXAxis || 'X Axis'
                }
              },
              y: {
                title: {
                  display: true,
                  text: selectedYAxis || 'Y Axis'
                }
              }
            }
          }
          
          return <Scatter data={scatterData} options={scatterOptions} />

        case 'pie':
          const pieData = {
            labels: chart_data.labels || [],
            datasets: [{
              data: chart_data.data || [],
              backgroundColor: colors,
              borderColor: colors.map(c => c.replace('0.6', '1')),
              borderWidth: 1
            }]
          }
          return <Pie data={pieData} options={{...chartOptions, scales: undefined}} />

        default:
          return <div className="text-red-500">Unsupported chart type: {chart_data.type}</div>
      }
    } catch (error) {
      console.error('Chart render error:', error)
      return <div className="text-red-500">Error rendering chart: {error.message}</div>
    }
  }

  if (!isOpen) return null

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-7xl max-h-[95vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Custom Charts: {fileMetadata?.original_filename || originalFile?.name}
          </DialogTitle>
          <Button variant="ghost" size="sm" onClick={handleClose}>
            <X className="h-4 w-4" />
          </Button>
        </DialogHeader>
        
        <div className="flex-1 overflow-auto">
          {isAnalyzing ? (
            <div className="p-6 text-center">
              <RefreshCw className="h-12 w-12 mx-auto mb-4 text-blue-500 animate-spin" />
              <h3 className="text-lg font-medium mb-2">Analyzing Data</h3>
              <p className="text-sm text-gray-500">Please wait...</p>
            </div>
          ) : analysisError ? (
            <Alert variant="destructive" className="m-6">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Analysis Error</AlertTitle>
              <AlertDescription>
                {analysisError}
                <Button onClick={analyzeFile} variant="outline" size="sm" className="mt-2">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry
                </Button>
              </AlertDescription>
            </Alert>
          ) : analysis ? (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 p-6">
              {/* Configuration Panel */}
              <div className="lg:col-span-1 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Dataset Info</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">{analysis.shape.rows.toLocaleString()}</div>
                        <div className="text-gray-600">Rows</div>
                      </div>
                      <div className="text-center p-3 bg-green-50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{analysis.shape.columns}</div>
                        <div className="text-gray-600">Columns</div>
                      </div>
                    </div>
                    
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Numeric:</span>
                        <Badge variant="outline">{analysis.column_categories.numeric.length}</Badge>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Categorical:</span>
                        <Badge variant="outline">{analysis.column_categories.categorical.length}</Badge>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Date:</span>
                        <Badge variant="outline">{analysis.column_categories.datetime.length}</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Simple Configuration */}
                <Card>
                  <CardHeader>
                    <CardTitle>Chart Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Chart Type - Simple buttons */}
                    <div>
                      <label className="block text-sm font-medium mb-2">Chart Type</label>
                      <div className="grid grid-cols-2 gap-2">
                        {['bar', 'line', 'scatter', 'pie'].map(type => (
                          <button
                            key={type}
                            type="button"
                            className={`p-2 text-sm border rounded capitalize ${
                              selectedChartType === type 
                                ? 'bg-blue-500 text-white border-blue-500' 
                                : 'hover:bg-gray-50'
                            }`}
                            onClick={() => {
                              console.log('Chart type selected:', type)
                              setSelectedChartType(type)
                              setChartData(null)
                              setChartError(null)
                            }}
                          >
                            {type}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Column Selection - Simple dropdowns */}
                    {selectedChartType !== 'pie' ? (
                      <>
                        <div>
                          <label className="block text-sm font-medium mb-2">X-Axis Column</label>
                          <select
                            value={selectedXAxis}
                            onChange={(e) => {
                              console.log('X-axis selected:', e.target.value)
                              setSelectedXAxis(e.target.value)
                              setChartData(null)
                              setChartError(null)
                            }}
                            className="w-full p-2 border rounded text-sm"
                          >
                            <option value="">Select column...</option>
                            {selectedChartType === 'scatter' 
                              ? analysis.column_categories.numeric?.map(col => (
                                  <option key={col} value={col}>{col} (numeric)</option>
                                ))
                              : Object.keys(analysis.columns || {}).map(col => (
                                  <option key={col} value={col}>{col}</option>
                                ))
                            }
                          </select>
                          {selectedChartType === 'scatter' && (
                            <p className="text-xs text-gray-500 mt-1">Scatter plots require numeric columns</p>
                          )}
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-2">Y-Axis Column</label>
                          <select
                            value={selectedYAxis}
                            onChange={(e) => {
                              console.log('Y-axis selected:', e.target.value)
                              setSelectedYAxis(e.target.value)
                              setChartData(null)
                              setChartError(null)
                            }}
                            className="w-full p-2 border rounded text-sm"
                          >
                            <option value="">Select column...</option>
                            {analysis.column_categories.numeric?.map(col => (
                              <option key={col} value={col}>{col}</option>
                            ))}
                          </select>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <label className="block text-sm font-medium mb-2">Category Column</label>
                          <select
                            value={selectedCategory}
                            onChange={(e) => {
                              console.log('Category selected:', e.target.value)
                              setSelectedCategory(e.target.value)
                              setChartData(null)
                              setChartError(null)
                            }}
                            className="w-full p-2 border rounded text-sm"
                          >
                            <option value="">Select column...</option>
                            {analysis.column_categories.categorical?.map(col => (
                              <option key={col} value={col}>{col}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-2">Value Column</label>
                          <select
                            value={selectedValue}
                            onChange={(e) => {
                              console.log('Value selected:', e.target.value)
                              setSelectedValue(e.target.value)
                              setChartData(null)
                              setChartError(null)
                            }}
                            className="w-full p-2 border rounded text-sm"
                          >
                            <option value="">Select column...</option>
                            {analysis.column_categories.numeric?.map(col => (
                              <option key={col} value={col}>{col}</option>
                            ))}
                          </select>
                        </div>
                      </>
                    )}

                    <Button
                      onClick={generateChart}
                      disabled={isGeneratingChart}
                      className="w-full"
                    >
                      {isGeneratingChart ? (
                        <>
                          <RefreshCw className="animate-spin mr-2 h-4 w-4" />
                          Generating...
                        </>
                      ) : (
                        'Generate Chart'
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </div>

              {/* Chart Display */}
              <div className="lg:col-span-3">
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>Chart Preview</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {chartError ? (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Chart Generation Error</AlertTitle>
                        <AlertDescription className="whitespace-pre-line">
                          {chartError}
                          <div className="mt-3 space-y-2">
                            <Button onClick={generateChart} variant="outline" size="sm">
                              <RefreshCw className="h-4 w-4 mr-2" />
                              Try Again
                            </Button>
                          </div>
                        </AlertDescription>
                      </Alert>
                    ) : chartData ? (
                      <div className="w-full h-[600px]">
                        {renderChart()}
                      </div>
                    ) : (
                      <div className="h-[600px] flex items-center justify-center text-gray-500">
                        <div className="text-center">
                          <BarChart3 className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                          <p className="text-lg font-medium">No Chart Generated</p>
                          <p className="text-sm">Configure your chart and click "Generate Chart"</p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                  
                  {chartData && (
                    <CardFooter>
                      <div className="flex items-center justify-between w-full text-sm text-gray-600">
                        <span>Data points: {chartData.data_points}</span>
                        {/* <Button variant="outline" size="sm">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button> */}
                      </div>
                    </CardFooter>
                  )}
                </Card>
              </div>
            </div>
          ) : (
            <div className="p-6 text-center">
              <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium mb-2">Ready to Analyze</h3>
              <p className="text-sm text-gray-500 mb-6">
                Click to analyze your data for chart creation
              </p>
              <Button onClick={analyzeFile}>
                <Activity className="h-4 w-4 mr-2" />
                Analyze Data
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}