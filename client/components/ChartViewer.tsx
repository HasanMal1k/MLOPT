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
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
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
  config_used: any
  data_points: number
}

export default function ChartViewer({ fileMetadata, originalFile, isOpen, onClose }: ChartViewerProps) {
  const [analysis, setAnalysis] = useState<FileAnalysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  
  const [selectedChart, setSelectedChart] = useState<any>(null)
  const [chartData, setChartData] = useState<ChartData | null>(null)
  const [isGeneratingChart, setIsGeneratingChart] = useState(false)
  const [chartError, setChartError] = useState<string | null>(null)
  
  const [customChartConfig, setCustomChartConfig] = useState({
    type: 'bar',
    x_axis: '',
    y_axis: '',
    category: '',
    value: '',
    title: 'Custom Chart'
  })

  // Get backend URL from environment or use default
  const getBackendUrl = () => {
    return process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  }

  // Analyze file when component opens
  useEffect(() => {
    if (isOpen && (originalFile || fileMetadata)) {
      analyzeFileForCharts()
    }
  }, [isOpen, originalFile, fileMetadata])

  const analyzeFileForCharts = async () => {
    setIsAnalyzing(true)
    setAnalysisError(null)

    try {
      const formData = new FormData()
      
      if (originalFile) {
        console.log('Using original file:', originalFile.name)
        formData.append('file', originalFile)
      } else if (fileMetadata) {
        console.log('Using file metadata:', fileMetadata.original_filename)
        
        // Try to get file from Supabase storage
        try {
          const supabase = createClient()
          const filePath = `${fileMetadata.user_id}/${fileMetadata.filename}`
          
          console.log('Attempting to download from storage path:', filePath)
          
          const { data, error } = await supabase.storage
            .from('data-files')
            .download(filePath)
          
          if (error) {
            throw new Error(`Storage download error: ${error.message}`)
          }
          
          if (!data) {
            throw new Error('No data received from storage')
          }
          
          const file = new File([data], fileMetadata.original_filename, { 
            type: fileMetadata.mime_type 
          })
          formData.append('file', file)
          console.log('Successfully created file from storage data')
          
        } catch (storageError) {
          console.error('Storage error:', storageError)
          throw new Error(`Failed to fetch file from storage: ${storageError.message}`)
        }
      } else {
        throw new Error('No file available for analysis')
      }

      const backendUrl = getBackendUrl()
      const analyzeUrl = `${backendUrl}/visualization/analyze-for-charts/`
      
      console.log('Sending request to:', analyzeUrl)

      const response = await fetch(analyzeUrl, {
        method: 'POST',
        body: formData
      })

      console.log('Response status:', response.status)
      console.log('Response headers:', response.headers)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Error response:', errorText)
        throw new Error(`Analysis failed (${response.status}): ${response.statusText}. ${errorText}`)
      }

      const result = await response.json()
      console.log('Analysis result:', result)
      
      if (result.success) {
        setAnalysis(result)
        
        // Set default chart config from first suggestion
        if (result.chart_suggestions && result.chart_suggestions.length > 0) {
          const firstSuggestion = result.chart_suggestions[0]
          setCustomChartConfig({
            type: firstSuggestion.type,
            x_axis: firstSuggestion.x_axis || '',
            y_axis: firstSuggestion.y_axis || '',
            category: firstSuggestion.category || '',
            value: firstSuggestion.value || '',
            title: firstSuggestion.description || 'Custom Chart'
          })
        }
      } else {
        throw new Error('Analysis was not successful')
      }
    } catch (error) {
      console.error('Chart analysis error:', error)
      setAnalysisError(error instanceof Error ? error.message : 'Failed to analyze file for charts')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const generateChart = async (config = customChartConfig) => {
    if (!originalFile && !fileMetadata) return

    console.log('Starting chart generation with config:', config)
    setIsGeneratingChart(true)
    setChartError(null)

    try {
      const formData = new FormData()
      
      if (originalFile) {
        console.log('Using original file for chart generation:', originalFile.name)
        formData.append('file', originalFile)
      } else if (fileMetadata) {
        console.log('Using file metadata for chart generation:', fileMetadata.original_filename)
        
        // Get file from Supabase storage (same logic as analysis)
        try {
          const supabase = createClient()
          const filePath = `${fileMetadata.user_id}/${fileMetadata.filename}`
          
          console.log('Downloading from Supabase path:', filePath)
          
          const { data, error } = await supabase.storage
            .from('data-files')
            .download(filePath)
          
          if (error) {
            throw new Error(`Storage download error: ${error.message}`)
          }
          
          const file = new File([data], fileMetadata.original_filename, { 
            type: fileMetadata.mime_type 
          })
          formData.append('file', file)
          console.log('Successfully created file from Supabase data')
          
        } catch (storageError) {
          console.error('Storage error during chart generation:', storageError)
          throw new Error(`Failed to fetch file from storage: ${storageError.message}`)
        }
      }

      formData.append('chart_config', JSON.stringify(config))

      const backendUrl = getBackendUrl()
      const chartUrl = `${backendUrl}/visualization/generate-chart-data/`
      
      console.log('Generating chart with config:', config)
      console.log('Sending request to:', chartUrl)

      const response = await fetch(chartUrl, {
        method: 'POST',
        body: formData
      })

      console.log('Chart generation response status:', response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Chart generation error response:', errorText)
        
        // Try to parse as JSON first
        try {
          const errorJson = JSON.parse(errorText)
          throw new Error(errorJson.detail || `Chart generation failed (${response.status})`)
        } catch {
          throw new Error(`Chart generation failed (${response.status}): ${response.statusText}`)
        }
      }

      const result = await response.json()
      console.log('Chart generation result:', result)
      
      if (result.success) {
        // Validate the chart data structure
        if (!result.chart_data) {
          throw new Error('No chart data returned from backend')
        }
        
        console.log('Chart data type:', result.chart_data.type)
        console.log('Chart data points:', result.chart_data.data)
        console.log('Chart labels:', result.chart_data.labels)
        
        // Additional validation based on chart type
        if (result.chart_data.type === 'scatter') {
          if (!Array.isArray(result.chart_data.data) || result.chart_data.data.length === 0) {
            throw new Error('Scatter plot requires data points')
          }
          // Check if scatter data has x,y format
          const firstPoint = result.chart_data.data[0]
          if (!firstPoint || typeof firstPoint.x === 'undefined' || typeof firstPoint.y === 'undefined') {
            throw new Error('Scatter plot data points must have x and y values')
          }
        } else if (['bar', 'line', 'pie'].includes(result.chart_data.type)) {
          if (!Array.isArray(result.chart_data.labels) || !Array.isArray(result.chart_data.data)) {
            throw new Error(`${result.chart_data.type} chart requires labels and data arrays`)
          }
          if (result.chart_data.labels.length !== result.chart_data.data.length) {
            console.warn('Labels and data arrays have different lengths')
          }
        }
        
        setChartData(result)
        console.log('Chart data set successfully')
      } else {
        throw new Error('Chart generation was not successful')
      }
    } catch (error) {
      console.error('Chart generation error:', error)
      setChartError(error instanceof Error ? error.message : 'Failed to generate chart')
    } finally {
      setIsGeneratingChart(false)
    }
  }

  const applyChartSuggestion = (suggestion: any) => {
    console.log('Applying chart suggestion:', suggestion)
    const newConfig = {
      type: suggestion.type,
      x_axis: suggestion.x_axis || '',
      y_axis: suggestion.y_axis || '',
      category: suggestion.category || '',
      value: suggestion.value || '',
      title: suggestion.description || 'Custom Chart'
    }
    console.log('New config from suggestion:', newConfig)
    setCustomChartConfig(newConfig)
    setChartData(null) // Clear previous chart
    setChartError(null)
    
    // Auto-generate chart if configuration is valid
    const validation = validateChartConfig(newConfig)
    if (validation.valid) {
      generateChart(newConfig)
    } else {
      setChartError(validation.error)
    }
  }

  // Add validation functions
  const validateChartConfig = (config: typeof customChartConfig) => {
    console.log('Validating chart config:', config)
    
    if (!config.type) {
      return { valid: false, error: 'Please select a chart type' }
    }

    if (config.type === 'pie') {
      if (!config.category) {
        return { valid: false, error: 'Please select a category column for pie chart' }
      }
      if (!config.value) {
        return { valid: false, error: 'Please select a value column for pie chart' }
      }
    } else {
      // For bar, line, scatter charts
      if (!config.x_axis) {
        return { valid: false, error: 'Please select an X-axis column' }
      }
      if (!config.y_axis) {
        return { valid: false, error: 'Please select a Y-axis column' }
      }
    }

    console.log('Configuration is valid')
    return { valid: true, error: null }
  }

  const isConfigurationValid = () => {
    return validateChartConfig(customChartConfig).valid
  }

  const getConfigurationStatus = () => {
    const validation = validateChartConfig(customChartConfig)
    if (validation.valid) {
      return '✅ Configuration ready'
    } else {
      return `❌ ${validation.error}`
    }
  }

  const testChartWithSampleData = () => {
    console.log('Testing chart with sample data')
    
    // Create sample data based on chart type
    const sampleData = {
      bar: {
        success: true,
        chart_data: {
          type: 'bar',
          labels: ['A', 'B', 'C', 'D', 'E'],
          data: [12, 19, 3, 5, 2]
        },
        config_used: customChartConfig,
        data_points: 5
      },
      line: {
        success: true,
        chart_data: {
          type: 'line',
          labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
          data: [65, 59, 80, 81, 56]
        },
        config_used: customChartConfig,
        data_points: 5
      },
      scatter: {
        success: true,
        chart_data: {
          type: 'scatter',
          data: [
            { x: 10, y: 20 },
            { x: 15, y: 25 },
            { x: 20, y: 30 },
            { x: 25, y: 35 },
            { x: 30, y: 40 }
          ]
        },
        config_used: customChartConfig,
        data_points: 5
      },
      pie: {
        success: true,
        chart_data: {
          type: 'pie',
          labels: ['Red', 'Blue', 'Yellow', 'Green'],
          data: [12, 19, 3, 5]
        },
        config_used: customChartConfig,
        data_points: 4
      }
    }

    const testData = sampleData[customChartConfig.type] || sampleData.bar
    console.log('Setting test chart data:', testData)
    
    setChartData(testData)
    setChartError(null)
  }

  const renderChart = () => {
    if (!chartData) {
      console.log('No chart data available')
      return null
    }

    console.log('Rendering chart with data:', chartData)
    console.log('Chart type:', chartData.chart_data.type)
    console.log('Chart data points:', chartData.chart_data.data)
    console.log('Chart labels:', chartData.chart_data.labels)

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: true,
          text: customChartConfig.title,
        },
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: customChartConfig.x_axis || 'X Axis'
          }
        },
        y: {
          display: true,
          title: {
            display: true,
            text: customChartConfig.y_axis || 'Y Axis'
          }
        }
      }
    }

    const chartColors = {
      backgroundColor: [
        'rgba(255, 99, 132, 0.6)',
        'rgba(54, 162, 235, 0.6)',
        'rgba(255, 205, 86, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(153, 102, 255, 0.6)',
        'rgba(255, 159, 64, 0.6)',
        'rgba(201, 203, 207, 0.6)',
        'rgba(255, 99, 255, 0.6)',
      ],
      borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 205, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(201, 203, 207, 1)',
        'rgba(255, 99, 255, 1)',
      ],
    }

    try {
      switch (chartData.chart_data.type) {
        case 'bar':
          if (!chartData.chart_data.labels || !chartData.chart_data.data) {
            console.error('Bar chart missing labels or data')
            return <div className="text-red-500">Bar chart data is incomplete</div>
          }
          
          const barData = {
            labels: chartData.chart_data.labels,
            datasets: [
              {
                label: customChartConfig.y_axis || 'Values',
                data: chartData.chart_data.data,
                backgroundColor: chartColors.backgroundColor.slice(0, chartData.chart_data.data.length),
                borderColor: chartColors.borderColor.slice(0, chartData.chart_data.data.length),
                borderWidth: 1,
              },
            ],
          }
          
          console.log('Bar chart data:', barData)
          return <Bar data={barData} options={chartOptions} />

        case 'scatter':
          if (!chartData.chart_data.data || chartData.chart_data.data.length === 0) {
            console.error('Scatter plot missing data points')
            return <div className="text-red-500">Scatter plot data is empty</div>
          }
          
          // Validate scatter data format
          const validScatterData = chartData.chart_data.data.filter(point => 
            point && typeof point.x !== 'undefined' && typeof point.y !== 'undefined' &&
            point.x !== null && point.y !== null && !isNaN(point.x) && !isNaN(point.y)
          )
          
          if (validScatterData.length === 0) {
            console.error('No valid scatter plot data points')
            return <div className="text-red-500">No valid data points for scatter plot</div>
          }
          
          const scatterData = {
            datasets: [
              {
                label: `${customChartConfig.y_axis} vs ${customChartConfig.x_axis}`,
                data: validScatterData,
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                borderColor: 'rgba(255, 99, 132, 1)',
                pointRadius: 4,
              },
            ],
          }
          
          console.log('Scatter chart data:', scatterData)
          return <Scatter data={scatterData} options={chartOptions} />

        case 'line':
          if (!chartData.chart_data.labels || !chartData.chart_data.data) {
            console.error('Line chart missing labels or data')
            return <div className="text-red-500">Line chart data is incomplete</div>
          }
          
          // Filter out null/undefined values
          const validIndices = chartData.chart_data.data
            .map((value, index) => value !== null && value !== undefined && !isNaN(value) ? index : -1)
            .filter(index => index !== -1)
          
          if (validIndices.length === 0) {
            console.error('No valid data points for line chart')
            return <div className="text-red-500">No valid data points for line chart</div>
          }
          
          const lineData = {
            labels: validIndices.map(i => chartData.chart_data.labels[i]),
            datasets: [
              {
                label: customChartConfig.y_axis || 'Values',
                data: validIndices.map(i => chartData.chart_data.data[i]),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                fill: false,
              },
            ],
          }
          
          console.log('Line chart data:', lineData)
          return <Line data={lineData} options={chartOptions} />

        case 'pie':
          if (!chartData.chart_data.labels || !chartData.chart_data.data) {
            console.error('Pie chart missing labels or data')
            return <div className="text-red-500">Pie chart data is incomplete</div>
          }
          
          // Filter out zero/negative values for pie chart
          const validPieIndices = chartData.chart_data.data
            .map((value, index) => value > 0 ? index : -1)
            .filter(index => index !== -1)
          
          if (validPieIndices.length === 0) {
            console.error('No positive values for pie chart')
            return <div className="text-red-500">No positive values available for pie chart</div>
          }
          
          const pieData = {
            labels: validPieIndices.map(i => chartData.chart_data.labels[i]),
            datasets: [
              {
                data: validPieIndices.map(i => chartData.chart_data.data[i]),
                backgroundColor: chartColors.backgroundColor.slice(0, validPieIndices.length),
                borderColor: chartColors.borderColor.slice(0, validPieIndices.length),
                borderWidth: 1,
              },
            ],
          }
          
          // Remove scales for pie chart
          const pieOptions = {
            ...chartOptions,
            scales: undefined
          }
          
          console.log('Pie chart data:', pieData)
          return <Pie data={pieData} options={pieOptions} />

        default:
          console.error('Unsupported chart type:', chartData.chart_data.type)
          return <div className="text-red-500">Unsupported chart type: {chartData.chart_data.type}</div>
      }
    } catch (error) {
      console.error('Error rendering chart:', error)
      return <div className="text-red-500">Error rendering chart: {error.message}</div>
    }
  }

  const getChartIcon = (type: string) => {
    switch (type) {
      case 'bar': return <BarChart3 className="h-4 w-4" />
      case 'line': return <LineChart className="h-4 w-4" />
      case 'scatter': return <ScatterChart className="h-4 w-4" />
      case 'pie': return <PieChart className="h-4 w-4" />
      default: return <Activity className="h-4 w-4" />
    }
  }

  if (!isOpen) return null

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-7xl max-h-[95vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Custom Charts: {fileMetadata?.original_filename || originalFile?.name}
          </DialogTitle>
          <DialogClose className="hover:bg-gray-100 p-2 rounded-full transition-colors">
            <X className="h-4 w-4" />
          </DialogClose>
        </DialogHeader>
        
        <div className="flex-1 overflow-auto">
          {isAnalyzing ? (
            <div className="p-6 text-center">
              <RefreshCw className="h-12 w-12 mx-auto mb-4 text-blue-500 animate-spin" />
              <h3 className="text-lg font-medium mb-2">Analyzing Data for Charts</h3>
              <p className="text-sm text-gray-500 mb-6">
                Please wait while we analyze your data structure for visualization options...
              </p>
              <Progress value={70} className="w-full max-w-md mx-auto" />
            </div>
          ) : analysisError ? (
            <Alert variant="destructive" className="m-6">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Analysis Error</AlertTitle>
              <AlertDescription>
                <div className="space-y-2">
                  <p>{analysisError}</p>
                  <details className="text-xs">
                    <summary className="cursor-pointer">Debug Information</summary>
                    <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto">
                      Backend URL: {getBackendUrl()}
                      {'\n'}File: {fileMetadata?.original_filename || originalFile?.name}
                      {'\n'}Has Original File: {!!originalFile}
                      {'\n'}Has File Metadata: {!!fileMetadata}
                      {fileMetadata && `\nUser ID: ${fileMetadata.user_id}`}
                      {fileMetadata && `\nFilename: ${fileMetadata.filename}`}
                    </pre>
                  </details>
                </div>
              </AlertDescription>
            </Alert>
          ) : analysis ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 p-6">
              {/* Chart Configuration Panel */}
              <div className="lg:col-span-1 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Dataset Overview</CardTitle>
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
                        <span>Numeric columns:</span>
                        <Badge variant="outline">{analysis.column_categories.numeric.length}</Badge>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Categorical columns:</span>
                        <Badge variant="outline">{analysis.column_categories.categorical.length}</Badge>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Date columns:</span>
                        <Badge variant="outline">{analysis.column_categories.datetime.length}</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Chart Suggestions */}
                {analysis.chart_suggestions && analysis.chart_suggestions.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Suggested Charts</CardTitle>
                      <CardDescription>Click to generate these recommended visualizations</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {analysis.chart_suggestions.slice(0, 6).map((suggestion, index) => (
                        <div
                          key={index}
                          className="p-3 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors"
                          onClick={() => applyChartSuggestion(suggestion)}
                        >
                          <div className="flex items-center gap-2 mb-1">
                            {getChartIcon(suggestion.type)}
                            <span className="font-medium text-sm capitalize">{suggestion.type} Chart</span>
                          </div>
                          <p className="text-xs text-gray-600">{suggestion.description}</p>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}

                {/* Custom Chart Configuration */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Custom Chart</CardTitle>
                    <CardDescription>Build your own visualization</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Chart Type */}
                    <div>
                      <label className="block text-sm font-medium mb-2">Chart Type</label>
                      <Select
                        value={customChartConfig.type}
                        onValueChange={(value) => {
                          console.log('Chart type changed to:', value)
                          const newConfig = {...customChartConfig, type: value}
                          setCustomChartConfig(newConfig)
                          // Clear previous chart data when type changes
                          setChartData(null)
                          setChartError(null)
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="bar">Bar Chart</SelectItem>
                          <SelectItem value="line">Line Chart</SelectItem>
                          <SelectItem value="scatter">Scatter Plot</SelectItem>
                          <SelectItem value="pie">Pie Chart</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {/* X-Axis (for non-pie charts) */}
                    {customChartConfig.type !== 'pie' && (
                      <div>
                        <label className="block text-sm font-medium mb-2">X-Axis</label>
                        <Select
                          value={customChartConfig.x_axis}
                          onValueChange={(value) => {
                            console.log('X-axis changed to:', value)
                            const newConfig = {...customChartConfig, x_axis: value}
                            setCustomChartConfig(newConfig)
                            // Clear chart data when axis changes
                            setChartData(null)
                            setChartError(null)
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select column..." />
                          </SelectTrigger>
                          <SelectContent>
                            {analysis && Object.keys(analysis.columns).map(col => (
                              <SelectItem key={col} value={col}>
                                {col} ({analysis.columns[col].category})
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}

                    {/* Y-Axis (for non-pie charts) */}
                    {customChartConfig.type !== 'pie' && (
                      <div>
                        <label className="block text-sm font-medium mb-2">Y-Axis</label>
                        <Select
                          value={customChartConfig.y_axis}
                          onValueChange={(value) => {
                            console.log('Y-axis changed to:', value)
                            const newConfig = {...customChartConfig, y_axis: value}
                            setCustomChartConfig(newConfig)
                            // Clear chart data when axis changes
                            setChartData(null)
                            setChartError(null)
                          }}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select column..." />
                          </SelectTrigger>
                          <SelectContent>
                            {analysis && analysis.column_categories.numeric.map(col => (
                              <SelectItem key={col} value={col}>{col}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}

                    {/* Pie Chart specific fields */}
                    {customChartConfig.type === 'pie' && (
                      <>
                        <div>
                          <label className="block text-sm font-medium mb-2">Category</label>
                          <Select
                            value={customChartConfig.category}
                            onValueChange={(value) => {
                              console.log('Category changed to:', value)
                              const newConfig = {...customChartConfig, category: value}
                              setCustomChartConfig(newConfig)
                              setChartData(null)
                              setChartError(null)
                            }}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select column..." />
                            </SelectTrigger>
                            <SelectContent>
                              {analysis && analysis.column_categories.categorical.map(col => (
                                <SelectItem key={col} value={col}>{col}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-2">Value</label>
                          <Select
                            value={customChartConfig.value}
                            onValueChange={(value) => {
                              console.log('Value changed to:', value)
                              const newConfig = {...customChartConfig, value: value}
                              setCustomChartConfig(newConfig)
                              setChartData(null)
                              setChartError(null)
                            }}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select column..." />
                            </SelectTrigger>
                            <SelectContent>
                              {analysis && analysis.column_categories.numeric.map(col => (
                                <SelectItem key={col} value={col}>{col}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </>
                    )}

                    <Button
                      onClick={() => {
                        console.log('Generate chart button clicked with config:', customChartConfig)
                        
                        // Validate configuration before generating
                        const isValid = validateChartConfig(customChartConfig)
                        if (!isValid.valid) {
                          setChartError(isValid.error)
                          return
                        }
                        
                        generateChart()
                      }}
                      disabled={isGeneratingChart || !isConfigurationValid()}
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
                    
                    {/* Configuration Status */}
                    <div className="text-xs text-gray-500">
                      {getConfigurationStatus()}
                    </div>
                    
                    {/* Test Chart Button - for debugging */}
                    <div className="border-t pt-4">
                      <Button
                        onClick={() => testChartWithSampleData()}
                        variant="outline"
                        size="sm"
                        className="w-full"
                      >
                        Test with Sample Data
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Chart Display Area */}
              <div className="lg:col-span-2">
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle className="text-lg">Chart Preview</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {chartError ? (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertTitle>Chart Generation Error</AlertTitle>
                        <AlertDescription>
                          <div>{chartError}</div>
                          <details className="mt-2">
                            <summary className="cursor-pointer text-sm">Debug Info</summary>
                            <pre className="text-xs mt-2 p-2 bg-gray-100 rounded overflow-auto">
                              Current Config: {JSON.stringify(customChartConfig, null, 2)}
                              {chartData && `\nChart Data: ${JSON.stringify(chartData, null, 2)}`}
                            </pre>
                          </details>
                        </AlertDescription>
                      </Alert>
                    ) : chartData ? (
                      <>
                        <div className="w-full h-96">
                          {renderChart()}
                        </div>
                        
                        {/* Debug information */}
                        <details className="mt-4">
                          <summary className="cursor-pointer text-sm text-gray-500">Chart Debug Info</summary>
                          <div className="mt-2 p-3 bg-gray-50 rounded text-xs">
                            <div><strong>Chart Type:</strong> {chartData.chart_data.type}</div>
                            <div><strong>Data Points:</strong> {chartData.data_points}</div>
                            <div><strong>Config Used:</strong></div>
                            <pre className="mt-1 overflow-auto max-h-20">{JSON.stringify(chartData.config_used, null, 2)}</pre>
                            {chartData.chart_data.labels && (
                              <>
                                <div className="mt-2"><strong>Labels:</strong></div>
                                <pre className="mt-1 overflow-auto max-h-20">{JSON.stringify(chartData.chart_data.labels, null, 2)}</pre>
                              </>
                            )}
                            <div className="mt-2"><strong>Data:</strong></div>
                            <pre className="mt-1 overflow-auto max-h-20">{JSON.stringify(chartData.chart_data.data, null, 2)}</pre>
                          </div>
                        </details>
                      </>
                    ) : (
                      <div className="h-96 flex items-center justify-center text-gray-500">
                        <div className="text-center">
                          <BarChart3 className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                          <p className="text-lg font-medium">No Chart Generated</p>
                          <p className="text-sm">Select chart options and click "Generate Chart" to create a visualization</p>
                          <div className="mt-4 text-xs text-gray-400">
                            Current configuration: {customChartConfig.type}
                            {customChartConfig.type === 'pie' 
                              ? ` (${customChartConfig.category} → ${customChartConfig.value})`
                              : ` (${customChartConfig.x_axis} → ${customChartConfig.y_axis})`
                            }
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                  
                  {chartData && (
                    <CardFooter>
                      <div className="flex items-center justify-between w-full text-sm text-gray-600">
                        <span>Data points: {chartData.data_points}</span>
                        <Button variant="outline" size="sm">
                          <Download className="h-4 w-4 mr-2" />
                          Download Chart
                        </Button>
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
                Click the button below to analyze your data for chart creation
              </p>
              <Button onClick={analyzeFileForCharts}>
                <Activity className="h-4 w-4 mr-2" />
                Analyze Data for Charts
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}