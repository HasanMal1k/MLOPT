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
        formData.append('file', originalFile)
      } else if (fileMetadata && fileMetadata.user_id !== "temporary") {
        // Fetch file from storage
        const supabase = createClient()
        const { data: { publicUrl } } = supabase.storage
          .from('data-files')
          .getPublicUrl(`${fileMetadata.user_id}/${fileMetadata.filename}`)
        
        const response = await fetch(publicUrl)
        if (!response.ok) throw new Error('Failed to fetch file from storage')
        
        const fileBlob = await response.blob()
        const file = new File([fileBlob], fileMetadata.original_filename, { 
          type: fileMetadata.mime_type 
        })
        formData.append('file', file)
      } else {
        throw new Error('No file available for analysis')
      }

      const response = await fetch('http://localhost:8000/visualization/analyze-for-charts/', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const result = await response.json()
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

    setIsGeneratingChart(true)
    setChartError(null)

    try {
      const formData = new FormData()
      
      if (originalFile) {
        formData.append('file', originalFile)
      } else if (fileMetadata && fileMetadata.user_id !== "temporary") {
        // Fetch file from storage
        const supabase = createClient()
        const { data: { publicUrl } } = supabase.storage
          .from('data-files')
          .getPublicUrl(`${fileMetadata.user_id}/${fileMetadata.filename}`)
        
        const response = await fetch(publicUrl)
        if (!response.ok) throw new Error('Failed to fetch file from storage')
        
        const fileBlob = await response.blob()
        const file = new File([fileBlob], fileMetadata.original_filename, { 
          type: fileMetadata.mime_type 
        })
        formData.append('file', file)
      }

      formData.append('chart_config', JSON.stringify(config))

      const response = await fetch('http://localhost:8000/visualization/generate-chart-data/', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Chart generation failed: ${response.statusText}`)
      }

      const result = await response.json()
      if (result.success) {
        setChartData(result)
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
    const newConfig = {
      type: suggestion.type,
      x_axis: suggestion.x_axis || '',
      y_axis: suggestion.y_axis || '',
      category: suggestion.category || '',
      value: suggestion.value || '',
      title: suggestion.description || 'Custom Chart'
    }
    setCustomChartConfig(newConfig)
    generateChart(newConfig)
  }

  const renderChart = () => {
    if (!chartData) return null

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

    switch (chartData.chart_data.type) {
      case 'bar':
        const barData = {
          labels: chartData.chart_data.labels,
          datasets: [
            {
              label: customChartConfig.y_axis,
              data: chartData.chart_data.data,
              backgroundColor: chartColors.backgroundColor,
              borderColor: chartColors.borderColor,
              borderWidth: 1,
            },
          ],
        }
        return <Bar data={barData} options={chartOptions} />

      case 'scatter':
        const scatterData = {
          datasets: [
            {
              label: `${customChartConfig.y_axis} vs ${customChartConfig.x_axis}`,
              data: chartData.chart_data.data,
              backgroundColor: 'rgba(255, 99, 132, 0.6)',
              borderColor: 'rgba(255, 99, 132, 1)',
            },
          ],
        }
        return <Scatter data={scatterData} options={chartOptions} />

      case 'line':
        const lineData = {
          labels: chartData.chart_data.labels,
          datasets: [
            {
              label: customChartConfig.y_axis,
              data: chartData.chart_data.data,
              borderColor: 'rgb(75, 192, 192)',
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              tension: 0.1,
            },
          ],
        }
        return <Line data={lineData} options={chartOptions} />

      case 'pie':
        const pieData = {
          labels: chartData.chart_data.labels,
          datasets: [
            {
              data: chartData.chart_data.data,
              backgroundColor: chartColors.backgroundColor,
              borderColor: chartColors.borderColor,
              borderWidth: 1,
            },
          ],
        }
        return <Pie data={pieData} options={chartOptions} />

      default:
        return <div>Unsupported chart type</div>
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
              <AlertDescription>{analysisError}</AlertDescription>
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
                        onValueChange={(value) => setCustomChartConfig({...customChartConfig, type: value})}
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
                          onValueChange={(value) => setCustomChartConfig({...customChartConfig, x_axis: value})}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select column..." />
                          </SelectTrigger>
                          <SelectContent>
                            {Object.keys(analysis.columns).map(col => (
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
                          onValueChange={(value) => setCustomChartConfig({...customChartConfig, y_axis: value})}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select column..." />
                          </SelectTrigger>
                          <SelectContent>
                            {analysis.column_categories.numeric.map(col => (
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
                            onValueChange={(value) => setCustomChartConfig({...customChartConfig, category: value})}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select column..." />
                            </SelectTrigger>
                            <SelectContent>
                              {analysis.column_categories.categorical.map(col => (
                                <SelectItem key={col} value={col}>{col}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-2">Value</label>
                          <Select
                            value={customChartConfig.value}
                            onValueChange={(value) => setCustomChartConfig({...customChartConfig, value: value})}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select column..." />
                            </SelectTrigger>
                            <SelectContent>
                              {analysis.column_categories.numeric.map(col => (
                                <SelectItem key={col} value={col}>{col}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </>
                    )}

                    <Button
                      onClick={() => generateChart()}
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
                        <AlertDescription>{chartError}</AlertDescription>
                      </Alert>
                    ) : chartData ? (
                      <div className="w-full h-96">
                        {renderChart()}
                      </div>
                    ) : (
                      <div className="h-96 flex items-center justify-center text-gray-500">
                        <div className="text-center">
                          <BarChart3 className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                          <p className="text-lg font-medium">No Chart Generated</p>
                          <p className="text-sm">Select chart options and click "Generate Chart" to create a visualization</p>
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