'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
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
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { 
  Brain,
  Plus,
  Search,
  Filter,
  Eye,
  Settings,
  MoreHorizontal,
  Database,
  Clock,
  BarChart3,
  Calendar,
  RefreshCw,
  Upload,
  TrendingUp,
  Play,
  Zap,
  Target,
  Award,
  GitBranch,
  Activity,
  Layers
} from "lucide-react"
import Link from 'next/link'
import { useToast } from "@/hooks/use-toast"
import { format } from 'date-fns'
import FilePreview from "@/components/FilePreview"
import { useRouter } from 'next/navigation'

interface ExtendedFileMetadata {
  id: string;
  user_id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  mime_type: string;
  column_names: string[];
  row_count: number;
  file_preview: any[];
  statistics: any;
  upload_date: string;
  preprocessing_info?: any;
  dataset_type: 'normal' | 'time_series';
  custom_cleaning_applied: boolean;
  custom_cleaning_config?: any;
  time_series_config?: any;
}

interface MLWorkflowStep {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  estimatedTime: string;
  requirements: string[];
}

const ML_WORKFLOW_STEPS: MLWorkflowStep[] = [
  {
    id: 'feature-selection',
    title: 'Feature Selection',
    description: 'Analyze and select the most important features using mutual information',
    icon: <Target className="h-5 w-5" />,
    estimatedTime: '1-2 min',
    requirements: ['Cleaned dataset', 'Numeric/categorical columns']
  },
  {
    id: 'model-config',
    title: 'Model Configuration',
    description: 'Configure training parameters like target column, train size, normalization',
    icon: <Settings className="h-5 w-5" />,
    estimatedTime: '2-3 min',
    requirements: ['Target column selection', 'Training preferences']
  },
  {
    id: 'auto-training',
    title: 'Auto Training',
    description: 'Train multiple ML models and compare performance automatically',
    icon: <Brain className="h-5 w-5" />,
    estimatedTime: '5-15 min',
    requirements: ['Configured parameters', 'Sufficient data']
  },
  {
    id: 'model-evaluation',
    title: 'Model Evaluation',
    description: 'View leaderboard, metrics, and download trained models',
    icon: <Award className="h-5 w-5" />,
    estimatedTime: '1-2 min',
    requirements: ['Completed training']
  }
]

export default function BlueprintsPage() {
  const [datasets, setDatasets] = useState<ExtendedFileMetadata[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('ml-ready')
  const [selectedDataset, setSelectedDataset] = useState<ExtendedFileMetadata | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  
  const { toast } = useToast()
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    fetchMLReadyDatasets()
  }, [])

  const fetchMLReadyDatasets = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser()
      
      if (user) {
        const { data, error } = await supabase
          .from('files')
          .select('*')
          .eq('user_id', user.id)
          .order('upload_date', { ascending: false })
        
        if (error) throw error
        
        setDatasets(data as ExtendedFileMetadata[])
      }
    } catch (error) {
      console.error('Error fetching datasets:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch datasets"
      })
    } finally {
      setLoading(false)
    }
  }

  // Filter datasets that are ready for ML training
  const getMLReadyDatasets = () => {
    return datasets.filter(dataset => {
      // Must have at least some rows and columns
      const hasData = dataset.row_count > 0 && dataset.column_names.length > 0
      
      // Must be processed or have sufficient data for auto-processing
      const isProcessed = dataset.preprocessing_info?.is_preprocessed || 
                         dataset.custom_cleaning_applied ||
                         dataset.row_count >= 10 // Minimum rows for ML
      
      return hasData && isProcessed
    })
  }

  const filteredDatasets = getMLReadyDatasets().filter(dataset => {
    const matchesSearch = (dataset.name || dataset.original_filename).toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = filterType === 'all' || dataset.dataset_type === filterType
    
    let matchesStatus = true
    if (filterStatus === 'ml-ready') {
      matchesStatus = dataset.row_count > 0 && dataset.column_names.length > 1
    } else if (filterStatus === 'processed') {
      matchesStatus = dataset.preprocessing_info?.is_preprocessed || dataset.custom_cleaning_applied
    } else if (filterStatus === 'large-dataset') {
      matchesStatus = dataset.row_count > 1000
    }
    
    return matchesSearch && matchesType && matchesStatus
  })

  const handlePreview = (dataset: ExtendedFileMetadata) => {
    setSelectedDataset(dataset)
    setIsPreviewOpen(true)
  }

  const handleStartMLWorkflow = (dataset: ExtendedFileMetadata) => {
    // Navigate to ML training workflow with the selected dataset
    router.push(`/dashboard/blueprints/train?file=${dataset.id}&filename=${encodeURIComponent(dataset.filename)}`)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getDatasetTypeIcon = (type: string) => {
    switch (type) {
      case 'time_series':
        return <Clock className="h-4 w-4" />
      default:
        return <Database className="h-4 w-4" />
    }
  }

  const getDatasetTypeBadge = (type: string) => {
    switch (type) {
      case 'time_series':
        return <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
          <Clock className="h-3 w-3 mr-1" />
          Time Series
        </Badge>
      default:
        return <Badge variant="outline" className="bg-gray-50 text-gray-700 border-gray-200">
          <Database className="h-3 w-3 mr-1" />
          Normal
        </Badge>
    }
  }

  const getMLReadinessBadge = (dataset: ExtendedFileMetadata) => {
    // Use actual quality rating from database
    const rating = (dataset as any).quality_rating || 'Fair';
    const score = (dataset as any).quality_score || 50;
    
    if (rating === 'Excellent') {
      return <Badge className="bg-green-500 text-white">
        <Zap className="h-3 w-3 mr-1" />
        Excellent ({Math.round(score)})
      </Badge>
    } else if (rating === 'Good') {
      return <Badge className="bg-blue-500 text-white">
        <Activity className="h-3 w-3 mr-1" />
        Good ({Math.round(score)})
      </Badge>
    } else if (rating === 'Fair') {
      return <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
        <GitBranch className="h-3 w-3 mr-1" />
        Fair ({Math.round(score)})
      </Badge>
    } else {
      return <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
        Poor ({Math.round(score)})
      </Badge>
    }
  }

  const getRecommendedModels = (dataset: ExtendedFileMetadata) => {
    // Use actual recommended models from database if available
    const recommendedModels = (dataset as any).recommended_models;
    if (recommendedModels && Array.isArray(recommendedModels) && recommendedModels.length > 0) {
      return recommendedModels.map((m: any) => m.model || m);
    }
    
    // Fallback to simple heuristic if not calculated
    const numericColumns = dataset.column_names.filter(col => 
      dataset.statistics?.[col]?.dtype?.includes('int') || 
      dataset.statistics?.[col]?.dtype?.includes('float')
    ).length
    
    const categoricalColumns = dataset.column_names.length - numericColumns
    
    if (dataset.dataset_type === 'time_series') {
      return ['Prophet', 'ARIMA', 'XGBoost', 'LSTM']
    } else if (numericColumns > categoricalColumns) {
      return ['Random Forest', 'XGBoost', 'Linear Models', 'SVM']
    } else {
      return ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
    }
  }

  const mlReadyDatasets = getMLReadyDatasets()
  const totalMLReady = mlReadyDatasets.length
  // Use actual quality ratings from database
  const excellentDatasets = mlReadyDatasets.filter(d => (d as any).quality_rating === 'Excellent').length
  const largeDatasets = mlReadyDatasets.filter(d => d.row_count > 1000).length

  return (
    <section className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold flex items-center gap-3">
            <Brain className="h-10 w-10 text-blue-600" />
            ML Blueprints
          </h1>
          <p className="text-muted-foreground mt-2">
            Transform your datasets into intelligent models with automated machine learning
          </p>
        </div>
        
        <div className="flex gap-3">
          <Button variant="outline" onClick={fetchMLReadyDatasets}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button asChild>
            <Link href="/dashboard/data-upload">
              <Plus className="h-4 w-4 mr-2" />
              Add Dataset
            </Link>
          </Button>
        </div>
      </div>

      {/* ML Workflow Overview */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            ML Training Workflow
          </CardTitle>
          <CardDescription>
            Our automated machine learning pipeline guides you through feature selection to model deployment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {ML_WORKFLOW_STEPS.map((step, index) => (
              <div key={step.id} className="relative">
                {index < ML_WORKFLOW_STEPS.length - 1 && (
                  <div className="hidden md:block absolute top-8 right-0 w-full h-0.5 bg-gray-200 z-0" />
                )}
                <Card className="relative z-10 bg-white">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                        {step.icon}
                      </div>
                      <div className="text-sm font-medium">{step.title}</div>
                    </div>
                    <p className="text-xs text-muted-foreground mb-2">{step.description}</p>
                    <div className="flex justify-between items-center">
                      <Badge variant="outline" className="text-xs">{step.estimatedTime}</Badge>
                      <div className="text-xs text-blue-600 font-medium">Step {index + 1}</div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ML Ready Datasets</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalMLReady}</div>
            <p className="text-xs text-muted-foreground">
              Ready for training
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Excellent Quality</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{excellentDatasets}</div>
            <p className="text-xs text-muted-foreground">
              High quality datasets
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Large Datasets</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{largeDatasets}</div>
            <p className="text-xs text-muted-foreground">
              1000+ rows
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Available Models</CardTitle>
            <Award className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">15+</div>
            <p className="text-xs text-muted-foreground">
              ML algorithms
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Find Your Dataset</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search datasets for ML training..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            
            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Dataset Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="normal">Normal</SelectItem>
                <SelectItem value="time_series">Time Series</SelectItem>
              </SelectContent>
            </Select>

            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="ML Readiness" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ml-ready">ML Ready</SelectItem>
                <SelectItem value="processed">Processed</SelectItem>
                <SelectItem value="large-dataset">Large Dataset</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Datasets Table */}
      <Card>
        <CardHeader>
          <CardTitle>ML Training Candidates ({filteredDatasets.length})</CardTitle>
          <CardDescription>
            Select a dataset to start your machine learning workflow
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
                <p>Loading ML-ready datasets...</p>
              </div>
            </div>
          ) : filteredDatasets.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-2">No ML-ready datasets found</h3>
              <p className="text-muted-foreground mb-6">
                {datasets.length === 0 
                  ? "Upload and process your first dataset to start ML training"
                  : "Process your datasets or adjust filters to find ML-ready data"
                }
              </p>
              <Button asChild>
                <Link href="/dashboard/data-upload">
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Dataset
                </Link>
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Quality</TableHead>
                  <TableHead>Dimensions</TableHead>
                  <TableHead>Recommended Models</TableHead>
                  <TableHead>Upload Date</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredDatasets.map((dataset) => (
                  <TableRow key={dataset.id}>
                    <TableCell>
                      <div className="flex items-center gap-3">
                        {getDatasetTypeIcon(dataset.dataset_type)}
                        <div>
                          <div className="font-medium">{dataset.name || dataset.original_filename}</div>
                          <div className="text-sm text-muted-foreground">
                            {formatFileSize(dataset.file_size)} • {dataset.mime_type.includes('csv') ? 'CSV' : 'Excel'}
                          </div>
                        </div>
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      {getDatasetTypeBadge(dataset.dataset_type)}
                    </TableCell>
                    
                    <TableCell>
                      {getMLReadinessBadge(dataset)}
                    </TableCell>
                    
                    <TableCell>
                      <div className="text-sm">
                        <div>{dataset.row_count.toLocaleString()} rows</div>
                        <div className="text-muted-foreground">
                          {dataset.column_names.length} features
                        </div>
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {getRecommendedModels(dataset).slice(0, 2).map((model, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {model}
                          </Badge>
                        ))}
                        {getRecommendedModels(dataset).length > 2 && (
                          <Badge variant="outline" className="text-xs">
                            +{getRecommendedModels(dataset).length - 2}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      <div className="text-sm">
                        {format(new Date(dataset.upload_date), 'MMM dd, yyyy')}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {format(new Date(dataset.upload_date), 'HH:mm')}
                      </div>
                    </TableCell>
                    
                    <TableCell className="text-right">
                      <div className="flex items-center gap-2 justify-end">
                        <Button 
                          size="sm" 
                          onClick={() => handleStartMLWorkflow(dataset)}
                          className="bg-green-400 hover:bg-green-500"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Start ML
                        </Button>
                        
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm">
                              <MoreHorizontal className="h-4 w-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuLabel>Actions</DropdownMenuLabel>
                            <DropdownMenuItem onClick={() => handlePreview(dataset)}>
                              <Eye className="h-4 w-4 mr-2" />
                              Preview Data
                            </DropdownMenuItem>
                            <DropdownMenuItem asChild>
                              <Link href={`/dashboard/datasets`}>
                                <Database className="h-4 w-4 mr-2" />
                                View in Datasets
                              </Link>
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => handleStartMLWorkflow(dataset)}>
                              <Brain className="h-4 w-4 mr-2" />
                              Configure ML Training
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* File Preview Modal */}
      {selectedDataset && (
        <FilePreview 
          fileMetadata={selectedDataset} 
          isOpen={isPreviewOpen} 
          onClose={() => setIsPreviewOpen(false)} 
        />
      )}
    </section>
  )
}