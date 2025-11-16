'use client'

import { useState, useEffect } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
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
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Input } from "@/components/ui/input"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { 
  Table,
  TableBody,
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
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { 
  Brain,
  ArrowLeft,
  Target,
  Settings,
  Play,
  Loader2,
  BarChart3,
  TrendingUp,
  Clock,
  Zap,
  CheckCircle,
  AlertCircle,
  Sparkles,
  Database,
  Award,
  Layers,
  RefreshCw,
  Download,
  Save,
  Package,
  FileCode2
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import SaveModelDialog from '@/components/SaveModelDialog'

interface FeatureImportance {
  feature: string
  importance: number
  relative_percent: number
  cumulative_percent: number
}

interface FeatureAnalysis {
  all_features: FeatureImportance[]
  recommended_features: FeatureImportance[]
  total_features: number
  recommended_count: number
}

interface TrainingConfig {
  target_column: string
  train_size: number
  session_id: number
  normalize: boolean
  transformation: boolean
  remove_outliers: boolean
  outliers_threshold: number
  feature_selection: boolean
  polynomial_features: boolean
}

export default function MLTrainingPage() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const { toast } = useToast()
  const supabase = createClient()

  // Core state
  const [currentStep, setCurrentStep] = useState(0)
  const fileId = searchParams.get('file') || ''
  const filename = decodeURIComponent(searchParams.get('filename') || '')
  
  // Data state
  const [fileMetadata, setFileMetadata] = useState<any>(null)
  const [taskType, setTaskType] = useState<'classification' | 'regression' | 'time_series' | null>(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [featureAnalysis, setFeatureAnalysis] = useState<FeatureAnalysis | null>(null)
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
  
  // Time series specific state
  const [timeColumn, setTimeColumn] = useState('')
  const [detectedTimeColumns, setDetectedTimeColumns] = useState<string[]>([])
  const [forecastingHorizon, setForecastingHorizon] = useState<number>(12)
  const [isDetectingTimeColumns, setIsDetectingTimeColumns] = useState(false)
  const [forecastingType, setForecastingType] = useState<'univariate' | 'multivariate' | 'exogenous' | null>(null)
  const [exogenousColumns, setExogenousColumns] = useState<string[]>([])
  
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    target_column: '',
    train_size: 0.8,
    session_id: 123,
    normalize: true,
    transformation: true,
    remove_outliers: true,
    outliers_threshold: 0.05,
    feature_selection: true,
    polynomial_features: false
  })

  // Advanced training parameters
  const [cvFolds, setCvFolds] = useState<number>(3)
  const [sortMetric, setSortMetric] = useState<string>('auto') // Performance metric to optimize
  const [hyperparameterTuning, setHyperparameterTuning] = useState<boolean>(false)
  const [tuningIterations, setTuningIterations] = useState<number>(10)
  const [ensembleMethods, setEnsembleMethods] = useState<boolean>(false)
  const [stackingEnabled, setStackingEnabled] = useState<boolean>(false)

  // Training state
  const [configId, setConfigId] = useState<string | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<string>('idle')
  const [trainingResults, setTrainingResults] = useState<any>(null)

  // Model testing state
  const [selectedModelForTest, setSelectedModelForTest] = useState<string | null>(null)
  const [testInputs, setTestInputs] = useState<Record<string, string>>({})
  const [testPrediction, setTestPrediction] = useState<any>(null)
  const [isTesting, setIsTesting] = useState(false)

  // Save model state
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [userId, setUserId] = useState<string>('')
  const [selectedModelToSave, setSelectedModelToSave] = useState<any>(null)

  // Loading states
  const [isLoadingFile, setIsLoadingFile] = useState(true)
  const [isAnalyzingFeatures, setIsAnalyzingFeatures] = useState(false)
  const [isConfiguringTraining, setIsConfiguringTraining] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const steps = [
    'Task Type & Target',
    'Feature Selection', 
    'Training Configuration',
    'Model Training',
    'Results'
  ]

  // Initialize
  useEffect(() => {
    if (fileId) {
      fetchFileMetadata()
    }
  }, [fileId])

  // Get user ID
  useEffect(() => {
    const fetchUser = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) setUserId(user.id)
    }
    fetchUser()
  }, [])

  // Load stored configId on component mount
  useEffect(() => {
    const stored = localStorage.getItem('ml_config_id')
    if (stored) {
      setConfigId(stored)
      console.log('Loaded stored configId:', stored)
    }
  }, [])

  const fetchFileMetadata = async () => {
    try {
      const { data, error } = await supabase
        .from('files')
        .select('*')
        .eq('id', fileId)
        .single()

      if (error) throw error
      setFileMetadata(data)
    } catch (error) {
      console.error('Error fetching file metadata:', error)
      setErrorMessage('Failed to load file metadata')
    } finally {
      setIsLoadingFile(false)
    }
  }

  const detectTimeColumns = async () => {
    setIsDetectingTimeColumns(true)
    setErrorMessage(null)
    
    try {
      // Get file content
      const response = await fetch(`/api/files/downloads?fileId=${fileId}`)
      if (!response.ok) throw new Error('Failed to get file')
      
      const fileBlob = await response.blob()
      
      // Send to backend for time column detection
      const formData = new FormData()
      formData.append('file', fileBlob, filename)

      const detectionResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/detect-time-columns/`, {
        method: 'POST',
        body: formData
      })

      if (!detectionResponse.ok) {
        const errorData = await detectionResponse.json()
        throw new Error(errorData.detail || 'Failed to detect time columns')
      }
      
      const result = await detectionResponse.json()
      if (result.success) {
        setDetectedTimeColumns(result.time_columns || [])
        
        // Auto-select the first detected time column
        if (result.time_columns && result.time_columns.length > 0) {
          setTimeColumn(result.time_columns[0])
        }
        
        toast({
          title: "Time Columns Detected",
          description: `Found ${result.time_columns?.length || 0} potential time columns`
        })
      }
    } catch (error) {
      console.error('Time column detection error:', error)
      setErrorMessage(`Time column detection failed: ${error}`)
    } finally {
      setIsDetectingTimeColumns(false)
    }
  }

  const analyzeFeatures = async () => {
    setIsAnalyzingFeatures(true)
    setErrorMessage(null)
    
    try {
      // Get file content
      const response = await fetch(`/api/files/downloads?fileId=${fileId}`)
      if (!response.ok) throw new Error('Failed to get file')
      
      const fileBlob = await response.blob()
      
      // Send to backend for analysis
      const formData = new FormData()
      formData.append('file', fileBlob, filename)
      formData.append('task_type', taskType!)
      formData.append('target_column', targetColumn)

      const analysisResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/analyze-target-with-file/`, {
        method: 'POST',
        body: formData
      })

      if (!analysisResponse.ok) {
        const errorData = await analysisResponse.json()
        throw new Error(errorData.detail || 'Failed to analyze features')
      }
      
      const result = await analysisResponse.json()
      if (result.success) {
        setFeatureAnalysis(result.feature_analysis)
        
        // Auto-select recommended features
        if (result.feature_analysis?.recommended_features) {
          const recommended = result.feature_analysis.recommended_features.map((f: FeatureImportance) => f.feature)
          setSelectedFeatures(recommended)
        }
      }
    } catch (error) {
      console.error('Feature analysis error:', error)
      setErrorMessage(`Feature analysis failed: ${error}`)
    } finally {
      setIsAnalyzingFeatures(false)
    }
  }

  const configureTraining = async () => {
    setIsConfiguringTraining(true)
    setErrorMessage(null)
    
    try {
      console.log('=== CONFIGURING TRAINING ===')
      
      // Get file content
      const response = await fetch(`/api/files/downloads?fileId=${fileId}`)
      if (!response.ok) throw new Error('Failed to get file')
      
      const fileBlob = await response.blob()
      console.log('File blob size:', fileBlob.size)
      
      const formData = new FormData()
      formData.append('file', fileBlob, filename)
      
      let configEndpoint = ''
      
      if (taskType === 'time_series') {
        // Time series configuration
        configEndpoint = `${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/configure-time-series-with-file/`
        
        formData.append('forecasting_type', forecastingType!)
        formData.append('target_column', targetColumn)
        formData.append('time_column', timeColumn)
        formData.append('exogenous_columns', JSON.stringify(exogenousColumns))
        formData.append('forecast_horizon', forecastingHorizon.toString())
        formData.append('train_split', trainingConfig.train_size.toString())
        formData.append('include_deep_learning', 'true')
        formData.append('include_statistical', 'true')
        formData.append('include_ml', 'true')
        formData.append('max_epochs', '10')
      } else {
        // Regular ML configuration
        configEndpoint = `${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/configure-training-with-file/`
        
        formData.append('task_type', taskType!)
        formData.append('target_column', targetColumn)
        formData.append('selected_features', JSON.stringify(selectedFeatures))
        formData.append('train_size', trainingConfig.train_size.toString())
        formData.append('session_id', trainingConfig.session_id.toString())
        formData.append('normalize', trainingConfig.normalize.toString())
        formData.append('transformation', trainingConfig.transformation.toString())
        formData.append('remove_outliers', trainingConfig.remove_outliers.toString())
        formData.append('outliers_threshold', trainingConfig.outliers_threshold.toString())
        formData.append('feature_selection', trainingConfig.feature_selection.toString())
        formData.append('polynomial_features', trainingConfig.polynomial_features.toString())
        
        // Add new advanced parameters
        formData.append('cv_folds', cvFolds.toString())
        formData.append('sort_metric', sortMetric)
        formData.append('hyperparameter_tuning', hyperparameterTuning.toString())
        formData.append('tuning_iterations', tuningIterations.toString())
        formData.append('ensemble_methods', ensembleMethods.toString())
        formData.append('stacking_enabled', stackingEnabled.toString())
      }

      console.log('Sending configuration to backend...')
      const configResponse = await fetch(configEndpoint, {
        method: 'POST',
        body: formData
      })

      console.log('Configuration response status:', configResponse.status)
      
      if (!configResponse.ok) {
        const errorData = await configResponse.json()
        throw new Error(errorData.detail || 'Configuration failed')
      }
      
      const result = await configResponse.json()
      console.log('Configuration result:', result)
      
      if (result.success && result.config_id) {
        // Store configId immediately in multiple places
        const newConfigId = result.config_id
        console.log('=== CONFIG ID RECEIVED ===', newConfigId)
        
        // Store in localStorage immediately
        localStorage.setItem('ml_config_id', newConfigId)
        
        // Set state
        setConfigId(newConfigId)
        
        // Move to next step
        setCurrentStep(3)
        
        console.log('Configuration complete, moving to training step')
        
        toast({
          title: "Configuration Complete",
          description: `${taskType === 'time_series' ? 'Time series' : 'ML'} training configured successfully`
        })
      } else {
        throw new Error('No config_id returned from backend')
      }
    } catch (error) {
      console.error('Configuration error:', error)
      setErrorMessage(`Configuration failed: ${error}`)
    } finally {
      setIsConfiguringTraining(false)
    }
  }

  const startTraining = async () => {
    // Get configId from state or localStorage
    const currentConfigId = configId || localStorage.getItem('ml_config_id')
    
    if (!currentConfigId) {
      setErrorMessage('No configuration ID available. Please reconfigure.')
      return
    }

    console.log('=== STARTING TRAINING ===', currentConfigId)
    
    setIsTraining(true)
    setTrainingStatus('starting')
    setErrorMessage(null)
    
    try {
      const formData = new FormData()
      formData.append('config_id', currentConfigId)

      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/start-training/`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Training failed to start')
      }
      
      const result = await response.json()
      console.log('Training started:', result)
      console.log('Result success:', result.success)
      console.log('Config ID:', currentConfigId)
      
      if (result.success) {
        // Use SSE streaming instead of polling
        console.log('📞 Calling streamTrainingResults with:', currentConfigId)
        streamTrainingResults(currentConfigId)
      } else {
        console.error('❌ Training start did not return success')
        setErrorMessage('Training failed to start properly')
        setIsTraining(false)
      }
    } catch (error) {
      console.error('Training start error:', error)
      setErrorMessage(`Training failed to start: ${error}`)
      setIsTraining(false)
    }
  }

  // Replace polling with Server-Sent Events for real-time updates
  const streamTrainingResults = (taskId: string) => {
    console.log('🚀 Starting SSE stream for task:', taskId)
    
    try {
      const eventSource = new EventSource(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/training-stream/${taskId}`
      )

      console.log('📡 EventSource created, waiting for connection...')

      // Initialize leaderboard array
      const liveLeaderboard: any[] = []

      eventSource.onopen = () => {
        console.log('✅ SSE connection opened successfully')
      }

      eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('📊 SSE Event received:', data.type, data)

        if (data.type === 'connected') {
          console.log('✅ Connected to training stream')
          setTrainingStatus('training')
        }
        
        else if (data.type === 'model_completed') {
          // Add model to live leaderboard
          const modelResult = data.model
          
          // Only add if not a failed model
          if (modelResult.status !== 'failed') {
            liveLeaderboard.push(modelResult)
            
            // Sort leaderboard by performance metric
            const sortedLeaderboard = [...liveLeaderboard].sort((a, b) => {
              // Determine which metric to use for sorting
              const metric = sortMetric === 'auto' 
                ? (taskType === 'regression' ? 'R2' : 'Accuracy')
                : sortMetric
              
              const metricA = a[metric] ?? 0
              const metricB = b[metric] ?? 0
              
              // For error metrics (MAE, RMSE, RMSLE), lower is better
              const lowerIsBetter = ['MAE', 'RMSE', 'RMSLE', 'MSE'].includes(metric)
              return lowerIsBetter ? metricA - metricB : metricB - metricA
            })
            
            // Update training results in real-time
            setTrainingResults((prev: any) => ({
              ...prev,
              leaderboard: sortedLeaderboard,
              total_models_tested: liveLeaderboard.length,
              current_model: modelResult.Model
            }))

            console.log(`✅ Model added to leaderboard: ${modelResult.Model} (${liveLeaderboard.length} total)`)
            
            // Show toast notification
            const displayMetric = sortMetric === 'auto' 
              ? (taskType === 'regression' ? 'R2' : 'Accuracy')
              : sortMetric
            const displayValue = modelResult[displayMetric] ?? 0
            
            toast({
              title: `Model Completed: ${modelResult.Model}`,
              description: `${displayMetric}: ${displayValue.toFixed(4)}`,
              duration: 2000
            })
          }
        }
        
        else if (data.type === 'completed') {
          console.log('🎉 Training completed!')
          
          setTrainingResults({
            leaderboard: data.leaderboard || liveLeaderboard,
            best_model_name: data.best_model_name,
            models_saved: data.models_saved,
            total_models_tested: data.total_models_tested,
            completed_at: new Date().toISOString()
          })
          
          setTrainingStatus('completed')
          setIsTraining(false)
          setCurrentStep(4) // Move to results
          
          toast({
            title: "Training Complete!",
            description: `${data.total_models_tested} models trained. Best: ${data.best_model_name}`,
          })
          
          eventSource.close()
        }
        
        else if (data.type === 'error') {
          console.error('❌ Training error:', data.error)
          setErrorMessage(data.error || 'Training failed')
          setIsTraining(false)
          setTrainingStatus('failed')
          
          toast({
            variant: "destructive",
            title: "Training Failed",
            description: data.error
          })
          
          eventSource.close()
        }
      } catch (error) {
        console.error('Error parsing SSE event:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error('❌ SSE connection error:', error)
      console.error('SSE readyState:', eventSource.readyState)
      eventSource.close()
      
      // Fallback to polling if SSE fails
      console.log('⚠️ Falling back to polling...')
      pollTrainingStatusFallback(taskId)
    }

    // Store event source for cleanup
    return eventSource
    } catch (error) {
      console.error('❌ Failed to create EventSource:', error)
      // Fallback to polling immediately
      pollTrainingStatusFallback(taskId)
      return null
    }
  }

  // Fallback polling function if SSE fails
  const pollTrainingStatusFallback = async (taskId: string) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/training-status/${taskId}`)
      
      if (!response.ok) throw new Error('Failed to check training status')
      
      const result = await response.json()
      setTrainingStatus(result.status)
      
      if (result.status === 'completed') {
        setTrainingResults(result)
        setIsTraining(false)
        setCurrentStep(4)
        toast({
          title: "Training Complete!",
          description: "Your models have been trained successfully."
        })
      } else if (result.status === 'failed') {
        setErrorMessage(result.error || 'Training failed')
        setIsTraining(false)
      } else {
        setTimeout(() => pollTrainingStatusFallback(taskId), 3000)
      }
    } catch (error) {
      console.error('Status check error:', error)
      setTimeout(() => pollTrainingStatusFallback(taskId), 5000)
    }
  }

  // Step navigation functions
  const handleTaskTypeNext = () => {
    if (!taskType) {
      toast({
        variant: "destructive",
        title: "Selection Required",
        description: "Please select a machine learning task type"
      })
      return
    }

    if (taskType === 'time_series') {
      if (!forecastingType) {
        toast({
          variant: "destructive",
          title: "Forecasting Type Required",
          description: "Please select univariate, multivariate, or exogenous forecasting"
        })
        return
      }

      if (!timeColumn) {
        toast({
          variant: "destructive",
          title: "Time Column Required",
          description: "Please select a time column"
        })
        return
      }

      if (forecastingType === 'univariate' && !targetColumn) {
        toast({
          variant: "destructive",
          title: "Target Column Required",
          description: "Please select a target column for univariate forecasting"
        })
        return
      }

      if (forecastingType === 'multivariate' && selectedFeatures.length === 0) {
        toast({
          variant: "destructive",
          title: "Target Columns Required", 
          description: "Please select at least one target column for multivariate forecasting"
        })
        return
      }

      if (forecastingType === 'exogenous') {
        if (!targetColumn) {
          toast({
            variant: "destructive",
            title: "Target Column Required",
            description: "Please select a target column for exogenous forecasting"
          })
          return
        }
        
        if (exogenousColumns.length === 0) {
          toast({
            variant: "destructive",
            title: "Exogenous Variables Required",
            description: "Please select at least one exogenous variable"
          })
          return
        }
      }
      
      // For time series, skip feature analysis and go directly to time series configuration
      setCurrentStep(2)
      return
    }

    // Regular ML validation
    if (!targetColumn) {
      toast({
        variant: "destructive",
        title: "Target Column Required",
        description: "Please select a target column"
      })
      return
    }
    
    // Start feature analysis for classification/regression
    analyzeFeatures()
    setCurrentStep(1)
  }

  const handleFeatureNext = () => {
    if (selectedFeatures.length === 0) {
      toast({
        variant: "destructive", 
        title: "Features Required",
        description: "Please select at least one feature"
      })
      return
    }
    
    setTrainingConfig(prev => ({ ...prev, target_column: targetColumn }))
    setCurrentStep(2)
  }

  const toggleFeature = (feature: string) => {
    setSelectedFeatures(prev => 
      prev.includes(feature) 
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    )
  }

  if (isLoadingFile) {
    return (
      <div className="h-screen w-full px-6 md:px-10 py-10">
        <Card className="max-w-2xl mx-auto">
          <CardContent className="flex flex-col items-center justify-center p-12">
            <Loader2 className="h-8 w-8 animate-spin mb-4" />
            <p>Loading dataset information...</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <section className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={() => router.push('/dashboard/blueprints')}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Blueprints
            </Button>
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-3">
                <Brain className="h-8 w-8 text-blue-600" />
                ML Training Workflow
              </h1>
              <p className="text-muted-foreground mt-1">
                Training models for: <span className="font-medium">{filename}</span>
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm text-muted-foreground">Step {currentStep + 1} of {steps.length}</div>
            <div className="text-2xl font-bold">{Math.round(((currentStep + 1) / steps.length) * 100)}%</div>
          </div>
        </div>

        {/* Progress */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => (
              <div key={index} className="flex items-center">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                  index < currentStep ? 'bg-green-500 border-green-500 text-white' :
                  index === currentStep ? 'bg-blue-500 border-blue-500 text-white' :
                  'bg-gray-100 border-gray-300 text-gray-400'
                }`}>
                  {index < currentStep ? <CheckCircle className="h-5 w-5" /> : index + 1}
                </div>
                <div className="ml-3 text-sm">
                  <div className={`font-medium ${
                    index === currentStep ? 'text-blue-600' : 
                    index < currentStep ? 'text-green-600' : 
                    'text-gray-400'
                  }`}>
                    {step}
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div className={`flex-1 h-0.5 mx-4 ${
                    index < currentStep ? 'bg-green-500' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
          <Progress value={((currentStep + 1) / steps.length) * 100} className="h-2" />
        </div>

        {/* Dataset Overview */}
        {fileMetadata && (
          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Dataset Overview
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {fileMetadata.row_count?.toLocaleString() || 0}
                  </div>
                  <div className="text-sm text-gray-600">Rows</div>
                </div>
                <div className="text-center p-3 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {fileMetadata.column_names?.length || 0}
                  </div>
                  <div className="text-sm text-gray-600">Features</div>
                </div>
                <div className="text-center p-3 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {fileMetadata.dataset_type === 'time_series' ? 'Time Series' : 'Tabular'}
                  </div>
                  <div className="text-sm text-gray-600">Type</div>
                </div>
                <div className="text-center p-3 bg-amber-50 rounded-lg">
                  <div className="text-2xl font-bold text-amber-600">
                    {configId ? 'Configured' : 'Pending'}
                  </div>
                  <div className="text-sm text-gray-600">Status</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Debug Info */}
        <div className="mb-4 p-2 bg-gray-100 rounded text-xs text-gray-600">
          Debug: Step={currentStep}, ConfigId={configId || 'None'}, TaskType={taskType || 'None'}, Target={targetColumn || 'None'}
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}

        {/* Step 0: Task Type & Target Selection */}
        {currentStep === 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Step 1: Select Task Type & Target Column
              </CardTitle>
              <CardDescription>
                Choose the type of machine learning problem and target column
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Task Type Selection */}
              <div>
                <h4 className="font-medium mb-3">Machine Learning Task Type</h4>
                <RadioGroup value={taskType || ''} onValueChange={(value: 'classification' | 'regression' | 'time_series') => setTaskType(value)}>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="flex items-center space-x-2 p-4 border-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                      <RadioGroupItem value="classification" id="classification" />
                      <Label htmlFor="classification" className="cursor-pointer flex-1">
                        <div className="flex items-center gap-3">
                          <BarChart3 className="h-6 w-6 text-blue-600" />
                          <div>
                            <div className="font-medium">Classification</div>
                            <div className="text-sm text-muted-foreground">
                              Predict categories (spam/not spam, yes/no)
                            </div>
                          </div>
                        </div>
                      </Label>
                    </div>
                    
                    <div className="flex items-center space-x-2 p-4 border-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                      <RadioGroupItem value="regression" id="regression" />
                      <Label htmlFor="regression" className="cursor-pointer flex-1">
                        <div className="flex items-center gap-3">
                          <TrendingUp className="h-6 w-6 text-green-600" />
                          <div>
                            <div className="font-medium">Regression</div>
                            <div className="text-sm text-muted-foreground">
                              Predict numbers (price, temperature)
                            </div>
                          </div>
                        </div>
                      </Label>
                    </div>
                    
                    <div className="flex items-center space-x-2 p-4 border-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                      <RadioGroupItem value="time_series" id="time_series" />
                      <Label htmlFor="time_series" className="cursor-pointer flex-1">
                        <div className="flex items-center gap-3">
                          <Clock className="h-6 w-6 text-purple-600" />
                          <div>
                            <div className="font-medium">Time Series</div>
                            <div className="text-sm text-muted-foreground">
                              Forecast future values (sales, stock prices)
                            </div>
                          </div>
                        </div>
                      </Label>
                    </div>
                  </div>
                </RadioGroup>
              </div>

              {/* Time Series Configuration */}
              {taskType === 'time_series' && (
                <div className="space-y-6">
                  {/* Forecasting Type Selection */}
                  <div>
                    <h4 className="font-medium mb-3">Forecasting Type</h4>
                    <RadioGroup 
                      value={forecastingType || ''} 
                      onValueChange={(value: 'univariate' | 'multivariate' | 'exogenous') => setForecastingType(value)}
                    >
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="flex items-center space-x-2 p-4 border-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                          <RadioGroupItem value="univariate" id="univariate" />
                          <Label htmlFor="univariate" className="cursor-pointer flex-1">
                            <div className="flex items-center gap-3">
                              <TrendingUp className="h-6 w-6 text-blue-600" />
                              <div>
                                <div className="font-medium">Univariate</div>
                                <div className="text-sm text-muted-foreground">
                                  Forecast single time series (sales over time)
                                </div>
                              </div>
                            </div>
                          </Label>
                        </div>
                        
                        <div className="flex items-center space-x-2 p-4 border-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                          <RadioGroupItem value="multivariate" id="multivariate" />
                          <Label htmlFor="multivariate" className="cursor-pointer flex-1">
                            <div className="flex items-center gap-3">
                              <BarChart3 className="h-6 w-6 text-green-600" />
                              <div>
                                <div className="font-medium">Multivariate</div>
                                <div className="text-sm text-muted-foreground">
                                  Forecast multiple related series (multiple products)
                                </div>
                              </div>
                            </div>
                          </Label>
                        </div>
                        
                        <div className="flex items-center space-x-2 p-4 border-2 rounded-lg hover:bg-gray-50 cursor-pointer transition-colors">
                          <RadioGroupItem value="exogenous" id="exogenous" />
                          <Label htmlFor="exogenous" className="cursor-pointer flex-1">
                            <div className="flex items-center gap-3">
                              <Layers className="h-6 w-6 text-purple-600" />
                              <div>
                                <div className="font-medium">Exogenous</div>
                                <div className="text-sm text-muted-foreground">
                                  Use external factors (weather, holidays, prices)
                                </div>
                              </div>
                            </div>
                          </Label>
                        </div>
                      </div>
                    </RadioGroup>
                  </div>

                  {/* Show explanation based on forecasting type */}
                  {forecastingType && (
                    <Alert>
                      <Sparkles className="h-4 w-4" />
                      <AlertTitle>
                        {forecastingType === 'univariate' && 'Univariate Forecasting'}
                        {forecastingType === 'multivariate' && 'Multivariate Forecasting'}
                        {forecastingType === 'exogenous' && 'Exogenous Forecasting'}
                      </AlertTitle>
                      <AlertDescription>
                        {forecastingType === 'univariate' && 
                          'You will forecast a single target variable using only its historical values and time patterns.'
                        }
                        {forecastingType === 'multivariate' && 
                          'You will forecast multiple target variables simultaneously, capturing relationships between different time series.'
                        }
                        {forecastingType === 'exogenous' && 
                          'You will forecast a target variable using both its historical values and external/exogenous variables that may influence it.'
                        }
                      </AlertDescription>
                    </Alert>
                  )}

                  {/* Time Column Detection - only show if forecasting type is selected */}
                  {forecastingType && (
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-medium">Time Column</h4>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={detectTimeColumns}
                          disabled={isDetectingTimeColumns}
                        >
                          {isDetectingTimeColumns ? (
                            <>
                              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              Detecting...
                            </>
                          ) : (
                            <>
                              <Clock className="h-4 w-4 mr-2" />
                              Auto-Detect
                            </>
                          )}
                        </Button>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {fileMetadata?.column_names?.map((column: string) => {
                          const isDetected = detectedTimeColumns.includes(column)
                          const isSelected = timeColumn === column
                          
                          return (
                            <div
                              key={column}
                              className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                                isSelected
                                  ? 'border-purple-500 bg-purple-50' 
                                  : isDetected
                                  ? 'border-purple-200 bg-purple-25 hover:bg-purple-50'
                                  : 'border-gray-200 hover:bg-gray-50'
                              }`}
                              onClick={() => setTimeColumn(column)}
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  <div className="font-medium flex items-center gap-2">
                                    {column}
                                    {isDetected && (
                                      <Badge variant="outline" className="text-xs bg-purple-100 text-purple-700">
                                        Auto-detected
                                      </Badge>
                                    )}
                                  </div>
                                  <div className="text-sm text-muted-foreground">
                                    {fileMetadata?.statistics?.[column]?.dtype || 'Unknown type'}
                                  </div>
                                </div>
                                {isSelected && (
                                  <CheckCircle className="h-5 w-5 text-purple-600" />
                                )}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* Target Column Selection */}
                  {timeColumn && (
                    <div>
                      <h4 className="font-medium mb-3">
                        {forecastingType === 'multivariate' 
                          ? 'Target Columns (Select multiple variables to forecast)' 
                          : 'Target Column (Variable to forecast)'
                        }
                      </h4>
                      
                      {forecastingType === 'multivariate' ? (
                        // Multiple selection for multivariate
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                          {fileMetadata?.column_names?.filter((col: string) => col !== timeColumn).map((column: string) => (
                            <div
                              key={column}
                              className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                                selectedFeatures.includes(column)
                                  ? 'border-blue-500 bg-blue-50' 
                                  : 'border-gray-200 hover:bg-gray-50'
                              }`}
                              onClick={() => {
                                if (selectedFeatures.includes(column)) {
                                  setSelectedFeatures(prev => prev.filter(f => f !== column))
                                } else {
                                  setSelectedFeatures(prev => [...prev, column])
                                }
                              }}
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  <div className="font-medium">{column}</div>
                                  <div className="text-sm text-muted-foreground">
                                    {fileMetadata?.statistics?.[column]?.dtype || 'Unknown type'}
                                  </div>
                                </div>
                                <Checkbox 
                                  checked={selectedFeatures.includes(column)}
                                  readOnly
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        // Single selection for univariate and exogenous
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                          {fileMetadata?.column_names?.filter((col: string) => col !== timeColumn).map((column: string) => (
                            <div
                              key={column}
                              className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                                targetColumn === column 
                                  ? 'border-blue-500 bg-blue-50' 
                                  : 'border-gray-200 hover:bg-gray-50'
                              }`}
                              onClick={() => setTargetColumn(column)}
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  <div className="font-medium">{column}</div>
                                  <div className="text-sm text-muted-foreground">
                                    {fileMetadata?.statistics?.[column]?.dtype || 'Unknown type'}
                                  </div>
                                </div>
                                {targetColumn === column && (
                                  <CheckCircle className="h-5 w-5 text-blue-600" />
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Exogenous Variables Selection */}
                  {forecastingType === 'exogenous' && targetColumn && (
                    <div>
                      <h4 className="font-medium mb-3">Exogenous Variables (External factors that influence the target)</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {fileMetadata?.column_names?.filter((col: string) => 
                          col !== timeColumn && col !== targetColumn
                        ).map((column: string) => (
                          <div
                            key={column}
                            className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                              exogenousColumns.includes(column)
                                ? 'border-green-500 bg-green-50' 
                                : 'border-gray-200 hover:bg-gray-50'
                            }`}
                            onClick={() => {
                              if (exogenousColumns.includes(column)) {
                                setExogenousColumns(prev => prev.filter(c => c !== column))
                              } else {
                                setExogenousColumns(prev => [...prev, column])
                              }
                            }}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="font-medium">{column}</div>
                                <div className="text-sm text-muted-foreground">
                                  {fileMetadata?.statistics?.[column]?.dtype || 'Unknown type'}
                                </div>
                              </div>
                              <Checkbox 
                                checked={exogenousColumns.includes(column)}
                                readOnly
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      
                      {exogenousColumns.length > 0 && (
                        <div className="mt-3 p-3 bg-green-50 rounded-lg">
                          <div className="text-sm font-medium text-green-900">
                            Selected Exogenous Variables: {exogenousColumns.length}
                          </div>
                          <div className="text-xs text-green-700">
                            {exogenousColumns.join(', ')}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Forecasting Configuration */}
                  {((forecastingType === 'univariate' && targetColumn) ||
                    (forecastingType === 'multivariate' && selectedFeatures.length > 0) ||
                    (forecastingType === 'exogenous' && targetColumn && exogenousColumns.length > 0)) && (
                    <div>
                      <h4 className="font-medium mb-3">Forecasting Configuration</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <Label htmlFor="forecast-horizon">Forecasting Horizon</Label>
                          <Select 
                            value={forecastingHorizon.toString()} 
                            onValueChange={(value) => setForecastingHorizon(parseInt(value))}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select horizon" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="7">7 periods</SelectItem>
                              <SelectItem value="12">12 periods</SelectItem>
                              <SelectItem value="24">24 periods</SelectItem>
                              <SelectItem value="30">30 periods</SelectItem>
                              <SelectItem value="52">52 periods</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div>
                          <Label htmlFor="train-split">Train/Test Split</Label>
                          <Select 
                            value={(trainingConfig.train_size || 0.8).toString()} 
                            onValueChange={(value) => setTrainingConfig(prev => ({ ...prev, train_size: parseFloat(value) }))}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="0.7">70% / 30%</SelectItem>
                              <SelectItem value="0.8">80% / 20%</SelectItem>
                              <SelectItem value="0.9">90% / 10%</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <div className="flex items-end">
                          <div className="p-3 bg-blue-50 rounded-lg w-full">
                            <div className="text-sm font-medium text-blue-900">Configuration Ready</div>
                            <div className="text-xs text-blue-700">
                              {forecastingType === 'multivariate' 
                                ? `${selectedFeatures.length} targets selected`
                                : `${targetColumn} → ${forecastingType}`
                              }
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Regular Target Column Selection (for classification/regression) */}
              {taskType && taskType !== 'time_series' && (
                <div>
                  <h4 className="font-medium mb-3">Select Target Column</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {fileMetadata?.column_names?.map((column: string) => (
                      <div
                        key={column}
                        className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                          targetColumn === column 
                            ? 'border-blue-500 bg-blue-50' 
                            : 'border-gray-200 hover:bg-gray-50'
                        }`}
                        onClick={() => setTargetColumn(column)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium">{column}</div>
                            <div className="text-sm text-muted-foreground">
                              {fileMetadata?.statistics?.[column]?.dtype || 'Unknown type'}
                            </div>
                          </div>
                          {targetColumn === column && (
                            <CheckCircle className="h-5 w-5 text-blue-600" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter>
              <Button 
                onClick={handleTaskTypeNext}
                disabled={!taskType || !targetColumn}
                className="ml-auto"
              >
                Continue to Feature Selection
                <Sparkles className="h-4 w-4 ml-2" />
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Step 1: Feature Selection */}
        {currentStep === 1 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
                Step 2: Feature Selection
              </CardTitle>
              <CardDescription>
                Select features for training. Recommended features are highlighted.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isAnalyzingFeatures ? (
                <div className="flex flex-col items-center justify-center p-12">
                  <Loader2 className="h-8 w-8 animate-spin mb-4" />
                  <p>Analyzing feature importance...</p>
                </div>
              ) : featureAnalysis ? (
                <div className="space-y-6">
                  {/* Feature Selection Actions */}
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        const recommended = featureAnalysis.recommended_features?.map(f => f.feature) || []
                        setSelectedFeatures(recommended)
                      }}
                    >
                      <Sparkles className="h-4 w-4 mr-2" />
                      Select Recommended
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => setSelectedFeatures([])}
                    >
                      Clear All
                    </Button>
                  </div>

                  {/* Feature Table */}
                  <div className="border rounded-lg overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-12">Select</TableHead>
                          <TableHead>Feature Name</TableHead>
                          <TableHead>Importance</TableHead>
                          <TableHead>Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {featureAnalysis.all_features
                          ?.filter(feature => feature.feature !== targetColumn)
                          ?.sort((a, b) => b.importance - a.importance)
                          ?.map((feature) => {
                            const isRecommended = featureAnalysis.recommended_features?.some(f => f.feature === feature.feature)
                            const isSelected = selectedFeatures.includes(feature.feature)
                            
                            return (
                              <TableRow 
                                key={feature.feature}
                                className={`cursor-pointer transition-colors ${
                                  isRecommended 
                                    ? 'bg-green-50 hover:bg-green-100' 
                                    : isSelected 
                                    ? 'bg-blue-50 hover:bg-blue-100' 
                                    : 'hover:bg-gray-50'
                                }`}
                                onClick={() => toggleFeature(feature.feature)}
                              >
                                <TableCell>
                                  <Checkbox
                                    checked={isSelected}
                                    onCheckedChange={() => toggleFeature(feature.feature)}
                                  />
                                </TableCell>
                                <TableCell>
                                  <div className="flex items-center gap-2">
                                    <span className="font-medium">{feature.feature}</span>
                                    {isRecommended && <Sparkles className="h-4 w-4 text-green-600" />}
                                  </div>
                                </TableCell>
                                <TableCell>
                                  <div className="text-sm font-medium">
                                    {feature.relative_percent?.toFixed(1) ?? (feature.importance * 100).toFixed(1)}%
                                  </div>
                                </TableCell>
                                <TableCell>
                                  {isRecommended ? (
                                    <Badge className="bg-green-500">Recommended</Badge>
                                  ) : isSelected ? (
                                    <Badge variant="outline">Selected</Badge>
                                  ) : (
                                    <Badge variant="outline" className="text-gray-500">Available</Badge>
                                  )}
                                </TableCell>
                              </TableRow>
                            )
                          })}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              ) : (
                <div className="text-center p-12">
                  <AlertCircle className="h-8 w-8 mx-auto mb-4 text-gray-400" />
                  <p>Feature analysis not available.</p>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={() => setCurrentStep(0)}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
              <Button 
                onClick={handleFeatureNext}
                disabled={selectedFeatures.length === 0}
              >
                Continue to Configuration
                <Settings className="h-4 w-4 ml-2" />
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Step 2: Training Configuration */}
        {currentStep === 2 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Step 3: Training Configuration
              </CardTitle>
              <CardDescription>
                Configure training parameters for your {taskType} model
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Basic Parameters */}
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Settings className="h-5 w-5" />
                    Basic Configuration
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <Label>Training Data Split</Label>
                        <Select 
                          value={trainingConfig.train_size.toString()} 
                          onValueChange={(value) => setTrainingConfig(prev => ({ ...prev, train_size: parseFloat(value) }))}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="0.6">60% Training, 40% Testing</SelectItem>
                            <SelectItem value="0.7">70% Training, 30% Testing</SelectItem>
                            <SelectItem value="0.8">80% Training, 20% Testing</SelectItem>
                            <SelectItem value="0.9">90% Training, 10% Testing</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          Split ratio for training and testing datasets
                        </p>
                      </div>
                      
                      <div>
                        <Label>Cross-Validation Folds</Label>
                        <Select 
                          value={cvFolds.toString()} 
                          onValueChange={(value) => setCvFolds(parseInt(value))}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="3">3 Folds (Fast)</SelectItem>
                            <SelectItem value="5">5 Folds (Balanced)</SelectItem>
                            <SelectItem value="10">10 Folds (Thorough)</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          More folds = more accurate but slower training
                        </p>
                      </div>

                      <div>
                        <Label>Optimization Metric</Label>
                        <Select 
                          value={sortMetric} 
                          onValueChange={setSortMetric}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="auto">Auto (Recommended)</SelectItem>
                            {taskType === 'regression' ? (
                              <>
                                <SelectItem value="R2">R² Score</SelectItem>
                                <SelectItem value="MAE">Mean Absolute Error</SelectItem>
                                <SelectItem value="RMSE">Root Mean Squared Error</SelectItem>
                                <SelectItem value="RMSLE">Root Mean Squared Log Error</SelectItem>
                              </>
                            ) : (
                              <>
                                <SelectItem value="Accuracy">Accuracy</SelectItem>
                                <SelectItem value="AUC">AUC</SelectItem>
                                <SelectItem value="F1">F1 Score</SelectItem>
                                <SelectItem value="Precision">Precision</SelectItem>
                                <SelectItem value="Recall">Recall</SelectItem>
                              </>
                            )}
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          Metric to optimize and rank models
                        </p>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <Label>Outlier Threshold</Label>
                        <Select 
                          value={trainingConfig.outliers_threshold.toString()} 
                          onValueChange={(value) => setTrainingConfig(prev => ({ ...prev, outliers_threshold: parseFloat(value) }))}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="0.01">1% (Very Strict)</SelectItem>
                            <SelectItem value="0.05">5% (Recommended)</SelectItem>
                            <SelectItem value="0.1">10% (Lenient)</SelectItem>
                            <SelectItem value="0.15">15% (Very Lenient)</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          Percentage of data to consider as outliers
                        </p>
                      </div>

                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <Label className="cursor-pointer">Data Normalization</Label>
                          <p className="text-xs text-muted-foreground">Scale features to 0-1 range</p>
                        </div>
                        <Checkbox
                          checked={trainingConfig.normalize}
                          onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, normalize: !!checked }))}
                        />
                      </div>
                      
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <Label className="cursor-pointer">Feature Transformation</Label>
                          <p className="text-xs text-muted-foreground">Apply power transforms</p>
                        </div>
                        <Checkbox
                          checked={trainingConfig.transformation}
                          onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, transformation: !!checked }))}
                        />
                      </div>
                      
                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <Label className="cursor-pointer">Remove Outliers</Label>
                          <p className="text-xs text-muted-foreground">Filter extreme values</p>
                        </div>
                        <Checkbox
                          checked={trainingConfig.remove_outliers}
                          onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, remove_outliers: !!checked }))}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Advanced Parameters */}
                <div>
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Zap className="h-5 w-5 text-amber-600" />
                    Advanced Options
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-3 border rounded-lg bg-gradient-to-r from-purple-50 to-pink-50">
                        <div>
                          <Label className="cursor-pointer font-semibold">Hyperparameter Tuning</Label>
                          <p className="text-xs text-muted-foreground">Optimize model parameters (slower)</p>
                        </div>
                        <Checkbox
                          checked={hyperparameterTuning}
                          onCheckedChange={(checked) => setHyperparameterTuning(!!checked)}
                        />
                      </div>

                      {hyperparameterTuning && (
                        <div>
                          <Label>Tuning Iterations</Label>
                          <Select 
                            value={tuningIterations.toString()} 
                            onValueChange={(value) => setTuningIterations(parseInt(value))}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="5">5 iterations (Fast)</SelectItem>
                              <SelectItem value="10">10 iterations (Balanced)</SelectItem>
                              <SelectItem value="20">20 iterations (Thorough)</SelectItem>
                              <SelectItem value="50">50 iterations (Exhaustive)</SelectItem>
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-muted-foreground mt-1">
                            Number of parameter combinations to try
                          </p>
                        </div>
                      )}

                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <Label className="cursor-pointer">Polynomial Features</Label>
                          <p className="text-xs text-muted-foreground">Create interaction features</p>
                        </div>
                        <Checkbox
                          checked={trainingConfig.polynomial_features}
                          onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, polynomial_features: !!checked }))}
                        />
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-3 border rounded-lg bg-gradient-to-r from-blue-50 to-cyan-50">
                        <div>
                          <Label className="cursor-pointer font-semibold">Ensemble Methods</Label>
                          <p className="text-xs text-muted-foreground">Combine multiple models</p>
                        </div>
                        <Checkbox
                          checked={ensembleMethods}
                          onCheckedChange={(checked) => setEnsembleMethods(!!checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between p-3 border rounded-lg bg-gradient-to-r from-green-50 to-emerald-50">
                        <div>
                          <Label className="cursor-pointer font-semibold">Model Stacking</Label>
                          <p className="text-xs text-muted-foreground">Stack best models for better accuracy</p>
                        </div>
                        <Checkbox
                          checked={stackingEnabled}
                          onCheckedChange={(checked) => setStackingEnabled(!!checked)}
                        />
                      </div>

                      <div className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <Label className="cursor-pointer">Feature Selection</Label>
                          <p className="text-xs text-muted-foreground">Auto-select best features</p>
                        </div>
                        <Checkbox
                          checked={trainingConfig.feature_selection}
                          onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, feature_selection: !!checked }))}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Information Alert */}
                <Alert className="bg-gradient-to-r from-amber-50 to-orange-50 border-amber-200">
                  <AlertCircle className="h-4 w-4 text-amber-600" />
                  <AlertTitle className="text-amber-900">Training Performance Note</AlertTitle>
                  <AlertDescription className="text-amber-800">
                    <ul className="list-disc list-inside space-y-1 text-sm mt-2">
                      <li>More CV folds = More accurate but slower (3 folds ≈ 1-2 min, 10 folds ≈ 3-5 min)</li>
                      <li>Hyperparameter tuning can increase training time by 5-10x</li>
                      <li>Ensemble methods and stacking train additional meta-models</li>
                      <li>Recommended: Start with default settings, then experiment</li>
                    </ul>
                  </AlertDescription>
                </Alert>

                {/* Configuration Summary */}
                <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 rounded-lg border-2 border-blue-200 dark:border-blue-800">
                  <h4 className="font-semibold mb-3 text-blue-900 dark:text-blue-100">Configuration Summary</h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-white dark:bg-gray-900">Target</Badge>
                      <span className="font-medium">{targetColumn}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-white dark:bg-gray-900">Task</Badge>
                      <span className="font-medium capitalize">{taskType}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-white dark:bg-gray-900">Features</Badge>
                      <span className="font-medium">{selectedFeatures.length}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-white dark:bg-gray-900">Train Split</Badge>
                      <span className="font-medium">{(trainingConfig.train_size * 100)}%</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-white dark:bg-gray-900">CV Folds</Badge>
                      <span className="font-medium">{cvFolds}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="bg-white dark:bg-gray-900">Metric</Badge>
                      <span className="font-medium">{sortMetric === 'auto' ? 'Auto' : sortMetric}</span>
                    </div>
                  </div>
                  {(hyperparameterTuning || ensembleMethods || stackingEnabled) && (
                    <div className="mt-3 pt-3 border-t border-blue-200 dark:border-blue-800">
                      <div className="text-xs font-medium text-blue-900 dark:text-blue-100 mb-2">Advanced Features Enabled:</div>
                      <div className="flex flex-wrap gap-2">
                        {hyperparameterTuning && (
                          <Badge className="bg-purple-500 text-white">Hyperparameter Tuning ({tuningIterations} iter)</Badge>
                        )}
                        {ensembleMethods && (
                          <Badge className="bg-blue-500 text-white">Ensemble Methods</Badge>
                        )}
                        {stackingEnabled && (
                          <Badge className="bg-green-500 text-white">Model Stacking</Badge>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={() => setCurrentStep(1)}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
              <Button 
                onClick={configureTraining}
                disabled={isConfiguringTraining}
              >
                {isConfiguringTraining ? (
                  <>
                    <Loader2 className="animate-spin h-4 w-4 mr-2" />
                    Configuring...
                  </>
                ) : (
                  <>
                    Start Training Setup
                    <Play className="h-4 w-4 ml-2" />
                  </>
                )}
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Step 3: Training */}
        {currentStep === 3 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Step 4: Model Training
              </CardTitle>
              <CardDescription>
                Training multiple models and comparing performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {!isTraining && (
                  <div className="text-center">
                    <Button 
                      onClick={startTraining}
                      size="lg" 
                      className="bg-blue-600 hover:bg-blue-700"
                      disabled={!configId && !localStorage.getItem('ml_config_id')}
                    >
                      <Play className="h-5 w-5 mr-2" />
                      Start Training Process
                    </Button>
                    
                    {!configId && !localStorage.getItem('ml_config_id') && (
                      <p className="text-red-500 text-sm mt-2">
                        No configuration available. Please reconfigure.
                      </p>
                    )}
                  </div>
                )}

                {isTraining && (
                  <div className="space-y-6">
                    {/* Training Info Card */}
                    <Card className="border-blue-200 bg-gradient-to-r from-blue-50 to-indigo-50">
                      <CardContent className="pt-6">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                          <div>
                            <div className="text-sm text-muted-foreground">Dataset Rows</div>
                            <div className="text-2xl font-bold text-blue-700">
                              {fileMetadata?.row_count?.toLocaleString() || 0}
                            </div>
                          </div>
                          <div>
                            <div className="text-sm text-muted-foreground">Features</div>
                            <div className="text-2xl font-bold text-green-700">
                              {selectedFeatures.length || fileMetadata?.column_names?.length || 0}
                            </div>
                          </div>
                          <div>
                            <div className="text-sm text-muted-foreground">CV Folds</div>
                            <div className="text-2xl font-bold text-purple-700">3</div>
                            <div className="text-xs text-muted-foreground">Cross-validation</div>
                          </div>
                          <div>
                            <div className="text-sm text-muted-foreground">Train/Test</div>
                            <div className="text-2xl font-bold text-amber-700">
                              {Math.round((trainingConfig.train_size || 0.8) * 100)}%
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {Math.round(fileMetadata?.row_count * (trainingConfig.train_size || 0.8))} / {Math.round(fileMetadata?.row_count * (1 - (trainingConfig.train_size || 0.8)))} rows
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-600" />
                      <h3 className="text-lg font-medium">Training Models in Real-Time...</h3>
                      <p className="text-muted-foreground">Status: {trainingStatus}</p>
                      {trainingResults?.current_model && (
                        <Badge className="mt-2 bg-blue-500">
                          <Zap className="h-3 w-3 mr-1" />
                          Currently: {trainingResults.current_model}
                        </Badge>
                      )}
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Models Completed</span>
                        <span>
                          {trainingResults?.total_models_tested || 0} models trained
                        </span>
                      </div>
                      <Progress 
                        value={trainingResults?.total_models_tested ? (trainingResults.total_models_tested * 5) : 10} 
                        className="h-2" 
                      />
                    </div>

                    {/* Live Leaderboard Preview */}
                    {trainingResults?.leaderboard && trainingResults.leaderboard.length > 0 && (() => {
                      // Determine display metric
                      const displayMetric = sortMetric === 'auto' 
                        ? (taskType === 'regression' ? 'R2' : 'Accuracy')
                        : sortMetric
                      
                      // Get metric display name
                      const metricNames: Record<string, string> = {
                        'R2': 'R² Score',
                        'MAE': 'MAE',
                        'RMSE': 'RMSE',
                        'RMSLE': 'RMSLE',
                        'MSE': 'MSE',
                        'Accuracy': 'Accuracy',
                        'AUC': 'AUC',
                        'F1': 'F1 Score',
                        'Precision': 'Precision',
                        'Recall': 'Recall'
                      }
                      const metricDisplayName = metricNames[displayMetric] || displayMetric
                      
                      return (
                      <Card className="border-blue-200 bg-blue-50/50">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-sm flex items-center gap-2">
                            <Sparkles className="h-4 w-4 text-blue-600" />
                            Live Results (Top 5)
                          </CardTitle>
                          <p className="text-xs text-muted-foreground mt-1">
                            Sorted by: {metricDisplayName}
                          </p>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            {trainingResults.leaderboard.slice(0, 5).map((model: any, index: number) => (
                              <div 
                                key={index} 
                                className="flex items-center justify-between p-2 bg-white dark:bg-gray-900 rounded border animate-in slide-in-from-top-2"
                              >
                                <div className="flex items-center gap-2">
                                  <Badge variant="outline" className="w-6 h-6 flex items-center justify-center">
                                    {index + 1}
                                  </Badge>
                                  <span className="font-medium text-sm">{model.Model}</span>
                                </div>
                                <div className="flex flex-col items-end">
                                  <div className="text-sm font-semibold text-blue-600">
                                    {(() => {
                                      const value = model[displayMetric] ?? 0
                                      // Classification metrics (0-1 range) should be shown as percentages
                                      const isPercentageMetric = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall'].includes(displayMetric)
                                      return isPercentageMetric 
                                        ? `${(value * 100).toFixed(2)}%`
                                        : value.toFixed(4)
                                    })()}
                                  </div>
                                  <div className="text-xs text-muted-foreground">
                                    {metricDisplayName}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                      )
                    })()}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 4: Results */}
        {currentStep === 4 && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="h-5 w-5" />
                  Training Results
                </CardTitle>
                <CardDescription>
                  Your models have been trained and evaluated successfully
                </CardDescription>
              </CardHeader>
              <CardContent>
                {trainingResults ? (
                  <div className="space-y-6">
                    {/* Success Summary */}
                    <div className="text-center p-6 bg-green-50 rounded-lg">
                      <CheckCircle className="h-12 w-12 text-green-600 mx-auto mb-4" />
                      <h3 className="text-xl font-semibold mb-2">Training Complete!</h3>
                      <p className="text-muted-foreground mb-4">
                        {trainingResults.total_models_tested} models tested, {trainingResults.models_saved} models saved
                      </p>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">{trainingResults.total_models_tested}</div>
                          <div className="text-sm text-gray-600">Models Tested</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">{trainingResults.models_saved}</div>
                          <div className="text-sm text-gray-600">Models Saved</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-600">
                            {taskType === 'classification' ? 'Accuracy' : 'R²'}
                          </div>
                          <div className="text-sm text-gray-600">Primary Metric</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-amber-600">
                            {trainingResults.completed_at ? new Date(trainingResults.completed_at).toLocaleTimeString() : 'N/A'}
                          </div>
                          <div className="text-sm text-gray-600">Completed</div>
                        </div>
                      </div>
                    </div>

                    {/* Best Model */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Award className="h-5 w-5 text-yellow-500" />
                          Best Performing Model
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="p-4 bg-yellow-50 rounded-lg">
                          <div className="font-medium text-lg mb-2">
                            {trainingResults.best_model_name?.split('(')[0] || 'Best Model'}
                          </div>
                          <div className="text-sm text-gray-600">
                            This model achieved the highest {taskType === 'classification' ? 'accuracy' : 'R² score'} on the test data
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Model Leaderboard */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5" />
                          Model Performance Leaderboard
                        </CardTitle>
                        <CardDescription>
                          Comparison of all trained models sorted by performance
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        {trainingResults.leaderboard && trainingResults.leaderboard.length > 0 ? (
                          <div className="overflow-x-auto">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Rank</TableHead>
                                  <TableHead>Model</TableHead>
                                  {taskType === 'classification' ? (
                                    <>
                                      <TableHead>Accuracy</TableHead>
                                      <TableHead>AUC</TableHead>
                                      <TableHead>Recall</TableHead>
                                      <TableHead>Precision</TableHead>
                                      <TableHead>F1</TableHead>
                                    </>
                                  ) : (
                                    <>
                                      <TableHead>R²</TableHead>
                                      <TableHead>RMSE</TableHead>
                                      <TableHead>MAE</TableHead>
                                      <TableHead>MSE</TableHead>
                                    </>
                                  )}
                                  <TableHead>Training Time</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {trainingResults.leaderboard.slice(0, 10).map((row: any, index: number) => (
                                  <TableRow key={index} className={index === 0 ? 'bg-yellow-50' : ''}>
                                    <TableCell>
                                      <div className="flex items-center gap-2">
                                        {index + 1}
                                        {index === 0 && <Award className="h-4 w-4 text-yellow-500" />}
                                      </div>
                                    </TableCell>
                                    <TableCell className="font-medium">{row.Model}</TableCell>
                                    {taskType === 'classification' ? (
                                      <>
                                        <TableCell>{(row.Accuracy * 100).toFixed(2)}%</TableCell>
                                        <TableCell>{row.AUC ? (row.AUC * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                                        <TableCell>{(row.Recall * 100).toFixed(2)}%</TableCell>
                                        <TableCell>{(row['Prec.'] * 100).toFixed(2)}%</TableCell>
                                        <TableCell>{(row.F1 * 100).toFixed(2)}%</TableCell>
                                      </>
                                    ) : (
                                      <>
                                        <TableCell>{row.R2?.toFixed(4) || 'N/A'}</TableCell>
                                        <TableCell>{row.RMSE?.toFixed(4) || 'N/A'}</TableCell>
                                        <TableCell>{row.MAE?.toFixed(4) || 'N/A'}</TableCell>
                                        <TableCell>{row.MSE?.toFixed(4) || 'N/A'}</TableCell>
                                      </>
                                    )}
                                    <TableCell>{row['TT (Sec)']?.toFixed(2) || 'N/A'}s</TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                        ) : (
                          <div className="text-center p-8">
                            <AlertCircle className="h-8 w-8 mx-auto mb-4 text-gray-400" />
                            <p>No leaderboard data available</p>
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {/* Download Options */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Database className="h-5 w-5" />
                          Download & Save Results
                        </CardTitle>
                        <CardDescription>
                          Download trained models, save to database, or get performance reports
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <Button 
                            variant="outline" 
                            onClick={() => {
                              const configIdToUse = configId || localStorage.getItem('ml_config_id')
                              if (configIdToUse) {
                                window.open(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/download-model/${configIdToUse}/best_model`, '_blank')
                              }
                            }}
                            className="h-auto p-4 flex flex-col items-start gap-2"
                          >
                            <Download className="h-5 w-5" />
                            <div className="text-left">
                              <div className="font-medium">Best Model</div>
                              <div className="text-sm text-muted-foreground">Download top performing model (.pkl)</div>
                            </div>
                          </Button>
                          
                          <Button 
                            variant="outline" 
                            onClick={() => {
                              const configIdToUse = configId || localStorage.getItem('ml_config_id')
                              if (configIdToUse) {
                                window.open(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/download-leaderboard/${configIdToUse}`, '_blank')
                              }
                            }}
                            className="h-auto p-4 flex flex-col items-start gap-2"
                          >
                            <Download className="h-5 w-5" />
                            <div className="text-left">
                              <div className="font-medium">Performance Report</div>
                              <div className="text-sm text-muted-foreground">Download metrics CSV</div>
                            </div>
                          </Button>

                          <Button 
                            variant="default"
                            onClick={() => {
                              setSelectedModelToSave(trainingResults?.leaderboard?.[0])
                              setShowSaveDialog(true)
                            }}
                            className="h-auto p-4 flex flex-col items-start gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                          >
                            <Save className="h-5 w-5" />
                            <div className="text-left">
                              <div className="font-medium">Save to Database</div>
                              <div className="text-sm opacity-90">Store best model in Supabase</div>
                            </div>
                          </Button>
                        </div>

                        {/* Download All Models Section */}
                        <div className="mt-6 pt-6 border-t">
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="font-medium flex items-center gap-2">
                              <Package className="h-4 w-4" />
                              Available Models
                            </h4>
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => {
                                const configIdToUse = configId || localStorage.getItem('ml_config_id')
                                if (configIdToUse) {
                                  window.open(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/download-all-models/${configIdToUse}`, '_blank')
                                }
                              }}
                              className="gap-2"
                            >
                              <Package className="h-4 w-4" />
                              Download All as ZIP
                            </Button>
                          </div>
                          <div className="space-y-2">
                            {trainingResults.leaderboard?.slice(0, 5).map((model: any, index: number) => {
                              // Use the selected metric for display
                              const displayMetric = sortMetric === 'auto' 
                                ? (taskType === 'regression' ? 'R2' : 'Accuracy')
                                : sortMetric
                              const metricValue = model[displayMetric] ?? 0
                              
                              // Get metric display name
                              const metricNames: Record<string, string> = {
                                'R2': 'R²',
                                'MAE': 'MAE',
                                'RMSE': 'RMSE',
                                'RMSLE': 'RMSLE',
                                'MSE': 'MSE',
                                'Accuracy': 'Accuracy',
                                'AUC': 'AUC',
                                'F1': 'F1',
                                'Precision': 'Precision',
                                'Recall': 'Recall'
                              }
                              const metricDisplayName = metricNames[displayMetric] || displayMetric
                              
                              // Format based on metric type
                              // Classification metrics (0-1 range) should be shown as percentages
                              const isPercentageMetric = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall'].includes(displayMetric)
                              const formattedValue = isPercentageMetric
                                ? `${(metricValue * 100).toFixed(2)}%`
                                : metricValue.toFixed(4)
                              
                              return (
                              <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg hover:bg-muted/80 transition-colors">
                                <div className="flex items-center gap-3">
                                  {index === 0 && <Award className="h-4 w-4 text-yellow-500" />}
                                  <div>
                                    <div className="font-medium text-sm">{model.Model}</div>
                                    <div className="text-xs text-muted-foreground">
                                      {metricDisplayName}: {formattedValue}
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center gap-2">
                                  <Button 
                                    variant="outline" 
                                    size="sm"
                                    onClick={() => {
                                      setSelectedModelToSave({
                                        ...model,
                                        modelIndex: index
                                      })
                                      setShowSaveDialog(true)
                                    }}
                                  >
                                    <Save className="h-4 w-4 mr-1" />
                                    Save
                                  </Button>
                                  <Button 
                                    variant="ghost" 
                                    size="sm"
                                    onClick={() => {
                                      const configIdToUse = configId || localStorage.getItem('ml_config_id')
                                      if (configIdToUse) {
                                        // Extract model ID from model name for download
                                        const modelId = index === 0 ? 'best_model' : model.Model.toLowerCase().replace(/\s+/g, '_')
                                        window.open(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/download-model/${configIdToUse}/${modelId}`, '_blank')
                                      }
                                    }}
                                  >
                                    <Download className="h-4 w-4 mr-1" />
                                    Download
                                  </Button>
                                </div>
                              </div>
                              )
                            })}
                          </div>
                          {trainingResults.leaderboard && trainingResults.leaderboard.length > 5 && (
                            <p className="text-xs text-muted-foreground mt-3">
                              Showing top 5 models. Download best model to get the top performer.
                            </p>
                          )}
                          
                          {/* View Saved Models Link */}
                          <div className="mt-4 pt-4 border-t">
                            <Button 
                              variant="link" 
                              className="w-full justify-center gap-2 text-blue-600 hover:text-blue-700"
                              onClick={() => router.push('/dashboard/models')}
                            >
                              <FileCode2 className="h-4 w-4" />
                              View All Saved Models in Database
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Model Testing Interface */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Play className="h-5 w-5 text-blue-600" />
                          Test Your Models
                        </CardTitle>
                        <CardDescription>
                          Enter values to get predictions from trained models
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-6">
                          {/* Model Selection */}
                          <div>
                            <label className="text-sm font-medium mb-2 block">Select Model to Test</label>
                            <Select 
                              value={selectedModelForTest || ''} 
                              onValueChange={(value) => {
                                setSelectedModelForTest(value)
                                setTestPrediction(null)
                                // Initialize inputs for all features
                                const inputs: Record<string, string> = {}
                                selectedFeatures.forEach(feature => {
                                  inputs[feature] = ''
                                })
                                setTestInputs(inputs)
                              }}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Choose a trained model..." />
                              </SelectTrigger>
                              <SelectContent>
                                {trainingResults.leaderboard?.map((model: any, index: number) => {
                                  // Use the selected metric for display
                                  const displayMetric = sortMetric === 'auto' 
                                    ? (taskType === 'regression' ? 'R2' : 'Accuracy')
                                    : sortMetric
                                  const metricValue = model[displayMetric] ?? 0
                                  
                                  // Format based on metric type
                                  // Classification metrics (0-1 range) should be shown as percentages
                                  const isPercentageMetric = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall'].includes(displayMetric)
                                  const formattedValue = isPercentageMetric
                                    ? `${(metricValue * 100).toFixed(1)}%`
                                    : metricValue.toFixed(3)
                                  
                                  return (
                                  <SelectItem key={index} value={model.Model}>
                                    <div className="flex items-center justify-between w-full">
                                      <span>{model.Model}</span>
                                      <Badge variant={index === 0 ? "default" : "outline"} className="ml-2">
                                        {formattedValue}
                                      </Badge>
                                    </div>
                                  </SelectItem>
                                  )
                                })}
                              </SelectContent>
                            </Select>
                          </div>

                          {/* Input Fields */}
                          {selectedModelForTest && (
                            <div className="space-y-4">
                              <div className="p-4 bg-blue-50 rounded-lg">
                                <h4 className="font-medium mb-3 flex items-center gap-2">
                                  <Target className="h-4 w-4" />
                                  Enter Feature Values
                                </h4>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  {selectedFeatures.map((feature) => (
                                    <div key={feature}>
                                      <label className="text-sm font-medium mb-1 block">{feature}</label>
                                      <Input
                                        type="number"
                                        step="any"
                                        placeholder={`Enter ${feature}...`}
                                        value={testInputs[feature] || ''}
                                        onChange={(e) => setTestInputs(prev => ({
                                          ...prev,
                                          [feature]: e.target.value
                                        }))}
                                      />
                                    </div>
                                  ))}
                                </div>
                              </div>

                              {/* Test Button */}
                              <Button 
                                onClick={async () => {
                                  setIsTesting(true)
                                  setTestPrediction(null)
                                  
                                  try {
                                    const configIdToUse = configId || localStorage.getItem('ml_config_id')
                                    if (!configIdToUse) {
                                      toast({
                                        variant: "destructive",
                                        title: "Error",
                                        description: "Configuration ID not found"
                                      })
                                      return
                                    }

                                    // Prepare data
                                    const formData = new FormData()
                                    formData.append('config_id', configIdToUse)
                                    formData.append('model_name', selectedModelForTest)
                                    formData.append('input_data', JSON.stringify(testInputs))

                                    const response = await fetch(
                                      `${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/predict/`,
                                      {
                                        method: 'POST',
                                        body: formData
                                      }
                                    )

                                    if (!response.ok) {
                                      const errorData = await response.json()
                                      throw new Error(errorData.detail || 'Prediction failed')
                                    }

                                    const result = await response.json()
                                    setTestPrediction(result)
                                    
                                    toast({
                                      title: "Prediction Complete",
                                      description: "Model inference successful"
                                    })
                                  } catch (error: any) {
                                    toast({
                                      variant: "destructive",
                                      title: "Prediction Failed",
                                      description: error.message
                                    })
                                  } finally {
                                    setIsTesting(false)
                                  }
                                }}
                                disabled={isTesting || Object.values(testInputs).some(v => !v)}
                                className="w-full"
                                size="lg"
                              >
                                {isTesting ? (
                                  <>
                                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                                    Running Prediction...
                                  </>
                                ) : (
                                  <>
                                    <Zap className="h-4 w-4 mr-2" />
                                    Get Prediction
                                  </>
                                )}
                              </Button>

                              {/* Prediction Result */}
                              {testPrediction && (
                                <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 rounded-lg border-2 border-green-200 dark:border-green-800">
                                  <div className="flex items-center gap-3 mb-4">
                                    <CheckCircle className="h-6 w-6 text-green-600 dark:text-green-400" />
                                    <h4 className="font-semibold text-lg">Prediction Result</h4>
                                  </div>
                                  <div className="space-y-3">
                                    <div className="flex items-center justify-between p-4 bg-white dark:bg-gray-900 rounded-lg">
                                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Predicted {targetColumn}:</span>
                                      <span className="text-2xl font-bold text-green-600 dark:text-green-400">
                                        {testPrediction.prediction?.toFixed(4) || testPrediction.prediction}
                                      </span>
                                    </div>
                                    <div className="flex items-center justify-between p-3 bg-white dark:bg-gray-900 rounded-lg">
                                      <span className="text-sm text-gray-600 dark:text-gray-400">Model Used:</span>
                                      <span className="text-sm font-medium">{selectedModelForTest}</span>
                                    </div>
                                    {testPrediction.confidence && (
                                      <div className="flex items-center justify-between p-3 bg-white dark:bg-gray-900 rounded-lg">
                                        <span className="text-sm text-gray-600 dark:text-gray-400">Confidence:</span>
                                        <Badge variant="outline">{(testPrediction.confidence * 100).toFixed(1)}%</Badge>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}

                          {!selectedModelForTest && (
                            <div className="text-center p-8 border-2 border-dashed rounded-lg">
                              <Zap className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                              <p className="text-gray-600">Select a model above to start testing</p>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>

                    {/* Next Steps */}
                    <Card>
                      <CardHeader>
                        <CardTitle>Next Steps</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <Button 
                            onClick={() => {
                              // Reset and start new training
                              setCurrentStep(0)
                              setConfigId(null)
                              setTrainingResults(null)
                              setTrainingStatus('idle')
                              localStorage.removeItem('ml_config_id')
                            }}
                            variant="outline"
                            className="w-full"
                          >
                            Train Another Model
                          </Button>
                          <Button 
                            onClick={() => router.push('/dashboard/blueprints')}
                            className="w-full"
                          >
                            Back to Blueprints
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                ) : (
                  <div className="text-center p-12">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                    <p>Loading results...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Save Model Dialog */}
      <SaveModelDialog
        open={showSaveDialog}
        onOpenChange={(open) => {
          setShowSaveDialog(open)
          if (!open) setSelectedModelToSave(null)
        }}
        taskId={configId || ''}
        modelInfo={{
          algorithm: selectedModelToSave?.Model || trainingResults?.best_model_name || 'Best Model',
          metrics: selectedModelToSave || trainingResults?.leaderboard?.[0] || {},
          taskType: taskType || 'classification',
          fileId: fileId
        }}
        userId={userId}
      />
    </section>
  )
}