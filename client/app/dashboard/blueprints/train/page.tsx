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
  Zap,
  CheckCircle,
  AlertCircle,
  Sparkles,
  Database,
  Award
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"

interface FeatureImportance {
  feature: string
  importance: number
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
  const [taskType, setTaskType] = useState<'classification' | 'regression' | null>(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [featureAnalysis, setFeatureAnalysis] = useState<FeatureAnalysis | null>(null)
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
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

  // Training state
  const [configId, setConfigId] = useState<string | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<string>('idle')
  const [trainingResults, setTrainingResults] = useState<any>(null)

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
      
      // Prepare configuration data
      const formData = new FormData()
      formData.append('file', fileBlob, filename)
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

      console.log('Sending configuration to backend...')
      const configResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/configure-training-with-file/`, {
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
          description: `Training configured with ID: ${newConfigId}`
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
      
      if (result.success) {
        pollTrainingStatus(currentConfigId)
      }
    } catch (error) {
      console.error('Training start error:', error)
      setErrorMessage(`Training failed to start: ${error}`)
      setIsTraining(false)
    }
  }

  const pollTrainingStatus = async (taskId: string) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/training-status/${taskId}`)
      
      if (!response.ok) throw new Error('Failed to check training status')
      
      const result = await response.json()
      setTrainingStatus(result.status)
      
      if (result.status === 'completed') {
        setTrainingResults(result)
        setIsTraining(false)
        setCurrentStep(4) // Move to results
        toast({
          title: "Training Complete!",
          description: "Your models have been trained successfully."
        })
      } else if (result.status === 'failed') {
        setErrorMessage(result.error || 'Training failed')
        setIsTraining(false)
      } else {
        // Continue polling
        setTimeout(() => pollTrainingStatus(taskId), 3000)
      }
    } catch (error) {
      console.error('Status check error:', error)
      setTimeout(() => pollTrainingStatus(taskId), 5000)
    }
  }

  // Step navigation functions
  const handleTaskTypeNext = () => {
    if (!taskType || !targetColumn) {
      toast({
        variant: "destructive",
        title: "Selection Required",
        description: "Please select task type and target column"
      })
      return
    }
    
    // Start feature analysis
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
                <RadioGroup value={taskType || ''} onValueChange={(value: 'classification' | 'regression') => setTaskType(value)}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                  </div>
                </RadioGroup>
              </div>

              {/* Target Column Selection */}
              {taskType && (
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
                                    {(feature.importance * 100).toFixed(1)}%
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
                        <SelectItem value="0.7">70% Training, 30% Testing</SelectItem>
                        <SelectItem value="0.8">80% Training, 20% Testing</SelectItem>
                        <SelectItem value="0.9">90% Training, 10% Testing</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
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
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>Data Normalization</Label>
                    <Checkbox
                      checked={trainingConfig.normalize}
                      onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, normalize: !!checked }))}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Feature Transformation</Label>
                    <Checkbox
                      checked={trainingConfig.transformation}
                      onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, transformation: !!checked }))}
                    />
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <Label>Remove Outliers</Label>
                    <Checkbox
                      checked={trainingConfig.remove_outliers}
                      onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, remove_outliers: !!checked }))}
                    />
                  </div>
                </div>
              </div>

              {/* Configuration Summary */}
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium mb-2">Configuration Summary</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>Target: <span className="font-medium">{targetColumn}</span></div>
                  <div>Task: <span className="font-medium">{taskType}</span></div>
                  <div>Features: <span className="font-medium">{selectedFeatures.length}</span></div>
                  <div>Train Split: <span className="font-medium">{(trainingConfig.train_size * 100)}%</span></div>
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
                  <div className="space-y-4">
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                      <h3 className="text-lg font-medium">Training Models...</h3>
                      <p className="text-muted-foreground">Status: {trainingStatus}</p>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Training Progress</span>
                        <span>This may take 5-15 minutes</span>
                      </div>
                      <Progress value={70} className="h-2" />
                    </div>
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
                          Download Results
                        </CardTitle>
                        <CardDescription>
                          Download trained models and performance reports
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <Button 
                            variant="outline" 
                            onClick={() => {
                              const configIdToUse = configId || localStorage.getItem('ml_config_id')
                              if (configIdToUse) {
                                window.open(`${process.env.NEXT_PUBLIC_BACKEND_URL}/ml/download-model/${configIdToUse}/best_model`, '_blank')
                              }
                            }}
                            className="h-auto p-4"
                          >
                            <div className="text-left">
                              <div className="font-medium">Best Model</div>
                              <div className="text-sm text-muted-foreground">Download the top performing model</div>
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
                            className="h-auto p-4"
                          >
                            <div className="text-left">
                              <div className="font-medium">Performance Report</div>
                              <div className="text-sm text-muted-foreground">Download detailed metrics CSV</div>
                            </div>
                          </Button>
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
    </section>
  )
}