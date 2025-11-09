'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  Download,
  Trash2,
  Search,
  FileCode2,
  Calendar,
  HardDrive,
  Filter,
  Loader2,
  TrendingUp,
} from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { createClient } from '@/utils/supabase/client'
import { listModels, downloadModelFile, deleteModel, getModelStats, type TrainedModel, type ModelStats } from '@/lib/api/models'

export default function SavedModelsPage() {
  const [models, setModels] = useState<TrainedModel[]>([])
  const [stats, setStats] = useState<ModelStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterStatus, setFilterStatus] = useState<string>('')
  const [userId, setUserId] = useState<string | null>(null)
  const { toast } = useToast()
  const supabase = createClient()

  useEffect(() => {
    const getUser = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser()
      if (user) {
        setUserId(user.id)
        await fetchModels(user.id)
        await fetchStats(user.id)
      }
    }
    getUser()
  }, [])

  const fetchModels = async (uid: string, status?: string) => {
    try {
      setLoading(true)
      const data = await listModels(uid, status)
      setModels(data.models)
    } catch (error) {
      console.error('Error fetching models:', error)
      toast({
        title: 'Failed to load models',
        description: 'Please try again',
        variant: 'destructive',
      })
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async (uid: string) => {
    try {
      const data = await getModelStats(uid)
      setStats(data)
    } catch (error) {
      console.error('Error fetching stats:', error)
    }
  }

  const handleDownload = async (model: TrainedModel) => {
    if (!userId) return

    try {
      await downloadModelFile(model.id, userId, model.model_name)
      toast({
        title: 'Download started',
        description: `Downloading ${model.model_name}.pkl`,
      })
    } catch (error) {
      toast({
        title: 'Download failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    }
  }

  const handleDelete = async (model: TrainedModel) => {
    if (!userId) return
    if (!confirm(`Are you sure you want to delete "${model.model_name}"?`)) return

    try {
      await deleteModel(model.id, userId)
      toast({
        title: 'Model deleted',
        description: `${model.model_name} has been removed`,
      })
      // Refresh list
      await fetchModels(userId, filterStatus || undefined)
      if (userId) await fetchStats(userId)
    } catch (error) {
      toast({
        title: 'Delete failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    }
  }

  const filteredModels = models.filter((model) =>
    model.model_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    model.algorithm.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getMetricBadgeColor = (key: string, value: number) => {
    // R2 and Accuracy: higher is better
    if (key === 'R2' || key === 'Accuracy') {
      if (value >= 0.9) return 'bg-green-500'
      if (value >= 0.7) return 'bg-yellow-500'
      return 'bg-red-500'
    }
    // MAE, RMSE, MSE: lower is better
    if (key === 'MAE' || key === 'RMSE' || key === 'MSE') {
      if (value <= 0.1) return 'bg-green-500'
      if (value <= 0.5) return 'bg-yellow-500'
      return 'bg-red-500'
    }
    return 'bg-blue-500'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    )
  }

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Saved Models</h1>
        <p className="text-muted-foreground">
          Manage and download your trained machine learning models
        </p>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid gap-4 md:grid-cols-4 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Models</CardTitle>
              <FileCode2 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_models}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Ready to Use</CardTitle>
              <TrendingUp className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.ready_models}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
              <HardDrive className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_storage_mb.toFixed(2)} MB</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Last Created</CardTitle>
              <Calendar className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-sm font-medium">
                {stats.last_model_created
                  ? new Date(stats.last_model_created).toLocaleDateString()
                  : 'N/A'}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search models by name or algorithm..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <div className="flex gap-2">
          <Button
            variant={filterStatus === '' ? 'default' : 'outline'}
            onClick={() => {
              setFilterStatus('')
              if (userId) fetchModels(userId)
            }}
          >
            All
          </Button>
          <Button
            variant={filterStatus === 'ready' ? 'default' : 'outline'}
            onClick={() => {
              setFilterStatus('ready')
              if (userId) fetchModels(userId, 'ready')
            }}
          >
            Ready
          </Button>
        </div>
      </div>

      {/* Models Grid */}
      {filteredModels.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <FileCode2 className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No models found</h3>
            <p className="text-sm text-muted-foreground text-center">
              {searchQuery
                ? 'Try adjusting your search query'
                : 'Train and save your first model to get started'}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map((model) => (
            <Card key={model.id} className="flex flex-col">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <CardTitle className="text-lg mb-2">{model.model_name}</CardTitle>
                    <CardDescription className="line-clamp-2">
                      {model.description || 'No description provided'}
                    </CardDescription>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 mt-3">
                  <Badge variant="secondary">{model.algorithm}</Badge>
                  <Badge variant="outline">{model.model_type}</Badge>
                  {model.status === 'ready' && (
                    <Badge className="bg-green-500">Ready</Badge>
                  )}
                </div>
              </CardHeader>

              <CardContent className="flex-1">
                {/* Metrics */}
                <div className="space-y-2 mb-4">
                  <p className="text-sm font-medium text-muted-foreground">Key Metrics:</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(model.metrics)
                      .slice(0, 4)
                      .map(([key, value]) => (
                        <Badge
                          key={key}
                          variant="outline"
                          className={`${getMetricBadgeColor(key, value)} text-white border-none`}
                        >
                          {key}: {typeof value === 'number' ? value.toFixed(4) : value}
                        </Badge>
                      ))}
                  </div>
                </div>

                {/* Tags */}
                {model.tags && model.tags.length > 0 && (
                  <div className="mb-4">
                    <p className="text-sm font-medium text-muted-foreground mb-2">Tags:</p>
                    <div className="flex flex-wrap gap-1">
                      {model.tags.map((tag, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Metadata */}
                <div className="text-xs text-muted-foreground space-y-1">
                  <div className="flex items-center gap-2">
                    <HardDrive className="h-3 w-3" />
                    <span>{(model.model_file_size / (1024 * 1024)).toFixed(2)} MB</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="h-3 w-3" />
                    <span>{new Date(model.created_at).toLocaleDateString()}</span>
                  </div>
                  {model.training_time_seconds && (
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-3 w-3" />
                      <span>Trained in {model.training_time_seconds.toFixed(2)}s</span>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex gap-2 mt-4">
                  <Button
                    size="sm"
                    onClick={() => handleDownload(model)}
                    className="flex-1"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => handleDelete(model)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
