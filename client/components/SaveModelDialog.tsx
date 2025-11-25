'use client'

import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Save, Download, X, Loader2, CheckCircle2 } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { saveModel, downloadModelFile } from '@/lib/api/models'

interface SaveModelDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  taskId: string
  modelInfo: {
    algorithm: string
    metrics: Record<string, number>
    taskType: string
    fileId?: string
  }
  userId: string
}

function SaveModelDialog({
  open,
  onOpenChange,
  taskId,
  modelInfo,
  userId,
}: SaveModelDialogProps) {
  const [modelName, setModelName] = useState('')
  const [description, setDescription] = useState('')
  const [tags, setTags] = useState('')
  const [saving, setSaving] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [saved, setSaved] = useState(false)
  const [savedModelId, setSavedModelId] = useState<string | null>(null)
  const { toast } = useToast()

  const handleSave = async () => {
    if (!modelName.trim()) {
      toast({
        title: 'Model name required',
        description: 'Please enter a name for your model',
        variant: 'destructive',
      })
      return
    }

    try {
      setSaving(true)

      const tagList = tags
        .split(',')
        .map((t) => t.trim())
        .filter((t) => t.length > 0)

      const isTimeSeries = modelInfo.taskType === 'time_series'
      
      const result = await saveModel(
        {
          task_id: taskId,
          model_name: modelName,
          description: description || undefined,
          tags: tagList.length > 0 ? tagList : undefined,
        },
        userId,
        isTimeSeries
      )

      if (result.success && result.model_id) {
        setSaved(true)
        setSavedModelId(result.model_id)
        toast({
          title: 'Model saved! 🎉',
          description: `${modelName} has been saved successfully`,
        })
      }
    } catch (error) {
      console.error('Error saving model:', error)
      toast({
        title: 'Failed to save model',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setSaving(false)
    }
  }

  const handleDownload = async () => {
    if (!savedModelId) return

    try {
      setDownloading(true)
      await downloadModelFile(savedModelId, userId, modelName)
      toast({
        title: 'Download started',
        description: 'Your model file is being downloaded',
      })
    } catch (error) {
      console.error('Error downloading model:', error)
      toast({
        title: 'Download failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      })
    } finally {
      setDownloading(false)
    }
  }

  const handleClose = () => {
    setModelName('')
    setDescription('')
    setTags('')
    setSaved(false)
    setSavedModelId(null)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {saved ? (
              <>
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                Model Saved Successfully!
              </>
            ) : (
              <>
                <Save className="h-5 w-5" />
                Save Trained Model
              </>
            )}
          </DialogTitle>
          <DialogDescription>
            {saved
              ? 'Your model has been saved. You can download it or close this dialog.'
              : 'Save this trained model for later use. You can download the pickle file anytime.'}
          </DialogDescription>
        </DialogHeader>

        {!saved ? (
          <div className="grid gap-4 py-4">
            {/* Model Info */}
            <div className="grid gap-2">
              <Label className="text-sm font-medium">Model Information</Label>
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">{modelInfo.algorithm}</Badge>
                <Badge variant="outline">{modelInfo.taskType}</Badge>
                {Object.entries(modelInfo.metrics)
                  .slice(0, 2)
                  .map(([key, value]) => (
                    <Badge key={key} variant="outline">
                      {key}: {typeof value === 'number' ? value.toFixed(4) : value}
                    </Badge>
                  ))}
              </div>
            </div>

            {/* Model Name */}
            <div className="grid gap-2">
              <Label htmlFor="model-name">
                Model Name <span className="text-red-500">*</span>
              </Label>
              <Input
                id="model-name"
                placeholder="e.g., Credit Card Fraud Detector v1"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={saving}
              />
            </div>

            {/* Description */}
            <div className="grid gap-2">
              <Label htmlFor="description">Description (Optional)</Label>
              <Textarea
                id="description"
                placeholder="What does this model do? What dataset was used?"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                disabled={saving}
                rows={3}
              />
            </div>

            {/* Tags */}
            <div className="grid gap-2">
              <Label htmlFor="tags">Tags (Optional)</Label>
              <Input
                id="tags"
                placeholder="e.g., production, fraud-detection, v1 (comma separated)"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                disabled={saving}
              />
              <p className="text-xs text-muted-foreground">
                Separate multiple tags with commas
              </p>
            </div>
          </div>
        ) : (
          <div className="py-6 text-center">
            <CheckCircle2 className="h-16 w-16 mx-auto text-green-500 mb-4" />
            <p className="text-lg font-medium mb-2">{modelName}</p>
            <p className="text-sm text-muted-foreground">
              Model ID: {savedModelId?.substring(0, 8)}...
            </p>
          </div>
        )}

        <DialogFooter>
          {!saved ? (
            <>
              <Button variant="outline" onClick={handleClose} disabled={saving}>
                Cancel
              </Button>
              <Button onClick={handleSave} disabled={saving}>
                {saving ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="mr-2 h-4 w-4" />
                    Save Model
                  </>
                )}
              </Button>
            </>
          ) : (
            <>
              <Button variant="outline" onClick={handleClose}>
                Close
              </Button>
              <Button onClick={handleDownload} disabled={downloading}>
                {downloading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="mr-2 h-4 w-4" />
                    Download Now
                  </>
                )}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default SaveModelDialog
