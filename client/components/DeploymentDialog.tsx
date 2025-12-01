/**
 * DeploymentDialog - Modal for deploying models to Azure ML
 */
"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { deployModel, DeployModelRequest, DeployModelResponse } from "@/lib/api/deployments"
import { CloudIcon, Loader2Icon, CheckCircle2Icon, AlertCircleIcon, ExternalLinkIcon } from "lucide-react"

interface DeploymentDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  modelId: string
  modelName: string
  userId: string
  onSuccess?: () => void
}

export default function DeploymentDialog({
  open,
  onOpenChange,
  modelId,
  modelName,
  userId,
  onSuccess,
}: DeploymentDialogProps) {
  const [deploymentName, setDeploymentName] = useState("")
  // Endpoint name is now fixed in backend
  const [instanceType, setInstanceType] = useState("Standard_DS1_v2")
  const [instanceCount, setInstanceCount] = useState(1)
  const [description, setDescription] = useState("")
  const [isDeploying, setIsDeploying] = useState(false)
  const [deploymentResult, setDeploymentResult] = useState<DeployModelResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleDeploy = async () => {
    if (!deploymentName.trim()) {
      setError("Deployment name is required")
      return
    }

    // Azure naming validation
    const azureNameRegex = /^[a-zA-Z][a-zA-Z0-9-]*$/
    if (!azureNameRegex.test(deploymentName.trim())) {
      setError("Deployment name must start with a letter and only contain alphanumeric characters and hyphens (-).")
      return
    }

    setIsDeploying(true)
    setError(null)

    try {
      const request: DeployModelRequest = {
        model_id: modelId,
        deployment_name: deploymentName.trim(),
        // endpoint_name is handled by backend
        instance_type: instanceType,
        instance_count: instanceCount,
        description: description.trim() || undefined,
      }

      const result = await deployModel(request, userId)
      setDeploymentResult(result)
      onSuccess?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to deploy model")
    } finally {
      setIsDeploying(false)
    }
  }

  const handleClose = () => {
    setDeploymentName("")
    setInstanceType("Standard_DS2_v2")
    setInstanceCount(1)
    setDescription("")
    setError(null)
    setDeploymentResult(null)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <CloudIcon className="h-5 w-5" />
            Deploy to Azure ML
          </DialogTitle>
          <DialogDescription>
            Deploy <span className="font-semibold">{modelName}</span> to Azure Machine Learning
          </DialogDescription>
        </DialogHeader>

        {!deploymentResult ? (
          <div className="space-y-4 py-4">
            {/* Deployment Name */}
            <div className="space-y-2">
              <Label htmlFor="deployment-name">
                Deployment Name <span className="text-red-500">*</span>
              </Label>
              <Input
                id="deployment-name"
                placeholder="e.g., my-model-deployment"
                value={deploymentName}
                onChange={(e) => setDeploymentName(e.target.value)}
                disabled={isDeploying}
              />
              <p className="text-xs text-muted-foreground">
                Must start with a letter and only contain alphanumeric characters and hyphens (-).
              </p>
            </div>

            {/* Endpoint Name - Hidden/Fixed */}
            
            {/* Instance Type */}
            <div className="space-y-2">
              <Label htmlFor="instance-type">Instance Type</Label>
              <Select
                value={instanceType}
                onValueChange={setInstanceType}
                disabled={isDeploying}
              >
                <SelectTrigger id="instance-type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Standard_DS1_v2">Standard_DS1_v2 (1 core, 3.5GB RAM)</SelectItem>
                  <SelectItem value="Standard_DS2_v2">Standard_DS2_v2 (2 cores, 7GB RAM)</SelectItem>
                  <SelectItem value="Standard_F2s_v2">Standard_F2s_v2 (2 cores, 4GB RAM)</SelectItem>
                  <SelectItem value="Standard_DS3_v2">Standard_DS3_v2 (4 cores, 14GB RAM)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Instance Count */}
            <div className="space-y-2">
              <Label htmlFor="instance-count">Instance Count</Label>
              <Input
                id="instance-count"
                type="number"
                min={1}
                max={10}
                value={instanceCount}
                onChange={(e) => setInstanceCount(parseInt(e.target.value) || 1)}
                disabled={isDeploying}
              />
              <p className="text-xs text-muted-foreground">
                Number of instances for load balancing (1-10)
              </p>
            </div>

            {/* Description */}
            <div className="space-y-2">
              <Label htmlFor="description">
                Description <span className="text-muted-foreground text-sm">(Optional)</span>
              </Label>
              <Textarea
                id="description"
                placeholder="Add a description for this deployment..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                disabled={isDeploying}
                rows={3}
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
                <AlertCircleIcon className="h-5 w-5 text-red-600 mt-0.5" />
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            {/* Deployment Info */}
            <div className="p-3 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg">
              <p className="text-sm text-blue-900 dark:text-blue-100">
                <span className="font-semibold">Note:</span> Deployment typically takes 5-10 minutes. 
                You can monitor progress in the Deployments page.
              </p>
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-2 pt-4">
              <Button
                variant="outline"
                onClick={handleClose}
                disabled={isDeploying}
              >
                Cancel
              </Button>
              <Button
                onClick={handleDeploy}
                disabled={isDeploying}
              >
                {isDeploying ? (
                  <>
                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                    Deploying...
                  </>
                ) : (
                  <>
                    <CloudIcon className="mr-2 h-4 w-4" />
                    Deploy to Azure
                  </>
                )}
              </Button>
            </div>
          </div>
        ) : (
          // Success State
          <div className="space-y-4 py-6">
            <div className="flex flex-col items-center justify-center gap-4">
              <div className="rounded-full bg-green-100 dark:bg-green-900/20 p-3">
                <CheckCircle2Icon className="h-10 w-10 text-green-600 dark:text-green-400" />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-semibold">Deployment Started!</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {deploymentResult.message}
                </p>
              </div>
            </div>

            {deploymentResult.endpoint_name && (
              <div className="p-4 bg-muted rounded-lg space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Endpoint Name:</span>
                  <span className="text-sm text-muted-foreground">{deploymentResult.endpoint_name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Status:</span>
                  <span className="text-sm text-yellow-600 dark:text-yellow-400 font-medium">Deploying</span>
                </div>
              </div>
            )}

            <div className="flex justify-end gap-2 pt-4">
              <Button variant="outline" onClick={handleClose}>
                Close
              </Button>
              <Button asChild>
                <a href="/dashboard/deployments">
                  View Deployments
                  <ExternalLinkIcon className="ml-2 h-4 w-4" />
                </a>
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
