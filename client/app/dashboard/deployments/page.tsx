/**
 * Deployments Dashboard - View and manage Azure ML deployments
 */
"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { listDeployments, getDeploymentStats, deleteDeployment, type Deployment, type DeploymentStats } from "@/lib/api/deployments"
import { CloudIcon, Loader2Icon, AlertCircleIcon, CheckCircle2Icon, ClockIcon, TrashIcon, ExternalLinkIcon, SearchIcon, RefreshCwIcon, RocketIcon } from "lucide-react"
import { createClient } from "@/utils/supabase/client"

export default function DeploymentsPage() {
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [stats, setStats] = useState<DeploymentStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [userId, setUserId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [filterStatus, setFilterStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const supabase = createClient()

  useEffect(() => {
    async function getUserId() {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        setUserId(user.id)
      }
    }
    getUserId()
  }, [])

  useEffect(() => {
    if (userId) {
      fetchDeployments()
      fetchStats()
    }
  }, [userId, filterStatus])

  const fetchDeployments = async () => {
    if (!userId) return

    setIsLoading(true)
    setError(null)

    try {
      const data = await listDeployments(
        userId,
        filterStatus as any
      )
      setDeployments(data.deployments)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load deployments")
    } finally {
      setIsLoading(false)
    }
  }

  const fetchStats = async () => {
    if (!userId) return

    try {
      const statsData = await getDeploymentStats(userId)
      setStats(statsData)
    } catch (err) {
      console.error("Failed to load stats:", err)
    }
  }

  const handleDelete = async (deploymentId: string) => {
    if (!userId) return
    if (!confirm("Are you sure you want to delete this deployment? This action cannot be undone.")) return

    try {
      await deleteDeployment(deploymentId, userId)
      fetchDeployments()
      fetchStats()
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete deployment")
    }
  }

  const handleRefresh = () => {
    fetchDeployments()
    fetchStats()
  }

  const filteredDeployments = deployments.filter((deployment) => {
    const matchesSearch =
      deployment.deployment_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      deployment.endpoint_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      deployment.trained_models?.model_name.toLowerCase().includes(searchQuery.toLowerCase())

    return matchesSearch
  })

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "active":
        return <Badge className="bg-green-500"><CheckCircle2Icon className="w-3 h-3 mr-1" />Active</Badge>
      case "deploying":
        return <Badge className="bg-yellow-500"><ClockIcon className="w-3 h-3 mr-1" />Deploying</Badge>
      case "failed":
        return <Badge variant="destructive"><AlertCircleIcon className="w-3 h-3 mr-1" />Failed</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <RocketIcon className="h-8 w-8" />
            Deployments
          </h1>
          <p className="text-muted-foreground mt-1">
            Manage your Azure ML model deployments
          </p>
        </div>
        <Button onClick={handleRefresh} variant="outline">
          <RefreshCwIcon className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Deployments</CardTitle>
              <CloudIcon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_deployments}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active</CardTitle>
              <CheckCircle2Icon className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{stats.active_deployments}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Deploying</CardTitle>
              <ClockIcon className="h-4 w-4 text-yellow-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-600">{stats.deploying_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Failed</CardTitle>
              <AlertCircleIcon className="h-4 w-4 text-red-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{stats.failed_deployments}</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Filters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search */}
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search deployments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Status Filter */}
          <div className="flex gap-2 flex-wrap">
            <Button
              variant={filterStatus === null ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterStatus(null)}
            >
              All
            </Button>
            <Button
              variant={filterStatus === "active" ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterStatus("active")}
            >
              Active
            </Button>
            <Button
              variant={filterStatus === "deploying" ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterStatus("deploying")}
            >
              Deploying
            </Button>
            <Button
              variant={filterStatus === "failed" ? "default" : "outline"}
              size="sm"
              onClick={() => setFilterStatus("failed")}
            >
              Failed
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Error State */}
      {error && (
        <Card className="border-red-200 dark:border-red-800">
          <CardContent className="flex items-center gap-2 p-4">
            <AlertCircleIcon className="h-5 w-5 text-red-600" />
            <p className="text-red-600">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {/* Empty State */}
      {!isLoading && filteredDeployments.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <RocketIcon className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No deployments found</h3>
            <p className="text-muted-foreground text-center max-w-md">
              {searchQuery
                ? "No deployments match your search criteria."
                : "Start deploying your trained models to Azure ML to see them here."}
            </p>
            <Button className="mt-4" asChild>
              <a href="/dashboard/models">View Models</a>
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Deployments List */}
      {!isLoading && filteredDeployments.length > 0 && (
        <div className="grid gap-4">
          {filteredDeployments.map((deployment) => (
            <Card key={deployment.id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="flex items-center gap-2">
                      {deployment.deployment_name}
                      {getStatusBadge(deployment.status)}
                    </CardTitle>
                    <CardDescription>
                      {deployment.trained_models?.model_name} • {deployment.trained_models?.algorithm}
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(deployment.id)}
                    disabled={deployment.status === "deploying"}
                  >
                    <TrashIcon className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {/* Deployment Info */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Endpoint:</span>
                    <p className="font-mono text-xs mt-1">{deployment.endpoint_name}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Instance:</span>
                    <p className="font-mono text-xs mt-1">
                      {deployment.instance_type} × {deployment.instance_count}
                    </p>
                  </div>
                </div>

                {/* Description */}
                {deployment.description && (
                  <div>
                    <span className="text-sm text-muted-foreground">Description:</span>
                    <p className="text-sm mt-1">{deployment.description}</p>
                  </div>
                )}

                {/* Scoring URI */}
                {deployment.scoring_uri && (
                  <div>
                    <span className="text-sm text-muted-foreground">Scoring URI:</span>
                    <div className="flex items-center gap-2 mt-1">
                      <code className="flex-1 text-xs bg-muted p-2 rounded overflow-x-auto">
                        {deployment.scoring_uri}
                      </code>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => window.open(deployment.swagger_uri, "_blank")}
                      >
                        <ExternalLinkIcon className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {deployment.error_message && (
                  <div className="p-3 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertCircleIcon className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
                      <div className="text-sm">
                        <p className="font-medium text-red-600">Deployment Failed</p>
                        <p className="text-red-600 mt-1">{deployment.error_message}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Timestamps */}
                <div className="flex gap-4 text-xs text-muted-foreground pt-2 border-t">
                  <span>Created: {new Date(deployment.created_at).toLocaleString()}</span>
                  {deployment.deployed_at && (
                    <span>Deployed: {new Date(deployment.deployed_at).toLocaleString()}</span>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
