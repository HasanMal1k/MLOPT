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
  Plus,
  Search,
  Filter,
  Eye,
  Download,
  Settings,
  MoreHorizontal,
  Database,
  Clock,
  BarChart3,
  Calendar,
  FileText,
  Trash2,
  Edit,
  RefreshCw,
  Upload,
  TrendingUp
} from "lucide-react"
import Link from 'next/link'
import { useToast } from "@/hooks/use-toast"
import { format } from 'date-fns'
import FilePreview from "@/components/FilePreview"
import type { FileMetadata } from '@/components/FilePreview'

interface DatasetSummary {
  dataset_type: string;
  total_files: number;
  custom_cleaned_files: number;
  preprocessed_files: number;
}

interface ExtendedFileMetadata extends FileMetadata {
  dataset_type: 'normal' | 'time_series';
  custom_cleaning_applied: boolean;
  custom_cleaning_config?: any;
  time_series_config?: any;
}

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<ExtendedFileMetadata[]>([])
  const [summary, setSummary] = useState<DatasetSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [selectedDataset, setSelectedDataset] = useState<ExtendedFileMetadata | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  
  const { toast } = useToast()
  const supabase = createClient()

  useEffect(() => {
    fetchDatasets()
    fetchSummary()
  }, [])

  const fetchDatasets = async () => {
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

  const fetchSummary = async () => {
    try {
      const { data, error } = await supabase
        .from('dataset_summary')
        .select('*')
      
      if (error) throw error
      setSummary(data || [])
    } catch (error) {
      console.error('Error fetching summary:', error)
    }
  }

  const filteredDatasets = datasets.filter(dataset => {
    const matchesSearch = dataset.original_filename.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = filterType === 'all' || dataset.dataset_type === filterType
    const matchesStatus = filterStatus === 'all' || 
      (filterStatus === 'processed' && dataset.preprocessing_info?.is_preprocessed) ||
      (filterStatus === 'unprocessed' && !dataset.preprocessing_info?.is_preprocessed) ||
      (filterStatus === 'custom_cleaned' && dataset.custom_cleaning_applied)
    
    return matchesSearch && matchesType && matchesStatus
  })

  const handlePreview = (dataset: ExtendedFileMetadata) => {
    setSelectedDataset(dataset)
    setIsPreviewOpen(true)
  }

  const handleDelete = async (datasetId: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return
    
    try {
      const { error } = await supabase
        .from('files')
        .delete()
        .eq('id', datasetId)
      
      if (error) throw error
      
      toast({
        title: "Success",
        description: "Dataset deleted successfully"
      })
      
      fetchDatasets()
      fetchSummary()
    } catch (error) {
      console.error('Error deleting dataset:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete dataset"
      })
    }
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

  const getProcessingStatusBadge = (dataset: ExtendedFileMetadata) => {
    if (dataset.preprocessing_info?.is_preprocessed) {
      return <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
        Processed
      </Badge>
    }
    if (dataset.custom_cleaning_applied) {
      return <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
        Custom Cleaned
      </Badge>
    }
    return <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
      Raw
    </Badge>
  }

  const totalDatasets = summary.reduce((acc, item) => acc + item.total_files, 0)
  const totalProcessed = summary.reduce((acc, item) => acc + item.preprocessed_files, 0)
  const totalCustomCleaned = summary.reduce((acc, item) => acc + item.custom_cleaned_files, 0)

  return (
    <section className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold">Datasets</h1>
          <p className="text-muted-foreground mt-2">
            Manage and analyze your data collections
          </p>
        </div>
        
        <div className="flex gap-3">
          <Button variant="outline" onClick={() => {
            fetchDatasets()
            fetchSummary()
          }}>
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

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalDatasets}</div>
            <p className="text-xs text-muted-foreground">
              {datasets.length} files uploaded
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Time Series</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {summary.find(s => s.dataset_type === 'time_series')?.total_files || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              Temporal datasets
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processed</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalProcessed}</div>
            <p className="text-xs text-muted-foreground">
              Ready for ML
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Custom Cleaned</CardTitle>
            <Settings className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalCustomCleaned}</div>
            <p className="text-xs text-muted-foreground">
              Custom preprocessing
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Filter & Search</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search datasets..."
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
                <SelectValue placeholder="Processing Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="processed">Processed</SelectItem>
                <SelectItem value="unprocessed">Raw</SelectItem>
                <SelectItem value="custom_cleaned">Custom Cleaned</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Datasets Table */}
      <Card>
        <CardHeader>
          <CardTitle>Your Datasets ({filteredDatasets.length})</CardTitle>
          <CardDescription>
            Manage your uploaded datasets and their processing status
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
                <p>Loading datasets...</p>
              </div>
            </div>
          ) : filteredDatasets.length === 0 ? (
            <div className="text-center py-12">
              <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-2">No datasets found</h3>
              <p className="text-muted-foreground mb-6">
                {datasets.length === 0 
                  ? "Upload your first dataset to get started"
                  : "No datasets match your current filters"
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
                  <TableHead>Size</TableHead>
                  <TableHead>Dimensions</TableHead>
                  <TableHead>Status</TableHead>
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
                          <div className="font-medium">{dataset.original_filename}</div>
                          <div className="text-sm text-muted-foreground">
                            {dataset.mime_type.includes('csv') ? 'CSV' : 'Excel'}
                          </div>
                        </div>
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      {getDatasetTypeBadge(dataset.dataset_type)}
                    </TableCell>
                    
                    <TableCell>
                      {formatFileSize(dataset.file_size)}
                    </TableCell>
                    
                    <TableCell>
                      <div className="text-sm">
                        <div>{dataset.row_count.toLocaleString()} rows</div>
                        <div className="text-muted-foreground">
                          {dataset.column_names.length} columns
                        </div>
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      {getProcessingStatusBadge(dataset)}
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
                            Preview
                          </DropdownMenuItem>
                          <DropdownMenuItem asChild>
                            <Link href={`/dashboard/preprocessing?file=${dataset.id}`}>
                              <Settings className="h-4 w-4 mr-2" />
                              Process
                            </Link>
                          </DropdownMenuItem>
                          {dataset.dataset_type === 'time_series' && (
                            <DropdownMenuItem asChild>
                              <Link href={`/dashboard/time-series?file=${dataset.id}`}>
                                <TrendingUp className="h-4 w-4 mr-2" />
                                Time Series Analysis
                              </Link>
                            </DropdownMenuItem>
                          )}
                          <DropdownMenuSeparator />
                          <DropdownMenuItem 
                            onClick={() => handleDelete(dataset.id)}
                            className="text-destructive"
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
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