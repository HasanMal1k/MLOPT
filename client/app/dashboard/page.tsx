'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import { 
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FileUp, Database, Settings2, ChevronRight, Eye, Braces, GitBranch, TrendingUp, Calendar, BarChart3 } from "lucide-react"
import type { FileMetadata } from '@/components/FilePreview'
import Link from 'next/link'
import { Badge } from "@/components/ui/badge"
import  FilePreview  from '@/components/FilePreview'
import { useTheme } from 'next-themes'
import { 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend
} from 'recharts'

export default function Dashboard() {
  const [userData, setUserData] = useState<{
    fileCount: number,
    preprocessedCount: number,
    recentFiles: FileMetadata[]
  }>({
    fileCount: 0,
    preprocessedCount: 0,
    recentFiles: []
  })
  const [loading, setLoading] = useState(true)
  const [userName, setUserName] = useState<string>('')
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const { theme } = useTheme()
  
  const supabase = createClient()
  
  useEffect(() => {
    async function fetchUserData() {
      try {
        const { data: { user } } = await supabase.auth.getUser()
        
        if (user) {
          setUserName(user.email || 'User')
          
          // Fetch files
          const { data, error } = await supabase
            .from('files')
            .select('*')
            .eq('user_id', user.id)
            .order('upload_date', { ascending: false })
          
          if (data) {
            const files = data as FileMetadata[]
            const preprocessedFiles = files.filter(file => file.preprocessing_info?.is_preprocessed)
            
            setUserData({
              fileCount: files.length,
              preprocessedCount: preprocessedFiles.length,
              recentFiles: files.slice(0, 3) // Get 3 most recent files
            })
          }
          
          if (error) {
            console.error('Error fetching files:', error.message)
          }
        }
      } catch (error) {
        console.error('Error in fetchUserData:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchUserData()
  }, [])
  
  const handlePreview = (file: FileMetadata) => {
    setSelectedFile(file)
    setIsPreviewOpen(true)
  }
  
  const closePreview = () => {
    setIsPreviewOpen(false)
  }
  
  // Format date to a readable string
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }
  
  // Calculate file size in readable format
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // Prepare chart data
  const getUploadTrendData = () => {
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const date = new Date()
      date.setDate(date.getDate() - (6 - i))
      return date.toISOString().split('T')[0]
    })

    return last7Days.map(date => {
      const count = userData.recentFiles.filter(file => 
        file.upload_date.split('T')[0] === date
      ).length
      return {
        date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        uploads: count
      }
    })
  }

  const getFileTypeData = () => {
    const csvCount = userData.recentFiles.filter(f => f.mime_type?.includes('csv')).length
    const excelCount = userData.recentFiles.filter(f => f.mime_type?.includes('sheet') || f.mime_type?.includes('excel')).length
    return [
      { name: 'CSV', value: csvCount, color: 'hsl(var(--chart-1))' },
      { name: 'Excel', value: excelCount, color: 'hsl(var(--chart-2))' }
    ]
  }

  const getProcessingStatusData = () => {
    return [
      { 
        name: 'Preprocessed', 
        value: userData.preprocessedCount,
        color: 'hsl(var(--chart-3))'
      },
      { 
        name: 'Raw', 
        value: userData.fileCount - userData.preprocessedCount,
        color: 'hsl(var(--chart-4))'
      }
    ]
  }

  const getFileSizeDistribution = () => {
    return userData.recentFiles.map(file => ({
      name: file.original_filename?.substring(0, 15) + '...',
      size: Number((file.file_size / (1024 * 1024)).toFixed(2)),
      rows: file.row_count
    })).slice(0, 5)
  }

  // Theme-aware colors
  const isDark = theme === 'dark'
  const textColor = isDark ? '#e5e7eb' : '#374151'
  const gridColor = isDark ? '#374151' : '#e5e7eb'
  const tooltipBg = isDark ? '#1f2937' : '#ffffff'
  const tooltipBorder = isDark ? '#374151' : '#e5e7eb'
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-6">
        Dashboard
      </div>
      
      {loading ? (
        <div className="flex justify-center items-center h-48">
          <p>Loading dashboard...</p>
        </div>
      ) : (
        <div className="grid gap-6">
          {/* Welcome Card */}
          <Card>
            <CardHeader>
              <CardTitle>Welcome, {userName}</CardTitle>
              <CardDescription>
                Here's an overview of your data preprocessing activities
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 border-blue-200 dark:border-blue-800">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">Total Files</CardTitle>
                      <Database className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-blue-700 dark:text-blue-300">{userData.fileCount}</div>
                    <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">Uploaded datasets</p>
                  </CardContent>
                </Card>
                
                <Card className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-950 dark:to-green-900 border-green-200 dark:border-green-800">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">Preprocessed</CardTitle>
                      <Settings2 className="h-4 w-4 text-green-600 dark:text-green-400" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-green-700 dark:text-green-300">{userData.preprocessedCount}</div>
                    <p className="text-xs text-green-600 dark:text-green-400 mt-1">Ready for training</p>
                  </CardContent>
                </Card>
                
                <Card className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-950 dark:to-purple-900 border-purple-200 dark:border-purple-800">
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-sm font-medium">ML Ready</CardTitle>
                      <TrendingUp className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-purple-700 dark:text-purple-300">{Math.min(userData.preprocessedCount, userData.fileCount)}</div>
                    <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                      {userData.fileCount > 0 ? `${Math.round((userData.preprocessedCount / userData.fileCount) * 100)}% processed` : 'No files yet'}
                    </p>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
            <CardFooter>
              <Button variant="outline" asChild className="w-full">
                <Link href="/dashboard/datasets">
                  View All Files
                  <ChevronRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          {/* Charts Section */}
          {userData.fileCount > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Upload Trend Chart */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Calendar className="h-5 w-5" />
                    Upload Activity (Last 7 Days)
                  </CardTitle>
                  <CardDescription>Track your daily upload patterns</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={getUploadTrendData()}>
                      <defs>
                        <linearGradient id="colorUploads" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis 
                        dataKey="date" 
                        stroke={textColor}
                        style={{ fontSize: '12px' }}
                      />
                      <YAxis 
                        stroke={textColor}
                        style={{ fontSize: '12px' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: tooltipBg, 
                          border: `1px solid ${tooltipBorder}`,
                          borderRadius: '8px',
                          color: textColor
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="uploads" 
                        stroke="hsl(var(--primary))" 
                        fillOpacity={1} 
                        fill="url(#colorUploads)" 
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* File Type Distribution */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileUp className="h-5 w-5" />
                    File Type Distribution
                  </CardTitle>
                  <CardDescription>Breakdown of your file formats</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={getFileTypeData()}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="hsl(var(--primary))"
                        dataKey="value"
                      >
                        {getFileTypeData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: tooltipBg, 
                          border: `1px solid ${tooltipBorder}`,
                          borderRadius: '8px',
                          color: textColor
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Processing Status */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings2 className="h-5 w-5" />
                    Processing Status
                  </CardTitle>
                  <CardDescription>Current preprocessing progress</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={getProcessingStatusData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis 
                        dataKey="name" 
                        stroke={textColor}
                        style={{ fontSize: '12px' }}
                      />
                      <YAxis 
                        stroke={textColor}
                        style={{ fontSize: '12px' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: tooltipBg, 
                          border: `1px solid ${tooltipBorder}`,
                          borderRadius: '8px',
                          color: textColor
                        }}
                      />
                      <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                        {getProcessingStatusData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* File Size Distribution */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Dataset Size Overview
                  </CardTitle>
                  <CardDescription>File sizes and row counts</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={getFileSizeDistribution()}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis 
                        dataKey="name" 
                        stroke={textColor}
                        style={{ fontSize: '10px' }}
                      />
                      <YAxis 
                        yAxisId="left"
                        stroke={textColor}
                        style={{ fontSize: '12px' }}
                        label={{ value: 'Size (MB)', angle: -90, position: 'insideLeft', fill: textColor }}
                      />
                      <YAxis 
                        yAxisId="right"
                        orientation="right"
                        stroke={textColor}
                        style={{ fontSize: '12px' }}
                        label={{ value: 'Rows', angle: 90, position: 'insideRight', fill: textColor }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: tooltipBg, 
                          border: `1px solid ${tooltipBorder}`,
                          borderRadius: '8px',
                          color: textColor
                        }}
                      />
                      <Legend wrapperStyle={{ color: textColor }} />
                      <Bar yAxisId="left" dataKey="size" fill="hsl(var(--chart-1))" radius={[8, 8, 0, 0]} />
                      <Bar yAxisId="right" dataKey="rows" fill="hsl(var(--chart-2))" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}
          
          {/* Quick Actions */}
          {/* <div>
            <h2 className="text-xl font-semibold mb-3">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Upload Data</CardTitle>
                  <FileUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    Upload new CSV or Excel files for processing
                  </p>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" asChild className="w-full">
                    <Link href="/dashboard/data-upload">Upload</Link>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Auto Preprocess</CardTitle>
                  <Settings2 className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    Clean and transform your data automatically
                  </p>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" asChild className="w-full">
                    <Link href="/dashboard/preprocessing">Preprocess</Link>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Custom Preprocessing</CardTitle>
                  <Braces className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    Define custom preprocessing steps for your data
                  </p>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" asChild className="w-full">
                    <Link href="/dashboard/custom-preprocessing">Customize</Link>
                  </Button>
                </CardFooter>
              </Card>
              
              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Browse Files</CardTitle>
                  <Database className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    View and manage your uploaded datasets
                  </p>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" asChild className="w-full">
                    <Link href="/dashboard/my-files">Browse</Link>
                  </Button>
                </CardFooter>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <CardTitle className="text-sm font-medium">Auto Transformations</CardTitle>
                  <GitBranch className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    Generate features automatically from your data
                  </p>
                </CardContent>
                <CardFooter>
                  <Button variant="outline" asChild className="w-full">
                    <Link href="/dashboard/transformations">Transform</Link>
                  </Button>
                </CardFooter>
              </Card>
            </div>
          </div>
           */}
          {/* Recent Files */}
          <div>
            <h2 className="text-xl font-semibold mb-3">Recent Files</h2>
            {userData.recentFiles.length === 0 ? (
              <Card>
                <CardContent className="flex flex-col items-center justify-center p-6">
                  <p className="text-muted-foreground mb-4">No files uploaded yet</p>
                  <Button asChild>
                    <Link href="/dashboard/data-upload">Upload Your First File</Link>
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4">
                {userData.recentFiles.map(file => (
                  <Card key={file.id}>
                    <CardHeader className="pb-2">
                      <div className="flex justify-between">
                        <CardTitle>{file.name || file.original_filename}</CardTitle>
                        <div className="text-xs text-muted-foreground">
                          {formatDate(file.upload_date)}
                        </div>
                      </div>
                      <CardDescription>
                        {file.row_count} rows, {file.column_names.length} columns, {formatFileSize(file.file_size)}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="pb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">Status:</span>
                        {file.preprocessing_info?.is_preprocessed ? (
                          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                            Preprocessed
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                            Raw
                          </Badge>
                        )}
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-end gap-2">
                      <Button variant="ghost" size="icon" onClick={() => handlePreview(file)}>
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button variant="outline" asChild>
                        <Link href={`/dashboard/transformations?file=${file.id}`}>Process</Link>
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
      
      {selectedFile && (
        <FilePreview 
          fileMetadata={selectedFile} 
          isOpen={isPreviewOpen} 
          onClose={closePreview} 
        />
      )}
    </section>
  )
}