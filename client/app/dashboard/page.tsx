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
import { FileUp, Database, Settings2, ChevronRight, Eye, Braces, GitBranch } from "lucide-react"
import type { FileMetadata } from '@/components/FilePreview'
import Link from 'next/link'
import { Badge } from "@/components/ui/badge"
import  FilePreview  from '@/components/FilePreview'

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
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Total Files</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{userData.fileCount}</div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Preprocessed Files</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{userData.preprocessedCount}</div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium">Files Ready for ML</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{Math.min(userData.preprocessedCount, userData.fileCount)}</div>
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
                        <CardTitle>{file.original_filename}</CardTitle>
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
                        <Link href={`/dashboard/preprocessing?file=${file.id}`}>Preprocess</Link>
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