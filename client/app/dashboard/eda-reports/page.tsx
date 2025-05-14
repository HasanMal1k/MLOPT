'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import { type FileMetadata } from '@/components/FilePreview'
import FilePreview from '@/components/FilePreview'
import EdaReportViewer from '@/components/EdaReportViewer'
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
import { 
  BarChart2,
  Eye,
  BarChart,
  FileBarChart,
  AlertCircle
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function EdaReportsPage() {
  const [files, setFiles] = useState<FileMetadata[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedFile, setSelectedFile] = useState<FileMetadata | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)
  const [isEdaReportOpen, setIsEdaReportOpen] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  
  const supabase = createClient()
  
  // Fetch user's files
  useEffect(() => {
    async function fetchFiles() {
      try {
        setIsLoading(true)
        const { data: { user } } = await supabase.auth.getUser()
        
        if (user) {
          const { data, error } = await supabase
            .from('files')
            .select('*')
            .eq('user_id', user.id)
            .order('upload_date', { ascending: false })
          
          if (data) {
            setFiles(data as FileMetadata[])
          }
          
          if (error) {
            console.error('Error fetching files:', error.message)
            setErrorMessage(`Error fetching files: ${error.message}`)
          }
        }
      } catch (error) {
        console.error('Error in fetchFiles:', error)
        setErrorMessage(`Error loading files: ${error}`)
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchFiles()
  }, [])
  
  // Handle file preview
  const handlePreview = (file: FileMetadata) => {
    setSelectedFile(file)
    setIsPreviewOpen(true)
  }
  
  // Handle generating EDA report
   const handleEdaReport = (file: FileMetadata) => {
    setSelectedFile(file)
    setIsEdaReportOpen(true)
  }
  
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10 overflow-y-auto">
      <div className="text-4xl font-bold mb-8">
        Data Analysis Reports
      </div>
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Exploratory Data Analysis</CardTitle>
          <CardDescription>
            Generate interactive reports to analyze and visualize your data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-6">
            Exploratory Data Analysis (EDA) reports provide comprehensive insights into your datasets.
            These reports include statistical summaries, distributions, correlations, and visualizations
            to help you better understand your data.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card>
              <CardContent className="pt-6">
                <BarChart2 className="h-8 w-8 text-primary mb-2" />
                <h3 className="font-medium">Data Profiling</h3>
                <p className="text-xs text-muted-foreground mt-1">
                  Generate comprehensive statistical profiles of your datasets
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <FileBarChart className="h-8 w-8 text-primary mb-2" />
                <h3 className="font-medium">Visualizations</h3>
                <p className="text-xs text-muted-foreground mt-1">
                  Auto-generated charts and plots to understand distributions
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <BarChart className="h-8 w-8 text-primary mb-2" />
                <h3 className="font-medium">Correlations</h3>
                <p className="text-xs text-muted-foreground mt-1">
                  Identify relationships between variables in your data
                </p>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
      
      {errorMessage && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      )}
      
      <Card>
        <CardHeader>
          <CardTitle>Your Files</CardTitle>
          <CardDescription>
            Select a file to generate an EDA report
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center items-center h-48">
              <p>Loading files...</p>
            </div>
          ) : files.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 gap-4">
              <p className="text-gray-500">No files found. Upload files to get started.</p>
              <Button variant="outline" asChild>
                <a href="/dashboard/data-upload">Upload Files</a>
              </Button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableCaption>Your available files for analysis</TableCaption>
                <TableHeader>
                  <TableRow>
                    <TableHead>Filename</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Rows</TableHead>
                    <TableHead>Columns</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {files.map((file) => (
                    <TableRow key={file.id}>
                      <TableCell className="font-medium">{file.original_filename}</TableCell>
                      <TableCell>{(file.file_size / 1048576).toFixed(2)} MB</TableCell>
                      <TableCell>{file.row_count}</TableCell>
                      <TableCell>{file.column_names.length}</TableCell>
                      <TableCell>
                        {file.preprocessing_info?.is_preprocessed ? (
                          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                            Preprocessed
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                            Raw
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Button variant="ghost" size="icon" onClick={() => handlePreview(file)} title="Preview file">
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="outline" 
                            className="flex items-center gap-2"
                            onClick={() => handleEdaReport(file)}
                          >
                            <BarChart className="h-4 w-4" />
                            Generate Report
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* File Preview Dialog */}
      {selectedFile && (
        <FilePreview 
          fileMetadata={selectedFile} 
          isOpen={isPreviewOpen} 
          onClose={() => setIsPreviewOpen(false)} 
        />
      )}
      
      {/* EDA Report Dialog */}
      {selectedFile && (
        <EdaReportViewer 
          fileMetadata={selectedFile} 
          isOpen={isEdaReportOpen} 
          onClose={() => setIsEdaReportOpen(false)} 
        />
      )}
    </section>
  )
}