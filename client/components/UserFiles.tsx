'use client'

import { useState, useEffect } from 'react'
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import FilePreview from "./FilePreview"
import { createClient } from '@/utils/supabase/client'

// Define types for your file data
interface FileMetadata {
  id: string;
  user_id: string;
  name?: string;
  filename: string;
  original_filename: string;
  file_size: number;
  mime_type: string;
  upload_date: string;
  column_names: string[];
  row_count: number;
  file_preview: Record<string, any>[];
  statistics: Record<string, any>;
}

export default function UserFiles() {
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [loading, setLoading] = useState<boolean>(true)
  const [previewFile, setPreviewFile] = useState<FileMetadata | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState<boolean>(false)
  
  const supabase = createClient()
  
  useEffect(() => {
    async function fetchFiles() {
      try {
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
          }
        }
      } catch (error) {
        console.error('Error in fetchFiles:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchFiles()
  }, [])
  
  const handlePreview = (file: FileMetadata) => {
    setPreviewFile(file)
    setIsPreviewOpen(true)
  }
  
  const closePreview = () => {
    setIsPreviewOpen(false)
    setPreviewFile(null)
  }
  
  if (loading) return <div>Loading your files...</div>
  
  return (
    <div className="container mx-auto my-8">
      <h2 className="text-2xl font-bold mb-4">Your Files</h2>
      
      {files.length === 0 ? (
        <div className="text-center p-8 border rounded-lg">
          <p>You haven't uploaded any files yet.</p>
        </div>
      ) : (
        <Table>
          <TableCaption>Your uploaded data files</TableCaption>
          <TableHeader>
            <TableRow>
              <TableHead>Filename</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Rows</TableHead>
              <TableHead>Columns</TableHead>
              <TableHead>Upload Date</TableHead>
              <TableHead className="text-center">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {files.map((file) => (
              <TableRow key={file.id}>
                <TableCell className="font-semibold">{file.original_filename}</TableCell>
                <TableCell>{(file.file_size / 1048576).toFixed(2)} MB</TableCell>
                <TableCell>{file.row_count}</TableCell>
                <TableCell>{file.column_names.length}</TableCell>
                <TableCell>{new Date(file.upload_date).toLocaleString()}</TableCell>
                <TableCell className="text-center">
                  <Button 
                    variant="outline" 
                    className="mr-2"
                    onClick={() => handlePreview(file)}
                  >
                    Preview
                  </Button>
                  <Button>Select for Analysis</Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      
      {previewFile && (
        <FilePreview 
          fileMetadata={previewFile} 
          isOpen={isPreviewOpen} 
          onClose={closePreview} 
        />
      )}
    </div>
  )
}