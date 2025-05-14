'use client'

import { useState } from 'react'
import { createClient } from '@/utils/supabase/client'
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle, Download, FileText, RefreshCw } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose, DialogFooter } from "@/components/ui/dialog"
import { X } from "lucide-react"
import type { FileMetadata } from '@/components/FilePreview'

interface EdaReportViewerProps {
  fileMetadata?: FileMetadata | null;
  onClose: () => void;
  isOpen: boolean;
  // Add original file for new uploads
  originalFile?: File | null;
}

export default function EdaReportViewer({ fileMetadata, onClose, isOpen, originalFile }: EdaReportViewerProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  
  // Function to generate the report
  const generateReport = async () => {
    if (!fileMetadata) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Create a FormData object to send the file
      const formData = new FormData();
      
      // Check if we have the original file object (for new uploads)
      if (originalFile) {
        // Use the original file object directly
        formData.append('file', originalFile);
        console.log("Using original file:", originalFile.name, originalFile.size);
      } else if (fileMetadata.user_id !== "temporary") {
        // This is an existing file in storage
        const supabase = createClient();
        const { data: { publicUrl } } = supabase.storage
          .from('data-files')
          .getPublicUrl(`${fileMetadata.user_id}/${fileMetadata.filename}`);
        
        // Fetch the file
        const response = await fetch(publicUrl);
        
        if (!response.ok) {
          throw new Error(`Failed to download file: ${response.statusText}`);
        }
        
        // Get the file content as a blob
        const fileBlob = await response.blob();
        const file = new File([fileBlob], fileMetadata.original_filename, { 
          type: fileMetadata.mime_type 
        });
        
        // Add the file to form data
        formData.append('file', file);
        console.log("Using file from storage:", file.name, file.size);
      } else {
        throw new Error("Cannot generate report: File not available");
      }
      
      // Send to API endpoint for EDA report
      const reportResponse = await fetch('/api/eda-report', {
        method: 'POST',
        body: formData,
      });
      
      if (!reportResponse.ok) {
        const errorText = await reportResponse.text();
        console.error("EDA report generation failed:", errorText);
        throw new Error(`Failed to generate EDA report: ${reportResponse.statusText}`);
      }
      
      // Get the HTML report
      const html = await reportResponse.text();
      setReportHtml(html);
    } catch (err) {
      console.error('Error generating report:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate EDA report');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader className="flex flex-row items-center justify-between">
          <DialogTitle>
            EDA Report: {fileMetadata?.original_filename}
          </DialogTitle>
          <DialogClose className="hover:bg-gray-100 p-2 rounded-full transition-colors">
            <X className="h-4 w-4" />
          </DialogClose>
        </DialogHeader>
        
        <div className="flex-1 overflow-auto">
          {!reportHtml && !isLoading && (
            <div className="p-6 text-center">
              <FileText className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium mb-2">Generate Exploratory Data Analysis Report</h3>
              <p className="text-sm text-gray-500 mb-6 max-w-md mx-auto">
                Generate a comprehensive report with statistics, visualizations, and insights about your data.
                This may take a few moments depending on the size of your data.
              </p>
              <Button 
                onClick={generateReport} 
                disabled={isLoading}
                size="lg"
              >
                Generate Report
              </Button>
            </div>
          )}
          
          {isLoading && (
            <div className="p-6 text-center">
              <RefreshCw className="h-12 w-12 mx-auto mb-4 text-blue-500 animate-spin" />
              <h3 className="text-lg font-medium mb-2">Generating Report</h3>
              <p className="text-sm text-gray-500 mb-6">
                Please wait while we analyze your data and create the report...
              </p>
              <Progress value={isLoading ? 70 : 0} className="w-full max-w-md mx-auto" />
            </div>
          )}
          
          {error && (
            <Alert variant="destructive" className="my-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {reportHtml && (
            <div className="relative w-full h-full" style={{ minHeight: '60vh' }}>
              <iframe 
                srcDoc={reportHtml}
                className="absolute inset-0 w-full h-full border-none"
                title="EDA Report"
                sandbox="allow-scripts allow-same-origin"
              />
            </div>
          )}
        </div>
        
        <DialogFooter className="flex justify-between pt-4 border-t">
          <p className="text-sm text-muted-foreground">
            Review this report to better understand your data before preprocessing
          </p>
          <Button onClick={onClose}>
            {reportHtml ? "Continue with Upload" : "Close"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}