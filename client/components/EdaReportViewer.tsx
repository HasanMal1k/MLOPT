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
import { AlertCircle, Download, FileText, RefreshCw, BarChart3 } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogClose, DialogFooter, DialogDescription } from "@/components/ui/dialog"
import { X } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import ChartViewer from '@/components/ChartViewer'
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
  const [activeTab, setActiveTab] = useState<string>("report");
  const [isChartViewerOpen, setIsChartViewerOpen] = useState(false);
  
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
      setActiveTab("report"); // Switch to report tab
    } catch (err) {
      console.error('Error generating report:', err);
      setError(err instanceof Error ? err.message : 'Failed to generate EDA report');
    } finally {
      setIsLoading(false);
    }
  };

  const openChartViewer = () => {
    setIsChartViewerOpen(true);
  };

  const handleClose = () => {
    setActiveTab("report");
    setReportHtml(null);
    setError(null);
    onClose();
  };
  
  return (
    <>
      <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader className="flex flex-row items-center justify-between">
            <DialogTitle>
              Data Analysis: {fileMetadata?.original_filename}
            </DialogTitle>
            <DialogDescription className="sr-only">
              Exploratory data analysis report with statistics and visualizations
            </DialogDescription>
            <DialogClose className="hover:bg-gray-100 p-2 rounded-full transition-colors">
              <X className="h-4 w-4" />
            </DialogClose>
          </DialogHeader>
          
          {/* Tabs for Report and Charts */}
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col overflow-hidden">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="report" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                EDA Report
              </TabsTrigger>
              <TabsTrigger value="charts" className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Custom Charts
              </TabsTrigger>
            </TabsList>
            
            {/* EDA Report Tab */}
            <TabsContent value="report" className="flex-1 overflow-auto mt-4">
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
                  <Progress value={70} className="w-full max-w-md mx-auto" />
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
            </TabsContent>

            {/* Custom Charts Tab */}
            <TabsContent value="charts" className="flex-1 overflow-auto mt-4">
              <div className="p-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Interactive Data Visualization
                    </CardTitle>
                    <CardDescription>
                      Create custom charts and visualizations from your data
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center py-8">
                      <BarChart3 className="h-16 w-16 mx-auto mb-4 text-blue-500" />
                      <h3 className="text-lg font-medium mb-2">Custom Chart Builder</h3>
                      <p className="text-sm text-gray-500 mb-6 max-w-md mx-auto">
                        Analyze your data structure and create interactive charts including bar charts, 
                        line charts, scatter plots, and pie charts. Our system will suggest the best 
                        visualizations based on your data types.
                      </p>
                      <Button onClick={openChartViewer} size="lg" className="gap-2">
                        <BarChart3 className="h-4 w-4" />
                        Open Chart Builder
                      </Button>
                    </div>
                    
                    <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 border rounded-lg">
                        <BarChart3 className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                        <div className="text-sm font-medium">Bar Charts</div>
                        <div className="text-xs text-gray-500">Compare categories</div>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <svg className="h-8 w-8 mx-auto mb-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4" />
                        </svg>
                        <div className="text-sm font-medium">Line Charts</div>
                        <div className="text-xs text-gray-500">Show trends</div>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <svg className="h-8 w-8 mx-auto mb-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <circle cx="12" cy="12" r="3"></circle>
                          <circle cx="5" cy="5" r="2"></circle>
                          <circle cx="19" cy="19" r="2"></circle>
                        </svg>
                        <div className="text-sm font-medium">Scatter Plots</div>
                        <div className="text-xs text-gray-500">Find correlations</div>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <svg className="h-8 w-8 mx-auto mb-2 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
                        </svg>
                        <div className="text-sm font-medium">Pie Charts</div>
                        <div className="text-xs text-gray-500">Show proportions</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
          
          <DialogFooter className="flex justify-between pt-4 border-t">
            <div className="flex items-center gap-2">
              {reportHtml && (
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Download Report
                </Button>
              )}
            </div>
            <div className="flex items-center gap-2">
              <p className="text-sm text-muted-foreground">
                {activeTab === "report" ? "Review this report to understand your data" : "Create visualizations from your data"}
              </p>
              <Button onClick={handleClose}>
                {reportHtml || activeTab === "charts" ? "Continue with Upload" : "Close"}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Chart Viewer Dialog */}
      <ChartViewer
        fileMetadata={fileMetadata}
        originalFile={originalFile}
        isOpen={isChartViewerOpen}
        onClose={() => setIsChartViewerOpen(false)}
      />
    </>
  )
}