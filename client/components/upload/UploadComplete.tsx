// components/upload/UploadComplete.tsx
'use client'

import { CheckCircle2, X, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import AutoPreprocessingReport from "@/components/AutoPreprocessingReport"

interface UploadResult {
  name: string;
  success: boolean;
}

interface UploadCompleteProps {
  uploadSummary: {
    totalFiles: number;
    successCount: number;
    filesProcessed: UploadResult[];
  }
  filePreprocessingResults: Record<string, any>
  customCleaningResults: any[]
  onFinish: (destination: 'dashboard' | 'feature-engineering') => void
  transformPreprocessingResults: (results: any) => any
}

export default function UploadComplete({
  uploadSummary,
  filePreprocessingResults,
  customCleaningResults,
  onFinish,
  transformPreprocessingResults
}: UploadCompleteProps) {
  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-green-500" />
          <span>Upload Complete</span>
        </CardTitle>
        <CardDescription>
          Your data is ready for analysis and machine learning
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-medium text-green-800 mb-2">Upload Summary</h3>
          <p className="text-green-700 mb-4">
            Successfully uploaded {uploadSummary.successCount} of {uploadSummary.totalFiles} files
          </p>
          
          {uploadSummary.filesProcessed.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-4">
              {uploadSummary.filesProcessed.map((file, index) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  {file.success ? (
                    <CheckCircle2 className="h-4 w-4 text-green-600" />
                  ) : (
                    <X className="h-4 w-4 text-red-600" />
                  )}
                  <span className={file.success ? "text-green-700" : "text-red-700"}>
                    {file.name}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Custom Cleaning Results Section */}
        {customCleaningResults.length > 0 && (
          <div className="mb-6">
            <h3 className="text-xl font-bold mb-4">Custom Data Cleaning Summary</h3>
            <div className="space-y-4">
              {customCleaningResults.map((result, index) => (
                <div key={index} className="border rounded-lg p-4 bg-blue-50">
                  <h4 className="font-medium text-lg mb-2">{uploadSummary.filesProcessed[index]?.name}</h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Columns Dropped:</span>{' '}
                      <Badge variant="outline" className="ml-1">
                        {result.report?.columns_dropped?.length || 0}
                      </Badge>
                    </div>
                    <div>
                      <span className="font-medium">Data Types Changed:</span>{' '}
                      <Badge variant="outline" className="ml-1">
                        {Object.keys(result.report?.data_types || {}).length}
                      </Badge>
                    </div>
                    <div>
                      <span className="font-medium">Transformations:</span>{' '}
                      <Badge variant="outline" className="ml-1">
                        {result.report?.transformations_applied?.length || 0}
                      </Badge>
                    </div>
                  </div>
                  {result.report?.columns_dropped?.length > 0 && (
                    <div className="mt-3">
                      <span className="text-sm font-medium">Dropped Columns: </span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {result.report.columns_dropped.map((col: string, idx: number) => (
                          <Badge key={idx} variant="secondary" className="text-xs bg-red-100 text-red-800">
                            {col}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Detailed Preprocessing Reports Section */}
        {uploadSummary.filesProcessed.some(file => filePreprocessingResults[file.name]) && (
          <div className="mb-6">
            <h3 className="text-xl font-bold mb-4">Data Preprocessing Report</h3>
            <div className="space-y-6">
              {uploadSummary.filesProcessed.map((file, index) => {
                if (file.success && filePreprocessingResults[file.name]) {
                  const transformedResults = transformPreprocessingResults(filePreprocessingResults[file.name]);
                  
                  if (!transformedResults) {
                    return (
                      <div key={index} className="border rounded-lg p-4 bg-amber-50">
                        <h4 className="font-medium text-lg mb-2">{file.name}</h4>
                        <p className="text-amber-700">No preprocessing details available for this file.</p>
                      </div>
                    );
                  }
                  
                  return (
                    <div key={index} className="border rounded-lg p-6">
                      <h4 className="font-medium text-lg mb-4">{file.name}</h4>
                      
                      {/* Preprocessing Summary Stats */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="bg-blue-50 p-3 rounded-md">
                          <div className="text-sm text-blue-700 font-medium">Original Rows</div>
                          <div className="text-xl font-bold">{transformedResults.original_shape?.[0] || 0}</div>
                        </div>
                        <div className="bg-blue-50 p-3 rounded-md">
                          <div className="text-sm text-blue-700 font-medium">Processed Rows</div>
                          <div className="text-xl font-bold">{transformedResults.processed_shape?.[0] || 0}</div>
                        </div>
                        <div className="bg-blue-50 p-3 rounded-md">
                          <div className="text-sm text-blue-700 font-medium">Original Columns</div>
                          <div className="text-xl font-bold">{transformedResults.original_shape?.[1] || 0}</div>
                        </div>
                        <div className="bg-blue-50 p-3 rounded-md">
                          <div className="text-sm text-blue-700 font-medium">Processed Columns</div>
                          <div className="text-xl font-bold">{transformedResults.processed_shape?.[1] || 0}</div>
                        </div>
                      </div>
                      
                      {/* Use AutoPreprocessingReport for detailed breakdown */}
                      <AutoPreprocessingReport
                        processingResults={transformedResults}
                        fileName={file.name}
                        isLoading={false}
                      />
                    </div>
                  );
                }
                return null;
              })}
            </div>
          </div>
        )}
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardContent className="pt-6">
              <h3 className="text-base font-medium mb-2">Feature Engineering</h3>
              <p className="text-sm text-muted-foreground mb-4">Create new features for ML</p>
              <p className="text-xs text-muted-foreground mb-4">
                Transform your data and create new features to improve machine learning model performance
              </p>
              <Button variant="outline" onClick={() => onFinish('feature-engineering')} className="w-full gap-2">
                <ChevronRight className="h-4 w-4" />
                <span>Go to Feature Engineering</span>
              </Button>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="pt-6">
              <h3 className="text-base font-medium mb-2">Dashboard</h3>
              <p className="text-sm text-muted-foreground mb-4">Return to dashboard</p>
              <p className="text-xs text-muted-foreground mb-4">
                Go back to the dashboard to view all your uploaded files and explore other options
              </p>
              <Button onClick={() => onFinish('dashboard')} className="w-full gap-2">
                <ChevronRight className="h-4 w-4" />
                <span>Return to Dashboard</span>
              </Button>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  )
}