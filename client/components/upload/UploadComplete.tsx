// components/upload/UploadComplete.tsx
'use client'

import { CheckCircle2, X, ChevronRight, Sparkles, Brain, Zap } from "lucide-react"
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
  onFinish: (destination: 'dashboard' | 'feature-engineering' | 'transformations' | 'blueprints') => void
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
        
        {/* Next Steps - Complete Pipeline */}
        <div className="mb-4">
          <h3 className="text-xl font-bold mb-2">What's Next?</h3>
          <p className="text-sm text-muted-foreground mb-6">
            Continue your data science pipeline with transformations and model training
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* Step 1: Transformations */}
          <Card className="border-2 hover:border-primary transition-colors">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-10 w-10 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
                  <Zap className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-base font-semibold">Transformations</h3>
                  <Badge variant="outline" className="text-xs">Step 1</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Apply mathematical and encoding transformations to engineer better features
              </p>
              <Button 
                variant="outline" 
                onClick={() => onFinish('transformations')} 
                className="w-full gap-2"
              >
                <Sparkles className="h-4 w-4" />
                <span>Transform Data</span>
              </Button>
            </CardContent>
          </Card>
          
          {/* Step 2: Model Training */}
          <Card className="border-2 hover:border-primary transition-colors">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-10 w-10 rounded-full bg-purple-100 dark:bg-purple-900 flex items-center justify-center">
                  <Brain className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h3 className="text-base font-semibold">Train Models</h3>
                  <Badge variant="outline" className="text-xs">Step 2</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Train ML models with automated hyperparameter tuning and comparison
              </p>
              <Button 
                onClick={() => onFinish('blueprints')} 
                className="w-full gap-2"
              >
                <Brain className="h-4 w-4" />
                <span>Start Training</span>
              </Button>
            </CardContent>
          </Card>

          {/* Dashboard Option */}
          <Card className="border-2 hover:border-primary transition-colors">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="h-10 w-10 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                  <ChevronRight className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-base font-semibold">Dashboard</h3>
                  <Badge variant="secondary" className="text-xs">Or Later</Badge>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Return to dashboard to manage files and explore other options
              </p>
              <Button 
                variant="ghost" 
                onClick={() => onFinish('dashboard')} 
                className="w-full gap-2"
              >
                <ChevronRight className="h-4 w-4" />
                <span>Go to Dashboard</span>
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Pipeline Flow Visualization */}
        <div className="bg-muted/50 rounded-lg p-4 border">
          <div className="flex items-center justify-center gap-2 text-sm">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-green-500 dark:bg-green-600 flex items-center justify-center text-white dark:text-white font-bold text-xs">
                ✓
              </div>
              <span className="font-medium">Upload</span>
            </div>
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-blue-500 dark:bg-blue-600 flex items-center justify-center text-white dark:text-white font-bold text-xs">
                1
              </div>
              <span className="font-medium">Transform</span>
            </div>
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-purple-500 dark:bg-purple-600 flex items-center justify-center text-white dark:text-white font-bold text-xs">
                2
              </div>
              <span className="font-medium">Train</span>
            </div>
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center text-muted-foreground font-bold text-xs">
                3
              </div>
              <span className="font-medium text-muted-foreground">Deploy</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}