'use client'

import { useState, useCallback, useEffect } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { 
  FilePlus2, 
  Link as LinkIcon, 
  BarChart2, 
  X, 
  FileText, 
  ChevronRight, 
  Upload, 
  CheckCircle2, 
  AlertCircle, 
  Eye
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "@/hooks/use-toast";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription, 
  DialogFooter 
} from "@/components/ui/dialog";
import { Stepper, Step, StepDescription, StepLabel, StepTitle } from "@/components/ui/steppar";

import FilePreview from "@/components/FilePreview";
import EdaReportViewer from "@/components/EdaReportViewer";
import KaggleUpload from "@/components/KaggleUpload";

interface ProcessingInfo {
  filename: string;
  status: {
    status: string;
    progress: number;
    message: string;
  };
}

export default function DataUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<Record<string, any>>({});
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<string>("upload");
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedFileForPreview, setSelectedFileForPreview] = useState<File | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [isEdaReportOpen, setIsEdaReportOpen] = useState(false);
  const [activeFileIndex, setActiveFileIndex] = useState(0);
  const [reviewedFiles, setReviewedFiles] = useState<Set<string>>(new Set());
  const [uploadComplete, setUploadComplete] = useState(false);
  const [uploadSummary, setUploadSummary] = useState<{
    totalFiles: number;
    successCount: number;
    filesProcessed: {name: string, success: boolean}[];
  }>({
    totalFiles: 0,
    successCount: 0,
    filesProcessed: []
  });

  const uploadData = async () => {
    if (files.length === 0) {
      setError("Please select at least one file to upload");
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);
    
    try {
      // Step 1: Upload to Python for preprocessing
      const pythonFormData = new FormData();
      files.forEach(file => {
        pythonFormData.append('files', file);
      });
      
      setUploadProgress(10);
      
      // Send to Python backend
      const pythonResponse = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: pythonFormData,
      });

      if (!pythonResponse.ok) {
        const errorData = await pythonResponse.json();
        throw new Error(errorData.detail || 'Preprocessing failed');
      }

      const preprocessingResult = await pythonResponse.json();
      setUploadProgress(50);
      
      // Initialize status tracking for preprocessing
      const processingInfo = (preprocessingResult.processing_info || []) as ProcessingInfo[];
      const initialStatus: Record<string, any> = {};
      processingInfo.forEach((info: ProcessingInfo) => {
        initialStatus[info.filename] = {
          status: info.status.status,
          progress: info.status.progress,
          message: info.status.message
        };
      });

      setProcessingStatus(initialStatus);
      
      // Step 2: Poll processing status
      const fileNames = processingInfo.map((info: ProcessingInfo) => info.filename);
      if (fileNames.length > 0) {
        await trackProcessingStatus(fileNames);
      }

      setUploadProgress(60);
      if (fileNames.length > 0 && Object.values(processingStatus).every(status => status.progress === 100)) {
        setUploadProgress(70);
        
        try {
          // Get preprocessing results to add engineered features to the metadata
          for (const fileName of fileNames) {
            const statusInfo = processingStatus[fileName];
            // Check if statusInfo and results exist before accessing
            if (statusInfo && statusInfo.results && statusInfo.results.engineered_features) {
              console.log(`${fileName} has ${statusInfo.results.engineered_features.length} engineered features`);
            }
          }
          
          setUploadProgress(80);
        } catch (err) {
          console.error('Error processing transformations:', err);
        }
      }
      
      // Step 3: Upload preprocessed files to database
      setUploadProgress(70);
      const uploadResults = [];
      
      for (const file of files) {
        // Create a new FormData for each file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('preprocessed', 'true');
        
        try {
          // Use relative URL for API endpoint
          const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
          });

          const result = await response.json();
          
          if (!response.ok) {
            uploadResults.push({
              name: file.name,
              success: false
            });
            console.error(`Upload failed for ${file.name}:`, result.error || result.details);
          } else {
            uploadResults.push({
              name: file.name,
              success: true
            });
          }
        } catch (err) {
          uploadResults.push({
            name: file.name,
            success: false
          });
          console.error(`Exception during upload for ${file.name}:`, err);
        }
      }
      
      // Prepare upload summary
      const successCount = uploadResults.filter(r => r.success).length;
      
      setUploadSummary({
        totalFiles: files.length,
        successCount: successCount,
        filesProcessed: uploadResults
      });
      
      setUploadProgress(100);
      setUploadComplete(true);
      
      if (successCount === files.length) {
        // All files uploaded successfully
        toast({
          title: "Success",
          description: `All ${files.length} files have been uploaded and preprocessed successfully`,
        });
      } else {
        // Some files failed
        toast({
          variant: "destructive",
          title: "Partial success",
          description: `Uploaded ${successCount} of ${files.length} files successfully`,
        });
      }
      
      // Move to next step in wizard
      setCurrentStep(3);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };
  
  // Function to poll processing status
  const trackProcessingStatus = async (fileNames: string[]) => {
    let allCompleted = false;
    let attempts = 0;
    const maxAttempts = 30; // Timeout after 30 attempts (5 minutes with 10-second interval)
    
    while (!allCompleted && attempts < maxAttempts) {
      attempts++;
      let completedCount = 0;
      
      for (const fileName of fileNames) {
        try {
          const response = await fetch(`http://localhost:8000/processing-status/${fileName}`);
          if (response.ok) {
            const status = await response.json();
            
            setProcessingStatus(prev => ({
              ...prev,
              [fileName]: {
                status: status.status,
                progress: status.progress,
                message: status.message,
                results: status.results
              }
            }));
            
            if (status.progress === 100 || status.progress === -1) {
              completedCount++;
            }
          }
        } catch (error) {
          console.error(`Error checking status for ${fileName}:`, error);
        }
      }
      
      if (completedCount === fileNames.length) {
        allCompleted = true;
      } else {
        // Wait 10 seconds before next poll
        await new Promise(resolve => setTimeout(resolve, 10000));
      }
    }
    
    return allCompleted;
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prevFiles) => [...prevFiles, ...acceptedFiles]);
    setError(null);
  }, []);

  const onDropRejected = useCallback((fileRejections: FileRejection[]) => {
    if (fileRejections.length > 0) {
      const rejection = fileRejections[0];
      if (rejection.errors[0]?.code === "file-invalid-type") {
        setError("Only CSV and XLSX files are allowed.");
      } else {
        setError(`Error: ${rejection.errors[0]?.message || "There was an error uploading your file."}`);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    onDropRejected,
    accept: {
      "text/csv": ['.csv'],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    maxSize: 10485760, // 10MB
    noClick: true
  });

  // Handler for when a file is imported from Kaggle
  const handleKaggleFileImported = (file: File) => {
    setFiles((prevFiles) => [...prevFiles, file]);
    setError(null);
    // Switch to upload tab to show the imported file
    setActiveTab("upload");
  };
  
  // Handle file preview
  const handlePreviewFile = (file: File) => {
    setSelectedFileForPreview(file);
    setIsPreviewOpen(true);
  };
  
  // Handle EDA report viewing
  const handleViewEDA = (fileIndex: number) => {
    setActiveFileIndex(fileIndex);
    setIsEdaReportOpen(true);
  };
  
  // Handle after EDA report is closed
  const handleEdaReportClosed = () => {
    setIsEdaReportOpen(false);
    
    // Mark the file as reviewed
    if (activeFileIndex >= 0 && activeFileIndex < files.length) {
      const fileName = files[activeFileIndex].name;
      setReviewedFiles(prev => new Set([...prev, fileName]));
    }
    
    // If all files have been reviewed, automatically go to next step
    if (files.length > 0 && reviewedFiles.size === files.length) {
      // Wait a brief moment to let the user see that all files are reviewed
      setTimeout(() => {
        toast({
          title: "All files reviewed",
          description: "Ready to proceed with upload and preprocessing",
        });
      }, 500);
    }
  };
  
  // Prepare for upload after selection
  const handleContinueToReview = () => {
    if (files.length === 0) {
      setError("Please select at least one file to upload");
      return;
    }
    setCurrentStep(1);
  };
  
  // Start upload after all files reviewed
  const handleStartUpload = () => {
    if (files.length === 0) {
      setError("Please select at least one file to upload");
      return;
    }
    
    setCurrentStep(2);
    uploadData();
  };
  
  // Remove file from selection
  const removeFile = (index: number) => {
    const newFiles = [...files];
    const removedFileName = newFiles[index].name;
    
    // Remove the file
    newFiles.splice(index, 1);
    setFiles(newFiles);
    
    // Remove from reviewed files if it exists
    if (reviewedFiles.has(removedFileName)) {
      const updatedReviewed = new Set(reviewedFiles);
      updatedReviewed.delete(removedFileName);
      setReviewedFiles(updatedReviewed);
    }
    
    toast({
      title: "File removed",
      description: `Removed ${removedFileName} from upload list`,
    });
  };
  
  // Check if all files are reviewed
  const areAllFilesReviewed = files.length > 0 && 
    files.every(file => reviewedFiles.has(file.name));
  
  // Navigate to dashboard or feature engineering
  const handleFinish = (destination: 'dashboard' | 'feature-engineering') => {
    // Redirect the user to the specified destination
    window.location.href = destination === 'dashboard' 
      ? '/dashboard' 
      : '/dashboard/feature-engineering';
  };

  return (
    <div {...getRootProps()} className="h-screen w-full px-6 md:px-10 py-10">
      {isDragActive &&
        <div className="absolute h-full w-full inset-0 bg-primary/10 flex items-center justify-center text-4xl backdrop-blur-sm z-50">
          <div className="bg-white dark:bg-gray-900 shadow-lg rounded-xl p-8 border border-primary">
            <FilePlus2 className="mx-auto h-16 w-16 text-primary mb-4" />
            <p className="text-center text-2xl font-medium">Drop Your Files Here</p>
          </div>
        </div>
      }
      
      <div className="max-w-5xl mx-auto">
        <div className="text-4xl font-bold mb-6">
          Upload Data
        </div>
        
        <Stepper currentStep={currentStep}>
          <Step>
            <StepTitle>Select Files</StepTitle>
            <StepDescription>Choose data files to upload</StepDescription>
          </Step>
          <Step>
            <StepTitle>Review Files</StepTitle>
            <StepDescription>Preview and analyze data</StepDescription>
          </Step>
          <Step>
            <StepTitle>Upload & Process</StepTitle>
            <StepDescription>Preprocess data for analysis</StepDescription>
          </Step>
          <Step>
            <StepTitle>Complete</StepTitle>
            <StepDescription>Use your data</StepDescription>
          </Step>
        </Stepper>
        
        <div className="mt-8">
          {/* Step 1: Select Files */}
          {currentStep === 0 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Select Your Data Files</CardTitle>
                <CardDescription>
                  Upload CSV or Excel files for data processing and analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-2">
                  <TabsList className="mb-4">
                    <TabsTrigger value="upload" className="flex items-center gap-2">
                      <FilePlus2 className="h-4 w-4" />
                      File Upload
                    </TabsTrigger>
                    <TabsTrigger value="kaggle" className="flex items-center gap-2">
                      <LinkIcon className="h-4 w-4" />
                      Import from Kaggle
                    </TabsTrigger>
                  </TabsList>
        
                  <TabsContent value="upload">
                    <input {...getInputProps()} />
                    <div className="cursor-pointer bg-muted/30 mt-4 h-40 rounded-lg border-2 border-dashed border-primary/20 flex items-center justify-center flex-col gap-3 hover:bg-muted/40 transition-all" onClick={open}>
                      <FilePlus2 className="h-10 w-10 text-primary/60" />
                      <div className="text-center">
                        <p className="text-muted-foreground font-medium mb-1">Click Here Or Drag And Drop Your Files</p>
                        <p className="text-xs text-muted-foreground">Accepts CSV and Excel (XLSX) files up to 10MB</p>
                      </div>
                    </div>
                  </TabsContent>
        
                  <TabsContent value="kaggle">
                    <KaggleUpload onFileImported={handleKaggleFileImported} />
                  </TabsContent>
                </Tabs>
                
                {files.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-lg font-medium mb-2">Selected Files ({files.length})</h3>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Size</TableHead>
                          <TableHead>Format</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {files.map((file, index) => (
                          <TableRow key={index}>
                            <TableCell className="font-medium">{file.name}</TableCell>
                            <TableCell>{(file.size / 1048576).toFixed(2)} MB</TableCell>
                            <TableCell>{file.type || file.name.split(".").pop()?.toUpperCase()}</TableCell>
                            <TableCell className="text-right">
                              <Button variant="ghost" size="icon" onClick={() => handlePreviewFile(file)} title="Preview file">
                                <Eye className="h-4 w-4" />
                              </Button>
                              <Button variant="ghost" size="icon" onClick={() => removeFile(index)} title="Remove file">
                                <X className="h-4 w-4" />
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
                
                {error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
                
                {files.length === 0 && !error && (
                  <Alert className="mt-4 border-purple-200 bg-purple-50 text-purple-800">
                    <FileText className="h-4 w-4" />
                    <AlertTitle>No files selected</AlertTitle>
                    <AlertDescription>
                      Select one or more data files to begin the upload and preprocessing workflow.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
              <CardFooter className="justify-between">
                <Button variant="outline" disabled>
                  Back
                </Button>
                <Button 
                  onClick={handleContinueToReview} 
                  disabled={files.length === 0}
                  className="gap-2"
                >
                  <span>Continue to Review</span>
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </CardFooter>
            </Card>
          )}
          
          {/* Step 2: Review Files */}
          {currentStep === 1 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Review Your Data</CardTitle>
                <CardDescription>
                  Preview and generate reports for your data before uploading
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Alert className="border-blue-200 bg-blue-50 text-blue-800 mb-6">
                  <BarChart2 className="h-4 w-4" />
                  <AlertTitle>Review Required</AlertTitle>
                  <AlertDescription>
                    Please review the EDA report for each file before proceeding with upload and preprocessing.
                    This helps ensure your data is suitable for analysis.
                  </AlertDescription>
                </Alert>
                
                <Table>
                  <TableCaption>Review all files before continuing</TableCaption>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Size</TableHead>
                      <TableHead>Format</TableHead>
                      <TableHead className="text-center">Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {files.map((file, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{file.name}</TableCell>
                        <TableCell>{(file.size / 1048576).toFixed(2)} MB</TableCell>
                        <TableCell>{file.type || file.name.split(".").pop()?.toUpperCase()}</TableCell>
                        <TableCell className="text-center">
                          {reviewedFiles.has(file.name) ? (
                            <Badge variant="outline" className="bg-green-50 text-green-700">
                              <CheckCircle2 className="h-3 w-3 mr-1" />
                              <span>Reviewed</span>
                            </Badge>
                          ) : (
                            <Badge variant="outline" className="bg-amber-50 text-amber-700">
                              <span>Needs Review</span>
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button 
                              variant="outline" 
                              size="sm" 
                              onClick={() => handlePreviewFile(file)}
                              className="gap-1"
                            >
                              <Eye className="h-3 w-3" />
                              <span>Preview</span>
                            </Button>
                            <Button 
                              variant={reviewedFiles.has(file.name) ? "outline" : "default"}
                              size="sm" 
                              onClick={() => handleViewEDA(index)}
                              className="gap-1"
                            >
                              <BarChart2 className="h-3 w-3" />
                              <span>{reviewedFiles.has(file.name) ? "View Report Again" : "Generate Report"}</span>
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
              <CardFooter className="justify-between">
                <Button variant="outline" onClick={() => setCurrentStep(0)}>
                  Back to Files
                </Button>
                <Button 
                  onClick={handleStartUpload} 
                  disabled={!areAllFilesReviewed}
                  className="gap-2"
                >
                  <Upload className="h-4 w-4" />
                  <span>Upload & Process Data</span>
                </Button>
              </CardFooter>
            </Card>
          )}
          
          {/* Step 3: Upload & Process */}
          {currentStep === 2 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Uploading & Processing Your Data</CardTitle>
                <CardDescription>
                  Please wait while we upload and preprocess your files
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mt-2 mb-8">
                  <p className="text-sm font-medium mb-2">Overall Progress</p>
                  <Progress value={uploadProgress} className="h-2 w-full" />
                  <p className="text-xs text-muted-foreground mt-2">
                    {uploadProgress < 100 
                      ? `Processing ${files.length} files...` 
                      : `Completed processing ${files.length} files`}
                  </p>
                </div>
                
                {Object.keys(processingStatus).length > 0 && (
                  <div className="mt-4">
                    <p className="text-sm font-medium mb-3">File Status:</p>
                    {Object.entries(processingStatus).map(([filename, status]) => (
                      <div key={filename} className="mb-3">
                        <div className="flex justify-between text-xs">
                          <span className="font-medium">{filename}</span>
                          <span>{status.message}</span>
                        </div>
                        <Progress 
                          value={status.progress < 0 ? 100 : status.progress} 
                          className={`h-2 w-full ${status.progress < 0 ? 'bg-red-300' : ''}`} 
                        />
                      </div>
                    ))}
                  </div>
                )}
                
                {error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error During Processing</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
              <CardFooter className="justify-between">
                <Button variant="outline" disabled={true}>
                  Back
                </Button>
                <Button 
                  onClick={() => setCurrentStep(3)} 
                  disabled={!uploadComplete && !error}
                >
                  Continue
                </Button>
              </CardFooter>
            </Card>
          )}
          
          {/* Step 4: Complete */}
          {currentStep === 3 && (
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
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Feature Engineering</CardTitle>
                      <CardDescription>Create new features for ML</CardDescription>
                    </CardHeader>
                                      <CardContent className="text-sm pb-2">
                      Transform your data and create new features to improve machine learning model performance
                    </CardContent>
                    <CardFooter>
                      <Button variant="outline" onClick={() => handleFinish('feature-engineering')} className="w-full gap-2">
                        <ChevronRight className="h-4 w-4" />
                        <span>Go to Feature Engineering</span>
                      </Button>
                    </CardFooter>
                  </Card>
                  
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">Dashboard</CardTitle>
                      <CardDescription>Return to dashboard</CardDescription>
                    </CardHeader>
                    <CardContent className="text-sm pb-2">
                      Go back to the dashboard to view all your uploaded files and explore other options
                    </CardContent>
                    <CardFooter>
                      <Button onClick={() => handleFinish('dashboard')} className="w-full gap-2">
                        <ChevronRight className="h-4 w-4" />
                        <span>Return to Dashboard</span>
                      </Button>
                    </CardFooter>
                  </Card>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
      
      {/* File Preview Dialog */}
      {selectedFileForPreview && (
        <FilePreview 
          fileMetadata={{
            id: Date.now().toString(),
            user_id: "temporary",
            filename: selectedFileForPreview.name,
            original_filename: selectedFileForPreview.name,
            file_size: selectedFileForPreview.size,
            mime_type: selectedFileForPreview.type,
            upload_date: new Date().toISOString(),
            column_names: [],
            row_count: 0,
            file_preview: [],
            statistics: {}
          }}
          isOpen={isPreviewOpen} 
          onClose={() => setIsPreviewOpen(false)} 
        />
      )}
      
      {/* EDA Report Dialog */}
      {files[activeFileIndex] && (
        <EdaReportViewer 
          fileMetadata={{
            id: Date.now().toString(),
            user_id: "temporary",
            filename: files[activeFileIndex].name,
            original_filename: files[activeFileIndex].name,
            file_size: files[activeFileIndex].size,
            mime_type: files[activeFileIndex].type,
            upload_date: new Date().toISOString(),
            column_names: [],
            row_count: 0,
            file_preview: [],
            statistics: {}
          }}
          originalFile={files[activeFileIndex]}
          isOpen={isEdaReportOpen} 
          onClose={handleEdaReportClosed} 
        />
      )}
    </div>
  );
}