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
import AutoPreprocessingReport from "@/components/AutoPreprocessingReport";

interface ProcessingInfo {
  filename: string;
  status: {
    status: string;
    progress: number;
    message: string;
  };
}

interface UploadResult {
  name: string;
  success: boolean;
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
  filesProcessed: UploadResult[];
}>({
  totalFiles: 0,
  successCount: 0,
  filesProcessed: []
});
  const [preprocessingResults, setPreprocessingResults] = useState<any>(null);
  const [filePreprocessingResults, setFilePreprocessingResults] = useState<Record<string, any>>({});

 // This is a utility function that can be added to dashboard/data-upload/page.tsx

/**
 * Enhanced version that transforms and normalizes the preprocessing results
 * from various formats, including those from the joblib file
 * @param serverResponse - The preprocessing results from the server
 * @returns Normalized preprocessing results
 */
function transformPreprocessingResults(serverResponse: any) {
  // If the response is null or undefined, return null
  if (!serverResponse) return null;

  console.log("Transforming server response:", serverResponse);

  // Handle the case where results come from joblib pipeline data
  if (serverResponse.preprocessing_info || 
      (serverResponse.original_shape && serverResponse.missing_value_stats)) {
    // Direct pipeline data format
    return {
      success: true,
      original_shape: serverResponse.original_shape || [0, 0],
      processed_shape: serverResponse.processed_shape || serverResponse.final_shape || [0, 0],
      columns_dropped: serverResponse.columns_dropped || [],
      dropped_by_unique_value: serverResponse.dropped_by_unique_value || [],
      date_columns_detected: serverResponse.date_columns_detected || [],
      columns_cleaned: serverResponse.columns_cleaned || [],
      missing_value_stats: serverResponse.missing_value_stats || {},
      engineered_features: serverResponse.engineered_features || [],
      transformation_details: serverResponse.transformation_details || {}
    };
  }
  
  // Check if results are nested under preprocessing_info
  if (serverResponse.preprocessing_info) {
    const info = serverResponse.preprocessing_info;
    return {
      success: serverResponse.success !== false,
      original_shape: serverResponse.original_shape || info.original_shape || [0, 0],
      processed_shape: serverResponse.processed_shape || info.processed_shape || [0, 0],
      columns_dropped: info.dropped_columns || [],
      dropped_by_unique_value: info.dropped_by_unique_value || [],
      date_columns_detected: info.auto_detected_dates || [],
      columns_cleaned: info.columns_cleaned || [],
      missing_value_stats: info.missing_value_stats || {},
      engineered_features: info.engineered_features || [],
      transformation_details: info.transformation_details || {}
    };
  }
  
  // Check if results come from processing_status endpoint via joblib file
  if (serverResponse.results) {
    return transformPreprocessingResults(serverResponse.results);
  }
  
  // Check for the "report" or "preprocessing_info" structure
  if (serverResponse.report && typeof serverResponse.report === 'object') {
    const report = serverResponse.report;
    return {
      success: serverResponse.success !== false,
      original_shape: report.original_shape || serverResponse.original_shape || [0, 0],
      processed_shape: report.processed_shape || serverResponse.processed_shape || [0, 0],
      columns_dropped: report.columns_dropped || [],
      dropped_by_unique_value: report.dropped_by_unique_value || [],
      date_columns_detected: report.date_columns_detected || [],
      columns_cleaned: report.columns_cleaned || [],
      missing_value_stats: report.missing_value_stats || {},
      engineered_features: report.engineered_features || [],
      transformation_details: report.transformation_details || {}
    };
  }
  
  // Try to extract directly from response
  return {
    success: serverResponse.success !== false,
    original_shape: serverResponse.original_shape || [0, 0],
    processed_shape: serverResponse.processed_shape || [0, 0],
    columns_dropped: serverResponse.columns_dropped || [],
    dropped_by_unique_value: serverResponse.dropped_by_unique_value || [],
    date_columns_detected: serverResponse.date_columns_detected || [],
    columns_cleaned: serverResponse.columns_cleaned || [],
    missing_value_stats: serverResponse.missing_value_stats || {},
    engineered_features: serverResponse.engineered_features || [],
    transformation_details: serverResponse.transformation_details || {}
  };
}

 const uploadData = async () => {
  if (files.length === 0) {
    setError("Please select at least one file to upload");
    return;
  }

  setIsUploading(true);
  setError(null);
  setUploadProgress(0);
  
  // First check if the server is available
  try {
    const serverCheckResponse = await fetch('http://localhost:8000/', { 
      method: 'GET',
      signal: AbortSignal.timeout(3000) // 3 second timeout
    }).catch(err => {
      console.error("Server check failed:", err);
      throw new Error("Cannot connect to preprocessing server. Please make sure it's running at http://localhost:8000");
    });
    
    if (!serverCheckResponse.ok) {
      throw new Error(`Preprocessing server is not responding properly: ${serverCheckResponse.statusText}`);
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : 'Server connection failed';
    setError(errorMessage);
    console.error('Server check error:', err);
    setIsUploading(false);
    return;
  }
  
  try {
    // Step 1: Upload to Python for preprocessing
    const pythonFormData = new FormData();
    files.forEach(file => {
      pythonFormData.append('files', file);
    });
    
    setUploadProgress(10);
    console.log("Uploading files to Python backend...");
    
    // Send to Python backend with timeout
    const pythonResponse = await fetch('http://localhost:8000/upload/', {
      method: 'POST',
      body: pythonFormData,
      signal: AbortSignal.timeout(30000) // 30 second timeout
    }).catch(err => {
      console.error("Error connecting to Python backend:", err);
      throw new Error("Failed to connect to preprocessing server. Please make sure it's running.");
    });

    if (!pythonResponse.ok) {
      const errorData = await pythonResponse.json();
      throw new Error(errorData.detail || `Preprocessing failed: ${pythonResponse.statusText}`);
    }

    const preprocessingResult = await pythonResponse.json();
    console.log("Received preprocessing result:", preprocessingResult);
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
      console.log("Tracking processing status for files:", fileNames);
      const processingSuccess = await trackProcessingStatus(fileNames);
      
      // If processing completely failed, we can try to continue with just the files
      if (!processingSuccess) {
        console.log("Processing status tracking failed. Continuing with upload only...");
        toast({
          variant: "warning",
          title: "Processing Incomplete",
          description: "Will continue with file upload without preprocessing.",
        });
      }
    }

    setUploadProgress(60);
    
    // Create a local copy of preprocessing data to use for upload
    const preprocessingData = {};
    for (const fileName of fileNames) {
      const statusInfo = processingStatus[fileName];
      if (statusInfo && (statusInfo.results || statusInfo.preprocessing_details)) {
        preprocessingData[fileName] = statusInfo.preprocessing_details || statusInfo.results;
      }
    }
    
    // Step 3: Upload to database even if preprocessing failed
    setUploadProgress(70);
    const uploadResults: UploadResult[] = [];

    for (const file of files) {
      // Find the matching processed file
      const processedFileName = fileNames.find(fn => fn.includes(file.name));
      
      // Create a new FormData for each file
      const formData = new FormData();
      formData.append('file', file);
      formData.append('preprocessed', Object.keys(preprocessingData).length > 0 ? 'true' : 'false');

      // Include preprocessing results if available
      const filePreprocessingResult = processedFileName ? preprocessingData[processedFileName] : null;
      
      if (filePreprocessingResult) {
        console.log(`Adding preprocessing results for ${file.name}`);
        formData.append('preprocessing_results', JSON.stringify(filePreprocessingResult));
      } else {
        console.log(`No preprocessing results found for ${file.name}. Uploading without preprocessing.`);
      }
      
      try {
        console.log(`Uploading ${file.name} to database...`);
        // Use relative URL for API endpoint
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
          signal: AbortSignal.timeout(30000) // 30 second timeout
        });

        const result = await response.json();
        
        if (!response.ok) {
          console.error(`Upload failed for ${file.name}:`, result);
          uploadResults.push({
            name: file.name,
            success: false
          });
        } else {
          console.log(`Successfully uploaded ${file.name} to database:`, result);
          uploadResults.push({
            name: file.name,
            success: true,
            metadata: result.metadata
          });
        }
      } catch (err) {
        console.error(`Exception during upload for ${file.name}:`, err);
        uploadResults.push({
          name: file.name,
          success: false
        });
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
        description: `All ${files.length} files have been uploaded successfully`,
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




// Helper function to debug the preprocessing results structure
const debugPreprocessingResults = () => {
  console.group("Debugging Preprocessing Results");
  
  // Log the overall state
  console.log("filePreprocessingResults state:", filePreprocessingResults);
  
  // Check each file's preprocessing results
  if (uploadSummary.filesProcessed.length > 0) {
    uploadSummary.filesProcessed.forEach(file => {
      const results = filePreprocessingResults[file.name];
      console.group(`File: ${file.name}`);
      
      if (!results) {
        console.log("No preprocessing results found for this file");
      } else {
        // Log the structure of results
        console.log("Raw results:", results);
        
        // Check for preprocessing_info
        if (results.preprocessing_info) {
          console.log("preprocessing_info found:", results.preprocessing_info);
          
          // Check for important properties
          console.log("- columns_dropped:", results.preprocessing_info.columns_dropped || "none");
          console.log("- date_columns_detected:", results.preprocessing_info.date_columns_detected || "none");
          console.log("- columns_cleaned:", results.preprocessing_info.columns_cleaned || "none");
          console.log("- missing_value_stats:", 
            Object.keys(results.preprocessing_info.missing_value_stats || {}).length, "columns");
        } else {
          console.log("No preprocessing_info found in results");
        }
        
        // Test the transformation function
        const transformed = transformPreprocessingResults(results);
        console.log("Transformed results:", transformed);
      }
      
      console.groupEnd();
    });
  } else {
    console.log("No processed files available");
  }
  
  console.groupEnd();
};

// Call this in useEffect after upload completes
useEffect(() => {
  if (uploadComplete && uploadSummary.filesProcessed.length > 0) {
    debugPreprocessingResults();
  }
}, [uploadComplete, uploadSummary.filesProcessed]);

useEffect(() => {
  console.log("Current filePreprocessingResults:", filePreprocessingResults);
  console.log("Current uploadSummary:", uploadSummary);
  
  if (uploadSummary.filesProcessed.length > 0) {
    const hasPreprocessingData = uploadSummary.filesProcessed.some(
      file => filePreprocessingResults[file.name]
    );
    console.log("Has preprocessing data:", hasPreprocessingData);
  }
}, [filePreprocessingResults, uploadSummary]);
  
  // Function to poll processing status
  // Enhanced trackProcessingStatus function with better results handling
const trackProcessingStatus = async (fileNames: string[]) => {
  let allCompleted = false;
  let attempts = 0;
  const maxAttempts = 30; // Timeout after 30 attempts

  // Add server availability check first
  try {
    // Simple health check to the root endpoint
    const healthCheck = await fetch('http://localhost:8000/', { 
      method: 'GET',
      // Add a short timeout to fail fast if server is unreachable
      signal: AbortSignal.timeout(3000)
    });
    
    if (!healthCheck.ok) {
      console.error("Python backend server is not responding properly:", await healthCheck.text());
      // Still continue since the server might be partially functioning
    }
  } catch (err) {
    console.error("Cannot connect to Python backend server:", err);
    // Create a toast notification to inform the user
    toast({
      variant: "destructive",
      title: "Server Connection Error",
      description: "Cannot connect to the preprocessing server. Please make sure it's running at http://localhost:8000",
    });
    
    // Return false to indicate failure
    return false;
  }
  
  while (!allCompleted && attempts < maxAttempts) {
    attempts++;
    let completedCount = 0;
    
    for (const fileName of fileNames) {
      try {
        console.log(`Checking processing status for ${fileName}, attempt ${attempts}...`);
        
        // Add timeout to the fetch request
        const response = await fetch(`http://localhost:8000/processing-status/${fileName}`, {
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });
        
        if (!response.ok) {
          console.error(`Error fetching status for ${fileName}: ${response.statusText}`);
          continue;
        }
        
        const status = await response.json();
        
        // Update the processing status in state
        setProcessingStatus(prev => {
          const updatedStatus = {
            ...prev,
            [fileName]: {
              status: status.status,
              progress: status.progress,
              message: status.message,
              results: status.results || status.preprocessing_details
            }
          };
          
          console.log(`Updated status for ${fileName}:`, updatedStatus[fileName]);
          
          return updatedStatus;
        });
        
        if (status.progress === 100 || status.progress === -1) {
          completedCount++;
          console.log(`Processing completed for ${fileName} (${completedCount}/${fileNames.length})`);
        }
      } catch (error) {
        console.error(`Error checking status for ${fileName}:`, error);
        // If we've tried a few times and still can't connect, notify the user
        if (attempts > 3) {
          toast({
            variant: "destructive",
            title: "Connection Error",
            description: "Having trouble connecting to the preprocessing server. Please check if it's running.",
          });
        }
      }
    }
    
    if (completedCount === fileNames.length) {
      console.log("All files completed processing");
      allCompleted = true;
      
      // Try to get final results, but don't fail if we can't
      try {
        // After all files are processed, do one final gathering of results
        const preprocessingData = {};
        
        for (const fileName of fileNames) {
          try {
            // Make one final status check for each file
            const response = await fetch(`http://localhost:8000/processing-status/${fileName}`, {
              signal: AbortSignal.timeout(5000) // 5 second timeout
            });
            
            if (response.ok) {
              const finalStatus = await response.json();
              const resultData = finalStatus.results || finalStatus.preprocessing_details;
              
              if (resultData) {
                console.log(`Final preprocessing results for ${fileName}:`, resultData);
                
                // Store the results
                preprocessingData[fileName] = resultData;
              }
            }
          } catch (error) {
            console.error(`Error getting final status for ${fileName}:`, error);
          }
        }
        
        // Update state with all preprocessing results
        if (Object.keys(preprocessingData).length > 0) {
          console.log("Updating filePreprocessingResults with:", preprocessingData);
          setFilePreprocessingResults(preprocessingData);
        }
      } catch (err) {
        console.error("Error getting final preprocessing results:", err);
      }
    } else {
      // Wait before next poll
      console.log(`Waiting for next status check... (${completedCount}/${fileNames.length} completed)`);
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
  
  if (attempts >= maxAttempts) {
    console.warn("Reached maximum polling attempts, some files may not have completed processing");
    toast({
      variant: "destructive",
      title: "Processing Timeout",
      description: "Processing is taking longer than expected. Proceeding with available results.",
    });
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
      
      {/* Detailed Preprocessing Reports Section */}
      {uploadSummary.filesProcessed.some(file => filePreprocessingResults[file.name]) && (
        <div className="mb-6">
          <h3 className="text-xl font-bold mb-4">Data Preprocessing Report</h3>
          <div className="space-y-6">
            {uploadSummary.filesProcessed.map((file, index) => {
              if (file.success && filePreprocessingResults[file.name]) {
                // Transform the results to match the expected format
                const transformedResults = transformPreprocessingResults(filePreprocessingResults[file.name]);
                
                // Debug logging to verify the structure
                console.log(`Transformed results for ${file.name}:`, transformedResults);
                
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
                    
                    {/* Columns Dropped */}
                    {transformedResults.columns_dropped && transformedResults.columns_dropped.length > 0 && (
                      <div className="mb-4">
                        <h5 className="text-base font-semibold mb-2">Columns Dropped</h5>
                        <div className="flex flex-wrap gap-2">
                          {transformedResults.columns_dropped.map((column, idx) => (
                            <Badge key={idx} variant="outline" className="bg-red-50 text-red-700">
                              {column}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Columns Dropped by Unique Value */}
                    {transformedResults.dropped_by_unique_value && transformedResults.dropped_by_unique_value.length > 0 && (
                      <div className="mb-4">
                        <h5 className="text-base font-semibold mb-2">Columns Dropped (Single Value)</h5>
                        <div className="flex flex-wrap gap-2">
                          {transformedResults.dropped_by_unique_value.map((column, idx) => (
                            <Badge key={idx} variant="outline" className="bg-orange-50 text-orange-700">
                              {column}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Date Columns Detected */}
                    {transformedResults.date_columns_detected && transformedResults.date_columns_detected.length > 0 && (
                      <div className="mb-4">
                        <h5 className="text-base font-semibold mb-2">Date Columns Detected</h5>
                        <div className="flex flex-wrap gap-2">
                          {transformedResults.date_columns_detected.map((column, idx) => (
                            <Badge key={idx} variant="outline" className="bg-blue-50 text-blue-700">
                              {column}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Columns with Missing Values Cleaned */}
                    {transformedResults.columns_cleaned && transformedResults.columns_cleaned.length > 0 && (
                      <div className="mb-4">
                        <h5 className="text-base font-semibold mb-2">Columns with Missing Values Handled</h5>
                        <div className="flex flex-wrap gap-2">
                          {transformedResults.columns_cleaned.map((column, idx) => (
                            <Badge key={idx} variant="outline" className="bg-green-50 text-green-700">
                              {column}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Missing Value Details */}
                    {transformedResults.missing_value_stats && Object.keys(transformedResults.missing_value_stats).length > 0 && (
                      <div className="mb-4">
                        <h5 className="text-base font-semibold mb-2">Missing Value Details</h5>
                        <div className="bg-gray-50 p-3 rounded-md max-h-40 overflow-y-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b">
                                <th className="text-left py-1 px-2">Column</th>
                                <th className="text-right py-1 px-2">Missing Count</th>
                                <th className="text-right py-1 px-2">Missing %</th>
                                <th className="text-right py-1 px-2">Imputation Method</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(transformedResults.missing_value_stats).map(([column, stats], idx) => (
                                <tr key={idx} className="border-b">
                                  <td className="py-1 px-2">{column}</td>
                                  <td className="text-right py-1 px-2">{stats.missing_count}</td>
                                  <td className="text-right py-1 px-2">{stats.missing_percentage}%</td>
                                  <td className="text-right py-1 px-2">
                                    <Badge variant="outline" className="bg-blue-50 text-blue-700">
                                      {stats.imputation_method || "None"}
                                    </Badge>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                    
                    {/* Engineered Features */}
                    {transformedResults.engineered_features && transformedResults.engineered_features.length > 0 && (
                      <div className="mb-4">
                        <h5 className="text-base font-semibold mb-2">Engineered Features</h5>
                        <div className="flex flex-wrap gap-2">
                          {transformedResults.engineered_features.map((feature, idx) => (
                            <Badge key={idx} variant="outline" className="bg-purple-50 text-purple-700">
                              {feature}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                    {(!transformedResults.columns_dropped || transformedResults.columns_dropped.length === 0) &&
                    (!transformedResults.date_columns_detected || transformedResults.date_columns_detected.length === 0) &&
                    (!transformedResults.columns_cleaned || transformedResults.columns_cleaned.length === 0) &&
                    (!transformedResults.missing_value_stats || Object.keys(transformedResults.missing_value_stats).length === 0) && (
                      <AutoPreprocessingReport
                        processingResults={transformedResults}
                        fileName={file.name}
                        isLoading={false}
                      />
                    )}
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