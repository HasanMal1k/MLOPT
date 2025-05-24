'use client'

import { useState, useCallback, useEffect, useRef } from "react";
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
  Eye,
  Settings2,
  ArrowLeft,
  Loader2
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
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

import FilePreview from "@/components/FilePreview";
import EdaReportViewer from "@/components/EdaReportViewer";
import KaggleUpload from "@/components/KaggleUpload";
import AutoPreprocessingReport from "@/components/AutoPreprocessingReport";

// Define interfaces for custom cleaning
interface ColumnAnalysis {
  name: string;
  current_type: string;
  suggested_type: string;
  sample_values: string[];
}

interface FileAnalysis {
  success: boolean;
  filename: string;
  row_count: number;
  column_count: number;
  columns_info: ColumnAnalysis[];
}

interface PreviewData {
  original: Record<string, any>[];
  transformed: Record<string, any>[];
  columns: {
    original: string[];
    transformed: string[];
  };
}

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

export default function EnhancedDataUpload() {
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

  // Custom cleaning states (always enabled now)
  const [fileAnalysisData, setFileAnalysisData] = useState<Record<string, FileAnalysis>>({});
  const [columnEdits, setColumnEdits] = useState<Record<string, Record<string, {
    newType?: string;
    drop?: boolean;
  }>>>({});
  const [isAnalyzingFiles, setIsAnalyzingFiles] = useState(false);
  const [previewData, setPreviewData] = useState<Record<string, PreviewData>>({});
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [previewNeedsUpdate, setPreviewNeedsUpdate] = useState(false);
  const [autoProcessingComplete, setAutoProcessingComplete] = useState(false);

  // Use refs for debouncing and tracking the latest edits
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const latestColumnEditsRef = useRef<Record<string, any>>({});

  const transformPreprocessingResults = (serverResponse) => {
    // If the response is null or undefined, return null
    if (!serverResponse) return null;

    console.log("Transforming server response:", serverResponse);

    // Check if results are nested under preprocessing_info
    if (serverResponse.preprocessing_info) {
      return {
        success: serverResponse.success !== false,
        original_shape: serverResponse.original_shape || [0, 0],
        processed_shape: serverResponse.processed_shape || [0, 0],
        columns_dropped: serverResponse.preprocessing_info.columns_dropped || [],
        date_columns_detected: serverResponse.preprocessing_info.date_columns_detected || [],
        columns_cleaned: serverResponse.preprocessing_info.columns_cleaned || [],
        missing_value_stats: serverResponse.preprocessing_info.missing_value_stats || {},
        engineered_features: serverResponse.preprocessing_info.engineered_features || [],
        transformation_details: serverResponse.preprocessing_info.transformation_details || {}
      };
    }
    
    // Check if it's a "results" wrapper with nested structure (from Python backend)
    if (serverResponse.results && typeof serverResponse.results === 'object') {
      return transformPreprocessingResults(serverResponse.results);
    }
    
    // Check for the "report" structure that might be returned from some endpoints
    if (serverResponse.report && typeof serverResponse.report === 'object') {
      const report = serverResponse.report;
      return {
        success: serverResponse.success !== false,
        original_shape: report.original_shape || serverResponse.original_shape || [0, 0],
        processed_shape: report.processed_shape || serverResponse.processed_shape || [0, 0],
        columns_dropped: report.columns_dropped || [],
        date_columns_detected: report.date_columns_detected || [],
        columns_cleaned: report.columns_cleaned || [],
        missing_value_stats: report.missing_value_stats || {},
        engineered_features: report.engineered_features || [],
        transformation_details: report.transformation_details || {}
      };
    }
    
    // Response is already in the expected format (or close enough)
    return {
      success: serverResponse.success !== false,
      original_shape: serverResponse.original_shape || [0, 0],
      processed_shape: serverResponse.processed_shape || [0, 0],
      columns_dropped: serverResponse.columns_dropped || [],
      date_columns_detected: serverResponse.date_columns_detected || [],
      columns_cleaned: serverResponse.columns_cleaned || [],
      missing_value_stats: serverResponse.missing_value_stats || {},
      engineered_features: serverResponse.engineered_features || [],
      transformation_details: serverResponse.transformation_details || {}
    };
  };

  // Analyze files for custom cleaning (always enabled now)
  const analyzeFilesForCustomCleaning = async () => {
    if (files.length === 0) return;

    setIsAnalyzingFiles(true);
    const analysisResults: Record<string, FileAnalysis> = {};
    const initialColumnEdits: Record<string, Record<string, any>> = {};

    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/custom-preprocessing/analyze-file/', {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          analysisResults[file.name] = result;

          // Initialize column edits for this file
          const fileColumnEdits: Record<string, any> = {};
          result.columns_info.forEach((col: ColumnAnalysis) => {
            fileColumnEdits[col.name] = {
              newType: col.suggested_type,
              drop: false
            };
          });
          initialColumnEdits[file.name] = fileColumnEdits;
        }
      } catch (error) {
        console.error(`Error analyzing ${file.name}:`, error);
      }
    }

    setFileAnalysisData(analysisResults);
    setColumnEdits(initialColumnEdits);
    latestColumnEditsRef.current = initialColumnEdits;
    setIsAnalyzingFiles(false);
  };

  // Update preview for custom cleaning
  const updatePreviewForFile = useCallback(async (fileName: string) => {
    const file = files.find(f => f.name === fileName);
    const fileAnalysis = fileAnalysisData[fileName];
    
    if (!file || !fileAnalysis) return;

    if (isLoadingPreview) {
      setPreviewNeedsUpdate(true);
      return;
    }

    setIsLoadingPreview(true);
    setPreviewNeedsUpdate(false);

    try {
      const transformationConfig = {
        data_types: {} as Record<string, string>,
        columns_to_drop: [] as string[]
      };

      const currentEdits = latestColumnEditsRef.current[fileName] || {};

      Object.entries(currentEdits).forEach(([colName, edit]) => {
        if (edit.newType) {
          transformationConfig.data_types[colName] = edit.newType;
        }
        if (edit.drop) {
          transformationConfig.columns_to_drop.push(colName);
        }
      });

      const formData = new FormData();
      formData.append('file', file);
      formData.append('transformations', JSON.stringify(transformationConfig));

      const previewResponse = await fetch('http://localhost:8000/custom-preprocessing/preview-transformation/', {
        method: 'POST',
        body: formData
      });

      if (previewResponse.ok) {
        const result = await previewResponse.json();
        if (result.success && result.preview) {
          setPreviewData(prev => ({
            ...prev,
            [fileName]: result.preview
          }));
        }
      }
    } catch (error) {
      console.error('Error generating preview:', error);
    } finally {
      setIsLoadingPreview(false);
      
      if (previewNeedsUpdate) {
        setTimeout(() => {
          updatePreviewForFile(fileName);
        }, 100);
      }
    }
  }, [files, fileAnalysisData, isLoadingPreview]);

  // Update column edit for custom cleaning
  const updateColumnEdit = (fileName: string, columnName: string, field: string, value: any) => {
    setColumnEdits(prev => {
      const newEdits = {
        ...prev,
        [fileName]: {
          ...prev[fileName],
          [columnName]: {
            ...prev[fileName]?.[columnName],
            [field]: value
          }
        }
      };
      
      latestColumnEditsRef.current = newEdits;
      return newEdits;
    });

    // Clear any existing timeout
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current);
    }

    // Update preview with debouncing
    if (field === 'drop') {
      setTimeout(() => {
        updatePreviewForFile(fileName);
      }, 10);
    } else {
      previewTimeoutRef.current = setTimeout(() => {
        updatePreviewForFile(fileName);
      }, 300);
    }
  };

  // Auto processing step - processes files for EDA and initial analysis
  const performAutoProcessing = async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);
    
    try {
      // Step 1: Upload to Python for initial auto preprocessing
      const pythonFormData = new FormData();
      files.forEach(file => {
        pythonFormData.append('files', file);
      });
      
      setUploadProgress(20);
      
      // Send to Python backend for auto preprocessing
      const pythonResponse = await fetch('http://localhost:8000/upload/', {
        method: 'POST',
        body: pythonFormData,
      });

      if (!pythonResponse.ok) {
        const errorData = await pythonResponse.json();
        throw new Error(errorData.detail || 'Auto preprocessing failed');
      }

      const preprocessingResult = await pythonResponse.json();
      setUploadProgress(60);
      
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
        await trackAutoProcessingStatus(fileNames);
      }

      setUploadProgress(100);
      setAutoProcessingComplete(true);
      
      toast({
        title: "Auto Processing Complete",
        description: `${files.length} files have been automatically processed and are ready for custom cleaning`,
      });
      
      // Move to custom cleaning step
      setCurrentStep(2);
      setIsUploading(false);
      
      // Start analysis for custom cleaning
      await analyzeFilesForCustomCleaning();
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Auto processing failed';
      setError(errorMessage);
      console.error('Auto processing error:', err);
      setIsUploading(false);
    }
  };

  // Track auto processing status
  const trackAutoProcessingStatus = async (fileNames: string[]) => {
    let allCompleted = false;
    let attempts = 0;
    const maxAttempts = 30;
    
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
        
        const preprocessingData = {};
        for (const fileName of fileNames) {
          const statusInfo = processingStatus[fileName];
          if (statusInfo && statusInfo.results) {
            preprocessingData[fileName] = statusInfo.results;
          }
        }
        
        setFilePreprocessingResults(preprocessingData);
      } else {
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
    
    return allCompleted;
  };

  // Apply custom transformations and final upload
  const applyCustomTransformationsAndUpload = async () => {
    setIsUploading(true);
    setError(null);
    setUploadProgress(0);
    
    try {
      // Step 1: Apply custom transformations
      setUploadProgress(20);
      const transformedFiles = await applyCustomTransformations();
      
      // Step 2: Upload to database
      setUploadProgress(50);
      const uploadResults: UploadResult[] = [];

      for (const file of transformedFiles) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('preprocessed', 'true');

        if (filePreprocessingResults[file.name]) {
          formData.append('preprocessing_results', JSON.stringify(filePreprocessingResults[file.name]));
        }
        
        try {
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
        totalFiles: transformedFiles.length,
        successCount: successCount,
        filesProcessed: uploadResults
      });
      
      setUploadProgress(100);
      setUploadComplete(true);
      
      if (successCount === transformedFiles.length) {
        toast({
          title: "Success",
          description: `All ${transformedFiles.length} files have been uploaded successfully with custom transformations applied`,
        });
      } else {
        toast({
          variant: "destructive",
          title: "Partial success",
          description: `Uploaded ${successCount} of ${transformedFiles.length} files successfully`,
        });
      }
      
      // Move to final step
      setCurrentStep(4);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  // Apply custom transformations (always enabled now)
  const applyCustomTransformations = async () => {
    const transformedFiles: File[] = [];

    for (const file of files) {
      const fileEdits = columnEdits[file.name];
      if (!fileEdits) {
        transformedFiles.push(file);
        continue;
      }

      try {
        const transformationConfig = {
          data_types: {} as Record<string, string>,
          columns_to_drop: [] as string[]
        };

        Object.entries(fileEdits).forEach(([colName, edit]) => {
          if (edit.newType) {
            transformationConfig.data_types[colName] = edit.newType;
          }
          if (edit.drop) {
            transformationConfig.columns_to_drop.push(colName);
          }
        });

        const formData = new FormData();
        formData.append('file', file);
        formData.append('transformations', JSON.stringify(transformationConfig));

        const response = await fetch('http://localhost:8000/custom-preprocessing/apply-transformations/', {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          const result = await response.json();
          if (result.success && result.transformed_file) {
            // Download the transformed file
            const downloadResponse = await fetch(`http://localhost:8000/preprocessing_results/${result.transformed_file}`);
            if (downloadResponse.ok) {
              const blob = await downloadResponse.blob();
              const transformedFile = new File([blob], file.name, { type: file.type });
              transformedFiles.push(transformedFile);
            } else {
              transformedFiles.push(file);
            }
          } else {
            transformedFiles.push(file);
          }
        } else {
          transformedFiles.push(file);
        }
      } catch (error) {
        console.error(`Error transforming ${file.name}:`, error);
        transformedFiles.push(file);
      }
    }

    return transformedFiles;
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
    
    if (activeFileIndex >= 0 && activeFileIndex < files.length) {
      const fileName = files[activeFileIndex].name;
      setReviewedFiles(prev => new Set([...prev, fileName]));
    }
    
    if (files.length > 0 && reviewedFiles.size === files.length) {
      setTimeout(() => {
        toast({
          title: "All files reviewed",
          description: "Ready to proceed with processing",
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

  // Continue to auto processing step
  const handleContinueToAutoProcessing = () => {
    setCurrentStep(1);
    performAutoProcessing();
  };
  
  // Start upload after all files reviewed
  const handleStartUpload = () => {
    if (files.length === 0) {
      setError("Please select at least one file to upload");
      return;
    }
    
    setCurrentStep(3);
    applyCustomTransformationsAndUpload();
  };
  
  // Remove file from selection
  const removeFile = (index: number) => {
    const newFiles = [...files];
    const removedFileName = newFiles[index].name;
    
    newFiles.splice(index, 1);
    setFiles(newFiles);
    
    if (reviewedFiles.has(removedFileName)) {
      const updatedReviewed = new Set(reviewedFiles);
      updatedReviewed.delete(removedFileName);
      setReviewedFiles(updatedReviewed);
    }
    
    // Clean up custom cleaning data
    setFileAnalysisData(prev => {
      const updated = { ...prev };
      delete updated[removedFileName];
      return updated;
    });
    
    setColumnEdits(prev => {
      const updated = { ...prev };
      delete updated[removedFileName];
      return updated;
    });
    
    setPreviewData(prev => {
      const updated = { ...prev };
      delete updated[removedFileName];
      return updated;
    });
    
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
    window.location.href = destination === 'dashboard' 
      ? '/dashboard' 
      : '/dashboard/feature-engineering';
  };

  // Format data type for display
  const formatDataType = (type: string) => {
    switch (type) {
      case 'int':
      case 'int64':
      case 'int32':
        return 'Integer'
      case 'float':
      case 'float64':
      case 'float32':
        return 'Decimal'
      case 'object':
      case 'string':
        return 'Text'
      case 'datetime64[ns]':
      case 'datetime':
        return 'Date & Time'
      default:
        return type
    }
  };

  // Check if any columns are marked as dropped for a file
  const hasDroppedColumns = (fileName: string) => {
    const fileEdits = columnEdits[fileName];
    if (!fileEdits) return false;
    return Object.values(fileEdits).some(edit => edit.drop);
  };

  // Check if any column types have been changed for a file
  const hasChangedTypes = (fileName: string) => {
    const fileAnalysis = fileAnalysisData[fileName];
    const fileEdits = columnEdits[fileName];
    if (!fileAnalysis || !fileEdits) return false;
    
    return fileAnalysis.columns_info.some(col => {
      const edit = fileEdits[col.name];
      return edit && edit.newType && edit.newType !== col.current_type;
    });
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current);
      }
    };
  }, []);

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
            <StepTitle>Auto Processing</StepTitle>
            <StepDescription>Automatic data cleaning</StepDescription>
          </Step>
          <Step>
            <StepTitle>Custom Cleaning</StepTitle>
            <StepDescription>Configure data transformations</StepDescription>
          </Step>
          <Step>
            <StepTitle>Upload & Finalize</StepTitle>
            <StepDescription>Apply changes and upload</StepDescription>
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
                  Preview and generate reports for your data before processing
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Alert className="border-blue-200 bg-blue-50 text-blue-800 mb-6">
                  <BarChart2 className="h-4 w-4" />
                  <AlertTitle>Review Required</AlertTitle>
                  <AlertDescription>
                    Please review the EDA report for each file before proceeding with processing.
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
                  onClick={handleContinueToAutoProcessing} 
                  disabled={!areAllFilesReviewed}
                  className="gap-2"
                >
                  <Settings2 className="h-4 w-4" />
                  <span>Start Auto Processing</span>
                </Button>
              </CardFooter>
            </Card>
          )}

          {/* Step 3: Auto Processing */}
          {currentStep === 2 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Auto Processing Your Data</CardTitle>
                <CardDescription>
                  Please wait while we automatically clean and preprocess your files
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mt-2 mb-8">
                  <p className="text-sm font-medium mb-2">Overall Progress</p>
                  <Progress value={uploadProgress} className="h-2 w-full" />
                  <p className="text-xs text-muted-foreground mt-2">
                    {uploadProgress < 100 
                      ? `Auto-processing ${files.length} files...` 
                      : `Completed auto-processing ${files.length} files`}
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
                  disabled={!autoProcessingComplete && !error}
                >
                  Continue to Custom Cleaning
                </Button>
              </CardFooter>
            </Card>
          )}

          {/* Step 4: Custom Cleaning */}
          {currentStep === 3 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Custom Data Cleaning</CardTitle>
                <CardDescription>
                  Configure data type conversions and column operations before final upload
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isAnalyzingFiles ? (
                  <div className="flex flex-col items-center justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin text-primary mb-4" />
                    <p className="text-center text-muted-foreground">
                      Analyzing files for custom cleaning options...
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <Alert className="border-orange-200 bg-orange-50 text-orange-800">
                      <Settings2 className="h-4 w-4" />
                      <AlertTitle>Custom Data Transformations</AlertTitle>
                      <AlertDescription>
                        Configure how each column should be processed. Changes are previewed in real-time.
                        Auto-processing has already been completed for these files.
                      </AlertDescription>
                    </Alert>

                    {files.map((file, fileIndex) => {
                      const fileAnalysis = fileAnalysisData[file.name];
                      const filePreview = previewData[file.name];
                      
                      if (!fileAnalysis) {
                        return (
                          <Card key={fileIndex}>
                            <CardHeader>
                              <CardTitle className="text-base">{file.name}</CardTitle>
                              <CardDescription>Analysis failed for this file</CardDescription>
                            </CardHeader>
                          </Card>
                        );
                      }

                      return (
                        <Card key={fileIndex} className="border-2">
                          <CardHeader>
                            <CardTitle className="text-base flex items-center gap-2">
                              <FileText className="h-4 w-4" />
                              {file.name}
                            </CardTitle>
                            <CardDescription>
                              {fileAnalysis.row_count} rows, {fileAnalysis.column_count} columns
                            </CardDescription>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4">
                              <div>
                                <h4 className="text-sm font-medium mb-3">Column Configuration</h4>
                                <div className="rounded-md border">
                                  <Table>
                                    <TableHeader>
                                      <TableRow>
                                        <TableHead>Column Name</TableHead>
                                        <TableHead>Current Type</TableHead>
                                        <TableHead>New Type</TableHead>
                                        <TableHead>Drop Column</TableHead>
                                        <TableHead>Sample Values</TableHead>
                                      </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                      {fileAnalysis.columns_info.map((column) => {
                                        const isDropped = columnEdits[file.name]?.[column.name]?.drop;
                                        return (
                                          <TableRow 
                                            key={column.name}
                                            className={`group ${isDropped ? "bg-red-100 hover:bg-red-200 dark:bg-red-900/30 dark:hover:bg-red-900/50" : "hover:bg-muted"}`}
                                          >
                                            <TableCell className={`font-medium ${isDropped ? "text-red-600 dark:text-red-400 line-through" : ""}`}>
                                              {column.name}
                                            </TableCell>
                                            <TableCell className={isDropped ? "text-red-600 dark:text-red-400" : ""}>
                                              {formatDataType(column.current_type)}
                                            </TableCell>
                                            <TableCell>
                                              <Select
                                                value={columnEdits[file.name]?.[column.name]?.newType || column.suggested_type}
                                                onValueChange={(value) => updateColumnEdit(file.name, column.name, 'newType', value)}
                                                disabled={isDropped}
                                              >
                                                <SelectTrigger className={`w-full ${
                                                  !isDropped && 
                                                  columnEdits[file.name]?.[column.name]?.newType !== column.current_type && 
                                                  columnEdits[file.name]?.[column.name]?.newType !== column.suggested_type 
                                                    ? 'border-amber-300 bg-amber-50 hover:bg-amber-100 dark:bg-amber-900/30 dark:border-amber-700' : ''
                                                }`}>
                                                  <SelectValue />
                                                </SelectTrigger>
                                                <SelectContent>
                                                  <SelectItem value="string">Text (String)</SelectItem>
                                                  <SelectItem value="int">Integer</SelectItem>
                                                  <SelectItem value="float">Decimal (Float)</SelectItem>
                                                  <SelectItem value="datetime">Date & Time</SelectItem>
                                                </SelectContent>
                                              </Select>
                                            </TableCell>
                                            <TableCell>
                                              <div className="flex items-center space-x-2">
                                                <Checkbox 
                                                  checked={!!isDropped}
                                                  onCheckedChange={(checked) => updateColumnEdit(file.name, column.name, 'drop', checked)}
                                                  id={`drop-${file.name}-${column.name}`}
                                                  className="data-[state=checked]:bg-red-600 data-[state=checked]:border-red-600 focus:ring-red-200"
                                                />
                                                <label 
                                                  htmlFor={`drop-${file.name}-${column.name}`}
                                                  className={`text-sm cursor-pointer select-none ${isDropped ? "text-red-600 dark:text-red-400 font-medium" : ""}`}
                                                >
                                                  Drop
                                                </label>
                                              </div>
                                            </TableCell>
                                            <TableCell>
                                              <div className="text-xs text-muted-foreground max-w-32 truncate">
                                                {column.sample_values.slice(0, 3).join(', ')}
                                              </div>
                                            </TableCell>
                                          </TableRow>
                                        );
                                      })}
                                    </TableBody>
                                  </Table>
                                </div>
                              </div>

                              {/* Live Preview */}
                              <div className="mt-6">
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="text-sm font-medium">Live Preview</h4>
                                  {isLoadingPreview && (
                                    <div className="flex items-center text-sm text-amber-600">
                                      <Loader2 className="animate-spin mr-2 h-4 w-4 text-amber-600" />
                                      Updating preview...
                                    </div>
                                  )}
                                </div>
                                <p className="text-xs text-muted-foreground mb-3">
                                  Preview of your data after applying the configured transformations (first 5 rows).
                                </p>
                                
                                <div className={`border rounded-md overflow-x-auto transition-opacity duration-200 ${isLoadingPreview ? 'opacity-50' : ''}`}>
                                  {filePreview ? (
                                    <Table>
                                      <TableHeader>
                                        <TableRow>
                                          {filePreview.columns.transformed.map((col, idx) => (
                                            <TableHead key={idx}>{col}</TableHead>
                                          ))}
                                        </TableRow>
                                      </TableHeader>
                                      <TableBody>
                                        {filePreview.transformed.map((row, rowIdx) => (
                                          <TableRow key={rowIdx}>
                                            {filePreview.columns.transformed.map((col, colIdx) => (
                                              <TableCell key={colIdx}>
                                                {row[col] !== null && row[col] !== undefined ? String(row[col]) : "null"}
                                              </TableCell>
                                            ))}
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                  ) : (
                                    <div className="p-6 text-center text-muted-foreground">
                                      {isLoadingPreview ? 
                                        "Loading preview data..." : 
                                        "Make changes to see a live preview."}
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* Status indicators */}
                              <div className="flex gap-4 text-sm">
                                {hasChangedTypes(file.name) && (
                                  <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
                                    Type changes applied
                                  </Badge>
                                )}
                                {hasDroppedColumns(file.name) && (
                                  <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                                    Columns will be dropped
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      );
                    })}
                  </div>
                )}
              </CardContent>
              <CardFooter className="justify-between">
                <Button variant="outline" onClick={() => setCurrentStep(2)}>
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Auto Processing
                </Button>
                <Button 
                  onClick={handleStartUpload} 
                  disabled={isAnalyzingFiles}
                  className="gap-2"
                >
                  <Upload className="h-4 w-4" />
                  <span>Apply Cleaning & Upload</span>
                </Button>
              </CardFooter>
            </Card>
          )}
          
          {/* Step 5: Upload & Finalize */}
          {currentStep === 4 && (
            <Card className="shadow-sm">
              <CardHeader>
                <CardTitle>Finalizing Your Data</CardTitle>
                <CardDescription>
                  Please wait while we apply custom transformations and upload your files
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mt-2 mb-8">
                  <p className="text-sm font-medium mb-2">Overall Progress</p>
                  <Progress value={uploadProgress} className="h-2 w-full" />
                  <p className="text-xs text-muted-foreground mt-2">
                    {uploadProgress < 100 
                      ? `Applying transformations and uploading ${files.length} files...` 
                      : `Completed processing ${files.length} files`}
                  </p>
                </div>
                
                {error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error During Upload</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
              <CardFooter className="justify-between">
                <Button variant="outline" disabled={true}>
                  Back
                </Button>
                <Button 
                  onClick={() => setCurrentStep(5)} 
                  disabled={!uploadComplete && !error}
                >
                  Continue
                </Button>
              </CardFooter>
            </Card>
          )}
          
          {/* Step 6: Complete */}
          {currentStep === 5 && (
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
                    with auto-processing and custom data cleaning applied
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
                    <h3 className="text-xl font-bold mb-4">Data Processing Report</h3>
                    <div className="space-y-6">
                      {uploadSummary.filesProcessed.map((file, index) => {
                        if (file.success && filePreprocessingResults[file.name]) {
                          const transformedResults = transformPreprocessingResults(filePreprocessingResults[file.name]);
                          
                          console.log(`Transformed results for ${file.name}:`, transformedResults);
                          
                          if (!transformedResults) {
                            return (
                              <div key={index} className="border rounded-lg p-4 bg-amber-50">
                                <h4 className="font-medium text-lg mb-2">{file.name}</h4>
                                <p className="text-amber-700">No processing details available for this file.</p>
                              </div>
                            );
                          }
                          
                          return (
                            <div key={index} className="border rounded-lg p-6">
                              <h4 className="font-medium text-lg mb-4">{file.name}</h4>
                              
                              {/* Processing Summary Stats */}
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
                              
                              {/* Use AutoPreprocessingReport for detailed display */}
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