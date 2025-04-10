'use client'
import { useState, useCallback } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { FilePlus2 } from "lucide-react";
import UploadedDataTable from "@/components/UploadDataTable";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

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
            // For the first error at line 52
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
            
            // Step 3: Upload preprocessed files to database
            setUploadProgress(70);
            for (const file of files) {
                // Create a new FormData for each file
                const formData = new FormData();
                formData.append('file', file);
                formData.append('preprocessed', 'true');
                
                // Use relative URL for API endpoint
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.details || result.error || `Upload failed: ${response.statusText}`);
                }
            }
            
            setUploadProgress(100);
            // All files uploaded successfully
            setFiles([]);
            
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

    return (
        <section {...getRootProps()} className="h-screen w-[100%] px-6 md:px-10 py-10" >
            {isDragActive &&
                <div className="absolute h-full w-full inset-0 bg-gray-100 flex items-center justify-center text-4xl opacity-65">
                    Drop Your Files
                </div>}
            <div className="text-4xl font-bold">
                Upload Data
            </div>
            <input {...getInputProps()} />
            <div className="cursor-pointer bg-gray-100 mt-10 h-24 rounded-lg border border-2 border-dashed border-zinc-300 flex items-center justify-center flex-col gap-2" onClick={open}>
                <FilePlus2 color="gray" />
                <p className="text-gray-600">Click Here Or Drag And Drop Your Files Anywhere</p>
            </div>

            {files.length === 0 && <p className="w-full flex justify-center mt-7 text-gray-400">No files uploaded</p>}
            {files.length > 0 && <UploadedDataTable files={files} setFiles={setFiles} />}
            
            {isUploading && (
                <div className="mt-6">
                    <p className="text-sm mb-2">Upload Progress</p>
                    <Progress value={uploadProgress} className="h-2 w-full" />
                    
                    {Object.keys(processingStatus).length > 0 && (
                        <div className="mt-4">
                            <p className="text-sm mb-2">Preprocessing Status:</p>
                            {Object.entries(processingStatus).map(([filename, status]) => (
                                <div key={filename} className="mb-3">
                                    <div className="flex justify-between text-xs">
                                        <span>{filename}</span>
                                        <span>{status.message}</span>
                                    </div>
                                    <Progress 
                                        value={status.progress < 0 ? 100 : status.progress} 
                                        className={`h-2 w-full ${status.progress < 0 ? 'bg-red-500' : ''}`} 
                                    />
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
            
            {error && <p className="text-red-500 text-center mt-4">{error}</p>}
            
            {files.length > 0 && (
                <div className="w-full flex justify-center mt-20">
                    <Button 
                        onClick={uploadData} 
                        disabled={isUploading}
                    >
                        {isUploading ? 'Processing...' : 'Upload Data'}
                    </Button>
                </div>
            )}
        </section>
    );
}