'use client'
import { useState, useCallback } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { FilePlus2 } from "lucide-react";
import UploadedDataTable from "@/components/UploadDataTable";
import { Button } from "@/components/ui/button";

export default function DataUpload() {
    const [files, setFiles] = useState<File[]>([])
    const [error, setError] = useState<string | null>(null)
    const [isUploading, setIsUploading] = useState(false)
    const [debug, setDebug] = useState<string | null>(null)  // For debugging issues

    const uploadData = async () => {
        if (files.length === 0) {
            setError("Please select at least one file to upload");
            return;
        }

        setIsUploading(true);
        setError(null);
        setDebug(null);
        
        try {
            for (const file of files) {
                // Create a new FormData for each file
                const formData = new FormData();
                formData.append('file', file);
                
                setDebug(`Uploading file: ${file.name}`);
                
                // Use relative URL for API endpoint
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.details || result.error || `Upload failed: ${response.statusText}`);
                }

                setDebug(`Uploaded ${file.name} successfully!`);
            }
            
            // All files uploaded successfully
            setFiles([]);  // Clear files after successful upload
            setDebug("All files uploaded successfully!");
            
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Upload failed';
            setError(errorMessage);
            console.error('Upload error:', err);
        } finally {
            setIsUploading(false);
        }
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
            {error && <p className="text-red-500 text-center mt-4">{error}</p>}
            {debug && <p className="text-blue-500 text-center mt-2 text-sm">{debug}</p>}
            {files.length > 0 && (
                <div className="w-full flex justify-center mt-20">
                    <Button 
                        onClick={uploadData} 
                        disabled={isUploading}
                    >
                        {isUploading ? 'Uploading...' : 'Upload Data'}
                    </Button>
                </div>
            )}
        </section>
    );
}