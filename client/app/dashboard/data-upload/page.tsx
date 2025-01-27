'use client'
import { useState, useCallback, useEffect } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { FilePlus2 } from "lucide-react";
import UploadedDataTable from "@/components/UploadDataTable";
import { Button } from "@/components/ui/button";

export default function Preprocessing() {
    const [files, setFiles] = useState<File[]>([])
    const [error, setError] = useState<string | null>(null)
    const [isUploading, setIsUploading] = useState(false)

    const uploadData = async () => {
        setIsUploading(true)
        setError(null)
        
        try {
            const formData = new FormData()
            files.forEach((file) => {
                formData.append('files', file)
            })

            const response = await fetch('http://127.0.0.1:8000/upload', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`)
            }

            const data = await response.json()
            // Handle successful upload
            setFiles([]) // Clear files after successful upload
            return data
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed')
            console.error('Upload error:', err)
        } finally {
            setIsUploading(false)
        }
    }

    const onDrop = useCallback((acceptedFiles: File[]) => {
        setFiles((prevFiles) => [...prevFiles, ...acceptedFiles])
    }, [])

    const onDropRejected = useCallback((fileRejections: FileRejection[]) => {
        const rejection = fileRejections[0]
        if (rejection.errors[0].code === "file-invalid-type") {
            setError("Only CSV and XLSX files are allowed.")
        } else {
            setError("There was an error uploading your file. Please try again.")
        }
    }, [])

    const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
        onDrop,
        onDropRejected,
        accept: {
            "text/csv": ['.csv'],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
        },
        maxSize: 10485760,
        noClick: true
    })

    useEffect(() => {
        console.log(files)
    }, [files])

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
            <div className="bg-gray-100 mt-10 h-24 rounded-lg border border-2 border-dashed border-zinc-300 flex items-center justify-center flex-col gap-2" onClick={open}>
                <FilePlus2 color="gray" />
                <p className="text-gray-600">Click Here Or Drag And Drop Your Files Anywhere</p>
            </div>

            {files.length > 0 || <p className="w-full flex justify-center mt-7 text-gray-400">No files uploaded</p>}
            {files.length > 0 && <UploadedDataTable files={files} setFiles={setFiles} />}
            {error && <p className="text-red-500 text-center mt-4">{error}</p>}
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