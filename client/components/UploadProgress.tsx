import React from 'react';
import { Progress } from "@/components/ui/progress";
import { LoaderCircle } from "lucide-react";

interface UploadProgressProps {
  isUploading: boolean;
  uploadProgress: number;
  message?: string;
}

const UploadProgress: React.FC<UploadProgressProps> = ({ isUploading, uploadProgress, message }) => {
  if (!isUploading) return null;
  
  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full">
        <div className="flex flex-col items-center gap-4">
          <LoaderCircle className="h-8 w-8 animate-spin text-primary" />
          <h3 className="text-lg font-medium">Processing Your Data</h3>
          <Progress value={uploadProgress} className="w-full h-2" />
          <p className="text-sm text-muted-foreground text-center">
            {message || getProgressMessage(uploadProgress)}
          </p>
        </div>
      </div>
    </div>
  );
};

// Helper function to generate appropriate messages based on progress
const getProgressMessage = (progress: number): string => {
  if (progress < 10) {
    return "Initializing upload...";
  } else if (progress < 40) {
    return "Processing your data on the server...";
  } else if (progress < 70) {
    return "Analyzing and preparing your data...";
  } else if (progress < 90) {
    return "Finalizing and saving your data...";
  } else {
    return "Almost done!";
  }
};

export default UploadProgress;
