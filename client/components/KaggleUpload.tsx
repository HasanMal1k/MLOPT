'use client'

import { useState } from "react";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, Download, ExternalLink, Loader2, CheckCircle2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface KaggleUploadProps {
  onFileImported: (file: File) => void;
}

// Helper function to convert base64 to File object
function base64ToFile(base64: string, filename: string, contentType: string): File {
  // Remove data URL prefix if present
  const base64Data = base64.includes(',') ? base64.split(',')[1] : base64;
  
  // Convert base64 to binary
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  
  // Create File object
  return new File([bytes], filename, { type: contentType });
}

export default function KaggleUpload({ onFileImported }: KaggleUploadProps) {
  const [kaggleUrl, setKaggleUrl] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const parseKaggleUrl = (url: string) => {
    const datasetPattern = /^https:\/\/www\.kaggle\.com\/datasets\/([\w-]+)\/([\w-]+)$/i;
    const competitionPattern = /^https:\/\/www\.kaggle\.com\/competitions\/([\w-]+)$/i;
    
    const datasetMatch = url.match(datasetPattern);
    if (datasetMatch) {
      return {
        type: 'dataset',
        owner: datasetMatch[1],
        name: datasetMatch[2],
        path: `${datasetMatch[1]}/${datasetMatch[2]}`
      };
    }
    
    const competitionMatch = url.match(competitionPattern);
    if (competitionMatch) {
      return {
        type: 'competition',
        name: competitionMatch[1],
        path: competitionMatch[1]
      };
    }
    
    return null;
  };

  const handleImport = async () => {
    const parsedUrl = parseKaggleUrl(kaggleUrl);
    
    if (!parsedUrl) {
      setError("Please enter a valid Kaggle URL. Examples:\n• https://www.kaggle.com/datasets/username/datasetname\n• https://www.kaggle.com/competitions/competitionname");
      return;
    }

    setIsImporting(true);
    setError(null);
    setSuccess(null);
    setImportProgress(10);

    try {
      console.log('Starting Kaggle import for:', parsedUrl);
      setImportProgress(20);
      
      const response = await fetch('/api/kaggle-import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: parsedUrl.type,
          path: parsedUrl.path,
          name: parsedUrl.name
        }),
      });

      setImportProgress(50);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `Failed to import from Kaggle: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Kaggle import response:', data);
      setImportProgress(70);
      
      // Convert the base64 content back to a File object
      if (data.content && data.filename) {
        try {
          const file = base64ToFile(data.content, data.filename, data.contentType || 'text/csv');
          console.log('Created File object:', {
            name: file.name,
            size: file.size,
            type: file.type
          });
          
          setImportProgress(90);
          
          // Call the parent callback with the File object
          onFileImported(file);
          
          setImportProgress(100);
          setSuccess(`Successfully imported "${data.filename}" (${(data.size / 1024 / 1024).toFixed(2)} MB)`);
          
          // Reset form after success
          setTimeout(() => {
            setKaggleUrl("");
            setImportProgress(0);
            setSuccess(null);
          }, 3000);
          
        } catch (fileError) {
          console.error('Error creating File object:', fileError);
          throw new Error('Failed to process the downloaded file content');
        }
      } else {
        throw new Error('Invalid response from server - missing file content');
      }
      
    } catch (err) {
      console.error('Error importing from Kaggle:', err);
      setError(err instanceof Error ? err.message : 'Failed to import dataset from Kaggle');
      setImportProgress(0);
    } finally {
      setIsImporting(false);
    }
  };

  const handleReset = () => {
    setKaggleUrl("");
    setError(null);
    setSuccess(null);
    setImportProgress(0);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Download className="h-5 w-5" />
          Import from Kaggle
        </CardTitle>
        <CardDescription>
          Import datasets directly from Kaggle without downloading them first
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid w-full gap-2">
            <Input
              type="url"
              placeholder="https://www.kaggle.com/datasets/username/datasetname"
              value={kaggleUrl}
              onChange={(e) => setKaggleUrl(e.target.value)}
              disabled={isImporting}
            />
            <p className="text-xs text-muted-foreground">
              Enter a Kaggle dataset URL (e.g., https://www.kaggle.com/datasets/username/datasetname) or 
              competition URL (e.g., https://www.kaggle.com/competitions/titanic)
            </p>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Import Error</AlertTitle>
              <AlertDescription className="whitespace-pre-line">{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="border-green-200 bg-green-50">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertTitle className="text-green-800">Import Successful</AlertTitle>
              <AlertDescription className="text-green-700">{success}</AlertDescription>
            </Alert>
          )}

          {isImporting && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">
                  {importProgress < 30 ? 'Contacting Kaggle API...' :
                   importProgress < 60 ? 'Downloading dataset...' :
                   importProgress < 90 ? 'Processing file...' :
                   'Finalizing import...'}
                </span>
              </div>
              <Progress value={importProgress} className="h-2" />
              <p className="text-xs text-muted-foreground">
                This may take a moment depending on the dataset size
              </p>
            </div>
          )}

          <div className="flex justify-between">
            <Button variant="outline" size="sm" asChild>
              <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
                <ExternalLink className="h-4 w-4" />
                Browse Datasets
              </a>
            </Button>
            <div className="flex gap-2">
              {(success || error) && (
                <Button variant="outline" onClick={handleReset} size="sm">
                  Reset
                </Button>
              )}
              <Button 
                onClick={handleImport} 
                disabled={!kaggleUrl || isImporting}
              >
                {isImporting ? (
                  <>
                    <Loader2 className="animate-spin mr-2 h-4 w-4" />
                    Importing...
                  </>
                ) : (
                  'Import Dataset'
                )}
              </Button>
            </div>
          </div>
          
          {/* Debug information */}
         
        </div>
      </CardContent>
    </Card>
  );
}