// components/KaggleUpload.tsx - Simple version that works with your existing system
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
import { AlertCircle, Download, ExternalLink, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface KaggleUploadProps {
  onFileImported: (file: File) => void;
}

export default function KaggleUpload({ onFileImported }: KaggleUploadProps) {
  const [kaggleUrl, setKaggleUrl] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

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
    setImportProgress(10);

    try {
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
      setImportProgress(70);
      
      // Download the file from our storage
      const fileResponse = await fetch(data.url);
      if (!fileResponse.ok) {
        throw new Error('Failed to download the imported file');
      }
      
      setImportProgress(90);
      
      const blob = await fileResponse.blob();
      const file = new File([blob], data.filename, { 
        type: data.contentType 
      });
      
      setImportProgress(100);
      
      // Call the parent callback
      onFileImported(file);
      
      // Reset form
      setKaggleUrl("");
      setImportProgress(0);
      
    } catch (err) {
      console.error('Error importing from Kaggle:', err);
      setError(err instanceof Error ? err.message : 'Failed to import dataset from Kaggle');
    } finally {
      setIsImporting(false);
    }
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
              <AlertTitle>Error</AlertTitle>
              <AlertDescription className="whitespace-pre-line">{error}</AlertDescription>
            </Alert>
          )}

          {isImporting && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">Importing from Kaggle...</span>
              </div>
              <Progress value={importProgress} className="h-2" />
            </div>
          )}

          <div className="flex justify-between">
            <Button variant="outline" size="sm" asChild>
              <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
                <ExternalLink className="h-4 w-4" />
                Browse Datasets
              </a>
            </Button>
            <Button onClick={handleImport} disabled={!kaggleUrl || isImporting}>
              {isImporting ? "Importing..." : "Import"}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}