// Modified KaggleUpload.tsx file
import { useState } from "react";
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle, Download, ExternalLink } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface KaggleUploadProps {
  onFileImported: (file: File) => void;
}

export default function KaggleUpload({ onFileImported }: KaggleUploadProps) {
  const [kaggleUrl, setKaggleUrl] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Updated validation and parsing for both dataset and competition URLs
  const parseKaggleUrl = (url: string) => {
    // Dataset URL pattern (e.g., https://www.kaggle.com/datasets/username/datasetname)
    const datasetPattern = /^https:\/\/www\.kaggle\.com\/datasets\/([\w-]+)\/([\w-]+)$/i;
    
    // Competition URL pattern (e.g., https://www.kaggle.com/competitions/titanic)
    const competitionPattern = /^https:\/\/www\.kaggle\.com\/competitions\/([\w-]+)$/i;
    
    // Check for dataset URL
    const datasetMatch = url.match(datasetPattern);
    if (datasetMatch) {
      return {
        type: 'dataset',
        owner: datasetMatch[1],
        name: datasetMatch[2],
        path: `${datasetMatch[1]}/${datasetMatch[2]}`
      };
    }
    
    // Check for competition URL
    const competitionMatch = url.match(competitionPattern);
    if (competitionMatch) {
      return {
        type: 'competition',
        name: competitionMatch[1],
        path: competitionMatch[1]
      };
    }
    
    // Not a valid URL
    return null;
  };

  const handleImport = async () => {
    const parsedUrl = parseKaggleUrl(kaggleUrl);
    
    if (!parsedUrl) {
      setError("Please enter a valid Kaggle URL. Examples:\n- https://www.kaggle.com/datasets/username/datasetname\n- https://www.kaggle.com/competitions/competitionname");
      return;
    }

    setIsImporting(true);
    setError(null);
    setImportProgress(10);

    try {
      setImportProgress(20);
      
      // Send the request to our Next.js API endpoint with improved data
      const response = await fetch('/api/kaggle-import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: parsedUrl.type,
          path: parsedUrl.path,
          // Add name information to help with better naming on server
          name: parsedUrl.name
        }),
      });

      setImportProgress(50);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `Failed to import from Kaggle: ${response.statusText}`);
      }

      // Get the response data
      const data = await response.json();
      setImportProgress(70);
      
      // Fetch the file from the provided URL
      const fileResponse = await fetch(data.url);
      if (!fileResponse.ok) {
        throw new Error('Failed to download the imported file');
      }
      
      setImportProgress(90);
      
      // Create a File object from the response
      const blob = await fileResponse.blob();
      const file = new File([blob], data.filename, { 
        type: data.contentType 
      });
      
      setImportProgress(100);
      
      // Let the parent component know we have a new file
      onFileImported(file);
      
      // Reset form
      setKaggleUrl("");
      
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
              <p className="text-sm">Importing from Kaggle...</p>
              <Progress value={importProgress} className="h-2" />
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" size="sm" asChild>
          <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
            <ExternalLink className="h-4 w-4" />
            Browse Kaggle Datasets
          </a>
        </Button>
        <Button onClick={handleImport} disabled={!kaggleUrl || isImporting}>
          {isImporting ? "Importing..." : "Import"}
        </Button>
      </CardFooter>
    </Card>
  );
}