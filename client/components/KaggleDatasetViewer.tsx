// Updated KaggleUpload.tsx - Simplified with Dataset Viewer
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
import { AlertCircle, Download, ExternalLink, Search, Loader2 } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import KaggleDatasetViewer from "./KaggleDatasetViewer";

interface KaggleFile {
  name: string;
  size: number;
  description?: string;
  columns?: number;
}

interface KaggleDatasetInfo {
  title: string;
  description: string;
  files: KaggleFile[];
  totalSize: number;
}

interface KaggleUploadProps {
  onFileImported: (file: File) => void;
}

export default function KaggleUpload({ onFileImported }: KaggleUploadProps) {
  const [kaggleUrl, setKaggleUrl] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<KaggleDatasetInfo | null>(null);
  const [selectedFile, setSelectedFile] = useState<string>("");

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

  const analyzeDataset = async () => {
    const parsedUrl = parseKaggleUrl(kaggleUrl);
    
    if (!parsedUrl) {
      setError("Please enter a valid Kaggle URL. Examples:\n• https://www.kaggle.com/datasets/username/datasetname\n• https://www.kaggle.com/competitions/competitionname");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setDatasetInfo(null);
    setSelectedFile("");

    try {
      const response = await fetch('/api/kaggle-analyze', {
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

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to analyze Kaggle dataset');
      }

      const data = await response.json();
      setDatasetInfo(data);
      
      // Auto-select first importable file if only one exists
      const importableFiles = data.files.filter((file: KaggleFile) => 
        file.name.toLowerCase().endsWith('.csv') || 
        file.name.toLowerCase().endsWith('.xlsx') ||
        file.name.toLowerCase().endsWith('.xls')
      );
      
      if (importableFiles.length === 1) {
        setSelectedFile(importableFiles[0].name);
      }
      
    } catch (err) {
      console.error('Error analyzing dataset:', err);
      setError(err instanceof Error ? err.message : 'Failed to analyze Kaggle dataset. Please check the URL and try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImport = async () => {
    if (!selectedFile || !datasetInfo) {
      setError("Please select a file to import");
      return;
    }

    const parsedUrl = parseKaggleUrl(kaggleUrl);
    if (!parsedUrl) return;

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
          name: parsedUrl.name,
          selectedFile: selectedFile
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
      setDatasetInfo(null);
      setSelectedFile("");
      setImportProgress(0);
      
    } catch (err) {
      console.error('Error importing from Kaggle:', err);
      setError(err instanceof Error ? err.message : 'Failed to import dataset from Kaggle');
    } finally {
      setIsImporting(false);
    }
  };

  const resetForm = () => {
    setKaggleUrl("");
    setDatasetInfo(null);
    setSelectedFile("");
    setError(null);
    setImportProgress(0);
  };

  return (
    <div className="space-y-6">
      {/* URL Input Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Import from Kaggle
          </CardTitle>
          <CardDescription>
            Import specific files from Kaggle datasets and competitions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* URL Input */}
            <div className="flex gap-2">
              <Input
                type="url"
                placeholder="https://www.kaggle.com/datasets/username/datasetname"
                value={kaggleUrl}
                onChange={(e) => setKaggleUrl(e.target.value)}
                disabled={isAnalyzing || isImporting}
                className="flex-1"
              />
              <Button 
                onClick={analyzeDataset} 
                disabled={!kaggleUrl || isAnalyzing || isImporting}
                variant={datasetInfo ? "outline" : "default"}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    {datasetInfo ? "Re-analyze" : "Analyze"}
                  </>
                )}
              </Button>
              {datasetInfo && (
                <Button variant="ghost" onClick={resetForm}>
                  Clear
                </Button>
              )}
            </div>
            
            <p className="text-xs text-muted-foreground">
              Paste a Kaggle dataset or competition URL above, then click "Analyze" to see available files
            </p>

            {/* Import Progress */}
            {isImporting && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="text-sm">Importing {selectedFile} from Kaggle...</span>
                </div>
                <Progress value={importProgress} className="h-2" />
              </div>
            )}

            {/* Error Display */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription className="whitespace-pre-line">{error}</AlertDescription>
              </Alert>
            )}
          </div>
        </CardContent>
        <CardFooter>
          <Button variant="outline" size="sm" asChild className="ml-auto">
            <a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1">
              <ExternalLink className="h-4 w-4" />
              Browse Kaggle Datasets
            </a>
          </Button>
        </CardFooter>
      </Card>

      {/* Dataset Analysis Results */}
      {datasetInfo && (
        <KaggleDatasetViewer
          datasetInfo={datasetInfo}
          selectedFile={selectedFile}
          onFileSelect={setSelectedFile}
          onImport={handleImport}
          isImporting={isImporting}
        />
      )}
    </div>
  );
}