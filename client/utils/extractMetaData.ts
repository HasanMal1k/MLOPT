import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import type { ParseStepResult } from 'papaparse';


interface ColumnStatistics {
  min?: number;
  max?: number;
  mean?: number;
  count?: number;
  type?: 'numeric' | 'categorical';
  uniqueCount?: number;
}

interface FileMetadata {
  columns: string[];
  rowCount: number;
  preview: Record<string, any>[];
  statistics: Record<string, ColumnStatistics>;
}

const extractMetadata = async (file: File): Promise<FileMetadata> => {
  const fileType = file.name.split('.').pop()?.toLowerCase();
  const metadata: FileMetadata = {
    columns: [],
    rowCount: 0,
    preview: [],
    statistics: {},
  };

  try {
    if (fileType === 'csv') {
      // Read CSV as text
      const text = await file.text();
      
      // Parse CSV with preview
      const previewResults = Papa.parse(text, {
        header: true,
        preview: 100,
        skipEmptyLines: true,
      });

      metadata.columns = previewResults.meta.fields || [];
      metadata.preview = previewResults.data as Record<string, any>[];

      // Count total rows (excluding header)
      const countResults = Papa.parse(text, {
        header: false,
        skipEmptyLines: true,
        step: (results: ParseStepResult<string[]>) => {
            if (results.data.length > 0) metadata.rowCount++;
        },
        });

      metadata.rowCount--; // Subtract header row
    } 
    else if (fileType === 'xlsx') {
      // Read Excel as ArrayBuffer
      const arrayBuffer = await file.arrayBuffer();
      const workbook = XLSX.read(arrayBuffer);
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];

      // Convert to JSON (auto-detects headers)
      const jsonData = XLSX.utils.sheet_to_json<Record<string, any>>(worksheet, {
        header: 'A',
        defval: '',
      });

      // Get columns from first row
      metadata.columns = jsonData.length > 0 ? Object.keys(jsonData[0]) : [];

      // Set row count (subtract 1 if header exists)
      metadata.rowCount = jsonData.length;

      // Get preview (first 100 rows)
      metadata.preview = jsonData.slice(0, 100);
    }

    // Calculate statistics
    metadata.statistics = calculateStatistics(metadata.preview, metadata.columns);
    return metadata;
  } catch (error) {
    console.error('Error parsing file:', error);
    throw new Error('Failed to parse file');
  }
};

const calculateStatistics = (
  data: Record<string, any>[],
  columns: string[]
): Record<string, ColumnStatistics> => {
  const stats: Record<string, ColumnStatistics> = {};

  columns.forEach((column) => {
    const values = data.map((row) => row[column]);
    const numericValues = values
      .map((val) => (typeof val === 'string' ? parseFloat(val) : val))
      .filter((val) => !isNaN(val) && typeof val === 'number');

    if (numericValues.length > 0) {
      stats[column] = {
        type: 'numeric',
        min: Math.min(...numericValues),
        max: Math.max(...numericValues),
        mean: numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length,
        count: numericValues.length,
      };
    } else {
      const uniqueValues = new Set(values.filter((val) => val !== ''));
      stats[column] = {
        type: 'categorical',
        uniqueCount: uniqueValues.size,
      };
    }
  });

  return stats;
};

export default extractMetadata