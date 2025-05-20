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
  console.log(`Extracting metadata for file: ${file.name}, size: ${file.size}`);
  const fileType = file.name.split('.').pop()?.toLowerCase();
  const metadata: FileMetadata = {
    columns: [],
    rowCount: 0,
    preview: [],
    statistics: {},
  };

  try {
    if (fileType === 'csv') {
      console.log("Processing as CSV");
      // Read CSV as text
      const text = await file.text();
      
      // Parse CSV with preview
      const previewResults = Papa.parse(text, {
        header: true,
        preview: 100,
        skipEmptyLines: true,
        dynamicTyping: true, // Automatically convert to appropriate types
      });

      metadata.columns = previewResults.meta.fields || [];
      metadata.preview = previewResults.data as Record<string, any>[];

      // Count total rows (excluding header)
      let rowCount = 0;
      
      try {
        Papa.parse(text, {
          header: false,
          skipEmptyLines: true,
          dynamicTyping: true,
          step: (results: ParseStepResult<string[]>) => {
            if (results.data.length > 0) rowCount++;
          },
          complete: () => {
            metadata.rowCount = rowCount > 0 ? rowCount - 1 : 0; // Subtract header row if there are rows
          }
        });
      } catch (err) {
        console.error("Error counting rows:", err);
        // Fallback to preview count
        metadata.rowCount = metadata.preview.length;
      }
    } 
    else if (fileType === 'xlsx') {
      console.log("Processing as XLSX");
      // Read Excel as ArrayBuffer
      const arrayBuffer = await file.arrayBuffer();
      const workbook = XLSX.read(arrayBuffer, { type: 'array' });
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];

      // Convert to JSON (auto-detects headers)
      const jsonData = XLSX.utils.sheet_to_json<Record<string, any>>(worksheet, {
        header: 1,
        defval: '',
      });

      if (jsonData.length > 0 && Array.isArray(jsonData[0])) {
        // Extract headers from first row
        metadata.columns = (jsonData[0] as any[]).map(col => String(col));
        
        // Convert data to format with column names as keys
        const records = jsonData.slice(1).map(row => {
          const record: Record<string, any> = {};
          if (Array.isArray(row)) {
            metadata.columns.forEach((col, i) => {
              record[col] = i < row.length ? row[i] : '';
            });
          }
          return record;
        });
        
        metadata.preview = records.slice(0, 100);
        metadata.rowCount = jsonData.length - 1; // Subtract header row
      }
    } else {
      console.warn(`Unsupported file type: ${fileType}`);
      throw new Error(`Unsupported file type: ${fileType}`);
    }

    // Calculate statistics
    metadata.statistics = calculateStatistics(metadata.preview, metadata.columns);
    console.log(`Metadata extraction complete for ${file.name}. Found ${metadata.columns.length} columns and ${metadata.rowCount} rows.`);
    return metadata;
  } catch (error) {
    console.error('Error parsing file:', error);
    // Return basic metadata rather than throwing an error
    return {
      columns: [],
      rowCount: 0,
      preview: [],
      statistics: {}
    };
  }
};

const calculateStatistics = (
  data: Record<string, any>[],
  columns: string[]
): Record<string, ColumnStatistics> => {
  console.log(`Calculating statistics for ${columns.length} columns`);
  const stats: Record<string, ColumnStatistics> = {};

  try {
    columns.forEach((column) => {
      const values = data.map((row) => row[column]);
      const numericValues = values
        .filter(val => val !== undefined && val !== null && val !== '')
        .map((val) => {
          const parsed = typeof val === 'string' ? parseFloat(val) : val;
          return typeof parsed === 'number' && !isNaN(parsed) ? parsed : null;
        })
        .filter((val): val is number => val !== null);

      if (numericValues.length > 0) {
        stats[column] = {
          type: 'numeric',
          min: Math.min(...numericValues),
          max: Math.max(...numericValues),
          mean: numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length,
          count: numericValues.length,
        };
      } else {
        // For non-numeric columns
        const nonEmptyValues = values.filter((val) => val !== null && val !== undefined && val !== '');
        const uniqueValues = new Set(nonEmptyValues);
        stats[column] = {
          type: 'categorical',
          uniqueCount: uniqueValues.size,
          count: nonEmptyValues.length
        };
      }
    });
  } catch (err) {
    console.error("Error calculating statistics:", err);
  }

  return stats;
};

export default extractMetadata;