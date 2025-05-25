import Papa from 'papaparse';
import * as XLSX from 'xlsx';

interface EnhancedColumnStatistics {
  min?: number;
  max?: number;
  mean?: number;
  count: number;
  type: 'numeric' | 'categorical' | 'datetime' | 'boolean';
  uniqueCount: number;
  nullCount: number;
  nullPercentage: number;
  sampleValues: any[];
}

interface EnhancedFileMetadata {
  columns: string[];
  rowCount: number;
  preview: Record<string, any>[];
  statistics: Record<string, EnhancedColumnStatistics>;
  fileInfo: {
    name: string;
    size: number;
    type: string;
    lastModified: number;
  };
  quality: {
    completeness: number;
    consistency: number;
    warnings: string[];
  };
}

const extractEnhancedMetadata = async (file: File): Promise<EnhancedFileMetadata> => {
  const fileType = file.name.split('.').pop()?.toLowerCase();
  const metadata: EnhancedFileMetadata = {
    columns: [],
    rowCount: 0,
    preview: [],
    statistics: {},
    fileInfo: {
      name: file.name,
      size: file.size,
      type: file.type,
      lastModified: file.lastModified
    },
    quality: {
      completeness: 0,
      consistency: 0,
      warnings: []
    }
  };

  try {
    let allData: Record<string, any>[] = [];
    
    if (fileType === 'csv') {
      const text = await file.text();
      
      // Parse CSV completely for accurate statistics
      const fullResults = Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: false, // Keep as strings initially for better type detection
        trimHeaders: true
      });

      if (fullResults.errors.length > 0) {
        metadata.quality.warnings.push(`CSV parsing warnings: ${fullResults.errors.length} issues found`);
      }

      allData = fullResults.data as Record<string, any>[];
      metadata.columns = fullResults.meta.fields || [];
      
    } else if (fileType === 'xlsx') {
      const arrayBuffer = await file.arrayBuffer();
      const workbook = XLSX.read(arrayBuffer, {
        cellDates: true,
        cellNF: false,
        cellText: false
      });
      
      const sheetName = workbook.SheetNames[0];
      if (!sheetName) {
        throw new Error('No worksheets found in Excel file');
      }
      
      const worksheet = workbook.Sheets[sheetName];
      
      // Convert to JSON with proper header detection
      const jsonData = XLSX.utils.sheet_to_json<Record<string, any>>(worksheet, {
        defval: '',
        blankrows: false
      });

      allData = jsonData;
      
      if (allData.length > 0) {
        metadata.columns = Object.keys(allData[0]);
      }
    } else {
      throw new Error(`Unsupported file type: ${fileType}`);
    }

    // Set basic counts
    metadata.rowCount = allData.length;
    metadata.preview = allData.slice(0, 50); // Show more rows in preview

    // Calculate enhanced statistics
    metadata.statistics = calculateEnhancedStatistics(allData, metadata.columns);
    metadata.quality = calculateDataQuality(allData, metadata.columns, metadata.statistics);

    return metadata;
  } catch (error) {
    console.error('Error parsing file:', error);
    
    // Return error information in the metadata
    metadata.quality.warnings.push(`Failed to parse file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return metadata;
  }
};

const detectColumnType = (values: any[]): 'numeric' | 'categorical' | 'datetime' | 'boolean' => {
  const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
  if (nonNullValues.length === 0) return 'categorical';

  // Check for boolean
  const booleanValues = nonNullValues.filter(v => 
    typeof v === 'boolean' || 
    (typeof v === 'string' && ['true', 'false', 'yes', 'no', '1', '0'].includes(v.toLowerCase()))
  );
  if (booleanValues.length / nonNullValues.length > 0.8) return 'boolean';

  // Check for datetime
  const dateValues = nonNullValues.filter(v => {
    if (v instanceof Date) return true;
    if (typeof v === 'string') {
      const dateTest = new Date(v);
      return !isNaN(dateTest.getTime()) && v.match(/\d{4}|\d{2}[-\/]\d{2}|\d{2}[-\/]\d{4}/);
    }
    return false;
  });
  if (dateValues.length / nonNullValues.length > 0.7) return 'datetime';

  // Check for numeric
  const numericValues = nonNullValues.filter(v => {
    if (typeof v === 'number') return !isNaN(v);
    if (typeof v === 'string') {
      const cleaned = v.replace(/[,$%]/g, '');
      return !isNaN(parseFloat(cleaned)) && isFinite(parseFloat(cleaned));
    }
    return false;
  });
  if (numericValues.length / nonNullValues.length > 0.8) return 'numeric';

  return 'categorical';
};

const calculateEnhancedStatistics = (
  data: Record<string, any>[],
  columns: string[]
): Record<string, EnhancedColumnStatistics> => {
  const stats: Record<string, EnhancedColumnStatistics> = {};

  columns.forEach((column) => {
    const values = data.map((row) => row[column]);
    const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
    const nullCount = values.length - nonNullValues.length;
    
    const type = detectColumnType(values);
    
    // Get sample values (up to 10 unique values)
    const uniqueValues = Array.from(new Set(nonNullValues));
    const sampleValues = uniqueValues.slice(0, 10);

    const baseStats: EnhancedColumnStatistics = {
      count: nonNullValues.length,
      type,
      uniqueCount: uniqueValues.length,
      nullCount,
      nullPercentage: values.length > 0 ? (nullCount / values.length) * 100 : 0,
      sampleValues
    };

    if (type === 'numeric') {
      const numericValues = nonNullValues.map(v => {
        if (typeof v === 'number') return v;
        if (typeof v === 'string') {
          const cleaned = v.replace(/[,$%]/g, '');
          return parseFloat(cleaned);
        }
        return NaN;
      }).filter(v => !isNaN(v));

      if (numericValues.length > 0) {
        stats[column] = {
          ...baseStats,
          min: Math.min(...numericValues),
          max: Math.max(...numericValues),
          mean: numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length
        };
      } else {
        stats[column] = baseStats;
      }
    } else {
      stats[column] = baseStats;
    }
  });

  return stats;
};

const calculateDataQuality = (
  data: Record<string, any>[],
  columns: string[],
  statistics: Record<string, EnhancedColumnStatistics>
): { completeness: number; consistency: number; warnings: string[] } => {
  const warnings: string[] = [];
  
  // Calculate completeness (percentage of non-null values)
  const totalCells = data.length * columns.length;
  const nullCells = Object.values(statistics).reduce((sum, stat) => sum + stat.nullCount, 0);
  const completeness = totalCells > 0 ? ((totalCells - nullCells) / totalCells) * 100 : 0;

  // Calculate consistency (how well data types are detected)
  const consistentColumns = Object.values(statistics).filter(stat => {
    if (stat.type === 'numeric') return stat.count > 0;
    if (stat.type === 'datetime') return stat.count > 0;
    return true;
  }).length;
  const consistency = columns.length > 0 ? (consistentColumns / columns.length) * 100 : 0;

  // Generate warnings
  if (completeness < 80) {
    warnings.push(`Low data completeness: ${completeness.toFixed(1)}% of cells have values`);
  }

  const highNullColumns = Object.entries(statistics)
    .filter(([_, stat]) => stat.nullPercentage > 50)
    .map(([col, _]) => col);
  
  if (highNullColumns.length > 0) {
    warnings.push(`Columns with >50% missing values: ${highNullColumns.join(', ')}`);
  }

  const lowUniqueColumns = Object.entries(statistics)
    .filter(([_, stat]) => stat.type === 'categorical' && stat.uniqueCount === 1 && stat.count > 1)
    .map(([col, _]) => col);
    
  if (lowUniqueColumns.length > 0) {
    warnings.push(`Columns with only one unique value: ${lowUniqueColumns.join(', ')}`);
  }

  return {
    completeness: Math.round(completeness * 10) / 10,
    consistency: Math.round(consistency * 10) / 10,
    warnings
  };
};

export default extractEnhancedMetadata;
export type { EnhancedFileMetadata, EnhancedColumnStatistics };