// utils/kaggle-error-handler.ts
export interface KaggleErrorResponse {
  error: string;
  message: string;
  suggestions?: string[];
}

export function handleKaggleError(error: any, context: 'analyze' | 'import'): KaggleErrorResponse {
  console.error(`Kaggle ${context} error:`, error);
  
  // Network/Connection errors
  if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
    return {
      error: 'Connection Error',
      message: 'Unable to connect to Kaggle API. Please check your internet connection.',
      suggestions: [
        'Check your internet connection',
        'Try again in a few moments',
        'Verify that Kaggle.com is accessible'
      ]
    };
  }
  
  // HTTP Status errors
  if (error.status) {
    switch (error.status) {
      case 401:
        return {
          error: 'Authentication Error',
          message: 'Invalid Kaggle API credentials. Please contact support.',
          suggestions: [
            'Verify Kaggle API credentials are configured correctly',
            'Check if your Kaggle account has API access enabled'
          ]
        };
        
      case 403:
        return {
          error: 'Access Denied',
          message: 'You don\'t have permission to access this dataset. It may be private or require acceptance of terms.',
          suggestions: [
            'Check if the dataset is public',
            'Visit the Kaggle page and accept any required terms',
            'Ensure you have joined the competition if it\'s a competition dataset'
          ]
        };
        
      case 404:
        return {
          error: 'Dataset Not Found',
          message: 'The specified dataset or file could not be found. Please check the URL.',
          suggestions: [
            'Verify the dataset URL is correct',
            'Check if the dataset still exists on Kaggle',
            'Try browsing to the dataset page manually first'
          ]
        };
        
      case 429:
        return {
          error: 'Rate Limited',
          message: 'Too many requests to Kaggle API. Please wait before trying again.',
          suggestions: [
            'Wait a few minutes before trying again',
            'Avoid making multiple requests in quick succession'
          ]
        };
        
      case 500:
      case 502:
      case 503:
        return {
          error: 'Kaggle Server Error',
          message: 'Kaggle\'s servers are experiencing issues. Please try again later.',
          suggestions: [
            'Try again in a few minutes',
            'Check Kaggle\'s status page for known issues'
          ]
        };
        
      default:
        return {
          error: `HTTP Error ${error.status}`,
          message: error.message || 'An unexpected error occurred while communicating with Kaggle.',
          suggestions: [
            'Try again in a few moments',
            'Verify the dataset URL is correct'
          ]
        };
    }
  }
  
  // Parse-specific errors
  if (error.message?.includes('Invalid URL')) {
    return {
      error: 'Invalid URL',
      message: 'The provided URL is not a valid Kaggle dataset or competition URL.',
      suggestions: [
        'Use format: https://www.kaggle.com/datasets/username/datasetname',
        'Or: https://www.kaggle.com/competitions/competitionname',
        'Copy the URL directly from Kaggle\'s website'
      ]
    };
  }
  
  // File-specific errors
  if (context === 'import' && error.message?.includes('file not found')) {
    return {
      error: 'File Not Found',
      message: 'The selected file was not found in the dataset.',
      suggestions: [
        'Re-analyze the dataset to see current files',
        'Check if the file name has changed',
        'Select a different file from the available options'
      ]
    };
  }
  
  // Generic fallback
  return {
    error: context === 'analyze' ? 'Analysis Failed' : 'Import Failed',
    message: error.message || `Failed to ${context} Kaggle dataset. Please try again.`,
    suggestions: [
      'Check your internet connection',
      'Verify the dataset URL is correct',
      'Try again in a few moments'
    ]
  };
}

// URL validation helper
export function validateKaggleUrl(url: string): { isValid: boolean; error?: string } {
  if (!url || typeof url !== 'string') {
    return { isValid: false, error: 'URL is required' };
  }
  
  const trimmedUrl = url.trim();
  
  if (!trimmedUrl.startsWith('https://www.kaggle.com/')) {
    return { 
      isValid: false, 
      error: 'URL must start with https://www.kaggle.com/' 
    };
  }
  
  const datasetPattern = /^https:\/\/www\.kaggle\.com\/datasets\/([\w-]+)\/([\w-]+)$/i;
  const competitionPattern = /^https:\/\/www\.kaggle\.com\/competitions\/([\w-]+)$/i;
  
  if (!datasetPattern.test(trimmedUrl) && !competitionPattern.test(trimmedUrl)) {
    return {
      isValid: false,
      error: 'Invalid Kaggle URL format. Expected dataset or competition URL.'
    };
  }
  
  return { isValid: true };
}

// Helper to check if file is importable
export function isImportableFile(filename: string): boolean {
  const extension = filename.split('.').pop()?.toLowerCase();
  return ['csv', 'xlsx', 'xls'].includes(extension || '');
}

// Helper to get file type description
export function getFileTypeDescription(filename: string): string {
  const extension = filename.split('.').pop()?.toLowerCase();
  
  switch (extension) {
    case 'csv':
      return 'Comma Separated Values';
    case 'xlsx':
      return 'Excel Spreadsheet (Modern)';
    case 'xls':
      return 'Excel Spreadsheet (Legacy)';
    case 'json':
      return 'JSON Data';
    case 'parquet':
      return 'Parquet File';
    case 'txt':
      return 'Text File';
    case 'zip':
      return 'Archive File';
    default:
      return 'Unknown File Type';
  }
}