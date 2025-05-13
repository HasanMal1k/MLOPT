/**
 * Utility functions for Kaggle URL parsing and validation
 */

interface KaggleDatasetInfo {
  type: 'dataset';
  owner: string;
  name: string;
  path: string;
}

interface KaggleCompetitionInfo {
  type: 'competition';
  name: string;
  path: string;
}

type KaggleUrlInfo = KaggleDatasetInfo | KaggleCompetitionInfo | null;

/**
 * Parse and validate a Kaggle URL
 * @param url The Kaggle URL to parse
 * @returns Parsed information or null if invalid
 */
export function parseKaggleUrl(url: string): KaggleUrlInfo {
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
}

/**
 * Get the display name for a Kaggle dataset or competition
 */
export function getKaggleDisplayName(info: KaggleUrlInfo): string {
  if (!info) return 'Unknown';
  
  if (info.type === 'dataset') {
    return `${info.owner}/${info.name}`;
  } else {
    return info.name;
  }
}

/**
 * Check if a URL is a valid Kaggle URL
 */
export function isValidKaggleUrl(url: string): boolean {
  return parseKaggleUrl(url) !== null;
}

/**
 * Get an appropriate filename for a Kaggle import
 */
export function getKaggleFilename(info: KaggleUrlInfo, originalFilename: string): string {
  if (!info) return originalFilename;
  
  // If we already have a filename, just prepend the source
  if (originalFilename) {
    return `kaggle_${info.type}_${originalFilename}`;
  }
  
  // Generate a default filename
  if (info.type === 'dataset') {
    return `kaggle_dataset_${info.owner}_${info.name}.csv`;
  } else {
    return `kaggle_competition_${info.name}.csv`;
  }
}