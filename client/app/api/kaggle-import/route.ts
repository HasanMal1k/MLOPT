// client/app/api/kaggle-import/route.ts - Fixed version with proper file handling
import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import JSZip from 'jszip'; // You'll need to install this: npm install jszip

// Utility to convert ReadableStream to Buffer
async function streamToBuffer(stream: ReadableStream<Uint8Array>): Promise<Buffer> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  
  return Buffer.concat(chunks);
}

// Check if content is likely a ZIP file
function isZipFile(buffer: Buffer): boolean {
  // ZIP files start with 'PK' (0x504B)
  return buffer.length >= 2 && buffer[0] === 0x50 && buffer[1] === 0x4B;
}

// Extract all CSV/Excel files from a ZIP archive
async function extractFilesFromZip(zipBuffer: Buffer): Promise<{ content: Buffer; filename: string }[]> {
  try {
    const zip = new JSZip();
    const zipContent = await zip.loadAsync(zipBuffer);
    const extractedFiles: { content: Buffer; filename: string }[] = [];
    
    // Find all CSV and Excel files in the ZIP (including nested folders)
    for (const [filepath, file] of Object.entries(zipContent.files)) {
      if (file.dir) continue;
      
      const filename = filepath.split('/').pop() || filepath;
      const lowerFilename = filename.toLowerCase();
      
      // Skip hidden files, metadata files, and system files
      if (filename.startsWith('.') || filename.startsWith('__MACOSX') || filename === 'desktop.ini') {
        continue;
      }
      
      // Accept CSV, Excel, and TXT files
      if (lowerFilename.endsWith('.csv') || 
          lowerFilename.endsWith('.xlsx') || 
          lowerFilename.endsWith('.xls') ||
          lowerFilename.endsWith('.txt')) {
        const content = await file.async('arraybuffer');
        extractedFiles.push({
          content: Buffer.from(content),
          filename: filename // Use just the filename without path
        });
      }
    }
    
    // If no data files found, try to find any readable file
    if (extractedFiles.length === 0) {
      for (const [filepath, file] of Object.entries(zipContent.files)) {
        if (!file.dir) {
          const filename = filepath.split('/').pop() || filepath;
          if (!filename.startsWith('.') && !filename.startsWith('__MACOSX')) {
            const content = await file.async('arraybuffer');
            extractedFiles.push({
              content: Buffer.from(content),
              filename: filename
            });
            break; // Just take the first one
          }
        }
      }
    }
    
    return extractedFiles;
  } catch (error) {
    console.error('Error extracting from ZIP:', error);
    return [];
  }
}

// Detect if content is likely CSV by checking the first few lines
function isLikelyCsv(buffer: Buffer): boolean {
  try {
    const text = buffer.toString('utf8', 0, Math.min(1000, buffer.length));
    const lines = text.split('\n').slice(0, 5);
    
    // Check if it looks like CSV
    const hasCommas = lines.some(line => line.includes(','));
    const hasConsistentColumns = lines.length > 1 && 
      lines.slice(0, 3).every(line => line.split(',').length === lines[0].split(',').length);
    
    return hasCommas && hasConsistentColumns;
  } catch {
    return false;
  }
}

export async function POST(request: Request) {
  try {
    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ 
        error: 'Unauthorized', 
        message: 'You must be logged in to import Kaggle datasets' 
      }, { status: 401 });
    }
    
    // Parse the request body
    const body = await request.json();
    const { type, path, name } = body;
    
    if (!type || !path) {
      return NextResponse.json({ 
        error: 'Bad Request', 
        message: 'Both type and path are required' 
      }, { status: 400 });
    }
    
    // Get Kaggle API credentials from environment variables
    const kaggleUsername = process.env.KAGGLE_USERNAME;
    const kaggleKey = process.env.KAGGLE_KEY;
    
    if (!kaggleUsername || !kaggleKey) {
      return NextResponse.json({ 
        error: 'Server Configuration Error', 
        message: 'Kaggle API credentials are not configured' 
      }, { status: 500 });
    }

    // Create base64 encoded credentials for Kaggle API authentication
    const credentials = Buffer.from(`${kaggleUsername}:${kaggleKey}`).toString('base64');
    
    let apiUrl;
    if (type === 'dataset') {
      // For datasets
      apiUrl = `https://www.kaggle.com/api/v1/datasets/download/${path}`;
    } else if (type === 'competition') {
      // For competitions
      apiUrl = `https://www.kaggle.com/api/v1/competitions/data/download/${path}`;
    } else {
      return NextResponse.json({ 
        error: 'Bad Request', 
        message: 'Invalid type. Must be "dataset" or "competition"' 
      }, { status: 400 });
    }
    
    console.log(`Fetching from Kaggle API: ${apiUrl}`);
    
    // Download the file from Kaggle
    const downloadResponse = await fetch(apiUrl, {
      headers: {
        'Authorization': `Basic ${credentials}`,
      },
    });
    
    if (!downloadResponse.ok) {
      return NextResponse.json({ 
        error: 'Kaggle API Error', 
        message: `Failed to download file: ${downloadResponse.statusText}` 
      }, { status: downloadResponse.status });
    }
    
    // Get the file content as buffer
    const fileBuffer = await streamToBuffer(downloadResponse.body!);
    console.log(`Downloaded ${fileBuffer.length} bytes from Kaggle`);
    
    // Process the downloaded content
    let filesToReturn: Array<{ content: Buffer; filename: string; contentType: string }> = [];
    
    // Check if it's a ZIP file
    if (isZipFile(fileBuffer)) {
      console.log('Detected ZIP file, extracting...');
      const extractedFiles = await extractFilesFromZip(fileBuffer);
      
      if (extractedFiles.length === 0) {
        return NextResponse.json({ 
          error: 'Processing Error', 
          message: 'Could not extract any data files from the downloaded archive. The ZIP might be empty or contain unsupported file types.' 
        }, { status: 500 });
      }
      
      console.log(`Extracted ${extractedFiles.length} file(s) from ZIP`);
      
      // Process each extracted file
      for (const extracted of extractedFiles) {
        let contentType = 'text/csv';
        if (extracted.filename.toLowerCase().endsWith('.xlsx')) {
          contentType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
        } else if (extracted.filename.toLowerCase().endsWith('.xls')) {
          contentType = 'application/vnd.ms-excel';
        } else if (extracted.filename.toLowerCase().endsWith('.json')) {
          contentType = 'application/json';
        }
        
        filesToReturn.push({
          content: extracted.content,
          filename: extracted.filename,
          contentType
        });
        
        console.log(`Processed: ${extracted.filename} (${extracted.content.length} bytes)`);
      }
    } else {
      // Not a ZIP file, use as-is
      console.log('File is not compressed, using directly');
      
      // Generate filename
      let finalFilename: string;
      if (type === 'dataset' && name) {
        finalFilename = `${name}.csv`;
      } else if (type === 'competition' && name) {
        finalFilename = `${name}.csv`;
      } else {
        finalFilename = `kaggle-${type}-${Date.now()}.csv`;
      }
      
      // Check if content looks like CSV
      if (!isLikelyCsv(fileBuffer)) {
        console.log('Warning: Content does not appear to be CSV format');
        // Try to detect actual content type
        const contentStart = fileBuffer.toString('utf8', 0, 100);
        if (contentStart.includes('<?xml') || contentStart.includes('<html')) {
          return NextResponse.json({ 
            error: 'Invalid Content', 
            message: 'The downloaded file appears to be HTML/XML instead of CSV. The dataset might be private or require special access.' 
          }, { status: 400 });
        }
      }
      
      filesToReturn.push({
        content: fileBuffer,
        filename: finalFilename,
        contentType: 'text/csv'
      });
    }
    
    // Validate that we have actual content
    if (filesToReturn.length === 0 || filesToReturn.some(f => f.content.length === 0)) {
      return NextResponse.json({ 
        error: 'Empty File', 
        message: 'The downloaded file(s) are empty' 
      }, { status: 400 });
    }
    
    // Validate CSV content for each file
    for (const file of filesToReturn) {
      if (file.contentType === 'text/csv') {
        try {
          const sampleText = file.content.toString('utf8', 0, Math.min(500, file.content.length));
          if (sampleText.includes('<!DOCTYPE') || sampleText.includes('<html>')) {
            return NextResponse.json({ 
              error: 'Access Denied', 
              message: 'Received HTML page instead of data file. The dataset might be private, require login, or need terms acceptance.' 
            }, { status: 403 });
          }
        } catch (error) {
          console.error('Error validating CSV content:', error);
        }
      }
    }
    
    // If multiple files, return array format
    if (filesToReturn.length > 1) {
      const filesData = filesToReturn.map(file => ({
        filename: file.filename,
        size: file.content.length,
        contentType: file.contentType,
        content: file.content.toString('base64')
      }));
      
      console.log(`Successfully processed ${filesToReturn.length} Kaggle files`);
      
      return NextResponse.json({
        success: true,
        message: `Successfully imported ${filesToReturn.length} files from Kaggle`,
        multipleFiles: true,
        files: filesData,
        totalSize: filesToReturn.reduce((sum, f) => sum + f.content.length, 0)
      });
    }
    
    // Single file - backward compatible response
    const singleFile = filesToReturn[0];
    console.log(`Successfully processed Kaggle file: ${singleFile.filename} (${singleFile.content.length} bytes)`);
    
    return NextResponse.json({
      success: true,
      message: 'Dataset imported successfully',
      filename: singleFile.filename,
      size: singleFile.content.length,
      contentType: singleFile.contentType,
      content: singleFile.content.toString('base64'),
      multipleFiles: false
    });
    
  } catch (err: any) {
    console.error('Kaggle import error:', err);
    return NextResponse.json({ 
      error: 'Import process failed', 
      message: err.message || 'Failed to import dataset from Kaggle'
    }, { status: 500 });
  }
}