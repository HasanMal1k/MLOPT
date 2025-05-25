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

// Extract the first CSV file from a ZIP archive
async function extractCsvFromZip(zipBuffer: Buffer): Promise<{ content: Buffer; filename: string } | null> {
  try {
    const zip = new JSZip();
    const zipContent = await zip.loadAsync(zipBuffer);
    
    // Find the first CSV file in the ZIP
    for (const [filename, file] of Object.entries(zipContent.files)) {
      if (!file.dir && (filename.toLowerCase().endsWith('.csv') || filename.toLowerCase().endsWith('.txt'))) {
        const content = await file.async('arraybuffer');
        return {
          content: Buffer.from(content),
          filename: filename
        };
      }
    }
    
    // If no CSV found, try the first file that's not a directory
    for (const [filename, file] of Object.entries(zipContent.files)) {
      if (!file.dir) {
        const content = await file.async('arraybuffer');
        return {
          content: Buffer.from(content),
          filename: filename
        };
      }
    }
    
    return null;
  } catch (error) {
    console.error('Error extracting from ZIP:', error);
    return null;
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
    let finalContent: Buffer;
    let finalFilename: string;
    let contentType = 'text/csv';
    
    // Check if it's a ZIP file
    if (isZipFile(fileBuffer)) {
      console.log('Detected ZIP file, extracting...');
      const extracted = await extractCsvFromZip(fileBuffer);
      
      if (extracted) {
        finalContent = extracted.content;
        finalFilename = extracted.filename;
        console.log(`Extracted file: ${finalFilename} (${finalContent.length} bytes)`);
        
        // Verify the extracted content is CSV-like
        if (!isLikelyCsv(finalContent)) {
          console.log('Extracted content does not appear to be CSV, using as-is');
        }
      } else {
        return NextResponse.json({ 
          error: 'Processing Error', 
          message: 'Could not extract CSV file from the downloaded archive' 
        }, { status: 500 });
      }
    } else {
      // Not a ZIP file, use as-is
      console.log('File is not compressed, using directly');
      finalContent = fileBuffer;
      
      // Generate filename
      if (type === 'dataset' && name) {
        finalFilename = `${name}.csv`;
      } else if (type === 'competition' && name) {
        finalFilename = `${name}.csv`;
      } else {
        finalFilename = `kaggle-${type}-${Date.now()}.csv`;
      }
      
      // Check if content looks like CSV
      if (!isLikelyCsv(finalContent)) {
        console.log('Warning: Content does not appear to be CSV format');
        // Try to detect actual content type
        const contentStart = finalContent.toString('utf8', 0, 100);
        if (contentStart.includes('<?xml') || contentStart.includes('<html')) {
          return NextResponse.json({ 
            error: 'Invalid Content', 
            message: 'The downloaded file appears to be HTML/XML instead of CSV. The dataset might be private or require special access.' 
          }, { status: 400 });
        }
      }
    }
    
    // Determine content type based on filename
    if (finalFilename.toLowerCase().endsWith('.xlsx')) {
      contentType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    } else if (finalFilename.toLowerCase().endsWith('.json')) {
      contentType = 'application/json';
    }
    
    // Validate that we have actual content
    if (finalContent.length === 0) {
      return NextResponse.json({ 
        error: 'Empty File', 
        message: 'The downloaded file is empty' 
      }, { status: 400 });
    }
    
    // Try to validate CSV content one more time
    if (contentType === 'text/csv') {
      try {
        const sampleText = finalContent.toString('utf8', 0, Math.min(500, finalContent.length));
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
    
    // Upload the file to Supabase Storage (optional - for debugging)
    const debugFilePath = `debug/${user.id}/kaggle-${Date.now()}-${finalFilename}`;
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('data-files')
      .upload(debugFilePath, finalContent, {
        contentType,
        upsert: false
      });
    
    if (uploadError) {
      console.error('Debug upload error:', uploadError);
      // Continue anyway - this is just for debugging
    }
    
    // Get the public URL for debugging
    const { data: { publicUrl } } = supabase.storage
      .from('data-files')
      .getPublicUrl(debugFilePath);
    
    console.log(`Successfully processed Kaggle file: ${finalFilename} (${finalContent.length} bytes)`);
    
    // Return file information that the frontend can use to create a File object
    return NextResponse.json({
      success: true,
      message: 'Dataset imported successfully',
      filename: finalFilename,
      size: finalContent.length,
      contentType,
      // Return the content as base64 so frontend can create File object
      content: finalContent.toString('base64'),
      debugUrl: uploadData ? publicUrl : null
    });
    
  } catch (err: any) {
    console.error('Kaggle import error:', err);
    return NextResponse.json({ 
      error: 'Import process failed', 
      message: err.message || 'Failed to import dataset from Kaggle'
    }, { status: 500 });
  }
}