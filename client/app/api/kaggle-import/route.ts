// Modified kaggle-import/route.ts
import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { Readable } from 'stream';

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

export async function POST(request: Request) {
  try {
    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized', message: 'You must be logged in to import Kaggle datasets' }, { status: 401 });
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
    
    // Get the file content
    const fileContent = await streamToBuffer(downloadResponse.body!);
    
    // Get filename from Content-Disposition header or use a more meaningful default based on provided name
    const contentDisposition = downloadResponse.headers.get('Content-Disposition');
    let filename = '';
    
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="(.+?)"/);
      if (filenameMatch && filenameMatch[1]) {
        filename = filenameMatch[1];
      }
    }
    
    // If filename couldn't be extracted from headers or is empty, create a better name
    if (!filename) {
      if (type === 'dataset' && name) {
        filename = `${name}.csv`;
      } else if (type === 'competition' && name) {
        filename = `${name}.csv`;
      } else {
        // Fallback to a generic name with a timestamp
        filename = `kaggle-${type}-${Date.now()}.csv`;
      }
    }
    
    // Determine content type
    let contentType = 'text/csv';
    if (filename.toLowerCase().endsWith('.xlsx')) {
      contentType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    }
    
    // Upload the file to Supabase Storage
    // Use the original filename from Kaggle instead of adding prefixes
    const filePath = `${user.id}/kaggle-${Date.now()}-${filename}`;
    
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('data-files')
      .upload(filePath, fileContent, {
        contentType,
        upsert: false
      });
    
    if (uploadError) {
      return NextResponse.json({ 
        error: 'Storage Error', 
        message: `Failed to upload file to storage: ${uploadError.message}` 
      }, { status: 500 });
    }
    
    // Get the public URL for the file
    const { data: { publicUrl } } = supabase.storage
      .from('data-files')
      .getPublicUrl(filePath);
    
    // Return file information to the frontend
    return NextResponse.json({
      success: true,
      message: 'Dataset imported successfully',
      filename,  // This is the original filename from Kaggle now
      url: publicUrl,
      size: fileContent.length,
      contentType
    });
  } catch (err: any) {
    console.error('Kaggle import error:', err);
    return NextResponse.json({ 
      error: 'Import process failed', 
      message: err.message || 'Failed to import dataset from Kaggle'
    }, { status: 500 });
  }
}