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
    const { type, path } = body;
    
    if (!type || !path) {
      return NextResponse.json({ 
        error: 'Bad Request', 
        message: 'Both type and path are required' 
      }, { status: 400 });
    }
    
    // Simulate Kaggle API response for development/testing
    // In production, you would use real Kaggle API credentials and endpoints
    
    // This is a mock implementation that simulates the Kaggle API
    // It generates a sample CSV file with the dataset name
    // const generateMockCSV = (datasetName: string) => {
    //   const header = "id,name,value\n";
    //   const rows = [
    //     "1,Item 1,100\n",
    //     "2,Item 2,200\n",
    //     "3,Item 3,300\n",
    //     "4,Item 4,400\n",
    //     "5,Item 5,500\n"
    //   ];
      
    //   return header + rows.join("");
    // };
    
    // // Create a sample filename based on the dataset or competition
    // const filename = `${path.replace(/\//g, '_')}_sample.csv`;
    
    // // Create a sample CSV file
    // const csvContent = generateMockCSV(path);
    // const fileBlob = new Blob([csvContent], { type: 'text/csv' });
    // const fileBuffer = Buffer.from(await fileBlob.arrayBuffer());
    
    // // Upload to Supabase Storage
    // const filePath = `${user.id}/kaggle-${Date.now()}-${filename}`;
    
    // const { data: uploadData, error: uploadError } = await supabase.storage
    //   .from('data-files')
    //   .upload(filePath, fileBuffer, {
    //     contentType: 'text/csv',
    //     upsert: false
    //   });
    
    // if (uploadError) {
    //   console.error('Storage upload error:', uploadError);
    //   return NextResponse.json({ 
    //     error: 'Storage Error', 
    //     message: `Failed to upload file to storage: ${uploadError.message}` 
    //   }, { status: 500 });
    // }
    
    // // Get the public URL
    // const { data: { publicUrl } } = supabase.storage
    //   .from('data-files')
    //   .getPublicUrl(filePath);
    
    // // Return success response
    // return NextResponse.json({
    //   success: true,
    //   message: `${type === 'dataset' ? 'Dataset' : 'Competition data'} imported successfully`,
    //   filename,
    //   url: publicUrl,
    //   size: fileBuffer.length,
    //   contentType: 'text/csv'
    // });
    
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
    
    // Get filename from Content-Disposition header or use a default
    const contentDisposition = downloadResponse.headers.get('Content-Disposition');
    let filename = 'kaggle-data.csv';
    
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="(.+?)"/);
      if (filenameMatch && filenameMatch[1]) {
        filename = filenameMatch[1];
      }
    }
    
    // Determine content type
    let contentType = 'text/csv';
    if (filename.toLowerCase().endsWith('.xlsx')) {
      contentType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
    }
    
    // Upload the file to Supabase Storage
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
      filename,
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