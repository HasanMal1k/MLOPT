// client/app/api/kaggle-analyze/route.ts
import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { handleKaggleError, validateKaggleUrl } from '@/utils/kaggle-error-handler';

export async function POST(request: Request) {
  try {
    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ 
        error: 'Unauthorized', 
        message: 'You must be logged in to analyze Kaggle datasets' 
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
    
    // Validate the constructed URL if possible
    const constructedUrl = type === 'dataset' 
      ? `https://www.kaggle.com/datasets/${path}`
      : `https://www.kaggle.com/competitions/${path}`;
      
    const validation = validateKaggleUrl(constructedUrl);
    if (!validation.isValid) {
      return NextResponse.json({
        error: 'Invalid URL',
        message: validation.error || 'Invalid Kaggle URL format'
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
      // For datasets, get metadata using the datasets API
      apiUrl = `https://www.kaggle.com/api/v1/datasets/view/${path}`;
    } else if (type === 'competition') {
      // For competitions, get data files list
      apiUrl = `https://www.kaggle.com/api/v1/competitions/data/list/${path}`;
    } else {
      return NextResponse.json({ 
        error: 'Bad Request', 
        message: 'Invalid type. Must be "dataset" or "competition"' 
      }, { status: 400 });
    }
    
    console.log(`Analyzing Kaggle dataset: ${apiUrl}`);
    
    // Get dataset/competition metadata
    const metadataResponse = await fetch(apiUrl, {
      headers: {
        'Authorization': `Basic ${credentials}`,
      },
    });
    
    if (!metadataResponse.ok) {
      return NextResponse.json({ 
        error: 'Kaggle API Error', 
        message: `Failed to get dataset info: ${metadataResponse.statusText}` 
      }, { status: metadataResponse.status });
    }
    
    const metadata = await metadataResponse.json();
    
    let datasetInfo;
    
    if (type === 'dataset') {
      // For datasets, also get the file list
      const filesApiUrl = `https://www.kaggle.com/api/v1/datasets/list/${path}/files`;
      
      const filesResponse = await fetch(filesApiUrl, {
        headers: {
          'Authorization': `Basic ${credentials}`,
        },
      });
      
      let files = [];
      if (filesResponse.ok) {
        const filesData = await filesResponse.json();
        files = filesData.datasetFiles || [];
      }
      
      datasetInfo = {
        title: metadata.title || name,
        description: metadata.subtitle || metadata.description || 'No description available',
        files: files.map((file: any) => ({
          name: file.name,
          size: file.totalBytes || 0,
          description: file.description,
          columns: file.columns
        })),
        totalSize: files.reduce((total: number, file: any) => total + (file.totalBytes || 0), 0)
      };
    } else {
      // For competitions, the metadata response already contains file info
      const files = metadata.map((file: any) => ({
        name: file.name,
        size: file.totalBytes || 0,
        description: file.description,
        columns: file.columns
      }));
      
      datasetInfo = {
        title: name,
        description: `Competition data for ${name}`,
        files: files,
        totalSize: files.reduce((total: number, file: any) => total + (file.size || 0), 0)
      };
    }
    
    return NextResponse.json(datasetInfo);
    
  } catch (err: any) {
    const errorResponse = handleKaggleError(err, 'analyze');
    return NextResponse.json(errorResponse, { 
      status: err.status || 500 
    });
  }
}


