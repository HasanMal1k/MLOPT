import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';

export async function GET(request: NextRequest) {
  try {
    // Get the fileId from the query parameters
    const { searchParams } = new URL(request.url);
    const fileId = searchParams.get('fileId');
    
    if (!fileId) {
      return NextResponse.json({ error: 'No file ID provided' }, { status: 400 });
    }
    
    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized', message: 'You must be logged in to download files' }, { status: 401 });
    }
    
    // Get the file metadata from the database
    const { data: fileData, error: fileError } = await supabase
      .from('files')
      .select('*')
      .eq('id', fileId)
      .eq('user_id', user.id)
      .single();
    
    if (fileError || !fileData) {
      return NextResponse.json({ error: 'File not found', details: fileError?.message }, { status: 404 });
    }
    
    // Get the file from storage
    const filePath = `${fileData.user_id}/${fileData.filename}`;
    const { data, error: storageError } = await supabase.storage
      .from('data-files')
      .download(filePath);
    
    if (storageError || !data) {
      return NextResponse.json({ 
        error: 'Failed to download file from storage', 
        details: storageError?.message 
      }, { status: 500 });
    }
    
    // Return the file
    return new NextResponse(data, {
      headers: {
        'Content-Type': fileData.mime_type,
        'Content-Disposition': `attachment; filename="${fileData.original_filename}"`,
      },
    });
    
  } catch (err: any) {
    console.error('File download error:', err);
    return NextResponse.json({ 
      error: 'Download failed', 
      message: err.message || 'Failed to download file' 
    }, { status: 500 });
  }
}