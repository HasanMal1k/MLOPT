import { createClient } from '@/utils/supabase/server';
import { NextResponse } from 'next/server';
import extractMetadata from '@/utils/extractMetaData';

export async function POST(request: Request) {
  try {
    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    // Parse the form data
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    const uniqueFilename = `${Date.now()}-${file.name}`;
    
    // Upload to Supabase storage
    const { data, error } = await supabase.storage
      .from('data-files')
      .upload(`${user.id}/${uniqueFilename}`, file, {
        contentType: file.type,
        upsert: false
      });
    
    if (error) {
      throw error;
    }
    
    // Extract metadata (you'll need to implement this function)
    // This should parse CSV/XLSX and extract columns, preview data, etc.
    const metadata = await extractMetadata(file);
    
    // Save metadata to database
    const { data: fileRecord, error: dbError } = await supabase
      .from('files')
      .insert({
        user_id: user.id,
        filename: uniqueFilename,
        original_filename: file.name,
        file_size: file.size,
        mime_type: file.type,
        column_names: metadata.columns,
        row_count: metadata.rowCount,
        file_preview: metadata.preview,
        statistics: metadata.statistics
      })
      .select();
    
    if (dbError) {
      throw dbError;
    }
    
    return NextResponse.json({
      success: true,
      fileId: fileRecord[0].id,
      metadata: {
        filename: file.name,
        size: file.size,
        rowCount: metadata.rowCount,
        columns: metadata.columns
      }
    });
    
  } catch (err: any) {
    console.error('Upload error:', err);
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}