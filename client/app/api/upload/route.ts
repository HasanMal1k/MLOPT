import { createClient } from '@/utils/supabase/server';
import { NextResponse } from 'next/server';
import extractMetadata from '@/utils/extractMetaData';

export async function POST(request: Request) {
  try {
    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      console.error('Authentication error:', authError);
      return NextResponse.json({ error: 'Unauthorized', details: authError?.message }, { status: 401 });
    }
    
    // Parse the form data
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const isPreprocessed = formData.get('preprocessed') === 'true';
    
    // Get preprocessing results if available
    const preprocessingResults = formData.get('preprocessing_results') ? 
      JSON.parse(formData.get('preprocessing_results') as string) : null;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    const uniqueFilename = `${Date.now()}-${file.name}`;
    const filePath = `${user.id}/${uniqueFilename}`;
    
    // Upload to Supabase storage
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('data-files')
      .upload(filePath, file, {
        contentType: file.type,
        upsert: false
      });
    
    if (uploadError) {
      console.error('Storage upload error:', uploadError);
      return NextResponse.json({ 
        error: 'Failed to upload file to storage', 
        details: uploadError.message 
      }, { status: 500 });
    }
    
    // Get the public URL (if needed)
    const { data: { publicUrl } } = supabase.storage
      .from('data-files')
      .getPublicUrl(filePath);
    
    // Extract metadata
    const metadata = await extractMetadata(file);
    
    // Prepare preprocessing info if available
    const preprocessingInfo = preprocessingResults ? {
      is_preprocessed: true,
      preprocessing_date: new Date().toISOString(),
      columns_cleaned: preprocessingResults.columns_cleaned || [],
      auto_detected_dates: preprocessingResults.date_columns_detected || [],
      dropped_columns: preprocessingResults.columns_dropped || [],
      missing_value_stats: preprocessingResults.missing_value_stats || {},
      engineered_features: preprocessingResults.engineered_features || [],
      transformation_details: preprocessingResults.transformation_details || {}
    } : {
      is_preprocessed: isPreprocessed,
      preprocessing_date: isPreprocessed ? new Date().toISOString() : null
    };
    
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
        statistics: metadata.statistics,
        preprocessing_info: preprocessingInfo
      })
      .select();
    
    if (dbError) {
      console.error('Database error:', dbError);
      // Try to delete the uploaded file if database insert fails
      await supabase.storage.from('data-files').remove([filePath]);
      return NextResponse.json({ 
        error: 'Failed to save file metadata', 
        details: dbError.message 
      }, { status: 500 });
    }
    
    return NextResponse.json({
      success: true,
      fileId: fileRecord?.[0]?.id,
      metadata: {
        filename: file.name,
        size: file.size,
        rowCount: metadata.rowCount,
        columns: metadata.columns,
        isPreprocessed: isPreprocessed
      }
    });
    
  } catch (err: any) {
    console.error('Upload error:', err);
    return NextResponse.json({ 
      error: 'Upload process failed', 
      details: err.message 
    }, { status: 500 });
  }
}