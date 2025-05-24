// app/api/upload/route.ts - Updated with Time Series Support
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
    const originalFilename = formData.get('original_filename') as string;
    
    // NEW: Get dataset type and processing configurations
    const datasetType = (formData.get('dataset_type') as string) || 'normal';
    const customCleaned = formData.get('custom_cleaned') === 'true';
    
    // Get preprocessing results if available
    const preprocessingResults = formData.get('preprocessing_results') ? 
      JSON.parse(formData.get('preprocessing_results') as string) : null;
    
    // NEW: Get time series configuration if available
    const timeSeriesConfig = formData.get('time_series_config') ? 
      JSON.parse(formData.get('time_series_config') as string) : null;
    
    // NEW: Get time series statistics if available
    const timeSeriesStatistics = formData.get('time_series_statistics') ? 
      JSON.parse(formData.get('time_series_statistics') as string) : null;
    
    // NEW: Get custom cleaning configuration if available
    const customCleaningConfig = formData.get('custom_cleaning_config') ? 
      JSON.parse(formData.get('custom_cleaning_config') as string) : null;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    const uniqueFilename = `${Date.now()}-${originalFilename || file.name}`;
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
    
    // Prepare preprocessing info based on dataset type
    let preprocessingInfo;
    
    if (datasetType === 'time_series') {
      // For time series data, combine preprocessing results with time series specific info
      preprocessingInfo = {
        is_preprocessed: true,
        is_time_series_processed: true,
        preprocessing_date: new Date().toISOString(),
        time_series_processing_date: new Date().toISOString(),
        columns_cleaned: preprocessingResults?.columns_cleaned || [],
        auto_detected_dates: preprocessingResults?.date_columns_detected || [],
        dropped_columns: preprocessingResults?.columns_dropped || [],
        missing_value_stats: preprocessingResults?.missing_value_stats || {},
        engineered_features: preprocessingResults?.engineered_features || [],
        transformation_details: preprocessingResults?.transformation_details || {},
        original_shape: preprocessingResults?.original_shape || [],
        processed_shape: preprocessingResults?.processed_shape || [],
        // Time series specific info
        time_series_frequency: timeSeriesConfig?.frequency,
        time_series_date_column: timeSeriesConfig?.dateColumn,
        time_series_excluded_columns: timeSeriesConfig?.columnsToExclude || [],
        time_series_imputation_method: timeSeriesConfig?.imputationMethod,
        time_series_statistics: timeSeriesStatistics
      };
    } else if (preprocessingResults) {
      // For normal preprocessed data
      preprocessingInfo = {
        is_preprocessed: isPreprocessed,
        preprocessing_date: isPreprocessed ? new Date().toISOString() : null,
        columns_cleaned: preprocessingResults.columns_cleaned || [],
        auto_detected_dates: preprocessingResults.date_columns_detected || [],
        dropped_columns: preprocessingResults.columns_dropped || [],
        missing_value_stats: preprocessingResults.missing_value_stats || {},
        engineered_features: preprocessingResults.engineered_features || [],
        transformation_details: preprocessingResults.transformation_details || {},
        original_shape: preprocessingResults.original_shape || [],
        processed_shape: preprocessingResults.processed_shape || []
      };
    } else {
      // For files without preprocessing
      preprocessingInfo = {
        is_preprocessed: isPreprocessed,
        preprocessing_date: isPreprocessed ? new Date().toISOString() : null
      };
    }
    
    // Save metadata to database with new schema fields
    const { data: fileRecord, error: dbError } = await supabase
      .from('files')
      .insert({
        user_id: user.id,
        filename: uniqueFilename,
        original_filename: originalFilename || file.name,
        file_size: file.size,
        mime_type: file.type,
        column_names: metadata.columns,
        row_count: metadata.rowCount,
        file_preview: metadata.preview,
        statistics: metadata.statistics,
        preprocessing_info: preprocessingInfo,
        // NEW: Add dataset type and processing configurations
        dataset_type: datasetType,
        custom_cleaning_applied: customCleaned,
        custom_cleaning_config: customCleaningConfig,
        time_series_config: timeSeriesConfig ? {
          date_column: timeSeriesConfig.dateColumn,
          frequency: timeSeriesConfig.frequency,
          imputation_method: timeSeriesConfig.imputationMethod,
          excluded_columns: timeSeriesConfig.columnsToExclude || [],
          processing_statistics: timeSeriesStatistics,
          processed_date: new Date().toISOString()
        } : null
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
    
    // Log successful upload with type information
    console.log(`Successfully uploaded ${datasetType} dataset: ${originalFilename || file.name}`);
    if (datasetType === 'time_series' && timeSeriesConfig) {
      console.log(`Time series config - Date column: ${timeSeriesConfig.dateColumn}, Frequency: ${timeSeriesConfig.frequency}`);
    }
    if (customCleaned && customCleaningConfig) {
      console.log(`Custom cleaning applied with ${customCleaningConfig.transformations?.length || 0} transformations`);
    }
    
    return NextResponse.json({
      success: true,
      fileId: fileRecord?.[0]?.id,
      metadata: {
        filename: originalFilename || file.name,
        size: file.size,
        rowCount: metadata.rowCount,
        columns: metadata.columns,
        isPreprocessed: isPreprocessed,
        datasetType: datasetType,
        customCleaningApplied: customCleaned,
        timeSeriesProcessed: datasetType === 'time_series'
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