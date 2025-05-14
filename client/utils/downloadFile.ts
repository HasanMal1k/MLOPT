import { createClient } from '@/utils/supabase/client';

export default async function downloadFile(fileId: string): Promise<File | null> {
  try {
    const supabase = createClient();
    
    // First, get the file metadata from the database
    const { data: fileData, error: fileError } = await supabase
      .from('files')
      .select('*')
      .eq('id', fileId)
      .single();
    
    if (fileError || !fileData) {
      console.error('Error fetching file metadata:', fileError);
      return null;
    }
    
    // Get the public URL for the file
    const { data: { publicUrl } } = supabase.storage
      .from('data-files')
      .getPublicUrl(`${fileData.user_id}/${fileData.filename}`);
    
    // Fetch the file
    const response = await fetch(publicUrl);
    
    if (!response.ok) {
      throw new Error(`Failed to download file: ${response.statusText}`);
    }
    
    // Create a File object
    const blob = await response.blob();
    const file = new File([blob], fileData.original_filename, { 
      type: fileData.mime_type 
    });
    
    return file;
  } catch (error) {
    console.error('Error downloading file:', error);
    return null;
  }
}