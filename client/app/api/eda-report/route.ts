import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';

export async function POST(request: NextRequest) {
  try {
    // Get the authenticated user
    const supabase = await createClient();
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      return NextResponse.json({ error: 'Unauthorized', message: 'You must be logged in to generate EDA reports' }, { status: 401 });
    }
    
    // Get the form data
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }
    
    // Forward the file to the Python backend for report generation
    const pythonBackendURL = 'http://localhost:8000/generate_report/';
    
    // Create a new FormData object to send to the Python backend
    const pythonFormData = new FormData();
    pythonFormData.append('file', file);
    
    try {
      // Send request to Python backend
      const pythonResponse = await fetch(pythonBackendURL, {
        method: 'POST',
        body: pythonFormData,
      });
      
      if (!pythonResponse.ok) {
        const errorText = await pythonResponse.text();
        console.error('Python backend error:', errorText);
        return NextResponse.json({ 
          error: 'Report generation failed', 
          message: `Python backend returned status ${pythonResponse.status}: ${pythonResponse.statusText}`
        }, { status: 500 });
      }
      
      // Get the HTML report
      const reportHtml = await pythonResponse.text();
      
      // Return the HTML report
      return new NextResponse(reportHtml, {
        headers: {
          'Content-Type': 'text/html',
        },
      });
    } catch (fetchError: any) {
      console.error('Error fetching from Python backend:', fetchError);
      
      return NextResponse.json({
        error: 'Connection error',
        message: 'Could not connect to the Python backend. Please make sure it is running.',
        details: fetchError.message
      }, { status: 503 });
    }
  } catch (err: any) {
    console.error('EDA report generation error:', err);
    return NextResponse.json({ 
      error: 'Report generation failed', 
      message: err.message || 'Failed to generate EDA report' 
    }, { status: 500 });
  }
}