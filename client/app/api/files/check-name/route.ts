// app/api/files/check-name/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/utils/supabase/server'

export async function GET(request: NextRequest) {
  try {
    const supabase = await createClient()
    const searchParams = request.nextUrl.searchParams
    const fileName = searchParams.get('name')

    if (!fileName) {
      return NextResponse.json(
        { error: 'Filename is required' },
        { status: 400 }
      )
    }

    // Check if filename exists in the files table (check the 'name' field for user's custom display name)
    const { data, error } = await supabase
      .from('files')
      .select('id')
      .eq('name', fileName)
      .maybeSingle()

    if (error) {
      console.error('Error checking filename:', error)
      return NextResponse.json(
        { error: 'Database error', exists: false },
        { status: 500 }
      )
    }

    return NextResponse.json({
      exists: data !== null,
      available: data === null
    })
  } catch (error) {
    console.error('Error in check-name API:', error)
    return NextResponse.json(
      { error: 'Internal server error', exists: false },
      { status: 500 }
    )
  }
}
