// First, create the directory structure:
// client/app/auth/callback/route.ts

import { createClient } from '@/utils/supabase/server'
import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get('code')
  
  if (code) {
    const supabase = await createClient()
    
    try {
      // Exchange the code for a session
      const { error } = await supabase.auth.exchangeCodeForSession(code)
      
      if (error) {
        console.error('Error in auth callback:', error)
        // Redirect to login page with error
        return NextResponse.redirect(`${requestUrl.origin}/login?error=auth_callback_error`)
      }
      
      // Debug - log when this callback is executed
      console.log('Auth callback executed successfully, redirecting to dashboard')
      
      // URL to redirect to after sign in process completes
      return NextResponse.redirect(`${requestUrl.origin}/dashboard`)
    } catch (error) {
      console.error('Exception in auth callback:', error)
      // Redirect to login page with error
      return NextResponse.redirect(`${requestUrl.origin}/login?error=auth_callback_error`)
    }
  }
  
  // If there's no code, redirect to homepage
  return NextResponse.redirect(`${requestUrl.origin}`)
}