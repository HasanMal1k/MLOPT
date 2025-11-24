import { createClient } from '@/utils/supabase/server'
import { NextResponse } from 'next/server'

export async function GET() {
  try {
    const supabase = await createClient()
    
    // Test 1: Check if Supabase client is created
    const config = {
      supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL,
      hasAnonKey: !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
      appUrl: process.env.NEXT_PUBLIC_APP_URL,
    }
    
    // Test 2: Try to get session
    const { data: sessionData, error: sessionError } = await supabase.auth.getSession()
    
    // Test 3: Try to construct OAuth URL manually
    const callbackUrl = `${process.env.NEXT_PUBLIC_APP_URL}/auth/callback`
    let oauthResult: any = null
    let oauthError: any = null
    
    try {
      const { data: oauthData, error: oauthErr } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: callbackUrl,
          skipBrowserRedirect: true, // Don't redirect, just get the URL
        }
      })
      oauthResult = oauthData
      oauthError = oauthErr
    } catch (e: any) {
      oauthError = { message: e.message, stack: e.stack }
    }
    
    return NextResponse.json({
      success: true,
      config,
      session: {
        hasSession: !!sessionData.session,
        error: sessionError?.message
      },
      oauth: {
        url: oauthResult?.url,
        provider: oauthResult?.provider,
        error: oauthError ? {
          message: oauthError.message,
          status: oauthError.status,
          code: oauthError.code
        } : null
      }
    })
  } catch (error: any) {
    return NextResponse.json({
      success: false,
      error: error.message,
      stack: error.stack
    }, { status: 500 })
  }
}
