'use server'

import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'
import { createClient } from '@/utils/supabase/server'

export async function login(formData: FormData) {
  const supabase = await createClient()

  // type-casting here for convenience
  // in practice, you should validate your inputs
  const data = {
    email: formData.get('email') as string,
    password: formData.get('password') as string,
  }

  const { error } = await supabase.auth.signInWithPassword(data)

  if (error) {
    return {error: error.message}
  }

  revalidatePath('/', 'layout')
  redirect('/')
}

export async function signup(formData: FormData) {
  const supabase = await createClient()
  
  // type-casting here for convenience
  // in practice, you should validate your inputs
  const data = {
    email: formData.get('email') as string,
    password: formData.get('password') as string,
  }

  const { error } = await supabase.auth.signUp(data)

  if (error) {
    return {error: error.message}
  }

  revalidatePath('/', 'layout')
  redirect('/')
}


// For signup with google
export async function signInWithOAuth(formData: FormData) {
  const supabase = await createClient()
  
  const provider = formData.get('provider') as 'google' | 'github'
  
  // Get the redirect URL - use the one passed from client or fallback
  const redirectTo = formData.get('redirectTo') as string | null
  const appUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'
  const callbackUrl = redirectTo || `${appUrl}/auth/callback`
  
  try {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: provider,
      options: {
        redirectTo: callbackUrl,
        queryParams: {
          prompt: 'select_account',
          access_type: 'offline'
        },
        skipBrowserRedirect: false
      }
    })
    
    if (error) {
      console.error('OAuth sign in error:', error)
      return { error: error.message }
    }
    
    return { url: data?.url }
  } catch (error: any) {
    console.error('Exception during OAuth:', error)
    return { error: error?.message || 'Failed to initiate OAuth login' }
  }
}