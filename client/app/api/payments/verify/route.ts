import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/utils/supabase/server'

// Verify payment from 2Checkout
export async function POST(request: NextRequest) {
  try {
    const { paymentIntentId, paymentToken, provider } = await request.json()
    
    const supabase = await createClient()
    
    // In production, verify the payment token with 2Checkout API
    // For now, we trust the token from frontend (only for demo)
    // PRODUCTION: Call 2Checkout API to verify token before updating database
    
    // Update payment intent to succeeded
    const { data: paymentIntent, error: updateError } = await supabase
      .from('payment_intents')
      .update({
        status: 'succeeded',
        payment_token: paymentToken,
        payment_method: provider || '2checkout',
        paid_at: new Date().toISOString()
      })
      .eq('id', paymentIntentId)
      .select()
      .single()
    
    if (updateError) {
      console.error('Update error:', updateError)
      return NextResponse.json(
        { error: 'Failed to update payment' },
        { status: 500 }
      )
    }
    
    // Create or update subscription
    const endDate = new Date()
    endDate.setMonth(endDate.getMonth() + 1) // 1 month subscription
    
    const { error: subError } = await supabase
      .from('subscriptions')
      .upsert({
        user_id: paymentIntent.user_id,
        plan_id: paymentIntent.plan_id,
        status: 'active',
        current_period_start: new Date().toISOString(),
        current_period_end: endDate.toISOString()
      })
    
    if (subError) {
      console.error('Subscription error:', subError)
      return NextResponse.json(
        { error: 'Failed to create subscription' },
        { status: 500 }
      )
    }
    
    return NextResponse.json({
      success: true,
      message: 'Payment completed successfully'
    })
    
  } catch (error) {
    console.error('Payment verification error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
