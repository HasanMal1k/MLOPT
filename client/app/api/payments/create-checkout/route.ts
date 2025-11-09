import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/utils/supabase/server'

// 2Checkout (Verifone) Integration
// Supports: Visa, Mastercard, PayPal, and more
// Works globally including Pakistan

export async function POST(request: NextRequest) {
  try {
    const { planId, userId } = await request.json()
    
    const supabase = await createClient()
    
    // Plan pricing
    const plans = {
      basic: { price: 4.99, name: 'Basic' },
      pro: { price: 9.99, name: 'Pro' },
      premium: { price: 19.99, name: 'Premium' }
    }
    
    const plan = plans[planId as keyof typeof plans]
    if (!plan) {
      return NextResponse.json(
        { error: 'Invalid plan' },
        { status: 400 }
      )
    }
    
    // Create payment intent in database
    const { data: paymentIntent, error } = await supabase
      .from('payment_intents')
      .insert({
        user_id: userId,
        plan_id: planId,
        amount: plan.price,
        currency: 'USD',
        status: 'pending',
        payment_method: '2checkout'
      })
      .select()
      .single()
    
    if (error) {
      console.error('Database error:', error)
      return NextResponse.json(
        { error: 'Failed to create payment intent' },
        { status: 500 }
      )
    }

    // Generate 2Checkout hosted checkout URL
    // In production, you'd use 2Checkout API to create order
    const sellerId = process.env.NEXT_PUBLIC_2CO_SELLER_ID || process.env.TWO_CHECKOUT_SELLER_ID
    const returnUrl = `${process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000'}/dashboard/payment/success?intent=${paymentIntent.id}`
    
    // 2Checkout Hosted Checkout URL format
    const checkoutUrl = sellerId 
      ? `https://secure.2checkout.com/checkout/buy?merchant=${sellerId}&prod=${planId}&qty=1&price=${plan.price}&return-url=${encodeURIComponent(returnUrl)}`
      : null
    
    return NextResponse.json({
      paymentIntentId: paymentIntent.id,
      amount: plan.price,
      currency: 'USD',
      planName: plan.name,
      checkoutUrl, // For hosted checkout redirect
      // For inline checkout, frontend uses 2Checkout.js
    })
    
  } catch (error) {
    console.error('Checkout error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
