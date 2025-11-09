'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Check, Zap, Rocket, Crown, Loader2, CreditCard } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/utils/supabase/client'
import { toast } from '@/hooks/use-toast'
import Script from 'next/script'

// Declare 2Checkout global
declare global {
  interface Window {
    TCO?: any
  }
}

interface PricingPlan {
  id: string
  name: string
  price: number
  description: string
  features: string[]
  icon: any
  color: string
  popular?: boolean
}

const plans: PricingPlan[] = [
  {
    id: 'basic',
    name: 'Basic',
    price: 4.99,
    description: 'Perfect for getting started',
    icon: Zap,
    color: 'text-blue-500',
    features: [
      '5 file uploads per month',
      'Basic data preprocessing',
      'Standard visualizations',
      'CSV/Excel support',
      'Email support'
    ]
  },
  {
    id: 'pro',
    name: 'Pro',
    price: 9.99,
    description: 'For professional data scientists',
    icon: Rocket,
    color: 'text-purple-500',
    popular: true,
    features: [
      'Unlimited file uploads',
      'Advanced preprocessing',
      'Custom transformations',
      'Time series analysis',
      'AutoML training',
      'Priority support',
      'Export trained models'
    ]
  },
  {
    id: 'premium',
    name: 'Premium',
    price: 19.99,
    description: 'For teams and enterprises',
    icon: Crown,
    color: 'text-amber-500',
    features: [
      'Everything in Pro',
      'Team collaboration',
      'API access',
      'Custom model deployment',
      'Advanced analytics',
      'Dedicated support',
      'SLA guarantee'
    ]
  }
]

export default function PricingPage() {
  const [loading, setLoading] = useState<string | null>(null)
  const [userId, setUserId] = useState<string | null>(null)
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null)
  const [tcoLoaded, setTcoLoaded] = useState(false)
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      setUserId(user?.id || null)
    }
    getUser()
  }, [])

  const handlePlanSelect = async (planId: string) => {
    if (!userId) {
      toast({
        title: "Please log in",
        description: "You need to be logged in to subscribe",
        variant: "destructive"
      })
      router.push('/login')
      return
    }

    if (!tcoLoaded || !window.TCO) {
      toast({
        title: "Loading payment system...",
        description: "Please wait a moment and try again",
        variant: "default"
      })
      return
    }

    try {
      setLoading(planId)
      const plan = plans.find(p => p.id === planId)
      if (!plan) return

      // Create payment intent first
      const checkoutRes = await fetch('/api/payments/create-checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          planId: planId, 
          userId 
        })
      })

      const { paymentIntentId } = await checkoutRes.json()

      // Get user email for 2Checkout
      const { data: { user } } = await supabase.auth.getUser()
      const userEmail = user?.email || ''

      // Launch 2Checkout inline checkout
      window.TCO.loadPubKey(
        process.env.NEXT_PUBLIC_2CO_ENVIRONMENT || 'sandbox',
        function() {
          // Card payment form
          const cardForm = {
            sellerId: process.env.NEXT_PUBLIC_2CO_SELLER_ID,
            publishableKey: process.env.NEXT_PUBLIC_2CO_PUBLISHABLE_KEY,
            ccNo: document.getElementById('ccNo')?.value || '',
            cvv: document.getElementById('cvv')?.value || '',
            expMonth: document.getElementById('expMonth')?.value || '',
            expYear: document.getElementById('expYear')?.value || ''
          }

          window.TCO.requestToken(
            async function(data: any) {
              // Success - got token
              await handlePaymentSuccess(paymentIntentId, data.response.token.token, plan)
            },
            function(error: any) {
              // Error
              console.error('2Checkout error:', error)
              toast({
                title: "Payment Failed",
                description: error.errorMsg || "Please check your card details",
                variant: "destructive"
              })
              setLoading(null)
            },
            cardForm
          )
        }
      )

    } catch (error) {
      console.error('Checkout error:', error)
      toast({
        title: "Error",
        description: "Failed to initialize payment",
        variant: "destructive"
      })
      setLoading(null)
    }
  }

  const handlePaymentSuccess = async (paymentIntentId: string, token: string, plan: PricingPlan) => {
    try {
      // Verify payment with your backend
      const verifyRes = await fetch('/api/payments/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paymentIntentId,
          paymentToken: token,
          provider: '2checkout'
        })
      })

      const result = await verifyRes.json()

      if (result.success) {
        toast({
          title: "Payment Successful! 🎉",
          description: `Welcome to ${plan.name} plan!`
        })
        setTimeout(() => {
          router.push('/dashboard?payment=success')
        }, 1500)
      } else {
        throw new Error('Payment verification failed')
      }
    } catch (error) {
      console.error('Verification error:', error)
      toast({
        title: "Payment Failed",
        description: "Please contact support",
        variant: "destructive"
      })
    } finally {
      setLoading(null)
      setSelectedPlan(null)
    }
  }

  const handleSimplePayment = async (planId: string) => {
    if (!userId) {
      toast({
        title: "Please log in",
        description: "You need to be logged in to subscribe",
        variant: "destructive"
      })
      router.push('/login')
      return
    }

    // Redirect to 2Checkout hosted checkout
    try {
      setLoading(planId)
      const plan = plans.find(p => p.id === planId)
      if (!plan) return

      // Create payment intent
      const checkoutRes = await fetch('/api/payments/create-checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          planId: planId, 
          userId 
        })
      })

      const { paymentIntentId, checkoutUrl } = await checkoutRes.json()

      // Redirect to 2Checkout hosted page
      if (checkoutUrl) {
        window.location.href = checkoutUrl
      } else {
        throw new Error('No checkout URL')
      }
    } catch (error) {
      console.error('Payment error:', error)
      toast({
        title: "Payment Failed",
        description: "Please try again",
        variant: "destructive"
      })
      setLoading(null)
    }
  }

  return (
    <>
      {/* Load 2Checkout script */}
      <Script 
        src="https://www.2checkout.com/checkout/api/2co.min.js" 
        strategy="afterInteractive"
        onLoad={() => setTcoLoaded(true)}
      />

      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950 py-12 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">Choose Your Plan</h1>
            <p className="text-xl text-muted-foreground">
              Select the perfect plan for your machine learning needs
            </p>
            <div className="mt-4 flex flex-col items-center gap-2">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-400">
                � Powered by 2Checkout - Visa, Mastercard, PayPal
              </Badge>
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-400">
                � Works Globally including Pakistan
              </Badge>
            </div>
          </div>

        {/* Pricing Cards */}
        <div className="grid md:grid-cols-3 gap-8">
          {plans.map((plan) => {
            const Icon = plan.icon
            return (
              <Card 
                key={plan.id}
                className={`relative ${
                  plan.popular 
                    ? 'border-purple-500 shadow-lg scale-105' 
                    : 'hover:shadow-md'
                } transition-all duration-300`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                    <Badge className="bg-purple-500 text-white">Most Popular</Badge>
                  </div>
                )}
                
                <CardHeader>
                  <div className="flex items-center gap-3 mb-2">
                    <div className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-800 ${plan.color}`}>
                      <Icon className="h-6 w-6" />
                    </div>
                    <CardTitle>{plan.name}</CardTitle>
                  </div>
                  <CardDescription>{plan.description}</CardDescription>
                  <div className="mt-4">
                    <span className="text-4xl font-bold">${plan.price}</span>
                    <span className="text-muted-foreground">/month</span>
                  </div>
                </CardHeader>

                <CardContent>
                  <ul className="space-y-3">
                    {plan.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <Check className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>

                <CardFooter>
                  <Button
                    onClick={() => handleSimplePayment(plan.id)}
                    disabled={loading === plan.id}
                    className={`w-full ${
                      plan.popular
                        ? 'bg-purple-500 hover:bg-purple-600'
                        : ''
                    }`}
                  >
                    {loading === plan.id ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <CreditCard className="mr-2 h-4 w-4" />
                        Subscribe Now
                      </>
                    )}
                  </Button>
                </CardFooter>
              </Card>
            )
          })}
        </div>

        {/* Info Section */}
        <div className="mt-12 text-center space-y-4">
          <div className="text-sm text-muted-foreground space-y-2">
            <p className="flex items-center justify-center gap-2">
              <Check className="h-4 w-4 text-green-500" />
              Secure checkout via 2Checkout
            </p>
            <p className="flex items-center justify-center gap-2">
              <Check className="h-4 w-4 text-green-500" />
              Cancel anytime • No hidden fees • Money-back guarantee
            </p>
          </div>
          <div className="inline-flex items-center gap-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg px-6 py-3">
            <span className="text-2xl">⚙️</span>
            <div className="text-left">
              <p className="font-semibold text-amber-900 dark:text-amber-100">Setup Required</p>
              <p className="text-xs text-amber-700 dark:text-amber-300">
                Add 2Checkout credentials to environment variables to go live
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
    </>
  )
}
