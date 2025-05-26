'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import { 
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { 
  CreditCard, 
  Calendar, 
  Download, 
  Settings,
  Crown,
  Zap,
  Shield,
  Star,
  AlertTriangle,
  Lock,
  ExternalLink,
  CheckCircle,
  XCircle
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { format, addMonths } from 'date-fns'

interface UserProfile {
  email: string
  user_metadata: {
    full_name?: string
  }
}

// Fake data for demonstration
const FAKE_SUBSCRIPTION = {
  id: 'sub_1Ox2K7LkdIwHu7ixeOpWPRGL',
  status: 'active',
  plan: 'Pro',
  amount: 2900, // $29.00
  interval: 'month',
  current_period_start: new Date(),
  current_period_end: addMonths(new Date(), 1),
  cancel_at_period_end: false
}

const FAKE_PAYMENT_METHOD = {
  id: 'pm_1Ox2K7LkdIwHu7ixeOpWPRGL',
  type: 'card',
  card: {
    brand: 'visa',
    last4: '4242',
    exp_month: 12,
    exp_year: 2027
  }
}

const FAKE_INVOICES = [
  {
    id: 'in_1Ox2K7LkdIwHu7ixeOpWPRGL',
    number: 'INV-2024-001',
    date: new Date('2024-01-15'),
    amount: 2900,
    status: 'paid',
    plan: 'Pro Plan'
  },
  {
    id: 'in_1Ox2K6LkdIwHu7ixeOpWPRGL',
    number: 'INV-2023-012',
    date: new Date('2023-12-15'),
    amount: 2900,
    status: 'paid',
    plan: 'Pro Plan'
  },
  {
    id: 'in_1Ox2K5LkdIwHu7ixeOpWPRGL',
    number: 'INV-2023-011',
    date: new Date('2023-11-15'),
    amount: 2900,
    status: 'paid',
    plan: 'Pro Plan'
  }
]

const PRICING_TIERS = [
  {
    name: 'Free',
    price: 0,
    interval: 'month',
    description: 'Perfect for getting started',
    features: [
      '5 file uploads per month',
      'Basic preprocessing',
      'Standard support',
      '1GB storage'
    ],
    current: false
  },
  {
    name: 'Pro',
    price: 29,
    interval: 'month',
    description: 'Best for professionals',
    features: [
      'Unlimited file uploads',
      'Advanced preprocessing',
      'Time series analysis',
      'Priority support',
      '100GB storage',
      'Custom transformations',
      'API access'
    ],
    current: true,
    popular: true
  },
  {
    name: 'Enterprise',
    price: 99,
    interval: 'month',
    description: 'For large organizations',
    features: [
      'Everything in Pro',
      'Custom integrations',
      'Dedicated support',
      'Unlimited storage',
      'SLA guarantee',
      'White-label options',
      'Advanced analytics'
    ],
    current: false
  }
]

export default function BillingPage() {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()
  const supabase = createClient()

  useEffect(() => {
    fetchUser()
  }, [])

  const fetchUser = async () => {
    try {
      const { data: { user }, error } = await supabase.auth.getUser()
      if (error) throw error
      setUser(user as UserProfile)
    } catch (error) {
      console.error('Error fetching user:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDisabledAction = (action: string) => {
    toast({
      variant: "destructive",
      title: "Feature Disabled",
      description: `${action} is currently disabled in this demo version.`,
    })
  }

  const formatPrice = (amount: number) => {
    return `$${(amount / 100).toFixed(2)}`
  }

  const getCardBrandIcon = (brand: string) => {
    return <CreditCard className="h-4 w-4" />
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      active: { variant: "outline" as const, className: "bg-green-50 text-green-700 border-green-200", icon: CheckCircle },
      cancelled: { variant: "outline" as const, className: "bg-red-50 text-red-700 border-red-200", icon: XCircle },
      past_due: { variant: "outline" as const, className: "bg-yellow-50 text-yellow-700 border-yellow-200", icon: AlertTriangle }
    }
    
    const config = variants[status as keyof typeof variants] || variants.active
    const Icon = config.icon
    
    return (
      <Badge variant={config.variant} className={config.className}>
        <Icon className="h-3 w-3 mr-1" />
        {status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ')}
      </Badge>
    )
  }

  if (loading) {
    return (
      <div className="h-screen w-full px-6 md:px-10 py-10 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p>Loading billing information...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-4xl font-bold">Billing & Usage</h1>
            <p className="text-muted-foreground mt-2">
              Manage your subscription and billing information
            </p>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-muted-foreground bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg px-3 py-2">
            <Lock className="h-4 w-4" />
            <span>Demo Mode - All actions disabled</span>
          </div>
        </div>

        {/* Current Subscription */}
        <Card>
          <CardHeader>
            <div className="flex justify-between items-start">
              <div>
                <CardTitle className="flex items-center gap-3">
                  <Crown className="h-5 w-5 text-yellow-600" />
                  Current Subscription
                </CardTitle>
                <CardDescription>
                  Your active plan and billing details
                </CardDescription>
              </div>
              {getStatusBadge(FAKE_SUBSCRIPTION.status)}
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="space-y-2">
                <div className="text-sm font-medium text-muted-foreground">Plan</div>
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-bold">{FAKE_SUBSCRIPTION.plan}</span>
                  <Badge variant="secondary">Popular</Badge>
                </div>
                <div className="text-sm text-muted-foreground">
                  {formatPrice(FAKE_SUBSCRIPTION.amount)}/{FAKE_SUBSCRIPTION.interval}
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="text-sm font-medium text-muted-foreground">Next billing date</div>
                <div className="flex items-center gap-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">
                    {format(FAKE_SUBSCRIPTION.current_period_end, 'MMM dd, yyyy')}
                  </span>
                </div>
                <div className="text-sm text-muted-foreground">
                  Auto-renewal enabled
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="text-sm font-medium text-muted-foreground">Monthly cost</div>
                <div className="text-2xl font-bold">
                  {formatPrice(FAKE_SUBSCRIPTION.amount)}
                </div>
                <div className="text-sm text-muted-foreground">
                  Billed monthly
                </div>
              </div>
            </div>

            <Separator />

            <div className="flex gap-3">
              <Button 
                variant="outline" 
                onClick={() => handleDisabledAction('Plan management')}
                disabled
              >
                <Settings className="h-4 w-4 mr-2" />
                Manage Plan
              </Button>
              <Button 
                variant="outline" 
                onClick={() => handleDisabledAction('Subscription cancellation')}
                disabled
              >
                Cancel Subscription
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Payment Method */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <CreditCard className="h-5 w-5" />
              Payment Method
            </CardTitle>
            <CardDescription>
              Your default payment method for billing
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div className="flex items-center gap-3">
                {getCardBrandIcon(FAKE_PAYMENT_METHOD.card.brand)}
                <div>
                  <div className="font-medium">
                    •••• •••• •••• {FAKE_PAYMENT_METHOD.card.last4}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Expires {FAKE_PAYMENT_METHOD.card.exp_month}/{FAKE_PAYMENT_METHOD.card.exp_year}
                  </div>
                </div>
              </div>
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                Default
              </Badge>
            </div>
            
            <div className="flex gap-3">
              <Button 
                variant="outline" 
                onClick={() => handleDisabledAction('Payment method update')}
                disabled
              >
                Update Payment Method
              </Button>
              <Button 
                variant="outline" 
                onClick={() => handleDisabledAction('Add payment method')}
                disabled
              >
                Add Payment Method
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Usage & Limits */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Zap className="h-5 w-5" />
              Usage This Month
            </CardTitle>
            <CardDescription>
              Your current usage and plan limits
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">File Uploads</span>
                  <span className="text-sm text-muted-foreground">847 / Unlimited</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{ width: '15%' }}></div>
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Storage Used</span>
                  <span className="text-sm text-muted-foreground">23.4 GB / 100 GB</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-green-600 h-2 rounded-full" style={{ width: '23%' }}></div>
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">API Calls</span>
                  <span className="text-sm text-muted-foreground">12,456 / Unlimited</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{ width: '8%' }}></div>
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Processing Time</span>
                  <span className="text-sm text-muted-foreground">145 hours / Unlimited</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-orange-600 h-2 rounded-full" style={{ width: '12%' }}></div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Billing History */}
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <div>
                <CardTitle className="flex items-center gap-3">
                  <Calendar className="h-5 w-5" />
                  Billing History
                </CardTitle>
                <CardDescription>
                  Your recent invoices and payments
                </CardDescription>
              </div>
              <Button 
                variant="outline" 
                onClick={() => handleDisabledAction('Invoice download')}
                disabled
              >
                <Download className="h-4 w-4 mr-2" />
                Download All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {FAKE_INVOICES.map((invoice) => (
                <div key={invoice.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <Calendar className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    </div>
                    <div>
                      <div className="font-medium">{invoice.number}</div>
                      <div className="text-sm text-muted-foreground">
                        {invoice.plan} • {format(invoice.date, 'MMM dd, yyyy')}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="font-medium">{formatPrice(invoice.amount)}</div>
                      {getStatusBadge(invoice.status)}
                    </div>
                    <Button 
                      variant="ghost" 
                      size="sm"
                      onClick={() => handleDisabledAction('Invoice download')}
                      disabled
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Available Plans */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Star className="h-5 w-5" />
              Available Plans
            </CardTitle>
            <CardDescription>
              Compare and switch between different pricing tiers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {PRICING_TIERS.map((tier) => (
                <div 
                  key={tier.name} 
                  className={`relative p-6 border rounded-lg ${
                    tier.current 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/10' 
                      : 'border-border'
                  }`}
                >
                  {tier.popular && (
                    <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                      <Badge className="bg-blue-600 hover:bg-blue-600">
                        Most Popular
                      </Badge>
                    </div>
                  )}
                  
                  <div className="text-center space-y-4">
                    <div>
                      <h3 className="text-lg font-semibold">{tier.name}</h3>
                      <p className="text-sm text-muted-foreground">{tier.description}</p>
                    </div>
                    
                    <div>
                      <span className="text-3xl font-bold">${tier.price}</span>
                      <span className="text-muted-foreground">/{tier.interval}</span>
                    </div>
                    
                    <div className="space-y-2 text-sm">
                      {tier.features.map((feature, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          <span>{feature}</span>
                        </div>
                      ))}
                    </div>
                    
                    <Button 
                      className="w-full" 
                      variant={tier.current ? "outline" : "default"}
                      onClick={() => handleDisabledAction(`${tier.name} plan selection`)}
                      disabled
                    >
                      {tier.current ? 'Current Plan' : `Upgrade to ${tier.name}`}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}