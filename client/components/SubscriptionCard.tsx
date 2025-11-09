'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { createClient } from '@/utils/supabase/client'
import { Crown, Calendar, CreditCard, ArrowRight } from 'lucide-react'
import Link from 'next/link'

interface Subscription {
  plan_id: string
  status: string
  current_period_start: string
  current_period_end: string
}

export default function SubscriptionCard() {
  const [subscription, setSubscription] = useState<Subscription | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadSubscription()
  }, [])

  const loadSubscription = async () => {
    try {
      const supabase = createClient()
      const { data: { user } } = await supabase.auth.getUser()
      
      if (!user) return

      const { data, error } = await supabase
        .from('subscriptions')
        .select('*')
        .eq('user_id', user.id)
        .eq('status', 'active')
        .single()

      if (error && error.code !== 'PGRST116') {
        console.error('Error loading subscription:', error)
        return
      }

      setSubscription(data)
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!subscription) {
    return (
      <Card className="border-dashed">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-amber-500" />
            No Active Subscription
          </CardTitle>
          <CardDescription>
            Upgrade to unlock premium features and unlimited access
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Link href="/dashboard/pricing">
            <Button className="w-full">
              View Plans
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </Link>
        </CardContent>
      </Card>
    )
  }

  const planColors = {
    basic: 'bg-blue-100 text-blue-800 border-blue-200',
    pro: 'bg-purple-100 text-purple-800 border-purple-200',
    premium: 'bg-amber-100 text-amber-800 border-amber-200'
  }

  const daysLeft = Math.ceil(
    (new Date(subscription.current_period_end).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
  )

  return (
    <Card className="border-purple-200 bg-gradient-to-br from-purple-50 to-white dark:from-purple-950/20 dark:to-gray-900">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Crown className="h-5 w-5 text-purple-500" />
            Current Subscription
          </CardTitle>
          <Badge 
            variant="outline" 
            className={planColors[subscription.plan_id as keyof typeof planColors]}
          >
            {subscription.plan_id.toUpperCase()}
          </Badge>
        </div>
        <CardDescription>
          Your premium features and benefits
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Calendar className="h-4 w-4" />
            <span>Renews on</span>
          </div>
          <span className="font-medium">
            {new Date(subscription.current_period_end).toLocaleDateString()}
          </span>
        </div>
        
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2 text-muted-foreground">
            <CreditCard className="h-4 w-4" />
            <span>Days remaining</span>
          </div>
          <Badge variant="secondary">
            {daysLeft} days
          </Badge>
        </div>

        <div className="pt-2 border-t">
          <Link href="/dashboard/pricing">
            <Button variant="outline" className="w-full">
              Manage Subscription
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  )
}
