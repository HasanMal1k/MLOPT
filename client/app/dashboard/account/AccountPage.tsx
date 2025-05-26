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
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { 
  User, 
  Mail, 
  Key, 
  Shield, 
  Calendar,
  CheckCircle,
  AlertCircle,
  Loader2,
  Eye,
  EyeOff
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { format } from 'date-fns'

interface UserProfile {
  id: string
  email: string
  created_at: string
  email_confirmed_at: string | null
  last_sign_in_at: string | null
  user_metadata: {
    avatar_url?: string
    full_name?: string
    provider?: string
  }
}

export default function AccountPage() {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [emailLoading, setEmailLoading] = useState(false)
  const [passwordLoading, setPasswordLoading] = useState(false)
  
  // Form states
  const [newEmail, setNewEmail] = useState('')
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  
  const { toast } = useToast()
  const supabase = createClient()

  useEffect(() => {
    fetchUserProfile()
  }, [])

  const fetchUserProfile = async () => {
    try {
      const { data: { user }, error } = await supabase.auth.getUser()
      
      if (error) throw error
      
      if (user) {
        setUser(user as UserProfile)
        setNewEmail(user.email || '')
      }
    } catch (error: any) {
      console.error('Error fetching user profile:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load user profile"
      })
    } finally {
      setLoading(false)
    }
  }

  const handleEmailUpdate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newEmail || newEmail === user?.email) return

    setEmailLoading(true)
    
    try {
      const { error } = await supabase.auth.updateUser({
        email: newEmail
      })

      if (error) throw error

      toast({
        title: "Email Update Requested",
        description: "Please check both your old and new email for confirmation links to complete the change.",
      })
    } catch (error: any) {
      console.error('Error updating email:', error)
      toast({
        variant: "destructive",
        title: "Failed to update email",
        description: error.message || "Something went wrong"
      })
    } finally {
      setEmailLoading(false)
    }
  }

  const handlePasswordUpdate = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!newPassword || !confirmPassword) {
      toast({
        variant: "destructive",
        title: "Missing fields",
        description: "Please fill in all password fields"
      })
      return
    }

    if (newPassword !== confirmPassword) {
      toast({
        variant: "destructive",
        title: "Passwords don't match",
        description: "New password and confirmation must match"
      })
      return
    }

    if (newPassword.length < 6) {
      toast({
        variant: "destructive",
        title: "Password too short",
        description: "Password must be at least 6 characters long"
      })
      return
    }

    setPasswordLoading(true)

    try {
      const { error } = await supabase.auth.updateUser({
        password: newPassword
      })

      if (error) throw error

      // Clear password fields
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')

      toast({
        title: "Password Updated",
        description: "Your password has been successfully updated.",
      })
    } catch (error: any) {
      console.error('Error updating password:', error)
      toast({
        variant: "destructive",
        title: "Failed to update password",
        description: error.message || "Something went wrong"
      })
    } finally {
      setPasswordLoading(false)
    }
  }

  const getUserInitials = () => {
    if (!user?.email) return "U"
    
    if (user.user_metadata?.full_name) {
      const nameParts = user.user_metadata.full_name.split(' ')
      if (nameParts.length >= 2) {
        return `${nameParts[0][0]}${nameParts[1][0]}`.toUpperCase()
      }
      return nameParts[0][0].toUpperCase()
    }
    
    return user.email[0].toUpperCase()
  }

  const getDisplayName = () => {
    if (user?.user_metadata?.full_name) {
      return user.user_metadata.full_name
    }
    return user?.email?.split('@')[0] || 'User'
  }

  if (loading) {
    return (
      <div className="h-screen w-full px-6 md:px-10 py-10 flex items-center justify-center">
        <div className="flex items-center gap-2">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span>Loading account settings...</span>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="h-screen w-full px-6 md:px-10 py-10 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="flex flex-col items-center p-6">
            <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">Unable to load account</h3>
            <p className="text-muted-foreground text-center mb-4">
              Please try logging out and logging back in.
            </p>
            <Button onClick={() => window.location.reload()}>
              Refresh Page
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold">Account Settings</h1>
          <p className="text-muted-foreground mt-2">
            Manage your account information and security settings
          </p>
        </div>

        {/* Profile Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <User className="h-5 w-5" />
              Profile Overview
            </CardTitle>
            <CardDescription>
              Your account information and current status
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center gap-4">
              <Avatar className="h-20 w-20">
                {user.user_metadata?.avatar_url ? (
                  <AvatarImage 
                    src={user.user_metadata.avatar_url} 
                    alt={getDisplayName()} 
                  />
                ) : (
                  <AvatarFallback className="text-xl bg-gradient-to-br from-blue-500 to-purple-500 text-white">
                    {getUserInitials()}
                  </AvatarFallback>
                )}
              </Avatar>
              
              <div className="space-y-2">
                <h3 className="text-xl font-semibold">{getDisplayName()}</h3>
                <div className="flex items-center gap-2">
                  <Mail className="h-4 w-4 text-muted-foreground" />
                  <span>{user.email}</span>
                  {user.email_confirmed_at ? (
                    <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      Verified
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200">
                      <AlertCircle className="h-3 w-3 mr-1" />
                      Unverified
                    </Badge>
                  )}
                </div>
                
                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Calendar className="h-4 w-4" />
                    <span>
                      Joined {format(new Date(user.created_at), 'MMM dd, yyyy')}
                    </span>
                  </div>
                  {user.last_sign_in_at && (
                    <div className="flex items-center gap-1">
                      <Shield className="h-4 w-4" />
                      <span>
                        Last login {format(new Date(user.last_sign_in_at), 'MMM dd, yyyy')}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Email Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Mail className="h-5 w-5" />
              Email Settings
            </CardTitle>
            <CardDescription>
              Update your email address
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleEmailUpdate} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="current-email">Current Email</Label>
                <Input
                  id="current-email"
                  type="email"
                  value={user.email || ''}
                  disabled
                  className="bg-muted"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="new-email">New Email</Label>
                <Input
                  id="new-email"
                  type="email"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder="Enter new email address"
                />
              </div>
              
              <div className="flex gap-3">
                <Button 
                  type="submit" 
                  disabled={emailLoading || newEmail === user.email || !newEmail}
                >
                  {emailLoading && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Update Email
                </Button>
                <Button 
                  type="button" 
                  variant="outline"
                  onClick={() => setNewEmail(user.email || '')}
                >
                  Reset
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Password Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Key className="h-5 w-5" />
              Password Settings
            </CardTitle>
            <CardDescription>
              Change your account password
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handlePasswordUpdate} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="new-password">New Password</Label>
                <div className="relative">
                  <Input
                    id="new-password"
                    type={showNewPassword ? "text" : "password"}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Enter new password"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                  >
                    {showNewPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="confirm-password">Confirm New Password</Label>
                <div className="relative">
                  <Input
                    id="confirm-password"
                    type={showConfirmPassword ? "text" : "password"}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm new password"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
              
              <div className="flex gap-3">
                <Button 
                  type="submit" 
                  disabled={passwordLoading || !newPassword || !confirmPassword}
                >
                  {passwordLoading && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Update Password
                </Button>
                <Button 
                  type="button" 
                  variant="outline"
                  onClick={() => {
                    setCurrentPassword('')
                    setNewPassword('')
                    setConfirmPassword('')
                  }}
                >
                  Clear
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Account Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Shield className="h-5 w-5" />
              Account Information
            </CardTitle>
            <CardDescription>
              View your account details and authentication status
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium">Account ID</Label>
                <div className="p-2 bg-muted rounded text-sm font-mono">
                  {user.id}
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm font-medium">Authentication Provider</Label>
                <div className="p-2 bg-muted rounded text-sm capitalize">
                  {user.user_metadata?.provider || 'Email'}
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm font-medium">Account Created</Label>
                <div className="p-2 bg-muted rounded text-sm">
                  {format(new Date(user.created_at), 'PPPp')}
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm font-medium">Email Confirmed</Label>
                <div className="p-2 bg-muted rounded text-sm">
                  {user.email_confirmed_at 
                    ? format(new Date(user.email_confirmed_at), 'PPPp')
                    : 'Not confirmed'
                  }
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}