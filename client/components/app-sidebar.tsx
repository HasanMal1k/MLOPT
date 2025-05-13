'use client'

import { useState, useEffect } from 'react'
import { createClient } from '@/utils/supabase/client'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"

import { 
  Upload, 
  Braces, 
  FileText, 
  Settings2, 
  GitBranch, 
  Database, 
  RefreshCw,
  LogOut,
  User,
  Settings,
  HelpCircle
} from "lucide-react"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"

const uploadAndPreprocessing = [
  {
    title: 'Data Upload',
    url: '/dashboard/data-upload',
    icon: Upload
  },
  {
    title: 'My Files',
    url: '/dashboard/my-files',
    icon: FileText
  },
  {
    title: 'Custom Cleaning',
    url: '/dashboard/preprocessing',
    icon: Settings2
  },
  {
    title: 'Feature Transformations',
    url: '/dashboard/transformations',
    icon: RefreshCw
  },
  {
    title: 'Feature Engineering',
    url: '/dashboard/feature-engineering',
    icon: GitBranch
  }
]

export function AppSidebar() {
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null)
  const { toast } = useToast()
  const supabase = createClient()

  useEffect(() => {
    async function getUser() {
      try {
        const { data: { user }, error } = await supabase.auth.getUser()
        if (error) {
          console.error('Error fetching user:', error)
          return
        }
        
        setUser(user)
        
        // Extract avatar URL from user metadata if it exists and is a valid string
        const userAvatarUrl = user?.user_metadata?.avatar_url
        if (userAvatarUrl && typeof userAvatarUrl === 'string' && userAvatarUrl.trim() !== '') {
          setAvatarUrl(userAvatarUrl)
        }
      } catch (error) {
        console.error('Error in getUser:', error)
      } finally {
        setLoading(false)
      }
    }
    
    getUser()
  }, [])
  
  const handleSignOut = async () => {
    try {
      const { error } = await supabase.auth.signOut()
      
      if (error) {
        toast({
          variant: "destructive",
          title: "Error signing out",
          description: error.message
        })
        return
      }
      
      // Redirect to login page after successful sign out
      window.location.href = "/login"
    } catch (error) {
      console.error('Error signing out:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "An unexpected error occurred while signing out."
      })
    }
  }
  
  // Helper to get user initials for avatar fallback
  const getUserInitials = () => {
    if (!user || !user.email) return "U"
    
    // Extract first letter of email or split name if available
    if (user.user_metadata?.full_name) {
      const nameParts = user.user_metadata.full_name.split(' ')
      if (nameParts.length >= 2) {
        return `${nameParts[0][0]}${nameParts[1][0]}`.toUpperCase()
      }
      return nameParts[0][0].toUpperCase()
    }
    
    return user.email[0].toUpperCase()
  }
  
  // Helper to get display name
  const getDisplayName = () => {
    if (!user) return "User"
    
    if (user.user_metadata?.full_name) {
      return user.user_metadata.full_name
    }
    
    // Use first part of email or full email
    return user.email?.split('@')[0] || user.email
  }

  return (
    <Sidebar>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Data Management</SidebarGroupLabel>
          <SidebarMenu>
            {uploadAndPreprocessing.map((item) => (
              <SidebarMenuItem key={item.title}>
                <SidebarMenuButton asChild>
                  <a href={item.url}>
                    <item.icon />
                    <span>{item.title}</span>
                  </a>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>
      
      <SidebarFooter>
        <div className="px-2 py-4">
          <Separator className="my-2" />
          
          {/* User profile section - Manual implementation without Avatar component */}
          <div className="flex flex-col gap-2 mt-2">
            <div className="flex items-center gap-2 px-2 py-1">
              {avatarUrl ? (
                // If we have an avatar URL, show the image
                <div className="h-8 w-8 rounded-full overflow-hidden">
                  <img 
                    src={avatarUrl} 
                    alt={getDisplayName()}
                    className="h-full w-full object-cover" 
                  />
                </div>
              ) : (
                // Otherwise show initials in a colored circle
                <div className="flex items-center justify-center h-8 w-8 rounded-full bg-primary text-primary-foreground text-sm font-medium">
                  {getUserInitials()}
                </div>
              )}
              
              <div className="flex flex-col text-xs">
                <span className="font-medium">{getDisplayName()}</span>
                <span className="text-muted-foreground truncate max-w-[120px]">
                  {user?.email}
                </span>
              </div>
            </div>
            
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <a href="/dashboard/account">
                    <User className="h-4 w-4" />
                    <span>Account</span>
                  </a>
                </SidebarMenuButton>
              </SidebarMenuItem>
              
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <a href="/dashboard/settings">
                    <Settings className="h-4 w-4" />
                    <span>Settings</span>
                  </a>
                </SidebarMenuButton>
              </SidebarMenuItem>
              
              <SidebarMenuItem>
                <SidebarMenuButton asChild>
                  <a href="/dashboard/help">
                    <HelpCircle className="h-4 w-4" />
                    <span>Help & Support</span>
                  </a>
                </SidebarMenuButton>
              </SidebarMenuItem>
              
              <SidebarMenuItem>
                <SidebarMenuButton onClick={handleSignOut}>
                  <LogOut className="h-4 w-4" />
                  <span>Sign Out</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </div>
          
          <div className="mt-4 text-xs text-center text-muted-foreground">
            © 2025 MLOpt
          </div>
        </div>
      </SidebarFooter> 
    </Sidebar>
  )
}