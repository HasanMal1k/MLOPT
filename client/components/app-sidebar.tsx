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
  SidebarHeader,
} from "@/components/ui/sidebar"
import { 
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import Link from 'next/link'
import { useTheme } from "next-themes"

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
  HelpCircle,
  LayoutDashboard,
  BarChart,
  Clock,
  ChevronDown,
  ChevronRight,
  Sparkles,
  Brain,
  Zap,
  Sun,
  Moon,
  CreditCard,
  FileCode2
} from "lucide-react"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { cn } from "@/lib/utils"
import { title } from 'process'

const navigation = [
  {
    title: 'Overview',
    items: [
      {
        title: 'Dashboard',
        url: '/dashboard',
        icon: LayoutDashboard
      }
    ]
  },
  {
    title: 'Data Management',
    items: [
      {
        title: 'My Files',
        url: '/dashboard/datasets',
        icon: Database
      },
      {
        title: 'Upload & Import',
        url: '/dashboard/data-upload',
        icon: Upload,
        badge: 'New'
      }
    ]
  },
  {
    title: 'Blueprints',
    items: [
      {
        title: 'My Models',
        url: '/dashboard/models',
        icon: FileCode2,
        badge: 'New'
      },
      {
        title: 'Transformations',
        url: '/dashboard/transformations',
        icon: Sparkles,
        badge: 'Beta'
      },
      {
        title: 'Models',
        url: '/dashboard/blueprints',
        icon: Braces,
        badge: 'New'
      }
    ]
  }
]


const accountItems = [
  
  {
    title: 'Subscription',
    url: '/dashboard/pricing',
    icon: CreditCard
  },
  {
    title: 'Account Settings',
    url: '/dashboard/account',
    icon: Settings
  },
  {
    title: 'Billing',
    url: '/dashboard/billing',
    icon: RefreshCw
  },
]

// Theme Toggle Component
function ThemeToggle() {
  const { setTheme } = useTheme()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <SidebarMenuButton 
          className="h-9 w-full justify-start gap-3"
          tooltip="Change theme"
        >
          <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute left-3 h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          <span>Theme</span>
        </SidebarMenuButton>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" side="right">
        <DropdownMenuItem onClick={() => setTheme("light")}>
          <Sun className="mr-2 h-4 w-4" />
          Light
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>
          <Moon className="mr-2 h-4 w-4" />
          Dark
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>
          <Settings className="mr-2 h-4 w-4" />
          System
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function AppSidebar() {
  const [user, setUser] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null)
  const [isAccountOpen, setIsAccountOpen] = useState(false)
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
    <Sidebar className="border-r border-border/40">
      {/* Header with Logo */}
      <SidebarHeader className="border-b border-border/40 p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-transparent text-white">
            <Brain className="h-5 w-5" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold text-gray-100">
              MLOpt
            </span>
            <span className="text-xs text-muted-foreground">
              Automate ML Workflow
            </span>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="p-2">
        {navigation.map((section) => (
          <SidebarGroup key={section.title}>
            <SidebarGroupLabel className="px-2 text-xs font-medium text-muted-foreground">
              {section.title}
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {section.items.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild className="h-9">
                      <Link href={item.url} className="flex items-center gap-3">
                        <item.icon className="h-4 w-4" />
                        <span className="flex-1">{item.title}</span>
                        {item.badge && (
                          <span className={cn(
                            "rounded-full px-2 py-0.5 text-xs font-medium",
                            item.badge === 'New' && "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
                            item.badge === 'Beta' && "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400"
                          )}>
                            {item.badge}
                          </span>
                        )}
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}

        {/* Theme Section */}
        <SidebarGroup>
          <SidebarGroupLabel className="px-2 text-xs font-medium text-muted-foreground">
            Preferences
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <ThemeToggle />
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      
      <SidebarFooter className="p-2 border-t border-border/40">
        
        
        <Separator className="my-2" />
        
        {/* Collapsible Account Section */}
        <Collapsible open={isAccountOpen} onOpenChange={setIsAccountOpen}>
          <CollapsibleTrigger asChild>
            <SidebarMenuButton className="h-12 p-2 data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground">
              <div className="flex items-center gap-3 flex-1">
                {avatarUrl ? (
                  <div className="h-8 w-8 rounded-full overflow-hidden border-2 border-border/40">
                    <img 
                      src={avatarUrl} 
                      alt={getDisplayName()}
                      className="h-full w-full object-cover" 
                    />
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-8 w-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 text-white text-sm font-medium border-2 border-border/40">
                    {getUserInitials()}
                  </div>
                )}
                
                <div className="flex flex-col text-left flex-1 min-w-0">
                  <span className="text-sm font-medium truncate">{getDisplayName()}</span>
                  <span className="text-xs text-muted-foreground truncate">
                    {user?.email}
                  </span>
                </div>
              </div>
              
              <ChevronDown className={cn(
                "h-4 w-4 transition-transform duration-200",
                isAccountOpen && "rotate-180"
              )} />
            </SidebarMenuButton>
          </CollapsibleTrigger>
          
          <CollapsibleContent className="space-y-1">
            <SidebarMenu>
              {accountItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild size="sm" className="h-8 pl-6">
                    <Link href={item.url} className="flex items-center gap-3">
                      <item.icon className="h-3 w-3" />
                      <span className="text-xs">{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
              
              <SidebarMenuItem>
                <SidebarMenuButton 
                  onClick={handleSignOut} 
                  size="sm" 
                  className="h-8 pl-6 text-destructive hover:text-destructive hover:bg-destructive/10"
                >
                  <LogOut className="h-3 w-3" />
                  <span className="text-xs">Sign Out</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </CollapsibleContent>
        </Collapsible>
        
        {/* Footer */}
        <div className="mt-2 pt-2 border-t border-border/20">
          <div className="text-xs text-center text-muted-foreground">
            © 2025 <span className="font-medium">MLOpt</span>
          </div>
        </div>
      </SidebarFooter> 
    </Sidebar>
  )
}