import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"


import { Upload, Braces, Move3D, FileText, Settings2, GitBranch, Shuffle, Database } from "lucide-react"

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
    title: 'Auto Preprocessing',
    url: '/dashboard/preprocessing',
    icon: Settings2
  },
  {
    title: 'Custom Preprocessing',
    url: '/dashboard/custom-preprocessing',
    icon: Braces
  },
  {
    title: 'Feature Engineering',
    url: '/dashboard/feature-engineering',
    icon: GitBranch
  }
]

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarContent>
      <SidebarGroup>
        <SidebarGroupLabel>Data Management</SidebarGroupLabel>
        <SidebarMenu>
          {uploadAndPreprocessing.map((item)=> (
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
    </Sidebar>
  )
}