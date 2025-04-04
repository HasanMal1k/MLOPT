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


import { Upload, Braces, Move3D, FileText } from "lucide-react"

const uploadAndPreprocessing = [
  {
    title: 'Data Upload',
    url: '/dashboard/data-upload',
    icon: Upload
  },
  {
    title: 'My Files',
    url: '/dashboard/my-files',
    icon: FileText // import this from lucide-react
  },
  {
    title: 'Preprocessing',
    url : '/dashboard/preprocessing',
    icon: Braces
  },
  {
    title: 'Transformation',
    url: '#',
    icon: Move3D
  }
]

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarContent>
      <SidebarGroup>
        <SidebarGroupLabel>Upload And Preprocessing</SidebarGroupLabel>
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