import { SidebarTrigger, SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";  

export default function Layout({ children }: { children: React.ReactNode }) {
    return (
      <SidebarProvider>
        <div className="flex h-screen w-screen overflow-hidden">
        <AppSidebar />
        <main className="flex-auto overflow-y-auto ">
          <SidebarTrigger className="absolute"/>
          {children}
        </main>
        </div>
      </SidebarProvider>
    )
  }