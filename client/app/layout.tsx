import type { Metadata } from "next";
import "./globals.css";
import { Geist } from 'next/font/google'
import { ThemeProvider } from "@/components/theme-provider"; 
import { Toaster } from "@/components/ui/toaster";

export const metadata: Metadata = {
  title: "Welcome to MLOpt",
  description: "Choose What You Want To Do",
};

const geist = Geist({subsets: ['latin']})

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={geist.className}
      >
        <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem
            disableTransitionOnChange
          >
        {children}
        <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
