import type { Metadata } from "next";
import "./globals.css";


export const metadata: Metadata = {
  title: "Welcome to MLOpt",
  description: "Choose What You Want To Do",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={``}
      >
        {children}
      </body>
    </html>
  );
}
