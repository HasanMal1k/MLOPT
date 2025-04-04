// app/dashboard/my-files/page.tsx
import UserFiles from "@/components/UserFiles"

export default function MyFilesPage() {
  return (
    <section className="h-screen w-[100%] px-6 md:px-10 py-10">
      <div className="text-4xl font-bold mb-8">
        My Files
      </div>
      <UserFiles />
    </section>
  )
}