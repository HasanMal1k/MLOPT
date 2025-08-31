import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"
import Image from "next/image"

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white">
      <section className="h-screen flex items-center justify-center">
        <div className="w-full max-w-7xl mx-auto grid lg:grid-cols-2 gap-8 px-8">
          {/* Left Content */}
          <div className="flex flex-col justify-center space-y-8">
            <div className="space-y-6">
              <div className="text-sm text-gray-400 font-medium">FYP-I Project 2024</div>
              <h1 className="text-6xl lg:text-7xl font-bold">Welcome To MLOPT</h1>
              <p className="text-xl text-gray-300 leading-relaxed">Reducing your machine learning workflow to clicks</p>
            </div>

            <Button size="lg" className="w-fit bg-white text-black hover:bg-gray-200 text-lg px-8 py-6">
              <Link href="/dashboard" className="flex items-center gap-2">
                Continue To Dashboard
                <ArrowRight className="h-5 w-5" />
              </Link>
            </Button>

            {/* Team Info moved to bottom */}
            <div className="pt-8 border-t border-gray-800">
              <div className="text-sm text-gray-500 space-y-1">
                <p>Team: Hasan Malik (SP22-BDS-026) & Jawad Idrees (SP22-BDS-020)</p>
                <p>Bachelor of Data Science | Final Year Project - I</p>
              </div>
            </div>
          </div>

          {/* Right Content - Image */}
          <div className="flex items-center justify-center">
            <div className="relative w-full h-full max-h-[600px]">
              <Image src="/img.jpg" alt="MLOpt Dashboard" fill className="object-cover rounded-lg" priority />
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
