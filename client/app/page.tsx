'use client'
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, Upload, Zap, Brain, Cloud, CheckCircle, Menu, X } from "lucide-react"
import { useState } from "react"
import Beams from "@/components/Beams"

// Beams Background Component
const BeamsBackground = () => {
  return (
    <div className="absolute inset-0 w-full h-full">
      <Beams
        beamWidth={2}
        beamHeight={15}
        beamNumber={12}
        lightColor="#ffffff"
        speed={2}
        noiseIntensity={1.75}
        scale={0.2}
        rotation={0}
      />
    </div>
  )
}

// Stylish Navbar Component
const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="fixed top-0 w-full z-50 bg-black/80 backdrop-blur-md border-b border-white/10">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex justify-between items-center h-16">
          <Link href="/" className="text-2xl font-bold text-white">
            MLOPT
          </Link>

          <div className="hidden md:flex items-center space-x-8">
            <Link href="#features" className="text-gray-300 hover:text-white transition-colors">
              Features
            </Link>
            <Link href="#how-it-works" className="text-gray-300 hover:text-white transition-colors">
              How It Works
            </Link>
            <Link href="#about" className="text-gray-300 hover:text-white transition-colors">
              About
            </Link>
            <Button size="sm" className="bg-white text-black hover:bg-gray-200">
              <Link href="/dashboard">Get Started</Link>
            </Button>
          </div>

          <button
            className="md:hidden text-white"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {isOpen && (
          <div className="md:hidden py-4 space-y-4 border-t border-white/10">
            <Link href="#features" className="block text-gray-300 hover:text-white transition-colors">
              Features
            </Link>
            <Link href="#how-it-works" className="block text-gray-300 hover:text-white transition-colors">
              How It Works
            </Link>
            <Link href="#about" className="block text-gray-300 hover:text-white transition-colors">
              About
            </Link>
            <Button size="sm" className="w-full bg-white text-black hover:bg-gray-200">
              <Link href="/dashboard">Get Started</Link>
            </Button>
          </div>
        )}
      </div>
    </nav>
  )
}

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Navbar />
      
      {/* Hero Section with Beams Background */}
      <section className="min-h-screen flex items-center justify-center relative overflow-hidden pt-16">
        <BeamsBackground />
        <div className="max-w-7xl mx-auto px-6 text-center relative z-10">
          <div className="space-y-8">
            <div className="inline-block px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full border border-white/20">
              <span className="text-sm text-gray-300">FYP-I Project 2024 • Data Science Automation</span>
            </div>
            
            <h1 className="text-6xl md:text-8xl font-bold leading-tight">
              Automate Your
              <br />
              <span className="text-white/80">Data Science</span>
              <br />
              Pipeline
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-400 max-w-3xl mx-auto leading-relaxed">
              From raw CSV files to deployed models in minutes. MLOPT transforms your data science workflow 
              with intelligent automation, cleaning, transformation, and deployment.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-8">
              <Button size="lg" className="bg-white text-black hover:bg-gray-200 text-lg px-8 py-4">
                <Link href="/dashboard" className="flex items-center gap-2">
                  Start Building
                  <ArrowRight className="h-5 w-5" />
                </Link>
              </Button>
              
              <Button size="lg" variant="outline" className="border-white/30 text-white hover:bg-white/10 text-lg px-8 py-4">
                Watch Demo
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 relative z-10">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">How MLOPT Works</h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Four simple steps to transform your data into intelligent models
            </p>
          </div>

          <div className="grid md:grid-cols-4 gap-8">
            <div className="text-center group">
              <div className="w-16 h-16 bg-white/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:bg-white/20 transition-colors">
                <Upload className="h-8 w-8" />
              </div>
              <h3 className="text-xl font-semibold mb-4">1. Upload Data</h3>
              <p className="text-gray-400">
                Simply drag and drop your CSV or Excel files. We support multiple formats and large datasets.
              </p>
            </div>

            <div className="text-center group">
              <div className="w-16 h-16 bg-white/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:bg-white/20 transition-colors">
                <Zap className="h-8 w-8" />
              </div>
              <h3 className="text-xl font-semibold mb-4">2. Auto Clean</h3>
              <p className="text-gray-400">
                Our AI automatically detects and cleans data issues. You can review and adjust the cleaning process.
              </p>
            </div>

            <div className="text-center group">
              <div className="w-16 h-16 bg-white/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:bg-white/20 transition-colors">
                <Brain className="h-8 w-8" />
              </div>
              <h3 className="text-xl font-semibold mb-4">3. Train Models</h3>
              <p className="text-gray-400">
                Choose from multiple ML algorithms. We automatically optimize hyperparameters for best performance.
              </p>
            </div>

            <div className="text-center group">
              <div className="w-16 h-16 bg-white/10 rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:bg-white/20 transition-colors">
                <Cloud className="h-8 w-8" />
              </div>
              <h3 className="text-xl font-semibold mb-4">4. Deploy</h3>
              <p className="text-gray-400">
                One-click deployment to Azure or other cloud platforms. Get your model live in minutes.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 relative z-10">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-4xl md:text-5xl font-bold mb-8">
                Built for Modern
                <br />
                Data Scientists
              </h2>
              
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <CheckCircle className="h-6 w-6 text-white mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Intelligent Data Cleaning</h3>
                    <p className="text-gray-400">
                      Automatically detect missing values, outliers, and inconsistencies with smart suggestions.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <CheckCircle className="h-6 w-6 text-white mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Advanced Transformations</h3>
                    <p className="text-gray-400">
                      Feature engineering, encoding, scaling, and dimensionality reduction made simple.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <CheckCircle className="h-6 w-6 text-white mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-lg font-semibold mb-2">AutoML Integration</h3>
                    <p className="text-gray-400">
                      Automated model selection and hyperparameter tuning for optimal performance.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4">
                  <CheckCircle className="h-6 w-6 text-white mt-1 flex-shrink-0" />
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Cloud Deployment</h3>
                    <p className="text-gray-400">
                      Seamless deployment to Azure, AWS, or GCP with monitoring and scaling.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-8">
                <div className="space-y-4">
                  <div className="h-4 bg-white/20 rounded w-3/4"></div>
                  <div className="h-4 bg-white/10 rounded w-1/2"></div>
                  <div className="h-4 bg-white/20 rounded w-full"></div>
                  <div className="h-4 bg-white/10 rounded w-2/3"></div>
                  <div className="grid grid-cols-2 gap-4 pt-4">
                    <div className="h-20 bg-white/10 rounded-xl"></div>
                    <div className="h-20 bg-white/20 rounded-xl"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About/Team Section */}
      <section id="about" className="py-24 relative z-10">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-8">About MLOPT</h2>
          <p className="text-xl text-gray-400 mb-12 leading-relaxed">
            MLOPT is a Final Year Project developed to democratize machine learning by automating 
            the complex data science pipeline. Our platform makes advanced ML accessible to everyone, 
            from beginners to experts.
          </p>
          
          <div className="bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-8">
            <h3 className="text-2xl font-semibold mb-6">Development Team</h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="text-center">
                <h4 className="text-lg font-semibold">Hasan Malik</h4>
                <p className="text-gray-400">SP22-BDS-026</p>
                <p className="text-sm text-gray-500 mt-1">Backend & ML Engineering</p>
              </div>
              <div className="text-center">
                <h4 className="text-lg font-semibold">Jawad Idrees</h4>
                <p className="text-gray-400">SP22-BDS-020</p>
                <p className="text-sm text-gray-500 mt-1">Frontend & UX Design</p>
              </div>
            </div>
            <p className="text-sm text-gray-500 mt-6">
              Bachelor of Data Science | Final Year Project - I | 2024
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative z-10">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Ready to Automate Your
            <br />
            Data Science Workflow?
          </h2>
          <p className="text-xl text-gray-400 mb-8">
            Join the future of machine learning automation. Start building your models today.
          </p>
          
          <Button size="lg" className="bg-white text-black hover:bg-gray-200 text-lg px-12 py-4">
            <Link href="/dashboard" className="flex items-center gap-2">
              Get Started Now
              <ArrowRight className="h-5 w-5" />
            </Link>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-8 relative z-10">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-2xl font-bold mb-4 md:mb-0">MLOPT</div>
            <div className="text-sm text-gray-500">
              © 2024 MLOPT. Built with ❤️ for FYP-I Project.
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}