'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  BookOpen, 
  Play, 
  Sparkles, 
  Brain, 
  TrendingUp,
  BarChart3,
  Settings,
  Calendar,
  GitBranch,
  Home
} from "lucide-react"

interface Tutorial {
  id: string
  title: string
  description: string
  videoId: string
  icon: React.ElementType
  color: string
  duration?: string
}

const tutorials: Tutorial[] = [
  {
    id: 'landing-login',
    title: 'Landing Page & Login',
    description: 'Get started with MLOPT - Learn how to sign up, log in, and navigate the dashboard',
    videoId: 'tw88arY6B1o',
    icon: Home,
    color: 'text-blue-500',
    duration: '5:30'
  },
  {
    id: 'auto-custom-preprocessing',
    title: 'Auto & Custom Preprocessing',
    description: 'Learn how to automatically clean your data and apply custom preprocessing steps',
    videoId: 'Y14MTUuC3y4',
    icon: Sparkles,
    color: 'text-purple-500',
    duration: '8:45'
  },
  {
    id: 'transformations',
    title: 'Transformations Activity',
    description: 'Discover how to transform your data with log, sqrt, binning, and encoding operations',
    videoId: 'znlODwYKlrI',
    icon: GitBranch,
    color: 'text-green-500',
    duration: '7:20'
  },
  {
    id: 'eda-chart-builder',
    title: 'EDA & Chart Builder',
    description: 'Explore your data visually with our interactive chart builder and EDA tools',
    videoId: 'mAPa38sAR0I',
    icon: BarChart3,
    color: 'text-amber-500',
    duration: '6:15'
  },
  {
    id: 'time-series-preprocessing',
    title: 'Time Series Preprocessing',
    description: 'Handle date parsing, frequency detection, and time series specific preprocessing',
    videoId: '0VqBUZTtYYs',
    icon: Calendar,
    color: 'text-cyan-500',
    duration: '9:30'
  },
  {
    id: 'auto-training-classification',
    title: 'Auto Training - Classification',
    description: 'Train classification models automatically with hyperparameter optimization',
    videoId: 'vTEQ2c_OuPY',
    icon: Brain,
    color: 'text-red-500',
    duration: '10:45'
  },
  {
    id: 'auto-training-regression',
    title: 'Auto Training - Regression',
    description: 'Build regression models with automated feature selection and model comparison',
    videoId: 'fBgMuqsSgB0',
    icon: TrendingUp,
    color: 'text-pink-500',
    duration: '11:20'
  },
  {
    id: 'auto-training-time-series',
    title: 'Auto Training - Time Series',
    description: 'Train time series forecasting models with ARIMA, Prophet, and LSTM',
    videoId: 'tNdKW_StDAQ',
    icon: TrendingUp,
    color: 'text-indigo-500',
    duration: '12:10'
  },
  {
    id: 'blueprint-classification',
    title: 'Blueprint Design - Classification',
    description: 'Design complete ML pipelines for classification problems with our blueprint system',
    videoId: '9ICbWWXQIF4',
    icon: Settings,
    color: 'text-orange-500',
    duration: '8:55'
  },
  {
    id: 'blueprint-regression',
    title: 'Blueprint Design - Regression',
    description: 'Create end-to-end regression pipelines with feature engineering and model selection',
    videoId: '1KXqaFrR6HY',
    icon: Settings,
    color: 'text-teal-500',
    duration: '9:40'
  }
]

export default function TutorialsPage() {
  return (
    <section className="h-screen w-full px-6 md:px-10 py-10 overflow-y-auto">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <BookOpen className="h-8 w-8 text-primary" />
            <h1 className="text-4xl font-bold">Video Tutorials</h1>
          </div>
          <p className="text-muted-foreground text-lg">
            Master MLOPT with our comprehensive video guides covering all features
          </p>
        </div>

        {/* Stats Banner */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Tutorials</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{tutorials.length}</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Topics Covered</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">5</div>
              <p className="text-xs text-muted-foreground mt-1">Preprocessing, Training, Analysis, Design & More</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Duration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">~90 min</div>
              <p className="text-xs text-muted-foreground mt-1">Complete walkthrough of all features</p>
            </CardContent>
          </Card>
        </div>

        {/* Tutorial Grid */}
        <div className="space-y-8">
          {tutorials.map((tutorial) => {
            const Icon = tutorial.icon
            return (
              <Card key={tutorial.id} className="overflow-hidden">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`h-12 w-12 rounded-lg bg-muted flex items-center justify-center`}>
                        <Icon className={`h-6 w-6 ${tutorial.color}`} />
                      </div>
                      <div>
                        <CardTitle className="text-xl">{tutorial.title}</CardTitle>
                        <CardDescription className="mt-1">{tutorial.description}</CardDescription>
                      </div>
                    </div>
                    {tutorial.duration && (
                      <Badge variant="secondary" className="flex items-center gap-1">
                        <Play className="h-3 w-3" />
                        {tutorial.duration}
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>
                    <iframe
                      className="absolute top-0 left-0 w-full h-full rounded-lg"
                      src={`https://www.youtube.com/embed/${tutorial.videoId}`}
                      title={tutorial.title}
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                    />
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Footer */}
        <Card className="mt-8 bg-muted/50">
          <CardContent className="py-6 text-center">
            <p className="text-muted-foreground">
              Have questions? Check out our{' '}
              <a href="/dashboard" className="text-primary hover:underline">
                documentation
              </a>{' '}
              or reach out to support.
            </p>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}
