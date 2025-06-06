'use client'

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { signup, signInWithOAuth } from "@/app/(auth)/auth-actions"
import { useState } from "react"
import Link from "next/link"
import { useToast } from "@/hooks/use-toast"

export function SignupForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const { toast } = useToast()
  const [isLoading, setIsLoading] = useState(false)

  async function handleSubmit(formData: FormData) {
    setIsLoading(true)
    
    try {
      const result = await signup(formData)
      
      if (result?.error) {
        toast({
          variant: "destructive",
          title: "Signup failed",
          description: result.error,
        })
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Something went wrong",
        description: "Please try again later.",
      })
    } finally {
      setIsLoading(false)
    }
  }

  async function handleOAuthSignIn(provider: 'google' | 'github') {
    setIsLoading(true)
    
    try {
      const formData = new FormData()
      formData.append('provider', provider)
      
      // Add explicit redirect to auth/callback
      const redirectTo = `${window.location.origin}/auth/callback`
      formData.append('redirectTo', redirectTo)
      
      const result = await signInWithOAuth(formData)
      
      if (result?.error) {
        toast({
          variant: "destructive",
          title: "Signup failed",
          description: result.error,
        })
      } else if (result?.url) {
        // Redirect the user to the provider's login page
        window.location.href = result.url
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Something went wrong",
        description: "Please try again later.",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card>
        <CardHeader className="text-center">
          <CardTitle className="text-xl">Signup to MLOPT</CardTitle>
          <CardDescription>
            Signin with your Github or Google account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form action={handleSubmit}>
            <div className="grid gap-6">
              {/* OAuth buttons */}
              <div className="flex flex-col gap-4">
                {/* <Button 
                  variant="outline" 
                  className="w-full" 
                  onClick={() => handleOAuthSignIn('github')}
                  type="button"
                  disabled={isLoading}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="26" height="26" fill="currentColor">
                    <path d="M12 2C6.477 2 2 6.485 2 12c0 4.418 2.865 8.167 6.839 9.49.5.093.682-.217.682-.483 0-.237-.009-.868-.014-1.703-2.782.603-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.462-1.11-1.462-.907-.62.069-.608.069-.608 1.004.071 1.532 1.033 1.532 1.033.89 1.528 2.335 1.087 2.902.832.09-.647.348-1.087.635-1.337-2.22-.252-4.555-1.113-4.555-4.951 0-1.092.39-1.986 1.032-2.684-.103-.253-.448-1.27.098-2.646 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.503.336 1.909-1.294 2.748-1.025 2.748-1.025.548 1.376.203 2.393.1 2.646.644.698 1.03 1.592 1.03 2.684 0 3.849-2.338 4.696-4.566 4.944.358.308.678.916.678 1.846 0 1.334-.012 2.41-.012 2.735 0 .268.18.579.688.481A10.017 10.017 0 0 0 22 12c0-5.515-4.477-10-10-10z"/>
                  </svg>
                  Login with Github
                </Button> */}
                <Button 
                  variant="outline" 
                  className="w-full" 
                  onClick={() => handleOAuthSignIn('google')}
                  type="button"
                  disabled={isLoading}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <path
                      d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z"
                      fill="currentColor"
                    />
                  </svg>
                  Login with Google
                </Button>
              </div>
              
              <div className="relative text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t after:border-border">
                <span className="relative z-10 bg-background px-2 text-muted-foreground">
                  Or continue with
                </span>
              </div>
              
              <div className="grid gap-6">
                <div className="grid gap-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    name="email"
                    placeholder="m@example.com"
                    required
                  />
                </div>
                <div className="grid gap-2">
                  <div className="flex items-center">
                    <Label htmlFor="password">Password</Label>
                  </div>
                  <Input id="password" name="password" type="password" required />
                </div>
                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? "Creating account..." : "Sign up"}
                </Button>
              </div>
              <div className="text-center text-sm">
                Already have an account?{" "}
                <Link href="/login" className="underline underline-offset-4">
                  Login
                </Link>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>
      <div className="text-balance text-center text-xs text-muted-foreground [&_a]:underline [&_a]:underline-offset-4 [&_a]:hover:text-primary">
        By clicking continue, you agree to our <a href="#">Terms of Service</a>{" "}
        and <a href="#">Privacy Policy</a>.
      </div>
    </div>
  )
}