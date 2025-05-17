"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { CheckCircle, Circle } from "lucide-react"

interface StepperProps extends React.HTMLAttributes<HTMLDivElement> {
  currentStep: number
  children: React.ReactNode
}

interface StepProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
}

interface StepLabelProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode
}

interface StepTitleProps extends React.HTMLAttributes<HTMLHeadingElement> {
  children: React.ReactNode
}

interface StepDescriptionProps extends React.HTMLAttributes<HTMLParagraphElement> {
  children: React.ReactNode
}

export function Stepper({ 
  currentStep = 0, 
  className, 
  children,
  ...props 
}: StepperProps) {
  const steps = React.Children.toArray(children).filter(
    (child) => React.isValidElement(child) && child.type === Step
  ) as React.ReactElement[]
  
  const totalSteps = steps.length
  
  return (
    <div
      className={cn("flex w-full", className)}
      {...props}
    >
      {steps.map((step, index) => {
        const isCompleted = index < currentStep
        const isCurrent = index === currentStep
        const isLast = index === totalSteps - 1
        
        // Clone the step with additional props
        const stepWithProps = React.cloneElement(step, {
          isCompleted,
          isCurrent,
          isLast,
          stepNumber: index + 1,
          ...step.props
        })
        
        return (
          <React.Fragment key={index}>
            {stepWithProps}
            {!isLast && (
              <div className="flex-1 h-0.5 self-center mx-2 bg-gray-200 dark:bg-gray-700">
                <div 
                  className="h-full bg-primary transition-all duration-300 ease-in-out" 
                  style={{ width: isCompleted ? '100%' : '0%' }}
                />
              </div>
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}

export function Step({ 
  isCompleted = false,
  isCurrent = false,
  stepNumber = 1,
  className, 
  children,
  ...props 
}: StepProps & {
  isCompleted?: boolean
  isCurrent?: boolean
  stepNumber?: number
}) {
  // Extract the StepLabel, StepTitle, and StepDescription from children
  const childArray = React.Children.toArray(children)
  
  const title = childArray.find(
    child => React.isValidElement(child) && child.type === StepTitle
  )
  
  const description = childArray.find(
    child => React.isValidElement(child) && child.type === StepDescription
  )
  
  return (
    <div 
      className={cn(
        "flex flex-col items-center relative z-10",
        className
      )}
      {...props}
    >
      <div 
        className={cn(
          "flex items-center justify-center w-8 h-8 rounded-full border transition-colors",
          isCompleted ? "bg-primary text-primary-foreground border-primary" :
          isCurrent ? "bg-primary/10 text-primary border-primary" :
          "bg-muted border-muted-foreground/30"
        )}
      >
        {isCompleted ? (
          <CheckCircle className="w-5 h-5" />
        ) : (
          <span className="text-sm font-medium">{stepNumber}</span>
        )}
      </div>
      <div className="flex flex-col items-center mt-2 space-y-0.5">
        {title}
        {description}
      </div>
    </div>
  )
}

export function StepLabel({ 
  className, 
  children,
  ...props 
}: StepLabelProps) {
  return (
    <div 
      className={cn("text-sm font-medium", className)}
      {...props}
    >
      {children}
    </div>
  )
}

export function StepTitle({ 
  className, 
  children,
  ...props 
}: StepTitleProps) {
  return (
    <h3 
      className={cn("text-sm font-medium text-center", className)}
      {...props}
    >
      {children}
    </h3>
  )
}

export function StepDescription({ 
  className, 
  children,
  ...props 
}: StepDescriptionProps) {
  return (
    <p 
      className={cn("text-xs text-muted-foreground text-center", className)}
      {...props}
    >
      {children}
    </p>
  )
}