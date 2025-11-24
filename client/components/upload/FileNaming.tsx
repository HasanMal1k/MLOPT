// components/upload/FileNaming.tsx
'use client'

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { ArrowLeft, ArrowRight, AlertCircle, CheckCircle2, FileText, Loader2 } from "lucide-react"
import { toast } from "@/hooks/use-toast"

interface FileNamingProps {
  files: File[]
  originalFiles: File[]
  onBack: () => void
  onContinue: (fileNames: Record<string, string>) => void
}

export default function FileNaming({
  files,
  originalFiles,
  onBack,
  onContinue
}: FileNamingProps) {
  const [fileNames, setFileNames] = useState<Record<string, string>>({})
  const [validationStatus, setValidationStatus] = useState<Record<string, { isValid: boolean; message: string }>>({})
  const [isValidating, setIsValidating] = useState(false)
  const [canContinue, setCanContinue] = useState(false)

  // Initialize with original filenames (without extension)
  useEffect(() => {
    const initialNames: Record<string, string> = {}
    originalFiles.forEach(file => {
      const nameWithoutExt = file.name.replace(/\.(csv|xlsx|xls)$/i, '')
      initialNames[file.name] = nameWithoutExt
    })
    setFileNames(initialNames)
  }, [originalFiles])

  // Validate filename uniqueness against database
  const validateFileName = async (originalFileName: string, newFileName: string) => {
    if (!newFileName || newFileName.trim() === '') {
      return { isValid: false, message: 'Filename cannot be empty' }
    }

    // Check for invalid characters
    const invalidChars = /[<>:"/\\|?*]/g
    if (invalidChars.test(newFileName)) {
      return { isValid: false, message: 'Filename contains invalid characters' }
    }

    // Check length
    if (newFileName.length > 100) {
      return { isValid: false, message: 'Filename too long (max 100 characters)' }
    }

    // Check for uniqueness within current batch
    const allNames = Object.entries(fileNames)
      .filter(([key]) => key !== originalFileName)
      .map(([_, value]) => value.toLowerCase())
    
    if (allNames.includes(newFileName.toLowerCase())) {
      return { isValid: false, message: 'Duplicate filename in current upload' }
    }

    // Check against database
    try {
      const response = await fetch(`/api/files/check-name?name=${encodeURIComponent(newFileName)}`)
      
      if (!response.ok) {
        console.error('Filename validation failed:', response.statusText)
        // If database check fails, allow the name (don't block user)
        return { isValid: true, message: 'Unable to verify uniqueness, proceeding anyway' }
      }
      
      const data = await response.json()
      
      if (data.error) {
        console.error('Database error:', data.error)
        // If there's an error, allow the name to proceed
        return { isValid: true, message: 'Unable to verify uniqueness, proceeding anyway' }
      }
      
      if (data.exists) {
        return { isValid: false, message: 'Filename already exists in database' }
      }

      return { isValid: true, message: 'Filename is available' }
    } catch (error) {
      console.error('Error validating filename:', error)
      // On network/other errors, allow the name (graceful degradation)
      return { isValid: true, message: 'Unable to verify uniqueness, proceeding anyway' }
    }
  }

  // Validate all filenames
  const validateAllFileNames = async () => {
    setIsValidating(true)
    const newValidationStatus: Record<string, { isValid: boolean; message: string }> = {}

    for (const [originalName, newName] of Object.entries(fileNames)) {
      const result = await validateFileName(originalName, newName)
      newValidationStatus[originalName] = result
    }

    setValidationStatus(newValidationStatus)
    
    // Check if all are valid
    const allValid = Object.values(newValidationStatus).every(status => status.isValid)
    setCanContinue(allValid)
    setIsValidating(false)

    if (allValid) {
      toast({
        title: "Success",
        description: "All filenames are valid and unique!",
      })
    } else {
      toast({
        title: "Validation Failed",
        description: "Some filenames are invalid or already exist.",
        variant: "destructive"
      })
    }
  }

  const handleFileNameChange = (originalFileName: string, newFileName: string) => {
    setFileNames(prev => ({
      ...prev,
      [originalFileName]: newFileName
    }))
    // Clear validation status for this file when name changes
    setValidationStatus(prev => ({
      ...prev,
      [originalFileName]: { isValid: false, message: '' }
    }))
    setCanContinue(false)
  }

  const handleContinue = () => {
    if (canContinue) {
      onContinue(fileNames)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Name Your Files
          </CardTitle>
          <CardDescription>
            Provide unique names for your processed files before uploading to the database.
            All names must be unique and cannot contain special characters.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Important</AlertTitle>
            <AlertDescription>
              File names must be unique across the entire database. Use descriptive names that help identify your dataset.
            </AlertDescription>
          </Alert>

          <div className="space-y-4">
            {originalFiles.map((file, index) => {
              const fileName = fileNames[file.name] || ''
              const validation = validationStatus[file.name]
              
              return (
                <div key={file.name} className="space-y-2">
                  <Label htmlFor={`filename-${index}`}>
                    File {index + 1}: <span className="text-muted-foreground text-sm">{file.name}</span>
                  </Label>
                  <div className="flex gap-2 items-start">
                    <div className="flex-1">
                      <Input
                        id={`filename-${index}`}
                        value={fileName}
                        onChange={(e) => handleFileNameChange(file.name, e.target.value)}
                        placeholder="Enter unique filename"
                        className={
                          validation?.isValid === false && fileName !== ''
                            ? "border-destructive focus-visible:ring-destructive"
                            : validation?.isValid === true
                            ? "border-green-500 focus-visible:ring-green-500"
                            : ""
                        }
                      />
                      {validation && fileName !== '' && (
                        <p className={`text-sm mt-1 flex items-center gap-1 ${
                          validation.isValid ? "text-green-600" : "text-destructive"
                        }`}>
                          {validation.isValid ? (
                            <CheckCircle2 className="h-3 w-3" />
                          ) : (
                            <AlertCircle className="h-3 w-3" />
                          )}
                          {validation.message}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              onClick={validateAllFileNames}
              disabled={isValidating || Object.values(fileNames).some(name => !name || name.trim() === '')}
              className="flex-1"
            >
              {isValidating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Validating...
                </>
              ) : (
                'Validate File Names'
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <Button 
          onClick={handleContinue}
          disabled={!canContinue}
        >
          Continue to Upload
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
