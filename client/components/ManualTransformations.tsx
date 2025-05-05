"use client"

import { useState } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  ArrowUpDown,
  Calendar,
  GitBranch,
  Code,
  SlidersHorizontal,
  Filter,
  Trash2,
  Check,
  X
} from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface ColumnInfo {
  name: string;
  type: string;
  sample_values: string[];
}

interface TransformationConfig {
  log_transform: string[];
  sqrt_transform: string[];
  squared_transform: string[];
  reciprocal_transform: string[];
  binning: {column: string; bins: number; labels?: string[]}[];
  one_hot_encoding: string[];
  datetime_features: {column: string; features: string[]}[];
}

interface ManualTransformationsProps {
  columns: ColumnInfo[];
  onApplyTransformations: (config: TransformationConfig) => void;
  isLoading: boolean;
}

export default function ManualTransformations({ 
  columns, 
  onApplyTransformations,
  isLoading
}: ManualTransformationsProps) {
  const [activeTab, setActiveTab] = useState<string>("numeric")
  const [transformationConfig, setTransformationConfig] = useState<TransformationConfig>({
    log_transform: [],
    sqrt_transform: [],
    squared_transform: [],
    reciprocal_transform: [],
    binning: [],
    one_hot_encoding: [],
    datetime_features: []
  })
  
  // Filter columns by type
  const numericColumns = columns.filter(col => 
    col.type === 'numeric' || col.type === 'int64' || col.type === 'float64' || 
    col.type === 'integer' || col.type === 'float'
  )
  
  const categoricalColumns = columns.filter(col => 
    col.type === 'categorical' || col.type === 'object' || col.type === 'string' || 
    col.type === 'category'
  )
  
  const dateColumns = columns.filter(col => 
    col.type === 'datetime' || col.type === 'datetime64[ns]' || 
    col.type === 'date' || col.name.toLowerCase().includes('date')
  )
  
  // Helper function to toggle a column in a transformation array
  const toggleColumnInTransformation = (transformType: keyof TransformationConfig, columnName: string) => {
    setTransformationConfig(prev => {
      // Handle different transformation types
      if (transformType === 'binning') {
        const exists = prev.binning.some(item => item.column === columnName)
        
        if (exists) {
          return {
            ...prev,
            binning: prev.binning.filter(item => item.column !== columnName)
          }
        } else {
          return {
            ...prev,
            binning: [...prev.binning, {column: columnName, bins: 4}]
          }
        }
      } else if (transformType === 'datetime_features') {
        const exists = prev.datetime_features.some(item => item.column === columnName)
        
        if (exists) {
          return {
            ...prev,
            datetime_features: prev.datetime_features.filter(item => item.column !== columnName)
          }
        } else {
          return {
            ...prev,
            datetime_features: [...prev.datetime_features, {
              column: columnName, 
              features: ["year", "month", "day", "dayofweek"]
            }]
          }
        }
      } else {
        // For array type transformations like log_transform
        const array = prev[transformType] as string[]
        
        if (array.includes(columnName)) {
          return {
            ...prev,
            [transformType]: array.filter(col => col !== columnName)
          }
        } else {
          return {
            ...prev,
            [transformType]: [...array, columnName]
          }
        }
      }
    })
  }
  
  // Helper function to update bins count for a column
  const updateBinCount = (columnName: string, binCount: number) => {
    setTransformationConfig(prev => {
      const binIndex = prev.binning.findIndex(item => item.column === columnName)
      
      if (binIndex === -1) return prev
      
      const newBinning = [...prev.binning]
      newBinning[binIndex] = { ...newBinning[binIndex], bins: binCount }
      
      return {
        ...prev,
        binning: newBinning
      }
    })
  }
  
  // Helper function to toggle a datetime feature
  const toggleDatetimeFeature = (columnName: string, feature: string) => {
    setTransformationConfig(prev => {
      const dateIndex = prev.datetime_features.findIndex(item => item.column === columnName)
      
      if (dateIndex === -1) return prev
      
      const newDateFeatures = [...prev.datetime_features]
      const currentFeatures = newDateFeatures[dateIndex].features
      
      if (currentFeatures.includes(feature)) {
        newDateFeatures[dateIndex] = {
          ...newDateFeatures[dateIndex],
          features: currentFeatures.filter(f => f !== feature)
        }
      } else {
        newDateFeatures[dateIndex] = {
          ...newDateFeatures[dateIndex],
          features: [...currentFeatures, feature]
        }
      }
      
      return {
        ...prev,
        datetime_features: newDateFeatures
      }
    })
  }
  
  // Check if a column is selected for a transformation
  const isColumnSelected = (transformType: keyof TransformationConfig, columnName: string) => {
    if (transformType === 'binning') {
      return transformationConfig.binning.some(item => item.column === columnName)
    } else if (transformType === 'datetime_features') {
      return transformationConfig.datetime_features.some(item => item.column === columnName)
    } else {
      const array = transformationConfig[transformType] as string[]
      return array.includes(columnName)
    }
  }
  
  // Check if a date feature is selected
  const isDateFeatureSelected = (columnName: string, feature: string) => {
    const dateFeature = transformationConfig.datetime_features.find(item => item.column === columnName)
    return dateFeature?.features.includes(feature) || false
  }
  
  // Get bin count for a column
  const getBinCount = (columnName: string) => {
    const binConfig = transformationConfig.binning.find(item => item.column === columnName)
    return binConfig?.bins || 4
  }
  
  // Handle apply transformations
  const handleApply = () => {
    onApplyTransformations(transformationConfig)
  }
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Manual Feature Transformations</CardTitle>
        <CardDescription>
          Select specific transformations to apply to your columns
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full mb-4">
            <TabsTrigger value="numeric" className="flex-1">
              <ArrowUpDown className="mr-2 h-4 w-4" />
              Numeric Transformations
            </TabsTrigger>
            <TabsTrigger value="categorical" className="flex-1">
              <Code className="mr-2 h-4 w-4" />
              Categorical Encodings
            </TabsTrigger>
            <TabsTrigger value="datetime" className="flex-1">
              <Calendar className="mr-2 h-4 w-4" />
              DateTime Features
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="numeric">
            <div className="space-y-4">
              <div className="bg-muted/50 p-4 rounded-md">
                <p className="text-sm">
                  Select transformations to apply to numeric columns. These transformations can help normalize skewed data and create features that better represent the data distribution.
                </p>
              </div>
              
              <div className="grid gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-2">Logarithmic Transformation</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Apply log(x+1) to handle positively skewed data. Good for variables with exponential growth patterns.
                  </p>
                  
                  <ScrollArea className="h-[200px] border rounded-md p-4">
                    <div className="space-y-2">
                      {numericColumns.map(column => (
                        <div key={`log-${column.name}`} className="flex items-center space-x-2">
                          <Checkbox 
                            id={`log-${column.name}`}
                            checked={isColumnSelected('log_transform', column.name)}
                            onCheckedChange={() => toggleColumnInTransformation('log_transform', column.name)}
                          />
                          <label 
                            htmlFor={`log-${column.name}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                          >
                            {column.name}
                          </label>
                        </div>
                      ))}
                      
                      {numericColumns.length === 0 && (
                        <p className="text-sm text-muted-foreground">No numeric columns available.</p>
                      )}
                    </div>
                  </ScrollArea>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Square Root Transformation</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Apply sqrt(x) to moderate positive skew. Less aggressive than log transformation.
                  </p>
                  
                  <ScrollArea className="h-[200px] border rounded-md p-4">
                    <div className="space-y-2">
                      {numericColumns.map(column => (
                        <div key={`sqrt-${column.name}`} className="flex items-center space-x-2">
                          <Checkbox 
                            id={`sqrt-${column.name}`}
                            checked={isColumnSelected('sqrt_transform', column.name)}
                            onCheckedChange={() => toggleColumnInTransformation('sqrt_transform', column.name)}
                          />
                          <label 
                            htmlFor={`sqrt-${column.name}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                          >
                            {column.name}
                          </label>
                        </div>
                      ))}
                      
                      {numericColumns.length === 0 && (
                        <p className="text-sm text-muted-foreground">No numeric columns available.</p>
                      )}
                    </div>
                  </ScrollArea>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Square Transformation</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Apply x² to handle negative skew or emphasize larger values.
                  </p>
                  
                  <ScrollArea className="h-[200px] border rounded-md p-4">
                    <div className="space-y-2">
                      {numericColumns.map(column => (
                        <div key={`squared-${column.name}`} className="flex items-center space-x-2">
                          <Checkbox 
                            id={`squared-${column.name}`}
                            checked={isColumnSelected('squared_transform', column.name)}
                            onCheckedChange={() => toggleColumnInTransformation('squared_transform', column.name)}
                          />
                          <label 
                            htmlFor={`squared-${column.name}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                          >
                            {column.name}
                          </label>
                        </div>
                      ))}
                      
                      {numericColumns.length === 0 && (
                        <p className="text-sm text-muted-foreground">No numeric columns available.</p>
                      )}
                    </div>
                  </ScrollArea>
                </div>
                
                <div>
                  <h3 className="text-lg font-medium mb-2">Binning (Discretization)</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Convert continuous variables into categorical bins. Useful for capturing non-linear effects.
                  </p>
                  
                  <ScrollArea className="h-[280px] border rounded-md p-4">
                    <div className="space-y-4">
                      {numericColumns.map(column => (
                        <div key={`bin-${column.name}`} className="space-y-2">
                          <div className="flex items-center space-x-2">
                            <Checkbox 
                              id={`bin-${column.name}`}
                              checked={isColumnSelected('binning', column.name)}
                              onCheckedChange={() => toggleColumnInTransformation('binning', column.name)}
                            />
                            <label 
                              htmlFor={`bin-${column.name}`}
                              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                            >
                              {column.name}
                            </label>
                          </div>
                          
                          {isColumnSelected('binning', column.name) && (
                            <div className="ml-6 mt-2 grid grid-cols-2 gap-2">
                              <div>
                                <Label htmlFor={`bins-${column.name}`} className="text-xs">Number of bins</Label>
                                <Select
                                  value={getBinCount(column.name).toString()}
                                  onValueChange={(value) => updateBinCount(column.name, parseInt(value))}
                                >
                                  <SelectTrigger id={`bins-${column.name}`} className="h-8">
                                    <SelectValue placeholder="Select bin count" />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectItem value="2">2 bins</SelectItem>
                                    <SelectItem value="3">3 bins</SelectItem>
                                    <SelectItem value="4">4 bins (quartiles)</SelectItem>
                                    <SelectItem value="5">5 bins (quintiles)</SelectItem>
                                    <SelectItem value="10">10 bins (deciles)</SelectItem>
                                  </SelectContent>
                                </Select>
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                      
                      {numericColumns.length === 0 && (
                        <p className="text-sm text-muted-foreground">No numeric columns available.</p>
                      )}
                    </div>
                  </ScrollArea>
                </div>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="categorical">
            <div className="space-y-4">
              <div className="bg-muted/50 p-4 rounded-md">
                <p className="text-sm">
                  Convert categorical variables into numerical features for machine learning. One-hot encoding creates binary columns for each category value.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">One-Hot Encoding</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Creates binary (0/1) columns for each category. Best for nominal variables with few unique values.
                </p>
                
                <ScrollArea className="h-[280px] border rounded-md p-4">
                  <div className="space-y-2">
                    {categoricalColumns.map(column => (
                      <div key={`onehot-${column.name}`} className="flex items-center space-x-2">
                        <Checkbox 
                          id={`onehot-${column.name}`}
                          checked={isColumnSelected('one_hot_encoding', column.name)}
                          onCheckedChange={() => toggleColumnInTransformation('one_hot_encoding', column.name)}
                        />
                        <label 
                          htmlFor={`onehot-${column.name}`}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                        >
                          {column.name}
                        </label>
                      </div>
                    ))}
                    
                    {categoricalColumns.length === 0 && (
                      <p className="text-sm text-muted-foreground">No categorical columns available.</p>
                    )}
                  </div>
                </ScrollArea>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="datetime">
            <div className="space-y-4">
              <div className="bg-muted/50 p-4 rounded-md">
                <p className="text-sm">
                  Extract useful components from date and time columns. These features can help capture patterns related to time.
                </p>
              </div>
              
              <div>
                <h3 className="text-lg font-medium mb-2">Extract Date Components</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Create separate features for year, month, day, etc. from datetime columns.
                </p>
                
                <ScrollArea className="h-[320px] border rounded-md p-4">
                  <div className="space-y-6">
                    {dateColumns.map(column => (
                      <div key={`date-${column.name}`} className="space-y-3">
                        <div className="flex items-center space-x-2">
                          <Checkbox 
                            id={`date-${column.name}`}
                            checked={isColumnSelected('datetime_features', column.name)}
                            onCheckedChange={() => toggleColumnInTransformation('datetime_features', column.name)}
                          />
                          <label 
                            htmlFor={`date-${column.name}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                          >
                            {column.name}
                          </label>
                        </div>
                        
                        {isColumnSelected('datetime_features', column.name) && (
                          <div className="ml-6 grid grid-cols-2 gap-x-4 gap-y-2">
                            <div className="flex items-center space-x-2">
                              <Checkbox 
                                id={`year-${column.name}`}
                                checked={isDateFeatureSelected(column.name, 'year')}
                                onCheckedChange={() => toggleDatetimeFeature(column.name, 'year')}
                              />
                              <label 
                                htmlFor={`year-${column.name}`}
                                className="text-xs leading-none"
                              >
                                Year
                              </label>
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <Checkbox 
                                id={`month-${column.name}`}
                                checked={isDateFeatureSelected(column.name, 'month')}
                                onCheckedChange={() => toggleDatetimeFeature(column.name, 'month')}
                              />
                              <label 
                                htmlFor={`month-${column.name}`}
                                className="text-xs leading-none"
                              >
                                Month
                              </label>
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <Checkbox 
                                id={`day-${column.name}`}
                                checked={isDateFeatureSelected(column.name, 'day')}
                                onCheckedChange={() => toggleDatetimeFeature(column.name, 'day')}
                              />
                              <label 
                                htmlFor={`day-${column.name}`}
                                className="text-xs leading-none"
                              >
                                Day
                              </label>
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <Checkbox 
                                id={`dayofweek-${column.name}`}
                                checked={isDateFeatureSelected(column.name, 'dayofweek')}
                                onCheckedChange={() => toggleDatetimeFeature(column.name, 'dayofweek')}
                              />
                              <label 
                                htmlFor={`dayofweek-${column.name}`}
                                className="text-xs leading-none"
                              >
                                Day of Week
                              </label>
                            </div>
                            
                            <div className="flex items-center space-x-2">
                              <Checkbox 
                                id={`quarter-${column.name}`}
                                checked={isDateFeatureSelected(column.name, 'quarter')}
                                onCheckedChange={() => toggleDatetimeFeature(column.name, 'quarter')}
                              />
                              <label 
                                htmlFor={`quarter-${column.name}`}
                                className="text-xs leading-none"
                              >
                                Quarter
                              </label>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                    
                    {dateColumns.length === 0 && (
                      <p className="text-sm text-muted-foreground">No date/time columns available or detected.</p>
                    )}
                  </div>
                </ScrollArea>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-end">
        <Button onClick={handleApply} disabled={isLoading}>
          {isLoading ? "Processing..." : "Apply Transformations"}
        </Button>
      </CardFooter>
    </Card>
  )
}