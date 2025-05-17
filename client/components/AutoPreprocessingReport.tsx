'use client'

import { useEffect, useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import {
  CheckCircle2,
  X,
  Calendar,
  ArrowDownUp,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  FileDown,
  GitBranch,
  Lightbulb,
  Dices
} from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface PreprocessingDetail {
  success: boolean;
  original_shape: [number, number];
  processed_shape: [number, number];
  columns_dropped: string[];
  date_columns_detected: string[];
  columns_cleaned: string[];
  missing_value_stats: Record<string, {
    missing_count: number;
    missing_percentage: number;
    imputation_method: string;
  }>;
  dropped_by_unique_value?: string[];
  engineered_features?: string[];
  transformation_details?: Record<string, any>;
}

interface AutoPreprocessingReportProps {
  processingResults: PreprocessingDetail | null;
  fileName: string;
  isLoading?: boolean;
}

export default function AutoPreprocessingReport({ 
  processingResults, 
  fileName,
  isLoading = false
}: AutoPreprocessingReportProps) {
  const [activeTab, setActiveTab] = useState("summary")
  
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Processing Report</CardTitle>
          <CardDescription>
            Generating preprocessing report for {fileName}...
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center p-6">
          <Progress value={70} className="w-full mb-4" />
          <p className="text-center text-muted-foreground">
            Analyzing data and applying automated preprocessing...
          </p>
        </CardContent>
      </Card>
    )
  }
  
  if (!processingResults) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Processing Report</CardTitle>
          <CardDescription>
            No preprocessing information available for {fileName}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>No Report Available</AlertTitle>
            <AlertDescription>
              Processing information is not available. This might happen if the file was uploaded without preprocessing.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }
  
  // Calculate stats
  const droppedColumnsCount = processingResults.columns_dropped?.length || 0
  const dateColumnsCount = processingResults.date_columns_detected?.length || 0
  const cleanedColumnsCount = processingResults.columns_cleaned?.length || 0
  const totalColumns = processingResults.original_shape?.[1] || 0
  const totalRows = processingResults.original_shape?.[0] || 0
  const resultRows = processingResults.processed_shape?.[0] || 0
  const rowsRemoved = totalRows - resultRows
  const engineeredFeaturesCount = processingResults.engineered_features?.length || 0
  
  // Get column names with highest missing percentage
  const getMostMissingColumns = () => {
    if (!processingResults.missing_value_stats) return [];
    
    return Object.entries(processingResults.missing_value_stats)
      .sort((a, b) => b[1].missing_percentage - a[1].missing_percentage)
      .slice(0, 5);
  }
  
  const mostMissingColumns = getMostMissingColumns()
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Auto Preprocessing Report</CardTitle>
        <CardDescription>
          What happened to your data during automated preprocessing
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full mb-4">
            <TabsTrigger value="summary" className="flex-1">Summary</TabsTrigger>
            <TabsTrigger value="columns" className="flex-1">Column Changes</TabsTrigger>
            <TabsTrigger value="missing" className="flex-1">Missing Values</TabsTrigger>
            <TabsTrigger value="features" className="flex-1">Feature Engineering</TabsTrigger>
          </TabsList>
          
          <TabsContent value="summary">
            <div className="grid gap-4">
              <Alert className="bg-blue-50 text-blue-800 border-blue-200">
                <Lightbulb className="h-4 w-4" />
                <AlertTitle>Automated Intelligence</AlertTitle>
                <AlertDescription>
                  Our system automatically processed your data to make it ready for machine learning. 
                  Here's what happened.
                </AlertDescription>
              </Alert>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{totalColumns}</div>
                    <p className="text-xs text-muted-foreground">Original Columns</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{droppedColumnsCount}</div>
                    <p className="text-xs text-muted-foreground">Columns Dropped</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{totalRows}</div>
                    <p className="text-xs text-muted-foreground">Original Rows</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{rowsRemoved}</div>
                    <p className="text-xs text-muted-foreground">Rows Removed</p>
                  </CardContent>
                </Card>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
                <div className="flex flex-col space-y-2">
                  <p className="text-sm font-medium">Missing Values Handled</p>
                  <div className="flex items-center gap-2">
                    <Progress 
                      value={cleanedColumnsCount / (totalColumns || 1) * 100} 
                      className="h-2" 
                    />
                    <span className="text-sm">{cleanedColumnsCount} columns</span>
                  </div>
                </div>
                
                <div className="flex flex-col space-y-2">
                  <p className="text-sm font-medium">Date Columns Detected</p>
                  <div className="flex items-center gap-2">
                    <Progress 
                      value={dateColumnsCount / (totalColumns || 1) * 100} 
                      className="h-2" 
                    />
                    <span className="text-sm">{dateColumnsCount} columns</span>
                  </div>
                </div>
                
                <div className="flex flex-col space-y-2">
                  <p className="text-sm font-medium">Features Engineered</p>
                  <div className="flex items-center gap-2">
                    <Progress 
                      value={engineeredFeaturesCount / (totalColumns || 1) * 100} 
                      className="h-2" 
                    />
                    <span className="text-sm">{engineeredFeaturesCount} features</span>
                  </div>
                </div>
              </div>
              
              {mostMissingColumns.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium mb-2">Top Columns with Missing Values</h3>
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Column</TableHead>
                          <TableHead>Missing %</TableHead>
                          <TableHead>Imputation Method</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {mostMissingColumns.map(([column, stats]) => (
                          <TableRow key={column}>
                            <TableCell className="font-medium">{column}</TableCell>
                            <TableCell>{stats.missing_percentage}%</TableCell>
                            <TableCell>
                              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                                {stats.imputation_method || "None"}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="columns">
            <div className="grid gap-4">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Column Modifications</h3>
              </div>
              
              <div className="rounded-md border">
                <Accordion type="multiple" className="w-full">
                  {processingResults.columns_dropped && processingResults.columns_dropped.length > 0 && (
                    <AccordionItem value="dropped">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <X className="h-4 w-4 text-red-500" />
                          <span>Dropped Columns ({processingResults.columns_dropped.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {processingResults.columns_dropped.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}
                  
                  {processingResults.columns_cleaned && processingResults.columns_cleaned.length > 0 && (
                    <AccordionItem value="cleaned">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                          <span>Cleaned Columns ({processingResults.columns_cleaned.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {processingResults.columns_cleaned.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}
                  
                  {processingResults.date_columns_detected && processingResults.date_columns_detected.length > 0 && (
                    <AccordionItem value="dates">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <Calendar className="h-4 w-4 text-blue-500" />
                          <span>Date Columns Detected ({processingResults.date_columns_detected.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {processingResults.date_columns_detected.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}
                  
                  {processingResults.dropped_by_unique_value && processingResults.dropped_by_unique_value.length > 0 && (
                    <AccordionItem value="unique">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <ArrowDownUp className="h-4 w-4 text-orange-500" />
                          <span>Dropped by Unique Value ({processingResults.dropped_by_unique_value.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {processingResults.dropped_by_unique_value.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}
                </Accordion>
              </div>
              
              <div className="bg-muted/30 p-4 rounded-md">
                <h4 className="text-sm font-medium mb-2">Why were columns dropped?</h4>
                <ul className="text-sm text-muted-foreground space-y-2">
                  <li>• Columns with all missing values were removed</li>
                  <li>• Columns with over 95% missing values were removed</li>
                  <li>• Columns with just one unique value were removed (no information value)</li>
                  <li>• Duplicated columns were removed</li>
                </ul>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="missing">
            {processingResults.missing_value_stats && Object.keys(processingResults.missing_value_stats).length > 0 ? (
              <div className="grid gap-4">
                <Alert className="border-amber-200 bg-amber-50 text-amber-800">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Missing Data Handling</AlertTitle>
                  <AlertDescription>
                    Missing values can cause problems for machine learning models. We've automatically addressed these issues.
                  </AlertDescription>
                </Alert>
                
                <h3 className="text-sm font-medium">Missing Value Statistics</h3>
                <div className="rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Column</TableHead>
                        <TableHead>Missing Count</TableHead>
                        <TableHead>Missing %</TableHead>
                        <TableHead>Imputation Method</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Object.entries(processingResults.missing_value_stats).map(([column, stats]: [string, any]) => (
                        <TableRow key={column}>
                          <TableCell className="font-medium">{column}</TableCell>
                          <TableCell>{stats.missing_count}</TableCell>
                          <TableCell>{stats.missing_percentage}%</TableCell>
                          <TableCell>
                            <Badge variant="outline" className={stats.imputation_method !== "None" 
                              ? "bg-green-50 text-green-700 border-green-200" 
                              : "bg-gray-50 text-gray-700 border-gray-200"
                            }>
                              {stats.imputation_method || "None"}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                
                <div className="bg-muted/30 p-4 rounded-md">
                  <h4 className="text-sm font-medium mb-2">Imputation Methods Explained</h4>
                  <ul className="text-sm text-muted-foreground space-y-2">
                    <li><span className="font-medium">Mode Imputation:</span> Fills missing values with the most frequent value in the column</li>
                    <li><span className="font-medium">KNN Imputation:</span> Uses K-Nearest Neighbors algorithm to predict missing values based on similar rows</li>
                    <li><span className="font-medium">Mean Imputation:</span> Fills missing values with the average of the column (for numeric data)</li>
                    <li><span className="font-medium">Median Imputation:</span> Fills missing values with the median of the column (for numeric data)</li>
                  </ul>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <CheckCircle2 className="mx-auto h-8 w-8 text-green-500 mb-2" />
                <p className="text-lg font-medium mb-1">No Missing Values</p>
                <p className="text-muted-foreground">Your data doesn't have any missing values - that's great!</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="features">
            {processingResults.engineered_features && processingResults.engineered_features.length > 0 ? (
              <div className="grid gap-4">
                <Alert className="border-purple-200 bg-purple-50 text-purple-800">
                  <Dices className="h-4 w-4" />
                  <AlertTitle>Automated Feature Engineering</AlertTitle>
                  <AlertDescription>
                    We've automatically created new features from your existing data to improve model performance.
                  </AlertDescription>
                </Alert>
                
                <div className="rounded-md border p-4">
                  <h3 className="text-sm font-medium mb-4">New Features Created ({processingResults.engineered_features.length})</h3>
                  <div className="flex flex-wrap gap-2">
                    {processingResults.engineered_features.map((feature, index) => (
                      <Badge key={index} variant="outline" className="bg-purple-50 text-purple-700 border-purple-200">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
                
                {processingResults.transformation_details && (
                  <div className="rounded-md border">
                    <Accordion type="multiple" className="w-full">
                      {processingResults.transformation_details.datetime_features && 
                       processingResults.transformation_details.datetime_features.length > 0 && (
                        <AccordionItem value="datetime">
                          <AccordionTrigger className="px-4">
                            <div className="flex items-center gap-2">
                              <Calendar className="h-4 w-4 text-blue-500" />
                              <span>Date & Time Features</span>
                            </div>
                          </AccordionTrigger>
                          <AccordionContent className="px-4 pb-4">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Source Column</TableHead>
                                  <TableHead>New Features</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {processingResults.transformation_details.datetime_features.map((item: any, index: number) => (
                                  <TableRow key={index}>
                                    <TableCell className="font-medium">{item.source_column}</TableCell>
                                    <TableCell>
                                      <div className="flex flex-wrap gap-1">
                                        {item.derived_features.map((feature: string, fidx: number) => (
                                          <Badge key={fidx} variant="outline" className="bg-blue-50 text-blue-700">
                                            {feature}
                                          </Badge>
                                        ))}
                                      </div>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </AccordionContent>
                        </AccordionItem>
                      )}
                      
                      {processingResults.transformation_details.categorical_encodings && 
                       processingResults.transformation_details.categorical_encodings.length > 0 && (
                        <AccordionItem value="categorical">
                          <AccordionTrigger className="px-4">
                            <div className="flex items-center gap-2">
                              <GitBranch className="h-4 w-4 text-amber-500" />
                              <span>Categorical Encodings</span>
                            </div>
                          </AccordionTrigger>
                          <AccordionContent className="px-4 pb-4">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Source Column</TableHead>
                                  <TableHead>Encoding Type</TableHead>
                                  <TableHead>New Features</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {processingResults.transformation_details.categorical_encodings.map((item: any, index: number) => (
                                  <TableRow key={index}>
                                    <TableCell className="font-medium">{item.source_column}</TableCell>
                                    <TableCell>{item.encoding_type}</TableCell>
                                    <TableCell>
                                      <div className="flex flex-wrap gap-1">
                                        {item.derived_features.map((feature: string, fidx: number) => (
                                          <Badge key={fidx} variant="outline" className="bg-amber-50 text-amber-700">
                                            {feature}
                                          </Badge>
                                        ))}
                                      </div>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </AccordionContent>
                        </AccordionItem>
                      )}
                      
                      {processingResults.transformation_details.numeric_transformations && 
                       processingResults.transformation_details.numeric_transformations.length > 0 && (
                        <AccordionItem value="numeric">
                          <AccordionTrigger className="px-4">
                            <div className="flex items-center gap-2">
                              <ArrowDownUp className="h-4 w-4 text-green-500" />
                              <span>Numeric Transformations</span>
                            </div>
                          </AccordionTrigger>
                          <AccordionContent className="px-4 pb-4">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Source Column</TableHead>
                                  <TableHead>Skew</TableHead>
                                  <TableHead>New Features</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {processingResults.transformation_details.numeric_transformations.map((item: any, index: number) => (
                                  <TableRow key={index}>
                                    <TableCell className="font-medium">{item.source_column}</TableCell>
                                    <TableCell>{item.skew.toFixed(2)}</TableCell>
                                    <TableCell>
                                      <div className="flex flex-wrap gap-1">
                                        {item.derived_features.map((feature: string, fidx: number) => (
                                          <Badge key={fidx} variant="outline" className="bg-green-50 text-green-700">
                                            {feature}
                                          </Badge>
                                        ))}
                                      </div>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </AccordionContent>
                        </AccordionItem>
                      )}
                    </Accordion>
                  </div>
                )}
                
                <div className="bg-muted/30 p-4 rounded-md">
                  <h4 className="text-sm font-medium mb-2">Why Engineer Features?</h4>
                  <p className="text-sm text-muted-foreground mb-2">
                    Feature engineering creates new data columns that help machine learning models learn patterns more effectively.
                  </p>
                  <ul className="text-sm text-muted-foreground space-y-2">
                    <li><span className="font-medium">Date Features:</span> Extract year, month, day etc. to capture seasonal patterns</li>
                    <li><span className="font-medium">One-Hot Encoding:</span> Convert categories to binary columns that ML models can process</li>
                    <li><span className="font-medium">Numeric Transformations:</span> Apply mathematical functions to handle skewed distributions</li>
                  </ul>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <GitBranch className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                <p className="text-lg font-medium mb-1">No Features Engineered</p>
                <p className="text-muted-foreground">No additional features were created for this dataset.</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}