"use client"

import { useState } from "react"
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
  CheckCircle2,
  X,
  Calendar,
  ArrowDownUp,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  FileDown,
  GitBranch
} from "lucide-react"
import { Progress } from "@/components/ui/progress"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"

interface MissingValueStat {
  missing_count: number;
  missing_percentage: number;
  imputation_method: string;
}

interface PreprocessingReportData {
  columns_dropped?: string[];
  date_columns_detected?: string[];
  columns_cleaned?: string[];
  original_shape?: [number, number];
  processed_shape?: [number, number];
  missing_value_stats?: Record<string, MissingValueStat>;
  dropped_by_unique_value?: string[];
  engineered_features?: string[];
  transformation_details?: Record<string, any>;
}

interface PreprocessingReportProps {
  report: PreprocessingReportData;
  onDownload?: () => void;
}
export default function PreprocessingReport({ report, onDownload }: PreprocessingReportProps) {
  const [activeTab, setActiveTab] = useState("summary")
  
  // Calculate stats
  const droppedColumnsCount = report.columns_dropped?.length || 0
  const dateColumnsCount = report.date_columns_detected?.length || 0
  const cleanedColumnsCount = report.columns_cleaned?.length || 0
  const totalColumns = (report.original_shape?.[1] || 0)
  const totalRows = report.original_shape?.[0] || 0
  const resultRows = report.processed_shape?.[0] || 0
  const rowsRemoved = totalRows - resultRows
  
  // Get column names with highest missing percentage
  const getMostMissingColumns = () => {
    if (!report.missing_value_stats) return [];
    
    return Object.entries(report.missing_value_stats)
      .sort((a, b) => b[1].missing_percentage - a[1].missing_percentage)
      .slice(0, 5);
  }
  
  const mostMissingColumns = getMostMissingColumns()
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Preprocessing Report</CardTitle>
        <CardDescription>
          Detailed report of the preprocessing operations performed on your data
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full mb-4">
            <TabsTrigger value="summary" className="flex-1">Summary</TabsTrigger>
            <TabsTrigger value="columns" className="flex-1">Columns</TabsTrigger>
            <TabsTrigger value="missing" className="flex-1">Missing Values</TabsTrigger>
            <TabsTrigger value="dates" className="flex-1">Date Detection</TabsTrigger>
          </TabsList>
          
          <TabsContent value="summary">
            <div className="grid gap-4">
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
                      value={cleanedColumnsCount / totalColumns * 100} 
                      className="h-2" 
                    />
                    <span className="text-sm">{cleanedColumnsCount} columns</span>
                  </div>
                </div>
                
                <div className="flex flex-col space-y-2">
                  <p className="text-sm font-medium">Columns Dropped</p>
                  <div className="flex items-center gap-2">
                    <Progress 
                      value={droppedColumnsCount / totalColumns * 100} 
                      className="h-2" 
                    />
                    <span className="text-sm">{droppedColumnsCount} columns</span>
                  </div>
                </div>
                
                <div className="flex flex-col space-y-2">
                  <p className="text-sm font-medium">Date Columns Detected</p>
                  <div className="flex items-center gap-2">
                    <Progress 
                      value={dateColumnsCount / totalColumns * 100} 
                      className="h-2" 
                    />
                    <span className="text-sm">{dateColumnsCount} columns</span>
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
              
              <div className="flex justify-end mt-4">
                <Button variant="outline" onClick={onDownload}>
                  <FileDown className="mr-2 h-4 w-4" /> Download Report
                </Button>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="columns">
            <div className="grid gap-4">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Column Modifications</h3>
              </div>
              
              <div className="rounded-md border">
                <Accordion type="multiple" className="w-full">
                  {report.columns_dropped && report.columns_dropped.length > 0 && (
                    <AccordionItem value="dropped">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <X className="h-4 w-4 text-red-500" />
                          <span>Dropped Columns ({report.columns_dropped.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {report.columns_dropped.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}
                  
                  {report.columns_cleaned && report.columns_cleaned.length > 0 && (
                    <AccordionItem value="cleaned">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                          <span>Cleaned Columns ({report.columns_cleaned.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {report.columns_cleaned.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}
                  
                  {report.date_columns_detected && report.date_columns_detected.length > 0 && (
                    <AccordionItem value="dates">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <Calendar className="h-4 w-4 text-blue-500" />
                          <span>Date Columns Detected ({report.date_columns_detected.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {report.date_columns_detected.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}
                  
                  {report.dropped_by_unique_value && report.dropped_by_unique_value.length > 0 && (
                    <AccordionItem value="unique">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <ArrowDownUp className="h-4 w-4 text-orange-500" />
                          <span>Dropped by Unique Value ({report.dropped_by_unique_value.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {report.dropped_by_unique_value.map((column: string) => (
                              <div key={column} className="py-1 px-2 rounded-md bg-muted/50">
                                {column}
                              </div>
                            ))}
                          </div>
                        </ScrollArea>
                      </AccordionContent>
                    </AccordionItem>
                  )}

                  {report.engineered_features && report.engineered_features.length > 0 && (
                    <AccordionItem value="engineered">
                      <AccordionTrigger className="px-4">
                        <div className="flex items-center gap-2">
                          <GitBranch className="h-4 w-4 text-indigo-500" />
                          <span>Engineered Features ({report.engineered_features.length})</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent className="px-4 pb-4">
                        <ScrollArea className="h-[200px]">
                          <div className="space-y-1">
                            {report.engineered_features.map((column: string) => (
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
            </div>
          </TabsContent>
          
          <TabsContent value="missing">
            {report.missing_value_stats && Object.keys(report.missing_value_stats).length > 0 ? (
              <div className="grid gap-4">
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
                      {Object.entries(report.missing_value_stats).map(([column, stats]: [string, any]) => (
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
              </div>
            ) : (
              <div className="text-center py-8">
                <AlertTriangle className="mx-auto h-8 w-8 text-amber-500 mb-2" />
                <p>No missing values information available</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="dates">
            {report.date_columns_detected && report.date_columns_detected.length > 0 ? (
              <div className="grid gap-4">
                <h3 className="text-sm font-medium">Date/Time Columns</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  The system automatically detected and converted the following columns to date/time format:
                </p>
                <div className="rounded-md border p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {report.date_columns_detected.map((column: string) => (
                      <div key={column} className="flex items-center gap-2 bg-muted/50 p-2 rounded-md">
                        <Calendar className="h-4 w-4 text-blue-500" />
                        <span>{column}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-blue-50 rounded-md p-4 mt-2">
                  <p className="text-sm text-blue-800">
                    <strong>Note:</strong> Date columns can be used for time-based analysis and feature engineering.
                    Visit the Feature Engineering page to extract year, month, day components from these columns.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <Calendar className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                <p>No date columns were detected in this dataset</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}