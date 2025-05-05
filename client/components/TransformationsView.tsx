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
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import {
  GitBranch,
  Calendar,
  ArrowUpDown,
  Code,
  SlidersHorizontal,
  Binary
} from "lucide-react"

interface TransformationDetails {
  datetime_features?: any[];
  categorical_encodings?: any[];
  numeric_transformations?: any[];
  binned_features?: any[];
}

interface TransformationsViewProps {
  engineeredFeatures?: string[];
  transformationDetails?: TransformationDetails;
}

export default function TransformationsView({ 
  engineeredFeatures = [], 
  transformationDetails = {} 
}: TransformationsViewProps) {
  const [activeTab, setActiveTab] = useState<string>("summary")
  
  // Group features by type for display
  const datetimeFeatures: string[] = [];
  const categoricalFeatures: string[] = [];
  const numericFeatures: string[] = [];
  const binnedFeatures: string[] = [];
  
  // Populate from transformation details if available
  if (transformationDetails.datetime_features) {
    transformationDetails.datetime_features.forEach(item => {
      if (item.derived_features) {
        datetimeFeatures.push(...item.derived_features);
      }
    });
  }
  
  if (transformationDetails.categorical_encodings) {
    transformationDetails.categorical_encodings.forEach(item => {
      if (item.derived_features) {
        categoricalFeatures.push(...item.derived_features);
      }
    });
  }
  
  if (transformationDetails.numeric_transformations) {
    transformationDetails.numeric_transformations.forEach(item => {
      if (item.derived_features) {
        numericFeatures.push(...item.derived_features);
      }
    });
  }
  
  if (transformationDetails.binned_features) {
    transformationDetails.binned_features.forEach(item => {
      if (item.derived_feature) {
        binnedFeatures.push(item.derived_feature);
      }
    });
  }
  
  // For features that don't have detailed transformation info
  const unknownFeatures = engineeredFeatures.filter(f => 
    !datetimeFeatures.includes(f) && 
    !categoricalFeatures.includes(f) && 
    !numericFeatures.includes(f) && 
    !binnedFeatures.includes(f)
  );
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Automated Feature Transformations</CardTitle>
        <CardDescription>
          Features automatically generated during preprocessing
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="w-full mb-4">
            <TabsTrigger value="summary" className="flex-1">Summary</TabsTrigger>
            <TabsTrigger value="datetime" className="flex-1">DateTime</TabsTrigger>
            <TabsTrigger value="categorical" className="flex-1">Categorical</TabsTrigger>
            <TabsTrigger value="numeric" className="flex-1">Numeric</TabsTrigger>
          </TabsList>
          
          <TabsContent value="summary">
            <div className="grid gap-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{engineeredFeatures.length}</div>
                    <p className="text-xs text-muted-foreground">Total Engineered Features</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{datetimeFeatures.length}</div>
                    <p className="text-xs text-muted-foreground">DateTime Features</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{categoricalFeatures.length}</div>
                    <p className="text-xs text-muted-foreground">Categorical Encodings</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="pt-4">
                    <div className="text-2xl font-bold">{numericFeatures.length + binnedFeatures.length}</div>
                    <p className="text-xs text-muted-foreground">Numeric Transformations</p>
                  </CardContent>
                </Card>
              </div>
              
              {engineeredFeatures.length > 0 ? (
                <div className="rounded-md border mt-4">
                  <Accordion type="single" collapsible className="w-full">
                    {datetimeFeatures.length > 0 && (
                      <AccordionItem value="datetime">
                        <AccordionTrigger className="px-4">
                          <div className="flex items-center gap-2">
                            <Calendar className="h-4 w-4 text-blue-500" />
                            <span>DateTime Features ({datetimeFeatures.length})</span>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="px-4 pb-4">
                          <div className="flex flex-wrap gap-2">
                            {datetimeFeatures.map(feature => (
                              <Badge key={feature} variant="secondary" className="bg-blue-50 text-blue-700">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    )}
                    
                    {categoricalFeatures.length > 0 && (
                      <AccordionItem value="categorical">
                        <AccordionTrigger className="px-4">
                          <div className="flex items-center gap-2">
                            <Code className="h-4 w-4 text-purple-500" />
                            <span>Categorical Encodings ({categoricalFeatures.length})</span>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="px-4 pb-4">
                          <div className="flex flex-wrap gap-2">
                            {categoricalFeatures.map(feature => (
                              <Badge key={feature} variant="secondary" className="bg-purple-50 text-purple-700">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    )}
                    
                    {numericFeatures.length > 0 && (
                      <AccordionItem value="numeric">
                        <AccordionTrigger className="px-4">
                          <div className="flex items-center gap-2">
                            <ArrowUpDown className="h-4 w-4 text-green-500" />
                            <span>Numeric Transformations ({numericFeatures.length})</span>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="px-4 pb-4">
                          <div className="flex flex-wrap gap-2">
                            {numericFeatures.map(feature => (
                              <Badge key={feature} variant="secondary" className="bg-green-50 text-green-700">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    )}
                    
                    {binnedFeatures.length > 0 && (
                      <AccordionItem value="binned">
                        <AccordionTrigger className="px-4">
                          <div className="flex items-center gap-2">
                            <SlidersHorizontal className="h-4 w-4 text-orange-500" />
                            <span>Binned Features ({binnedFeatures.length})</span>
                          </div>
                        </AccordionTrigger>
                        <AccordionContent className="px-4 pb-4">
                          <div className="flex flex-wrap gap-2">
                            {binnedFeatures.map(feature => (
                              <Badge key={feature} variant="secondary" className="bg-orange-50 text-orange-700">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    )}
                  </Accordion>
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  No automatic transformations were applied to this file.
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="datetime">
            {transformationDetails.datetime_features && transformationDetails.datetime_features.length > 0 ? (
              <div>
                <h3 className="text-sm font-medium mb-4">DateTime Features</h3>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Source Column</TableHead>
                      <TableHead>Extracted Features</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {transformationDetails.datetime_features.map((item, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{item.source_column}</TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {item.derived_features.map((feature: string) => (
                              <Badge key={feature} variant="secondary" className="bg-blue-50 text-blue-700">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                
                <div className="mt-6 p-4 bg-blue-50 rounded-md">
                  <p className="text-sm text-blue-800">
                    <strong>Note:</strong> DateTime features extract temporal components like year, month, day, etc. from date columns. These features can be useful for identifying seasonal patterns and time-based trends in your data.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No datetime features were generated for this file.
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="categorical">
            {transformationDetails.categorical_encodings && transformationDetails.categorical_encodings.length > 0 ? (
              <div>
                <h3 className="text-sm font-medium mb-4">Categorical Encodings</h3>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Source Column</TableHead>
                      <TableHead>Encoding Type</TableHead>
                      <TableHead>Cardinality</TableHead>
                      <TableHead>Encoded Features</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {transformationDetails.categorical_encodings.map((item, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{item.source_column}</TableCell>
                        <TableCell>{item.encoding_type}</TableCell>
                        <TableCell>{item.cardinality}</TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {item.derived_features.map((feature: string) => (
                              <Badge key={feature} variant="secondary" className="bg-purple-50 text-purple-700">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                
                <div className="mt-6 p-4 bg-purple-50 rounded-md">
                  <p className="text-sm text-purple-800">
                    <strong>Note:</strong> One-hot encoding converts categorical variables into a form that can be provided to machine learning algorithms. It creates a binary column for each category, which is especially useful for algorithms that cannot directly handle categorical data.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No categorical encodings were generated for this file.
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="numeric">
            <div className="space-y-8">
              {transformationDetails.numeric_transformations && transformationDetails.numeric_transformations.length > 0 ? (
                <div>
                  <h3 className="text-sm font-medium mb-4">Numeric Transformations</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Source Column</TableHead>
                        <TableHead>Skew</TableHead>
                        <TableHead>Derived Features</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {transformationDetails.numeric_transformations.map((item, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-medium">{item.source_column}</TableCell>
                          <TableCell>{item.skew.toFixed(2)}</TableCell>
                          <TableCell>
                            <div className="flex flex-wrap gap-1">
                              {item.derived_features.map((feature: string) => (
                                <Badge key={feature} variant="secondary" className="bg-green-50 text-green-700">
                                  {feature}
                                </Badge>
                              ))}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  
                  <div className="mt-6 p-4 bg-green-50 rounded-md">
                    <p className="text-sm text-green-800">
                      <strong>Note:</strong> Numeric transformations like logarithm and square root help normalize skewed data distributions, which can improve model performance for many machine learning algorithms.
                    </p>
                  </div>
                </div>
              ) : null}
              
              {transformationDetails.binned_features && transformationDetails.binned_features.length > 0 ? (
                <div className="mt-8">
                  <h3 className="text-sm font-medium mb-4">Binned Features</h3>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Source Column</TableHead>
                        <TableHead>Bins</TableHead>
                        <TableHead>Method</TableHead>
                        <TableHead>Derived Feature</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {transformationDetails.binned_features.map((item, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-medium">{item.source_column}</TableCell>
                          <TableCell>{item.bins}</TableCell>
                          <TableCell>{item.method}</TableCell>
                          <TableCell>
                            <Badge variant="secondary" className="bg-orange-50 text-orange-700">
                              {item.derived_feature}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  
                  <div className="mt-6 p-4 bg-orange-50 rounded-md">
                    <p className="text-sm text-orange-800">
                      <strong>Note:</strong> Binning converts continuous variables into categorical features by grouping values into bins. This can help capture non-linear relationships and make models more robust to outliers.
                    </p>
                  </div>
                </div>
              ) : null}
              
              {(!transformationDetails.numeric_transformations || transformationDetails.numeric_transformations.length === 0) && 
               (!transformationDetails.binned_features || transformationDetails.binned_features.length === 0) && (
                <div className="text-center py-8 text-muted-foreground">
                  No numeric transformations were generated for this file.
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}