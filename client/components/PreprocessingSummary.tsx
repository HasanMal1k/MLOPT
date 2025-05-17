import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, X, Calendar, ArrowDownUp } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface PreprocessingSummaryProps {
  processingResults: {
    columns_dropped: string[];
    date_columns_detected: string[];
    columns_cleaned: string[];
    original_shape: [number, number];
    processed_shape: [number, number];
    missing_value_stats?: Record<string, {
      missing_count: number;
      missing_percentage: number;
      imputation_method: string;
    }>;
  };
  fileName: string;
}

export default function PreprocessingSummary({ processingResults, fileName }: PreprocessingSummaryProps) {
  // Calculate stats
  const droppedColumnsCount = processingResults.columns_dropped?.length || 0;
  const dateColumnsCount = processingResults.date_columns_detected?.length || 0;
  const cleanedColumnsCount = processingResults.columns_cleaned?.length || 0;
  const totalColumns = processingResults.original_shape[1];
  const totalRows = processingResults.original_shape[0];
  const resultRows = processingResults.processed_shape[0];
  const rowsRemoved = totalRows - resultRows;
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Preprocessing Summary for {fileName}</CardTitle>
        <CardDescription>
          Changes made to your data during automatic preprocessing
        </CardDescription>
      </CardHeader>
      <CardContent>
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
          </div>
          
          {processingResults.columns_dropped?.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium mb-2">Dropped Columns</h3>
              <div className="flex flex-wrap gap-1">
                {processingResults.columns_dropped.map(column => (
                  <Badge key={column} variant="secondary" className="bg-red-50 text-red-700">
                    {column}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          
          {processingResults.date_columns_detected?.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium mb-2">Detected Date Columns</h3>
              <div className="flex flex-wrap gap-1">
                {processingResults.date_columns_detected.map(column => (
                  <Badge key={column} variant="secondary" className="bg-blue-50 text-blue-700">
                    {column}
                  </Badge>
                ))}
              </div>
            </div>
          )}
          
          {processingResults.missing_value_stats && Object.keys(processingResults.missing_value_stats).length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-medium mb-2">Missing Value Handling</h3>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Column</TableHead>
                    <TableHead>Missing Values</TableHead>
                    <TableHead>Percentage</TableHead>
                    <TableHead>Method</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(processingResults.missing_value_stats).map(([column, stats]) => (
                    <TableRow key={column}>
                      <TableCell className="font-medium">{column}</TableCell>
                      <TableCell>{stats.missing_count}</TableCell>
                      <TableCell>{stats.missing_percentage.toFixed(2)}%</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="bg-green-50 text-green-700">
                          {stats.imputation_method}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}