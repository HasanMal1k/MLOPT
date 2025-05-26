from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import uuid
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import logging

# Import the robust file reading functions
from universal_file_handler import read_any_file_universal
from robust_csv_reader import read_csv_with_robust_handling

# Configure logging
logger = logging.getLogger("data_visualization")

# Create router for data visualization
router = APIRouter(prefix="/visualization", tags=["visualization"])

# Create folder to store visualization data
viz_folder = Path("visualization_data")
viz_folder.mkdir(exist_ok=True)

def clean_for_json(obj):
    """Clean data for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        if pd.isna(obj) or np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def read_file_robustly(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Use the existing robust file reading infrastructure
    """
    try:
        # First try the universal file handler (handles both CSV and Excel)
        df, success, message = read_any_file_universal(file_content, filename)
        
        if success and not df.empty:
            logger.info(f"Successfully read file with universal handler: {message}")
            return df
        else:
            logger.warning(f"Universal handler failed: {message}")
            
        # If universal handler fails and it's a CSV, try the robust CSV reader
        if filename.lower().endswith('.csv'):
            logger.info("Trying robust CSV reader as fallback...")
            df = read_csv_with_robust_handling(file_content)
            if not df.empty:
                logger.info("Robust CSV reader succeeded")
                return df
            else:
                logger.warning("Robust CSV reader returned empty dataframe")
        
        # If all else fails, raise an error
        raise ValueError(f"Could not read file {filename} with any method")
        
    except Exception as e:
        logger.error(f"All file reading methods failed for {filename}: {e}")
        raise Exception(f"Error reading file {filename}: {str(e)}")

@router.post("/analyze-for-charts/")
async def analyze_file_for_charts(file: UploadFile = File(...)):
    """
    Analyze a file and return column information suitable for chart creation
    """
    try:
        # Read file content
        file_content = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(file_content)} bytes")
        
        # Use robust file reading
        try:
            df = read_file_robustly(file_content, file.filename)
            logger.info(f"Successfully read file: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to read file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Validate dataframe
        if df.empty:
            raise HTTPException(status_code=400, detail="File appears to be empty")
        
        if len(df.columns) == 0:
            raise HTTPException(status_code=400, detail="No columns found in file")
        
        # Analyze columns for chart suitability
        column_analysis = {}
        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        
        for col in df.columns:
            try:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "unique_count": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": []
                }
                
                # Get sample values safely
                try:
                    sample_vals = df[col].dropna().head(5)
                    col_info["sample_values"] = [str(val) for val in sample_vals if not pd.isna(val)]
                except Exception as sample_error:
                    logger.warning(f"Error getting sample values for {col}: {sample_error}")
                    col_info["sample_values"] = []
                
                # Categorize columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info["category"] = "numeric"
                    numeric_columns.append(col)
                    try:
                        col_info["min"] = clean_for_json(df[col].min())
                        col_info["max"] = clean_for_json(df[col].max())
                        col_info["mean"] = clean_for_json(df[col].mean())
                    except Exception as stats_error:
                        logger.warning(f"Error calculating stats for {col}: {stats_error}")
                        
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_info["category"] = "datetime"
                    datetime_columns.append(col)
                else:
                    col_info["category"] = "categorical"
                    categorical_columns.append(col)
                    # For categorical, check if it's suitable for grouping
                    try:
                        unique_count = df[col].nunique()
                        col_info["suitable_for_grouping"] = unique_count <= 20
                    except:
                        col_info["suitable_for_grouping"] = False
                
                column_analysis[col] = col_info
                
            except Exception as col_error:
                logger.warning(f"Error analyzing column {col}: {col_error}")
                # Add basic info even if analysis fails
                column_analysis[col] = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "category": "unknown",
                    "unique_count": 0,
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": []
                }
        
        logger.info(f"Column analysis complete: {len(numeric_columns)} numeric, {len(categorical_columns)} categorical, {len(datetime_columns)} datetime")
        
        # Suggest chart types based on data
        chart_suggestions = []
        
        try:
            # Bar charts - categorical x numeric
            if categorical_columns and numeric_columns:
                for cat_col in categorical_columns[:3]:  # Limit suggestions
                    try:
                        if df[cat_col].nunique() <= 15:  # Not too many categories
                            for num_col in numeric_columns[:3]:
                                chart_suggestions.append({
                                    "type": "bar",
                                    "x_axis": cat_col,
                                    "y_axis": num_col,
                                    "description": f"Bar chart showing {num_col} by {cat_col}"
                                })
                    except Exception as e:
                        logger.warning(f"Error creating bar chart suggestion for {cat_col}: {e}")
            
            # Scatter plots - numeric x numeric
            if len(numeric_columns) >= 2:
                for i, x_col in enumerate(numeric_columns[:3]):
                    for y_col in numeric_columns[i+1:4]:
                        chart_suggestions.append({
                            "type": "scatter",
                            "x_axis": x_col,
                            "y_axis": y_col,
                            "description": f"Scatter plot of {y_col} vs {x_col}"
                        })
            
            # Line charts - especially good for time series
            if datetime_columns and numeric_columns:
                for date_col in datetime_columns[:2]:
                    for num_col in numeric_columns[:3]:
                        chart_suggestions.append({
                            "type": "line",
                            "x_axis": date_col,
                            "y_axis": num_col,
                            "description": f"Time series line chart of {num_col} over {date_col}"
                        })
            
            # Pie charts - categorical with aggregation
            for cat_col in categorical_columns[:3]:
                try:
                    unique_count = df[cat_col].nunique()
                    if 2 <= unique_count <= 10:  # Good range for pie chart
                        if numeric_columns:
                            chart_suggestions.append({
                                "type": "pie",
                                "category": cat_col,
                                "value": numeric_columns[0],
                                "description": f"Pie chart showing distribution of {numeric_columns[0]} by {cat_col}"
                            })
                except Exception as e:
                    logger.warning(f"Error creating pie chart suggestion for {cat_col}: {e}")
        
        except Exception as e:
            logger.warning(f"Error generating chart suggestions: {e}")
        
        response_data = {
            "success": True,
            "filename": file.filename,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": clean_for_json(column_analysis),
            "column_categories": {
                "numeric": numeric_columns,
                "categorical": categorical_columns,
                "datetime": datetime_columns
            },
            "chart_suggestions": chart_suggestions[:10]  # Limit to top 10 suggestions
        }
        
        logger.info(f"Analysis complete, returning {len(chart_suggestions)} chart suggestions")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error analyzing file for charts: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@router.post("/generate-chart-data/")
async def generate_chart_data(
    file: UploadFile = File(...),
    chart_config: str = Form(...)
):
    """
    Generate data for frontend chart libraries (Chart.js, Recharts, etc.)
    This is the recommended approach for Next.js frontend
    """
    try:
        # Parse chart configuration
        try:
            config = json.loads(chart_config)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid chart configuration JSON: {str(e)}")
        
        chart_type = config.get("type")
        x_axis = config.get("x_axis")
        y_axis = config.get("y_axis")
        
        logger.info(f"Generating {chart_type} chart with x_axis={x_axis}, y_axis={y_axis}")
        
        # Read file content
        file_content = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(file_content)} bytes")
        
        # Use robust file reading
        try:
            df = read_file_robustly(file_content, file.filename)
            logger.info(f"Successfully read file: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to read file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Validate dataframe
        if df.empty:
            raise HTTPException(status_code=400, detail="File appears to be empty")
        
        # Generate chart data based on type
        chart_data = {}
        
        try:
            if chart_type == "bar":
                if x_axis in df.columns and y_axis in df.columns:
                    # Group by x_axis and aggregate y_axis
                    try:
                        grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
                        chart_data = {
                            "labels": [str(x) for x in grouped[x_axis].tolist()],
                            "data": clean_for_json(grouped[y_axis].tolist()),
                            "type": "bar"
                        }
                        logger.info(f"Generated bar chart data with {len(chart_data['labels'])} points")
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error creating bar chart: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Required columns not found: {x_axis}, {y_axis}")
            
            elif chart_type == "scatter":
                if x_axis in df.columns and y_axis in df.columns:
                    try:
                        # Remove null values and limit data points for performance
                        clean_df = df[[x_axis, y_axis]].dropna()
                        if len(clean_df) > 1000:  # Limit to 1000 points for performance
                            clean_df = clean_df.sample(1000)
                        
                        chart_data = {
                            "data": [
                                {"x": clean_for_json(row[x_axis]), "y": clean_for_json(row[y_axis])}
                                for _, row in clean_df.iterrows()
                                if not (pd.isna(row[x_axis]) or pd.isna(row[y_axis]))
                            ],
                            "type": "scatter"
                        }
                        logger.info(f"Generated scatter plot data with {len(chart_data['data'])} points")
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error creating scatter plot: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Required columns not found: {x_axis}, {y_axis}")
            
            elif chart_type == "line":
                if x_axis in df.columns and y_axis in df.columns:
                    try:
                        # Sort by x_axis for proper line chart
                        sorted_df = df[[x_axis, y_axis]].dropna().sort_values(x_axis)
                        if len(sorted_df) > 1000:  # Limit for performance
                            sorted_df = sorted_df.iloc[::len(sorted_df)//1000]  # Sample evenly
                        
                        chart_data = {
                            "labels": [str(x) for x in sorted_df[x_axis].tolist()],
                            "data": clean_for_json(sorted_df[y_axis].tolist()),
                            "type": "line"
                        }
                        logger.info(f"Generated line chart data with {len(chart_data['labels'])} points")
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error creating line chart: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Required columns not found: {x_axis}, {y_axis}")
            
            elif chart_type == "pie":
                category_col = config.get("category")
                value_col = config.get("value")
                if category_col in df.columns and value_col in df.columns:
                    try:
                        # Group by category and sum values
                        grouped = df.groupby(category_col)[value_col].sum().reset_index()
                        # Limit to top 10 categories for readability
                        if len(grouped) > 10:
                            grouped = grouped.nlargest(10, value_col)
                        
                        chart_data = {
                            "labels": [str(x) for x in grouped[category_col].tolist()],
                            "data": clean_for_json(grouped[value_col].tolist()),
                            "type": "pie"
                        }
                        logger.info(f"Generated pie chart data with {len(chart_data['labels'])} categories")
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error creating pie chart: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Required columns not found: {category_col}, {value_col}")
            
            elif chart_type == "histogram":
                if x_axis in df.columns:
                    try:
                        # Create histogram bins
                        clean_series = df[x_axis].dropna()
                        if len(clean_series) == 0:
                            raise HTTPException(status_code=400, detail=f"No data available for histogram in column {x_axis}")
                        
                        hist, bins = np.histogram(clean_series, bins=20)
                        chart_data = {
                            "labels": [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)],
                            "data": clean_for_json(hist.tolist()),
                            "type": "bar"  # Histogram is rendered as bar chart
                        }
                        logger.info(f"Generated histogram data with {len(chart_data['labels'])} bins")
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Error creating histogram: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Required column not found: {x_axis}")
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported chart type: {chart_type}")
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating chart data: {str(e)}")
        
        if not chart_data:
            raise HTTPException(status_code=400, detail="No chart data generated")
        
        response_data = {
            "success": True,
            "chart_data": chart_data,
            "config_used": config,
            "data_points": len(chart_data.get("data", [])) if isinstance(chart_data.get("data"), list) else len(chart_data.get("labels", []))
        }
        
        logger.info(f"Chart data generation complete, returning {response_data['data_points']} data points")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart data: {str(e)}")

@router.post("/generate-chart-image/")
async def generate_chart_image(
    file: UploadFile = File(...),
    chart_config: str = Form(...)
):
    """
    Generate chart as base64 image (alternative approach for simple use cases)
    """
    try:
        # Parse chart configuration
        try:
            config = json.loads(chart_config)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid chart configuration JSON: {str(e)}")
        
        chart_type = config.get("type")
        x_axis = config.get("x_axis")
        y_axis = config.get("y_axis")
        title = config.get("title", f"{chart_type.title()} Chart")
        
        # Read file content
        file_content = await file.read()
        
        # Use robust file reading
        try:
            df = read_file_robustly(file_content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Create matplotlib figure
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if chart_type == "bar":
                if x_axis in df.columns and y_axis in df.columns:
                    grouped = df.groupby(x_axis)[y_axis].mean()
                    grouped.plot(kind='bar', ax=ax)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
            
            elif chart_type == "scatter":
                if x_axis in df.columns and y_axis in df.columns:
                    clean_df = df[[x_axis, y_axis]].dropna()
                    ax.scatter(clean_df[x_axis], clean_df[y_axis], alpha=0.6)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
            
            elif chart_type == "line":
                if x_axis in df.columns and y_axis in df.columns:
                    df_sorted = df[[x_axis, y_axis]].dropna().sort_values(x_axis)
                    ax.plot(df_sorted[x_axis], df_sorted[y_axis])
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
            
            elif chart_type == "pie":
                category_col = config.get("category")
                value_col = config.get("value")
                if category_col in df.columns and value_col in df.columns:
                    grouped = df.groupby(category_col)[value_col].sum()
                    ax.pie(grouped.values, labels=grouped.index, autopct='%1.1f%%')
            
            ax.set_title(title)
            plt.tight_layout()
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close(fig)
            
            return JSONResponse(content={
                "success": True,
                "image": f"data:image/png;base64,{img_base64}",
                "config_used": config
            })
            
        except Exception as plot_error:
            plt.close(fig)
            raise HTTPException(status_code=400, detail=f"Error creating chart image: {str(plot_error)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating chart image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.post("/get-column-stats/")
async def get_column_statistics(
    file: UploadFile = File(...),
    column_name: str = Form(...)
):
    """
    Get detailed statistics for a specific column to help with chart configuration
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Use robust file reading
        try:
            df = read_file_robustly(file_content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' not found")
        
        col_data = df[column_name]
        stats = {
            "column_name": column_name,
            "dtype": str(col_data.dtype),
            "count": int(len(col_data)),
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique())
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "min": clean_for_json(col_data.min()),
                "max": clean_for_json(col_data.max()),
                "mean": clean_for_json(col_data.mean()),
                "median": clean_for_json(col_data.median()),
                "std": clean_for_json(col_data.std()),
                "quartiles": {
                    "q25": clean_for_json(col_data.quantile(0.25)),
                    "q50": clean_for_json(col_data.quantile(0.50)),
                    "q75": clean_for_json(col_data.quantile(0.75))
                }
            })
        else:
            # For categorical data
            try:
                value_counts = col_data.value_counts().head(10)
                stats.update({
                    "most_common": {
                        str(k): int(v) for k, v in value_counts.items()
                    }
                })
            except Exception as e:
                logger.warning(f"Error getting value counts for {column_name}: {e}")
                stats["most_common"] = {}
        
        return JSONResponse(content={
            "success": True,
            "statistics": stats
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting column statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@router.get("/chart-types/")
async def get_available_chart_types():
    """
    Return available chart types and their configurations
    """
    chart_types = {
        "bar": {
            "name": "Bar Chart",
            "description": "Compare categories with rectangular bars",
            "required_fields": ["x_axis", "y_axis"],
            "suitable_for": "categorical x-axis, numeric y-axis"
        },
        "scatter": {
            "name": "Scatter Plot",
            "description": "Show relationship between two numeric variables",
            "required_fields": ["x_axis", "y_axis"],
            "suitable_for": "numeric x-axis, numeric y-axis"
        },
        "line": {
            "name": "Line Chart",
            "description": "Show trends over time or ordered categories",
            "required_fields": ["x_axis", "y_axis"],
            "suitable_for": "ordered x-axis (time/numeric), numeric y-axis"
        },
        "pie": {
            "name": "Pie Chart",
            "description": "Show parts of a whole",
            "required_fields": ["category", "value"],
            "suitable_for": "categorical data with numeric values"
        },
        "histogram": {
            "name": "Histogram",
            "description": "Show distribution of a numeric variable",
            "required_fields": ["x_axis"],
            "suitable_for": "single numeric variable"
        }
    }
    
    return JSONResponse(content={
        "success": True,
        "chart_types": chart_types
    })

# Add a test endpoint for CORS verification (as mentioned in the debugging utilities)
@router.get("/test-cors")
async def test_cors():
    """Test endpoint to verify CORS is working"""
    return {
        "status": "success",
        "message": "CORS is working correctly",
        "timestamp": pd.Timestamp.now().isoformat()
    }