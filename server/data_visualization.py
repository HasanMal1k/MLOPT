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

@router.post("/analyze-for-charts/")
async def analyze_file_for_charts(file: UploadFile = File(...)):
    """
    Analyze a file and return column information suitable for chart creation
    """
    temp_file_path = None
    try:
        # Save file temporarily
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Read the file
        try:
            if file.filename.lower().endswith('.csv'):
                df = pd.read_csv(temp_file_path)
            elif file.filename.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(temp_file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Analyze columns for chart suitability
        column_analysis = {}
        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": df[col].dropna().head(5).astype(str).tolist()
            }
            
            # Categorize columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["category"] = "numeric"
                numeric_columns.append(col)
                col_info["min"] = clean_for_json(df[col].min())
                col_info["max"] = clean_for_json(df[col].max())
                col_info["mean"] = clean_for_json(df[col].mean())
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info["category"] = "datetime"
                datetime_columns.append(col)
            else:
                col_info["category"] = "categorical"
                categorical_columns.append(col)
                # For categorical, check if it's suitable for grouping
                if df[col].nunique() <= 20:  # Reasonable number for grouping
                    col_info["suitable_for_grouping"] = True
                else:
                    col_info["suitable_for_grouping"] = False
            
            column_analysis[col] = col_info
        
        # Suggest chart types based on data
        chart_suggestions = []
        
        # Bar charts - categorical x numeric
        if categorical_columns and numeric_columns:
            for cat_col in categorical_columns[:3]:  # Limit suggestions
                if df[cat_col].nunique() <= 15:  # Not too many categories
                    for num_col in numeric_columns[:3]:
                        chart_suggestions.append({
                            "type": "bar",
                            "x_axis": cat_col,
                            "y_axis": num_col,
                            "description": f"Bar chart showing {num_col} by {cat_col}"
                        })
        
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
            if 2 <= df[cat_col].nunique() <= 10:  # Good range for pie chart
                if numeric_columns:
                    chart_suggestions.append({
                        "type": "pie",
                        "category": cat_col,
                        "value": numeric_columns[0],
                        "description": f"Pie chart showing distribution of {numeric_columns[0]} by {cat_col}"
                    })
        
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
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing file for charts: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/generate-chart-data/")
async def generate_chart_data(
    file: UploadFile = File(...),
    chart_config: str = Form(...)
):
    """
    Generate data for frontend chart libraries (Chart.js, Recharts, etc.)
    This is the recommended approach for Next.js frontend
    """
    temp_file_path = None
    try:
        # Parse chart configuration
        config = json.loads(chart_config)
        chart_type = config.get("type")
        x_axis = config.get("x_axis")
        y_axis = config.get("y_axis")
        
        # Save and read file
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate chart data based on type
        chart_data = {}
        
        if chart_type == "bar":
            if x_axis in df.columns and y_axis in df.columns:
                # Group by x_axis and aggregate y_axis
                grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
                chart_data = {
                    "labels": grouped[x_axis].astype(str).tolist(),
                    "data": clean_for_json(grouped[y_axis].tolist()),
                    "type": "bar"
                }
        
        elif chart_type == "scatter":
            if x_axis in df.columns and y_axis in df.columns:
                # Remove null values
                clean_df = df[[x_axis, y_axis]].dropna()
                chart_data = {
                    "data": [
                        {"x": clean_for_json(row[x_axis]), "y": clean_for_json(row[y_axis])}
                        for _, row in clean_df.iterrows()
                    ],
                    "type": "scatter"
                }
        
        elif chart_type == "line":
            if x_axis in df.columns and y_axis in df.columns:
                # Sort by x_axis for proper line chart
                sorted_df = df[[x_axis, y_axis]].dropna().sort_values(x_axis)
                chart_data = {
                    "labels": sorted_df[x_axis].astype(str).tolist(),
                    "data": clean_for_json(sorted_df[y_axis].tolist()),
                    "type": "line"
                }
        
        elif chart_type == "pie":
            category_col = config.get("category")
            value_col = config.get("value")
            if category_col in df.columns and value_col in df.columns:
                # Group by category and sum values
                grouped = df.groupby(category_col)[value_col].sum().reset_index()
                chart_data = {
                    "labels": grouped[category_col].astype(str).tolist(),
                    "data": clean_for_json(grouped[value_col].tolist()),
                    "type": "pie"
                }
        
        elif chart_type == "histogram":
            if x_axis in df.columns:
                # Create histogram bins
                hist, bins = np.histogram(df[x_axis].dropna(), bins=20)
                chart_data = {
                    "labels": [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)],
                    "data": clean_for_json(hist.tolist()),
                    "type": "bar"  # Histogram is rendered as bar chart
                }
        
        response_data = {
            "success": True,
            "chart_data": chart_data,
            "config_used": config,
            "data_points": len(chart_data.get("data", [])) if isinstance(chart_data.get("data"), list) else len(chart_data.get("labels", []))
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart data: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/generate-chart-image/")
async def generate_chart_image(
    file: UploadFile = File(...),
    chart_config: str = Form(...)
):
    """
    Generate chart as base64 image (alternative approach for simple use cases)
    """
    temp_file_path = None
    try:
        # Parse chart configuration
        config = json.loads(chart_config)
        chart_type = config.get("type")
        x_axis = config.get("x_axis")
        y_axis = config.get("y_axis")
        title = config.get("title", f"{chart_type.title()} Chart")
        
        # Save and read file
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
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
                    ax.scatter(df[x_axis], df[y_axis], alpha=0.6)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
            
            elif chart_type == "line":
                if x_axis in df.columns and y_axis in df.columns:
                    df_sorted = df.sort_values(x_axis)
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
            raise plot_error
            
    except Exception as e:
        logger.error(f"Error generating chart image: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/get-column-stats/")
async def get_column_statistics(
    file: UploadFile = File(...),
    column_name: str = Form(...)
):
    """
    Get detailed statistics for a specific column to help with chart configuration
    """
    temp_file_path = None
    try:
        # Save and read file
        file_content = await file.read()
        temp_file_path = f"temp_{uuid.uuid4()}.csv"
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
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
            value_counts = col_data.value_counts().head(10)
            stats.update({
                "most_common": {
                    str(k): int(v) for k, v in value_counts.items()
                }
            })
        
        return JSONResponse(content={
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting column statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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