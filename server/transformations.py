from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import uuid
import os

router = APIRouter(
    prefix="/transformations",
    tags=["transformations"],
)

# Create folder to store transformed files
transform_folder = Path("transformed_files")
transform_folder.mkdir(exist_ok=True)

@router.post("/auto-transform/")
async def auto_transform_file(file: UploadFile = File(...)):
    """
    Automatically apply transformations to a file and return the transformed file
    """
    try:
        # Generate a unique filename to prevent collisions
        unique_id = str(uuid.uuid4())[:8]
        original_filename = file.filename
        safe_filename = f"{unique_id}_{original_filename}"
        
        # Save the uploaded file temporarily
        temp_file_path = Path(f"temp_{safe_filename}")
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read the file into a pandas dataframe
        if original_filename.lower().endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif original_filename.lower().endswith('.xlsx'):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="File format not supported. Please upload CSV or XLSX files.")
        
        # Remove the temporary file
        os.remove(temp_file_path)
        
        # Apply automatic transformations
        transformed_df, transformation_results = engineer_features(df)
        
        # Save the transformed file
        output_filename = f"transformed_{safe_filename}"
        if original_filename.lower().endswith('.csv'):
            output_path = transform_folder / output_filename
            transformed_df.to_csv(output_path, index=False)
        else:
            output_path = transform_folder / output_filename.replace('.csv', '.xlsx')
            transformed_df.to_excel(output_path, index=False)
        
        # Return the results
        return {
            "success": True,
            "original_filename": original_filename,
            "transformed_filename": output_filename,
            "original_shape": df.shape,
            "transformed_shape": transformed_df.shape,
            "transformations": transformation_results
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

def engineer_features(df):
    """
    Automatically engineer features based on data types and patterns
    (Fixed version to eliminate warnings and improve performance)
    """
    import warnings
    from pandas.errors import PerformanceWarning
    
    # Suppress only specific warnings we know are unavoidable
    warnings.filterwarnings('ignore', category=PerformanceWarning)
    
    transformation_results = {
        "datetime_features": [],
        "categorical_encodings": [],
        "numeric_transformations": [],
        "binned_features": []
    }
    
    # Create a copy to avoid modifying the original dataframe during detection
    df_transformed = df.copy()
    
    # 1. Handle datetime features (extract components)
    # First try to convert string columns to datetime
    date_columns = []
    
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # FIX: Remove infer_datetime_format parameter
            temp_series = pd.to_datetime(df[col], errors='coerce')
            
            # If more than 50% of values were converted successfully, consider it a date column
            if temp_series.notna().sum() > 0.5 * len(df):
                df_transformed[col] = temp_series
                date_columns.append(col)
            else:
                # Revert if not mostly dates
                df_transformed[col] = df[col]
        except:
            # Keep as-is if conversion fails
            df_transformed[col] = df[col]
    
    # Add already datetime columns
    existing_date_columns = df.select_dtypes(include=['datetime64']).columns
    date_columns.extend(existing_date_columns)
    
    # FIX: Process datetime columns efficiently to avoid fragmentation
    if date_columns:
        # Collect all new date features at once
        all_date_features = {}
        for col in date_columns:
            # Create new feature names and values
            date_features = {
                f'{col}_year': df_transformed[col].dt.year,
                f'{col}_month': df_transformed[col].dt.month,
                f'{col}_day': df_transformed[col].dt.day,
                f'{col}_dayofweek': df_transformed[col].dt.dayofweek,
                f'{col}_quarter': df_transformed[col].dt.quarter
            }
            
            # Collect all features
            all_date_features.update(date_features)
            
            # Track transformations
            transformation_results["datetime_features"].append({
                "source_column": col,
                "derived_features": list(date_features.keys())
            })
        
        # FIX: Add all date features at once using assign()
        df_transformed = df_transformed.assign(**all_date_features)
    
    # 2. Handle categorical features (auto one-hot encode)
    # FIX: Ensure we're not processing date columns again
    remaining_cat_columns = df.select_dtypes(include=['object', 'category']).columns
    cat_columns = [col for col in remaining_cat_columns if col not in date_columns]
    
    for col in cat_columns:
        # Only one-hot encode if cardinality is reasonable
        if 1 < df[col].nunique() < 10:
            # Create dummies with column prefix
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            # Add dummy columns to the dataframe efficiently
            df_transformed = pd.concat([df_transformed, dummies], axis=1)
            
            # Track transformations
            transformation_results["categorical_encodings"].append({
                "source_column": col,
                "encoding_type": "one_hot",
                "derived_features": dummies.columns.tolist(),
                "cardinality": int(df[col].nunique())
            })
    
    # 3. FIX: Handle numeric features efficiently to avoid fragmentation
    num_columns = df.select_dtypes(include=['number']).columns
    all_numeric_features = {}  # Collect all numeric transformations
    
    for col in num_columns:
        derived_features = []
        
        # Skip columns with too many zeros or negative values
        if (df[col] > 0).sum() < 0.5 * len(df):
            continue
            
        # Compute temporary positive-only version for transformations
        positive_values = df[col][df[col] > 0]
        
        # Create transformations
        numeric_transformations = {}
        
        # Log transformation for skewed data
        if len(positive_values) > 0 and abs(positive_values.skew()) > 1:
            numeric_transformations[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
            derived_features.append(f'{col}_log')
            
        # Square root transformation for moderately skewed data
        if len(positive_values) > 0:
            numeric_transformations[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))
            derived_features.append(f'{col}_sqrt')
        
        # Square transformation for negatively skewed data
        if abs(df[col].skew()) < -0.5:
            numeric_transformations[f'{col}_squared'] = df[col] ** 2
            derived_features.append(f'{col}_squared')
        
        # Add reciprocal for appropriate distributions
        if len(positive_values) > 0:
            numeric_transformations[f'{col}_reciprocal'] = 1 / df[col].replace([np.inf, -np.inf, 0], np.nan)
            derived_features.append(f'{col}_reciprocal')
        
        # FIX: Collect all transformations instead of adding one by one
        all_numeric_features.update(numeric_transformations)
        
        # Track transformations if any were applied
        if derived_features:
            transformation_results["numeric_transformations"].append({
                "source_column": col,
                "derived_features": derived_features,
                "skew": float(df[col].skew())
            })
    
    # FIX: Add all numeric features at once using assign()
    if all_numeric_features:
        df_transformed = df_transformed.assign(**all_numeric_features)
    
    # 4. FIX: Create binned features efficiently
    binned_features = {}
    for col in num_columns:
        # Skip if too few unique values
        if df[col].nunique() < 10:
            continue
            
        # Create bins (quartiles)
        try:
            binned_feature = pd.qcut(df[col], q=4, labels=False, duplicates='drop')
            bin_col_name = f'{col}_binned'
            binned_features[bin_col_name] = binned_feature
            
            # Track transformation
            transformation_results["binned_features"].append({
                "source_column": col,
                "derived_feature": bin_col_name,
                "bins": 4,
                "method": "equal_frequency"
            })
        except:
            # Skip if binning fails
            pass
    
    # FIX: Add all binned features at once
    if binned_features:
        df_transformed = df_transformed.assign(**binned_features)
    
    # FIX: No need for additional copy at the end - df_transformed is already clean
    return df_transformed, transformation_results

# Alternative more efficient implementation
def engineer_features_optimized(df):
    """
    More optimized version of feature engineering with minimal dataframe operations
    """
    import warnings
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
    
    transformation_results = {
        "datetime_features": [],
        "categorical_encodings": [],
        "numeric_transformations": [],
        "binned_features": []
    }
    
    # Start with a copy
    result_df = df.copy()
    
    # Collect all new features in batches
    new_features = {}
    
    # 1. Datetime processing batch
    for col in df.select_dtypes(include=['object']).columns:
        try:
            temp_series = pd.to_datetime(df[col], errors='coerce')
            if temp_series.notna().sum() > 0.5 * len(df):
                new_features[col] = temp_series  # Convert column
                
                # Add date components
                new_features.update({
                    f'{col}_year': temp_series.dt.year,
                    f'{col}_month': temp_series.dt.month,
                    f'{col}_day': temp_series.dt.day,
                    f'{col}_dayofweek': temp_series.dt.dayofweek,
                    f'{col}_quarter': temp_series.dt.quarter
                })
                
                transformation_results["datetime_features"].append({
                    "source_column": col,
                    "derived_features": [f'{col}_year', f'{col}_month', f'{col}_day', 
                                       f'{col}_dayofweek', f'{col}_quarter']
                })
        except:
            pass
    
    # Process existing datetime columns
    for col in df.select_dtypes(include=['datetime64']).columns:
        new_features.update({
            f'{col}_year': df[col].dt.year,
            f'{col}_month': df[col].dt.month,
            f'{col}_day': df[col].dt.day,
            f'{col}_dayofweek': df[col].dt.dayofweek,
            f'{col}_quarter': df[col].dt.quarter
        })
        
        transformation_results["datetime_features"].append({
            "source_column": col,
            "derived_features": [f'{col}_year', f'{col}_month', f'{col}_day', 
                               f'{col}_dayofweek', f'{col}_quarter']
        })
    
    # 2. Categorical encoding batch
    already_converted = set(col for col in new_features.keys() if col in df.columns)
    cat_columns = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                   if col not in already_converted]
    
    for col in cat_columns:
        if 1 < df[col].nunique() < 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            new_features.update(dummies.to_dict(orient='list'))
            
            transformation_results["categorical_encodings"].append({
                "source_column": col,
                "encoding_type": "one_hot",
                "derived_features": dummies.columns.tolist(),
                "cardinality": int(df[col].nunique())
            })
    
    # 3. Numeric transformations batch
    for col in df.select_dtypes(include=['number']).columns:
        derived_features = []
        
        if (df[col] > 0).sum() < 0.5 * len(df):
            continue
            
        positive_values = df[col][df[col] > 0]
        
        # All transformations in one batch
        if len(positive_values) > 0:
            if abs(positive_values.skew()) > 1:
                new_features[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
                derived_features.append(f'{col}_log')
            
            new_features[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))
            derived_features.append(f'{col}_sqrt')
            
            new_features[f'{col}_reciprocal'] = 1 / df[col].replace([np.inf, -np.inf, 0], np.nan)
            derived_features.append(f'{col}_reciprocal')
        
        if abs(df[col].skew()) < -0.5:
            new_features[f'{col}_squared'] = df[col] ** 2
            derived_features.append(f'{col}_squared')
        
        if derived_features:
            transformation_results["numeric_transformations"].append({
                "source_column": col,
                "derived_features": derived_features,
                "skew": float(df[col].skew())
            })
    
    # 4. Binning batch
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].nunique() >= 10:
            try:
                binned_feature = pd.qcut(df[col], q=4, labels=False, duplicates='drop')
                bin_col_name = f'{col}_binned'
                new_features[bin_col_name] = binned_feature
                
                transformation_results["binned_features"].append({
                    "source_column": col,
                    "derived_feature": bin_col_name,
                    "bins": 4,
                    "method": "equal_frequency"
                })
            except:
                pass
    
    # Apply all transformations at once
    result_df = result_df.assign(**new_features)
    
    return result_df, transformation_results


@router.post("/analyze-columns/")
async def analyze_columns(file: UploadFile = File(...)):
    """
    Analyze columns in a file and return information about data types and samples
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = Path(f"temp_{file.filename}")
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read the file into a pandas dataframe
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="File format not supported. Please upload CSV or XLSX files.")
        
        # Remove the temporary file
        os.remove(temp_file_path)
        
        # Analyze columns
        columns_info = []
        
        for column in df.columns:
            # Determine column type
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                col_type = "datetime64[ns]"
            elif pd.api.types.is_numeric_dtype(df[column]):
                if pd.api.types.is_integer_dtype(df[column]):
                    col_type = "integer"
                else:
                    col_type = "float"
            elif pd.api.types.is_categorical_dtype(df[column]):
                col_type = "category"
            else:
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[column], errors='raise')
                    col_type = "datetime"
                except:
                    col_type = "string"
            
            # Get sample values (up to 5)
            sample_values = df[column].dropna().head(5).astype(str).tolist()
            
            columns_info.append({
                "name": column,
                "type": col_type,
                "sample_values": sample_values
            })
        
        return {
            "success": True,
            "filename": file.filename,
            "columns": columns_info
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/specific-transform/")
async def apply_specific_transformations(
    file: UploadFile = File(...),
    transformations: str = Form(...)
):
    """Apply specific transformations to a file"""
    try:
        # Parse the transformations
        transform_config = json.loads(transformations)
        
        # Generate a unique filename
        unique_id = str(uuid.uuid4())[:8]
        original_filename = file.filename
        safe_filename = f"{unique_id}_{original_filename}"
        
        # Save the uploaded file temporarily
        temp_file_path = Path(f"temp_{safe_filename}")
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read the file
        if original_filename.lower().endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif original_filename.lower().endswith('.xlsx'):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="File format not supported")
        
        # Remove the temporary file
        os.remove(temp_file_path)
        
        # Apply transformations based on the config
        transformed_df, applied_transforms = apply_transformations(df, transform_config)
        
        # Save the transformed file
        output_filename = f"transformed_{safe_filename}"
        if original_filename.lower().endswith('.csv'):
            output_path = transform_folder / output_filename
            transformed_df.to_csv(output_path, index=False)
        else:
            output_path = transform_folder / output_filename.replace('.csv', '.xlsx')
            transformed_df.to_excel(output_path, index=False)
        
        # Return the results
        return {
            "success": True,
            "original_filename": original_filename,
            "transformed_filename": output_filename,
            "transformations_applied": applied_transforms
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

def apply_transformations(df, config):
    """Apply specific transformations based on a configuration"""
    df_transformed = df.copy()
    applied_transforms = []
    
    # Process each transformation
    if "log_transform" in config:
        for col in config["log_transform"]:
            if col in df.columns:
                df_transformed[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
                applied_transforms.append(f"Log transform applied to {col}")
    
    if "sqrt_transform" in config:
        for col in config["sqrt_transform"]:
            if col in df.columns:
                df_transformed[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))
                applied_transforms.append(f"Square root transform applied to {col}")
    
    if "squared_transform" in config:
        for col in config["squared_transform"]:
            if col in df.columns:
                df_transformed[f'{col}_squared'] = df[col] ** 2
                applied_transforms.append(f"Square transform applied to {col}")
    
    if "reciprocal_transform" in config:
        for col in config["reciprocal_transform"]:
            if col in df.columns:
                df_transformed[f'{col}_reciprocal'] = 1 / df[col].replace([np.inf, -np.inf, 0], np.nan)
                applied_transforms.append(f"Reciprocal transform applied to {col}")
    
    if "binning" in config:
        for item in config["binning"]:
            col = item["column"]
            num_bins = item["bins"]
            labels = item.get("labels", None)
            
            if col in df.columns:
                try:
                    df_transformed[f'{col}_binned'] = pd.qcut(
                        df[col], q=num_bins, labels=labels, duplicates='drop'
                    )
                    applied_transforms.append(f"Binning applied to {col} with {num_bins} bins")
                except:
                    pass
    
    if "one_hot_encoding" in config:
        for col in config["one_hot_encoding"]:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=config.get("drop_first", False))
                df_transformed = pd.concat([df_transformed, dummies], axis=1)
                applied_transforms.append(f"One-hot encoding applied to {col}")
    
    if "datetime_features" in config:
        for item in config["datetime_features"]:
            col = item["column"]
            features = item["features"]
            
            if col in df.columns:
                try:
                    # Convert to datetime if not already
                    if df[col].dtype != 'datetime64[ns]':
                        df_transformed[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # Extract requested features
                    if "year" in features:
                        df_transformed[f'{col}_year'] = df_transformed[col].dt.year
                    if "month" in features:
                        df_transformed[f'{col}_month'] = df_transformed[col].dt.month
                    if "day" in features:
                        df_transformed[f'{col}_day'] = df_transformed[col].dt.day
                    if "dayofweek" in features:
                        df_transformed[f'{col}_dayofweek'] = df_transformed[col].dt.dayofweek
                    if "quarter" in features:
                        df_transformed[f'{col}_quarter'] = df_transformed[col].dt.quarter
                    
                    applied_transforms.append(f"Datetime features extracted from {col}")
                except:
                    pass
    
    return df_transformed, applied_transforms