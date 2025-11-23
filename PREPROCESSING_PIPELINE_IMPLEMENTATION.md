# Preprocessing Pipeline for Model Inference

## Overview
This implementation enables models to accept raw data inputs (with original column names) and automatically apply the same preprocessing transformations that were used during training. This solves the critical production issue where trained models expect preprocessed column names but users input raw data.

## Architecture

### 1. Pipeline Saving (Training Time)
When data is preprocessed, a comprehensive pipeline is saved containing:
- Original column names
- Missing value imputation strategies (mode for categorical, median for numeric)
- Date parsing and feature extraction rules
- One-hot encoding mappings
- Numeric transformations (log, sqrt, squared, reciprocal)
- Column type information (categorical vs numeric)

**Files Modified:**
- `server/data_preprocessing.py` - Lines 735-810 (enhanced pipeline saving)
- `server/custom_preprocessing.py` - Lines 318-340 (custom transformation pipeline saving)

### 2. Pipeline Loading and Application (Inference Time)
New module `preprocessing_pipeline_loader.py` provides:
- `load_preprocessing_pipeline()` - Loads saved pipeline from joblib file
- `apply_preprocessing_pipeline()` - Applies auto preprocessing transformations
- `apply_custom_transformation_pipeline()` - Applies user-defined transformations
- `preprocess_for_prediction()` - Main function that chains all transformations

**File Created:**
- `server/preprocessing_pipeline_loader.py` (287 lines)

### 3. Prediction Endpoint Integration
The `/ml/predict/` endpoint now:
1. Accepts raw column names from user
2. Loads the appropriate preprocessing pipeline(s)
3. Transforms raw data to match training format
4. Passes transformed data to model
5. Returns prediction

**Files Modified:**
- `server/ml_training.py` - Lines 1425-1505 (updated prediction endpoint)

## Pipeline Structure

### Auto Preprocessing Pipeline
```python
{
    "original_columns": ["Age", "Sex", "Pclass", ...],
    "final_columns": ["Age", "Sex_male", "Pclass", "Age_log", ...],
    "date_columns": ["DateOfBirth"],
    "categorical_modes": {"Sex": "male", "Embarked": "S"},
    "numeric_medians": {"Age": 28.0, "Fare": 14.45},
    "one_hot_columns": {
        "Sex": ["Sex_male"],
        "Embarked": ["Embarked_Q", "Embarked_S"]
    },
    "numeric_transformations": {
        "Age": ["Age_log", "Age_sqrt"],
        "Fare": ["Fare_log"]
    }
}
```

### Custom Transformation Pipeline
```python
{
    "original_columns": [...],
    "final_columns": [...],
    "transform_config": {
        "log_transform": ["column1"],
        "binning": [{"column": "column2", "bins": 4}],
        "one_hot_encoding": ["column3"]
    },
    "applied_transforms": [...]
}
```

## Usage

### From Backend
```python
from preprocessing_pipeline_loader import preprocess_for_prediction

# Raw data from user
raw_input = {
    "Age": 35,
    "Sex": "male",
    "Pclass": 1,
    "Fare": 50.0
}

# Preprocess for model
processed_df = preprocess_for_prediction(
    raw_data_dict=raw_input,
    file_id="1234567890",
    processed_folder=Path("processed_files")
)

# Now ready for model.predict(processed_df)
```

### From API
```bash
curl -X POST "http://localhost:8000/ml/predict/" \
  -F "config_id=abc123" \
  -F "model_name=Random Forest" \
  -F "file_id=1234567890" \
  -F 'input_data={"Age": 35, "Sex": "male", "Pclass": 1, "Fare": 50.0}'
```

## Transformation Steps Applied

1. **Column Validation** - Ensures all original columns are present
2. **Missing Value Handling** - Replaces indicators like "N/A", "Unknown" with NaN
3. **Date Parsing** - Converts date strings to datetime objects
4. **Date Feature Extraction** - Creates year, month, day, dayofweek, quarter columns
5. **Missing Value Imputation** - Uses saved mode (categorical) or median (numeric)
6. **Outlier Capping** - Clips values to training data bounds
7. **One-Hot Encoding** - Creates dummy variables matching training schema
8. **Numeric Transformations** - Applies log, sqrt, squared, reciprocal as configured
9. **Final Validation** - Fills any remaining NaN values

## Pipeline File Naming Convention

### Auto Preprocessing
- Pattern: `processed_{filename}_pipeline.joblib`
- Location: `server/processed_files/`
- Example: `processed_1234567890_titanic_pipeline.joblib`

### Custom Transformations
- Pattern: `transformed_{uuid}_{filename}_pipeline.joblib`
- Location: `server/transformed_files/`
- Example: `transformed_a1b2c3d4_titanic_pipeline.joblib`

## Error Handling

The system gracefully degrades if pipelines are not found:
1. Logs warning about missing pipeline
2. Falls back to using raw input directly
3. Prediction may fail if column names don't match

**Best Practice:** Always provide `file_id` parameter to prediction endpoint to ensure proper preprocessing.

## Robustness Features

### Multiple Pipeline Support
- Can apply both auto preprocessing AND custom transformations in sequence
- Automatically chains transformations in correct order

### Schema Flexibility
- If a one-hot encoded category wasn't seen during training, adds column with 0
- If a column is missing in raw input, fills with NaN and imputes using saved strategy

### Logging
- Comprehensive logging at each transformation step
- Helps debug prediction issues by showing what transformations were applied

## Testing Recommendations

1. **Test with training data columns** - Verify pipeline reproduces training transformations
2. **Test with missing columns** - Ensure graceful handling of incomplete inputs
3. **Test with new categorical values** - Check one-hot encoding handles unseen categories
4. **Test with extreme values** - Verify outlier capping works correctly
5. **Test custom transformations** - Ensure user-applied transforms are preserved

## Future Enhancements

1. **Pipeline Versioning** - Track which pipeline version was used for each model
2. **Schema Validation** - Explicit schema checking before prediction
3. **Pipeline Composition** - Better handling of multiple transformation layers
4. **Feature Store Integration** - Cache preprocessed features for faster inference
5. **Model-Pipeline Binding** - Store pipeline reference with model metadata

## Known Limitations

1. **KNN Imputation** - Currently not saved/applied during inference (falls back to median)
2. **Binning** - Bin edges not saved, approximates with quartiles during inference
3. **Complex Transformations** - Some edge cases may not perfectly replicate training preprocessing
4. **Pipeline Size** - Large pipelines can increase inference latency

## Debugging

### Check Pipeline Exists
```python
from pathlib import Path
import glob

# List all pipelines
pipelines = list(Path("processed_files").glob("*_pipeline.joblib"))
print(f"Found {len(pipelines)} pipelines")
for p in pipelines:
    print(f"  - {p.name}")
```

### Inspect Pipeline Contents
```python
import joblib

pipeline = joblib.load("processed_files/processed_123_file_pipeline.joblib")
print("Original columns:", pipeline["original_columns"])
print("Final columns:", pipeline["final_columns"])
print("One-hot mappings:", pipeline["one_hot_columns"])
print("Transformations:", pipeline["numeric_transformations"])
```

### Compare Raw vs Preprocessed
```python
raw_input = {"Age": 35, "Sex": "male"}
processed = preprocess_for_prediction(raw_input, "123", Path("processed_files"))

print("Raw shape:", pd.DataFrame([raw_input]).shape)
print("Processed shape:", processed.shape)
print("Processed columns:", list(processed.columns))
```

## Integration with Frontend

The frontend should:
1. Show raw column names to users for input
2. Pass `file_id` parameter to prediction endpoint
3. Backend automatically handles all preprocessing
4. User never sees preprocessed column names

This creates a seamless user experience where they input familiar column names and the system handles the complexity of preprocessing behind the scenes.
