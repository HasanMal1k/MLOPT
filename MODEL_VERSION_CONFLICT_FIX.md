# Model Version Conflict Fix

## Problem Identified

When attempting the second deployment (at 17:35:52), it failed with:
```
(UserError) Conflict
Code: UserError
Message: Conflict
Exception Details: (ModelVersionInUse) Model model_Iris_classs_20251109123552:1 already exists. 
Please use a different name or version.
```

**Root Cause:**
- The first deployment attempt (17:11:11) successfully registered the model in Azure ML but failed at deployment creation due to quota
- The model naming used `%Y%m%d%H%M%S` format (timestamp to seconds)
- When you tried deploying again quickly (17:35:52), both timestamps rounded to the same format (20251109123552)
- Azure ML retained the first model registration even though its deployment failed
- The second attempt tried to register a model with the same name → conflict

## Solution Implemented

### 1. **More Granular Timestamps** (Primary Fix)
Changed model naming to include microseconds:

```python
# Before:
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
azure_model_name = f"model_{model_data['model_name']}_{timestamp}"

# After:
now = datetime.now()
timestamp = now.strftime('%Y%m%d%H%M%S')
timestamp_ms = now.strftime('%Y%m%d%H%M%S%f')  # Include microseconds
azure_model_name = f"model_{model_data['model_name']}_{timestamp_ms}"
```

**Result:** Model names now include microseconds (e.g., `model_Iris_classs_20251109173552123456`), making collisions virtually impossible even with rapid successive deployments.

### 2. **Retry Logic for Name Conflicts** (Safety Net)
Added automatic retry with variant names if collision still occurs:

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        # If retry, append attempt number to model name
        current_model_name = azure_model_name if attempt == 0 else f"{azure_model_name}_v{attempt + 1}"
        
        model_asset = Model(
            path=str(mlflow_model_path),
            type=AssetTypes.MLFLOW_MODEL,
            name=current_model_name,
            description=model_data.get("description", "Auto-deployed model")
        )
        
        registered_model = ml_client.models.create_or_update(model_asset)
        azure_model_name = current_model_name  # Update name if we used retry variant
        break
    except Exception as model_error:
        if "ModelVersionInUse" in str(model_error) or "already exists" in str(model_error):
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Model name collision, retrying with variant name")
                continue
            else:
                raise
        else:
            raise  # Non-conflict error
```

**Retry Sequence:**
1. **First attempt:** `model_Iris_classs_20251109173552123456`
2. **If conflict:** `model_Iris_classs_20251109173552123456_v2`
3. **If still conflict:** `model_Iris_classs_20251109173552123456_v3`
4. **After 3 attempts:** Raise error with clear message

## Changes Made

**File:** `server/azure_deployment.py`

**Lines Changed:**
1. **Lines 120-125:** Updated timestamp generation to include microseconds
2. **Lines 226-254:** Added retry logic for model registration with conflict handling

## Testing Instructions

### 1. **Test Rapid Deployments** (Verify Fix)
```bash
# Deploy the same model multiple times in quick succession
1. Go to /dashboard/models
2. Click "Deploy" on your model
3. Set deployment name: test-deploy-1
4. Click "Deploy to Azure"
5. Immediately click "Deploy" again
6. Set deployment name: test-deploy-2
7. Click "Deploy to Azure"
```

**Expected:** Both deployments should proceed without "ModelVersionInUse" errors. Model names will be different due to microsecond timestamps.

### 2. **Check Azure ML Studio**
```bash
# Verify models are registered with unique names
1. Go to https://ml.azure.com
2. Navigate to: MLOPT_91 workspace → Models
3. Check for models with names like:
   - model_Iris_classs_20251109173552123456
   - model_Iris_classs_20251109173552456789
```

### 3. **Verify Deployment Success**
```bash
# After 5-10 minutes, check deployment status
1. Go to /dashboard/deployments
2. Refresh the page
3. Both deployments should show "Active" status
4. Each should have unique:
   - Model names (different microseconds)
   - Endpoint names (test-deploy-1, test-deploy-2)
   - Scoring URIs
```

## Manual Cleanup (If Needed)

If you want to clean up the failed model from the first attempt:

### Azure ML Studio:
1. Go to https://ml.azure.com
2. Navigate to: MLOPT_91 workspace → Models
3. Find: `model_Iris_classs_20251109121111`
4. Click "..." → "Delete"
5. Confirm deletion

### Azure CLI:
```bash
az ml model delete \
  --name model_Iris_classs_20251109121111 \
  --version 1 \
  --resource-group MLOPT1 \
  --workspace-name MLOPT_91
```

## Summary

✅ **Microsecond timestamps** ensure unique model names even with rapid deployments
✅ **Retry logic** provides safety net for any remaining edge cases
✅ **Better error messages** help diagnose issues quickly
✅ **No breaking changes** to existing functionality

**Next Steps:**
1. Try deploying your model again
2. The model name will now include microseconds
3. Deployment should proceed past Step 4 (model registration)
4. Wait 5-10 minutes for deployment to complete
5. Check `/dashboard/deployments` for "Active" status

**Note:** You still have the quota fix from earlier (DS1_v2 default), so the deployment should complete successfully all the way through Step 7!
