# 🐛 Model Deletion Bug Fix

## Problem
When users delete models from the dashboard, the UI shows "Model deleted" but the models still appear in the list after refresh. The models were being soft-deleted (marked as `status='deleted'`) but still being returned by the API.

## Root Cause
Two issues were found:

1. **Backend API (`server/model_management.py`)**: The `/list` endpoint was not filtering out deleted models
2. **Database View**: The `user_model_stats` view was counting deleted models in statistics

## Fix Applied

### 1. Backend - `server/model_management.py`
**Line 263**: Added filter to exclude deleted models

```python
# Before:
query = supabase.table('trained_models').select('*').eq('user_id', user_id)

# After:
query = supabase.table('trained_models').select('*').eq('user_id', user_id).neq('status', 'deleted')
```

### 2. Database View - Run in Supabase SQL Editor
**File**: `database/fix_model_stats_view.sql`

```sql
DROP VIEW IF EXISTS user_model_stats;

CREATE OR REPLACE VIEW user_model_stats AS
SELECT 
  user_id,
  COUNT(*) as total_models,
  COUNT(*) FILTER (WHERE status = 'ready') as ready_models,
  COUNT(*) FILTER (WHERE status = 'training') as training_models,
  COUNT(*) FILTER (WHERE status = 'failed') as failed_models,
  SUM(model_file_size) as total_storage_bytes,
  MAX(created_at) as last_model_created
FROM public.trained_models
WHERE status != 'deleted'  -- ✅ EXCLUDE DELETED MODELS
GROUP BY user_id;
```

## How to Apply

### Step 1: Backend is already fixed ✅
The Python file has been updated automatically.

### Step 2: Update Database View
1. Go to **Supabase Dashboard**: https://supabase.com/dashboard/project/ckiicaqieuoincwtzjky
2. Navigate to **SQL Editor**
3. Copy and paste the contents of `database/fix_model_stats_view.sql`
4. Click **Run**

### Step 3: Restart Backend Server
```bash
# Stop current server (Ctrl+C in terminal)
# Then restart:
cd server
docker run -d --rm -p 8000:8000 \
  --name mlopt \
  -v ${PWD}:/app \
  -e SUPABASE_URL="https://ckiicaqieuoincwtzjky.supabase.co" \
  -e SUPABASE_SERVICE_KEY="..." \
  mlopt:v3
```

## Testing
1. Go to `/dashboard/models`
2. Delete a model
3. Refresh the page
4. ✅ Deleted model should no longer appear
5. ✅ Model count in stats should decrease

## What Changed
- ✅ `/models/list` API now excludes deleted models
- ✅ Stats view excludes deleted models from counts
- ✅ Deleted models remain in database (soft delete) for audit purposes
- ✅ Frontend behavior unchanged (already working correctly)

## Notes
- This is a **soft delete** system - models marked as `status='deleted'` remain in the database
- To permanently delete: Manually update database or add a hard delete endpoint
- The fix maintains data integrity while improving user experience
