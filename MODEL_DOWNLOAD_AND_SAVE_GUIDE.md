# Model Download & Save Guide

Complete guide for downloading trained models and saving them to Supabase database after training completion.

## 🎯 Overview

After completing model training in the Blueprints page, you now have multiple options to:
1. **Download Best Model** - Get the top-performing model as a pickle file
2. **Download All Models (ZIP)** - Download all trained models at once
3. **Save to Database** - Store the best model in Supabase with metadata
4. **Download Individual Models** - Pick specific models from the leaderboard

---

## 📦 Features Added

### 1. Download & Save Results Section
Located in the training results page after training completes.

#### **Quick Actions (3 Main Buttons)**

**🏆 Best Model**
- Downloads the top-performing model as `.pkl` file
- Endpoint: `/ml/download-model/{task_id}/best_model`
- File format: `best_model_{task_id}.pkl`

**📊 Performance Report**
- Downloads complete leaderboard with all metrics
- Endpoint: `/ml/download-leaderboard/{task_id}`
- File format: `leaderboard_{task_id}.csv`
- Contains: Model names, accuracy/R², training time, all metrics

**💾 Save to Database** (NEW)
- Opens SaveModelDialog to store model in Supabase
- Uploads pickle file to Supabase Storage
- Saves metadata to `trained_models` table
- Features:
  - Custom model name
  - Description field
  - Tags for categorization
  - Automatic file size tracking
  - User-isolated storage

### 2. Download All Models as ZIP (NEW)

**Features:**
- Downloads all trained models + leaderboard in one ZIP file
- Endpoint: `/ml/download-all-models/{task_id}`
- Includes:
  - `best_model.pkl`
  - All individual model files (if saved)
  - `leaderboard.csv`
- Auto-cleanup: ZIP file deleted after download

**Button Location:** Above the individual models list

### 3. Individual Model Downloads

**Features:**
- Shows top 5 models from leaderboard
- Each model displays:
  - Model name
  - Accuracy/R² score
  - Gold award icon for #1 model
- Click "Download" button to get specific model
- Hover effect for better UX

---

## 🚀 Usage Flow

### Step 1: Complete Training
1. Configure and start model training in Blueprints
2. Wait for training to complete
3. View results in the "Results" tab

### Step 2: Choose Your Option

#### **Option A: Quick Download Best Model**
```typescript
// Click "Best Model" button
// Downloads: best_model_{task_id}.pkl
// Use case: You just want the top performer
```

#### **Option B: Download All Models**
```typescript
// Click "Download All as ZIP" button
// Downloads: all_models_{task_id}.zip
// Contains: All models + leaderboard
// Use case: Archive full training session
```

#### **Option C: Save to Database**
```typescript
// Click "Save to Database" button
// Opens SaveModelDialog
// Enter:
//   - Model name (required)
//   - Description (optional)
//   - Tags (optional, comma-separated)
// Click "Save Model"
// File uploads to Supabase
// Record created in database
// Download available from /dashboard/models
```

#### **Option D: Download Specific Model**
```typescript
// Scroll to "Available Models" section
// Browse top 5 models
// Click "Download" on desired model
// Downloads: {model_name}_{task_id}.pkl
```

### Step 3: View Saved Models
Navigate to `/dashboard/models` to:
- View all saved models
- Download from cloud storage
- Delete old models
- View statistics

---

## 🛠️ Technical Implementation

### Backend Changes

#### **1. New Endpoint: Download All Models**
**File:** `server/ml_training.py`

```python
@router.get("/download-all-models/{task_id}")
async def download_all_models(task_id: str):
    """Download all models as a zip file"""
    import zipfile
    
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = training_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed")
    
    models_dir = Path("models") / task_id
    if not models_dir.exists():
        raise HTTPException(status_code=404, detail="Models directory not found")
    
    # Create a temporary zip file
    zip_path = models_dir / f"all_models_{task_id}.zip"
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all .pkl files
            for pkl_file in models_dir.glob("*.pkl"):
                zipf.write(pkl_file, pkl_file.name)
            
            # Add leaderboard if exists
            leaderboard_path = models_dir / "leaderboard.csv"
            if leaderboard_path.exists():
                zipf.write(leaderboard_path, "leaderboard.csv")
        
        return FileResponse(
            path=zip_path,
            filename=f"all_models_{task_id}.zip",
            media_type="application/zip",
            background=lambda: os.unlink(zip_path) if zip_path.exists() else None
        )
    except Exception as e:
        logger.error(f"Error creating zip file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create zip file: {str(e)}")
```

**Features:**
- Creates temporary ZIP file
- Adds all `.pkl` files from models directory
- Includes leaderboard CSV
- Auto-deletes ZIP after sending (background task)
- Error handling with logging

### Frontend Changes

#### **1. Updated Training Results Page**
**File:** `client/app/dashboard/blueprints/train/page.tsx`

**New Imports:**
```typescript
import { Download, Save, Package } from "lucide-react"
import SaveModelDialog from '@/components/SaveModelDialog'
```

**New State:**
```typescript
const [showSaveDialog, setShowSaveDialog] = useState(false)
const [userId, setUserId] = useState<string>('')
```

**Get User ID:**
```typescript
useEffect(() => {
  const fetchUser = async () => {
    const { data: { user } } = await supabase.auth.getUser()
    if (user) setUserId(user.id)
  }
  fetchUser()
}, [])
```

**Updated UI Structure:**
1. **3 Main Action Buttons** (grid layout)
   - Best Model (outline)
   - Performance Report (outline)
   - Save to Database (gradient blue-purple)

2. **Download All Section**
   - Header with "Download All as ZIP" button
   - Individual models list (top 5)
   - Each model has download button

3. **SaveModelDialog Component**
   - Positioned at end before closing `</section>`
   - Props: `open`, `onOpenChange`, `taskId`, `modelInfo`, `userId`

---

## 📊 Data Flow

### Download Flow
```
User clicks download button
    ↓
Frontend sends GET request
    ↓
Backend validates task_id and status
    ↓
Backend finds model file
    ↓
FileResponse streams file to browser
    ↓
Browser triggers download
```

### Save to Database Flow
```
User clicks "Save to Database"
    ↓
SaveModelDialog opens
    ↓
User fills form (name, description, tags)
    ↓
User clicks "Save Model"
    ↓
Frontend POST to /models/save
    ↓
Backend:
  1. Validates task_id exists
  2. Reads training results
  3. Loads model pickle file
  4. Uploads to Supabase Storage
  5. Creates database record
    ↓
Frontend shows success
    ↓
User can download or view in /dashboard/models
```

---

## 🗄️ Database Schema

### trained_models Table
```sql
- id (uuid, primary key)
- user_id (uuid, references auth.users)
- file_id (uuid, references files)
- model_name (text, user-defined name)
- algorithm (text, e.g., "Random Forest Classifier")
- task_type (text: classification/regression/time_series)
- metrics (jsonb, all performance metrics)
- file_path (text, Supabase Storage path)
- file_size (bigint, bytes)
- description (text, optional)
- tags (text[], optional)
- status (text, default: 'ready')
- created_at (timestamp)
- updated_at (timestamp)
- deleted (boolean, default: false)
```

### Supabase Storage
**Bucket:** `model-files` (private)
**Path Structure:** `{user_id}/{model_name}_{timestamp}.pkl`

---

## 🔐 Security

### Row Level Security (RLS)
- Users can only access their own models
- Storage bucket is private
- Signed URLs expire after 1 hour
- Delete is soft-delete (marks as deleted)

### Authentication
- All endpoints require user authentication
- User ID passed in request headers
- Supabase validates session tokens

---

## 📝 Usage Examples

### Example 1: Download Best Model
```typescript
// After training completes
onClick={() => {
  const configId = localStorage.getItem('ml_config_id')
  window.open(`${BACKEND_URL}/ml/download-model/${configId}/best_model`, '_blank')
}}
```

### Example 2: Save Model with Metadata
```typescript
// User fills form:
{
  model_name: "Customer Churn Predictor",
  description: "Random Forest model for predicting customer churn with 94% accuracy",
  tags: "churn, classification, production"
}

// SaveModelDialog sends:
await saveModel({
  task_id: configId,
  model_name: "Customer Churn Predictor",
  description: "Random Forest model...",
  tags: ["churn", "classification", "production"]
}, userId)

// Result: Model saved in database, accessible at /dashboard/models
```

### Example 3: Download All Models as ZIP
```typescript
onClick={() => {
  const configId = localStorage.getItem('ml_config_id')
  window.open(`${BACKEND_URL}/ml/download-all-models/${configId}`, '_blank')
}}

// Downloads: all_models_{configId}.zip
// Contains:
//   - best_model.pkl
//   - leaderboard.csv
//   - (any other saved model files)
```

---

## 🎨 UI Components

### Main Download Card
```
┌─────────────────────────────────────────┐
│ 💾 Download & Save Results             │
│ Download trained models, save to        │
│ database, or get performance reports    │
├─────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│ │📥 Best  │ │📥 Perf  │ │💾 Save  │   │
│ │  Model  │ │ Report  │ │   DB    │   │
│ └─────────┘ └─────────┘ └─────────┘   │
├─────────────────────────────────────────┤
│ 📦 Available Models    [Download All ZIP]│
│                                         │
│ 🏆 Random Forest   Acc: 94.2%  [⬇️]    │
│ 🔹 XGBoost         Acc: 93.8%  [⬇️]    │
│ 🔹 LightGBM        Acc: 93.1%  [⬇️]    │
└─────────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Issue: "Model not found"
**Cause:** Training didn't save models properly
**Solution:** 
- Check training completed successfully
- Verify `models/{task_id}/` directory exists
- Check server logs for save errors

### Issue: "Task not found"
**Cause:** Invalid or expired task_id
**Solution:**
- Re-train the model
- Check localStorage for `ml_config_id`
- Verify training_tasks cache not cleared

### Issue: SaveModelDialog fails
**Cause:** Supabase not configured or user not authenticated
**Solution:**
- Ensure SUPABASE_URL and SUPABASE_KEY in server/.env
- Check user is logged in
- Verify `supabase` Python package installed
- Check server logs for detailed error

### Issue: ZIP download incomplete
**Cause:** Large files timing out
**Solution:**
- Increase nginx timeout settings
- Use individual downloads for large models
- Check server disk space

---

## 📚 Related Files

### Frontend
- `client/app/dashboard/blueprints/train/page.tsx` - Main training page with download UI
- `client/components/SaveModelDialog.tsx` - Dialog for saving models to database
- `client/lib/api/models.ts` - API client for model operations
- `client/app/dashboard/models/page.tsx` - View saved models

### Backend
- `server/ml_training.py` - Training endpoints and model download
- `server/model_management.py` - Model CRUD operations for database
- `server/requirements.txt` - Python dependencies (includes supabase)

### Database
- `database/setup_models_table.sql` - Schema for trained_models table

### Documentation
- `REALTIME_MODEL_SAVING_IMPLEMENTATION.md` - Complete model management guide
- `MODEL_DOWNLOAD_AND_SAVE_GUIDE.md` - This file

---

## ✅ Testing Checklist

- [ ] Train a classification model
- [ ] Click "Best Model" - verify download
- [ ] Click "Performance Report" - verify CSV download
- [ ] Click "Save to Database" - verify dialog opens
- [ ] Fill form and save - verify upload to Supabase
- [ ] Click "Download All as ZIP" - verify ZIP contains files
- [ ] Click individual model download - verify specific model downloads
- [ ] Navigate to /dashboard/models - verify saved model appears
- [ ] Download from saved models page - verify signed URL works
- [ ] Delete saved model - verify soft delete
- [ ] Train regression model - repeat tests for regression
- [ ] Test with different users - verify RLS isolation

---

## 🚀 Future Enhancements

1. **Bulk Save to Database**
   - Save multiple models at once
   - Batch upload with progress bar

2. **Model Versioning**
   - Track different versions of same model
   - Compare versions side-by-side

3. **Model Deployment**
   - One-click deploy to API endpoint
   - Auto-generate prediction API

4. **Model Analytics**
   - Track download count
   - Most popular models
   - Usage statistics

5. **Collaborative Features**
   - Share models with team
   - Public model gallery
   - Model comments and ratings

---

## 📞 Support

For issues or questions:
1. Check server logs: `docker logs <container-name>`
2. Check browser console for frontend errors
3. Verify Supabase configuration
4. Review this documentation
5. Check related implementation guides

---

**Last Updated:** November 9, 2025
**Version:** 1.0.0
