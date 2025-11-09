# Supabase Configuration for Model Management

## 🚨 Issue: 503 Service Unavailable

You're seeing this error because the Supabase credentials are not configured in your server:

```
503 Service Unavailable
Model saving is not configured. Supabase module not installed or credentials not set.
```

## ✅ Solution: Configure Supabase Credentials

### Step 1: Get Your Supabase Credentials

1. Go to your Supabase project dashboard
2. Click on **Settings** (gear icon in sidebar)
3. Click on **API** in the settings menu
4. Copy these two values:
   - **Project URL** (looks like: `https://xxxxx.supabase.co`)
   - **anon public key** (long string starting with `eyJ...`)

### Step 2: Create .env File

**Option A: Docker (Recommended)**

If running in Docker, add these to your Docker run command:

```powershell
docker run -it --rm -p 8000:8000 `
  -v ${PWD}:/app `
  -e SUPABASE_URL="https://your-project.supabase.co" `
  -e SUPABASE_KEY="your-anon-public-key-here" `
  mlopt:v3
```

**Option B: Local Development**

Create a `.env` file in the `server/` directory:

```bash
# server/.env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-public-key-here
```

### Step 3: Restart Your Server

After adding credentials:
- Docker: Stop and restart the container
- Local: Restart uvicorn

### Step 4: Verify Configuration

Check server logs for:
```
✅ Supabase client initialized for model management
```

Instead of:
```
⚠️ Supabase credentials not configured
```

## 📋 What This Enables

Once configured, you can:
- ✅ Save trained models to database
- ✅ View saved models at `/dashboard/models`
- ✅ Download models from cloud storage
- ✅ Delete old models
- ✅ Search and filter your model collection

## 🔒 Security Notes

- Never commit `.env` file to Git (it's in `.gitignore`)
- Use **anon public key**, not the service role key
- The anon key is safe for client-side use with RLS policies

## 🧪 Test Configuration

After setup, test with:

```bash
curl http://localhost:8000/models/list?user_id=YOUR_USER_ID
```

Should return:
```json
{
  "models": [],
  "total": 0,
  "message": "Success"
}
```

NOT:
```json
{
  "detail": "Model saving is not configured..."
}
```

## 📚 Database Setup Required

Make sure you've also run the database setup script:

```sql
-- In Supabase SQL Editor
-- Run: database/setup_models_table.sql
```

This creates:
- `trained_models` table
- `model-files` storage bucket
- RLS policies for security

## 🆘 Still Having Issues?

1. **Check Supabase project is active** - Visit your dashboard
2. **Verify credentials are correct** - Copy/paste carefully
3. **Check .env file location** - Must be in `server/` directory
4. **Restart server** - Changes require restart
5. **Check logs** - Look for Supabase initialization messages

---

**Quick Docker Command (with credentials):**

```powershell
cd server
docker run -it --rm -p 8000:8000 `
  -v ${PWD}:/app `
  -e SUPABASE_URL="YOUR_SUPABASE_URL" `
  -e SUPABASE_KEY="YOUR_SUPABASE_KEY" `
  mlopt:v3
```

Replace `YOUR_SUPABASE_URL` and `YOUR_SUPABASE_KEY` with your actual values!
