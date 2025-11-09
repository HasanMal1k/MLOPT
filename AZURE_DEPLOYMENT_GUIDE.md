# Azure ML Deployment Implementation Guide

## Overview
Complete implementation of Azure Machine Learning deployment feature allowing users to deploy trained models to Azure ML workspace with managed online endpoints.

## ✅ Completed Components

### 1. Database Setup
**File:** `database/setup_deployments_table.sql`

**Features:**
- `deployments` table with 18 columns
- Foreign keys to `auth.users` and `trained_models`
- Status tracking: deploying, active, failed, deleted
- RLS policies for user isolation
- Automatic timestamp management
- User statistics view

**To Setup:**
```sql
-- Run this in Supabase SQL Editor
-- File: database/setup_deployments_table.sql
-- Creates deployments table, RLS policies, and stats view
```

### 2. Backend API
**File:** `server/azure_deployment.py`

**Endpoints:**
- `POST /deployments/deploy` - Deploy model to Azure ML
- `GET /deployments/list` - List user's deployments
- `GET /deployments/stats` - Get deployment statistics
- `DELETE /deployments/{id}` - Soft delete deployment

**Features:**
- Background task processing (5-10 min deployments)
- MLflow model conversion
- Azure ML workspace integration
- Managed online endpoints
- Error handling with database updates
- Service role key authentication

**Environment Variables Required:**
```bash
AZURE_TENANT_ID=75df096c-8b72-48e4-9b91-cbf79d87ee3a
AZURE_CLIENT_ID=b83e920b-886b-4ca5-8674-1e82be0572d7
AZURE_CLIENT_SECRET=n.z8Q~nHgqOrzvzo8jNLfoy3~Qvv3vWIOSjvObaB
AZURE_SUBSCRIPTION_ID=262f6ecf-518e-4f93-ba94-ba1bae0b8940
AZURE_RESOURCE_GROUP=MLOPT1
AZURE_WORKSPACE_NAME=MLOPT_91
```

### 3. Python Dependencies
**Updated:** `server/requirements.txt`

**Added Packages:**
```
azure-ai-ml>=1.11.0
azure-identity>=1.14.0
mlflow>=2.8.0
```

### 4. FastAPI Integration
**Updated:** `server/main.py`

**Changes:**
- Import azure_deployment router
- Register router with app
- Accessible at `/deployments/*` endpoints

### 5. Frontend API Client
**File:** `client/lib/api/deployments.ts`

**Functions:**
```typescript
deployModel(request, userId) → DeployModelResponse
listDeployments(userId, status?) → { deployments, total }
getDeploymentStats(userId) → DeploymentStats
deleteDeployment(deploymentId, userId) → { message }
```

**TypeScript Interfaces:**
- Deployment
- DeploymentStats
- DeployModelRequest
- DeployModelResponse

### 6. DeploymentDialog Component
**File:** `client/components/DeploymentDialog.tsx`

**Features:**
- Modal dialog for deployment configuration
- Form fields:
  * Deployment name (required)
  * Endpoint name (optional, auto-generates if empty)
  * Instance type (dropdown: DS2_v2, DS3_v2, DS4_v2, F4s_v2)
  * Instance count (1-10)
  * Description (optional)
- Two-stage flow: Configure → Deploy → Success
- Error handling with user-friendly messages
- Link to deployments page on success

### 7. Deployments Dashboard
**File:** `client/app/dashboard/deployments/page.tsx`

**Features:**
- Stats cards: Total, Active, Deploying, Failed
- Search functionality
- Status filter buttons (All, Active, Deploying, Failed)
- Auto-refresh button
- Deployment cards showing:
  * Status badge with icon
  * Model name and algorithm
  * Endpoint name
  * Instance configuration
  * Scoring URI (copyable)
  * Swagger URI (external link)
  * Error messages (for failed deployments)
  * Timestamps
- Delete functionality
- Empty state with call-to-action

### 8. Models Page Enhancement
**Updated:** `client/app/dashboard/models/page.tsx`

**Changes:**
- Added "Deploy" button next to Download
- Deploy button opens DeploymentDialog
- Cloud icon for visual clarity
- Toast notification on deployment start
- State management for dialog

### 9. Sidebar Navigation
**Updated:** `client/components/app-sidebar.tsx`

**Changes:**
- Added "Deployments" menu item in Account section
- Cloud icon
- "New" badge
- Links to `/dashboard/deployments`
- Positioned between "My Models" and "Subscription"

## 🔧 Setup Instructions

### Step 1: Database Setup
1. Open Supabase SQL Editor
2. Copy contents of `database/setup_deployments_table.sql`
3. Execute the script
4. Verify success messages appear
5. Check that `deployments` table exists
6. Test RLS policies with a test user

### Step 2: Update Docker Image
```powershell
# Navigate to server directory
cd server

# Rebuild Docker image with Azure packages
docker build -t mlopt:v3 --no-cache .

# Verify build success
docker images | grep mlopt
```

### Step 3: Update Docker Run Command
```powershell
# Stop existing container
docker stop mlopt-server

# Run with Azure environment variables
docker run -d --rm -p 8000:8000 `
  --name mlopt-server `
  -v ${PWD}:/app `
  -e SUPABASE_URL="https://ckiicaqieuoincwtzjky.supabase.co" `
  -e SUPABASE_SERVICE_KEY="YOUR_SERVICE_ROLE_KEY" `
  -e AZURE_TENANT_ID="75df096c-8b72-48e4-9b91-cbf79d87ee3a" `
  -e AZURE_CLIENT_ID="b83e920b-886b-4ca5-8674-1e82be0572d7" `
  -e AZURE_CLIENT_SECRET="n.z8Q~nHgqOrzvzo8jNLfoy3~Qvv3vWIOSjvObaB" `
  -e AZURE_SUBSCRIPTION_ID="262f6ecf-518e-4f93-ba94-ba1bae0b8940" `
  -e AZURE_RESOURCE_GROUP="MLOPT1" `
  -e AZURE_WORKSPACE_NAME="MLOPT_91" `
  mlopt:v3

# Check logs
docker logs -f mlopt-server
```

### Step 4: Install Frontend Dependencies
```powershell
# Navigate to client directory
cd client

# Install dependencies (if needed)
pnpm install

# Run development server
pnpm dev
```

### Step 5: Test Deployment Flow

#### A. Train a Model
1. Go to `/dashboard/upload`
2. Upload a CSV file
3. Go to `/dashboard/blueprints/train`
4. Train a classification or regression model
5. Save model to database

#### B. Deploy Model
1. Go to `/dashboard/models`
2. Find a saved model
3. Click "Deploy" button
4. Fill deployment form:
   - Deployment name: `my-first-deployment`
   - Leave endpoint empty (auto-generate)
   - Instance type: `Standard_DS2_v2`
   - Instance count: `1`
   - Description: "Test deployment"
5. Click "Deploy to Azure"
6. Wait for success message
7. Click "View Deployments"

#### C. Monitor Deployment
1. Should see deployment with "Deploying" status
2. Refresh page every 2-3 minutes
3. Status should change to "Active" after 5-10 minutes
4. Scoring URI will appear when active
5. Click external link to view Swagger docs

#### D. Test Scoring Endpoint
```powershell
# Get scoring URI from deployment page
$scoringUri = "https://iris-endpoint-20251108095359.eastus2.inference.ml.azure.com/score"

# Prepare test data
$body = @{
    "data" = @(
        @{
            "sepal_length" = 5.1
            "sepal_width" = 3.5
            "petal_length" = 1.4
            "petal_width" = 0.2
        }
    )
} | ConvertTo-Json

# Make prediction request
Invoke-RestMethod -Uri $scoringUri -Method POST -Body $body -ContentType "application/json" -Headers @{"Authorization"="Bearer YOUR_API_KEY"}
```

## 🎯 User Flow

### Scenario 1: First-Time Deployment
1. User trains model → Saves to database
2. Goes to "My Models" page
3. Clicks "Deploy" on a model
4. Fills deployment form
5. Clicks "Deploy to Azure"
6. Sees "Deployment Started" success message
7. Navigates to "Deployments" page
8. Sees deployment with "Deploying" status
9. Waits 5-10 minutes
10. Refreshes page
11. Deployment shows "Active" with scoring URI
12. Copies scoring URI for production use

### Scenario 2: Multiple Deployments
1. User has 3 trained models
2. Deploys Model A to endpoint "production-endpoint"
3. Deploys Model B to same endpoint (creates new deployment)
4. Deploys Model C to new endpoint "testing-endpoint"
5. Views all deployments in dashboard
6. Filters by "Active" status
7. Each deployment isolated by user

### Scenario 3: Failed Deployment
1. User deploys model with invalid config
2. Deployment starts (status: "Deploying")
3. Azure ML fails to create endpoint
4. Status changes to "Failed"
5. Error message displayed: "Authentication failed"
6. User fixes credentials
7. Tries deployment again

### Scenario 4: Deployment Cleanup
1. User has 5 deployments
2. 3 are active, 2 are failed
3. Selects failed deployment
4. Clicks delete button
5. Confirms deletion
6. Deployment soft-deleted (status: "deleted")
7. No longer appears in list
8. Stats updated: Total = 5, Active = 3, Failed = 0

## 📊 Deployment Process (Backend)

### Phase 1: Initiation (1-2 seconds)
1. Receive deployment request
2. Validate user and model
3. Generate unique names
4. Create database record (status: "deploying")
5. Return deployment ID immediately
6. Start background task

### Phase 2: Model Preparation (30-60 seconds)
1. Download pickle file from Supabase Storage
2. Load model with joblib
3. Convert to MLflow format
4. Save MLflow model locally
5. Log progress

### Phase 3: Azure Authentication (5-10 seconds)
1. Create ClientSecretCredential
2. Initialize MLClient
3. Verify workspace access
4. Log authentication success

### Phase 4: Model Registration (1-2 minutes)
1. Create Model entity
2. Upload MLflow model to Azure
3. Register in workspace
4. Get model version
5. Log registration success

### Phase 5: Endpoint Setup (2-5 minutes)
1. Check if endpoint exists
2. If not, create new endpoint
3. Configure authentication (key-based)
4. Wait for endpoint provisioning
5. Get endpoint details

### Phase 6: Deployment Creation (2-5 minutes)
1. Create ManagedOnlineDeployment
2. Allocate compute instances
3. Deploy model to endpoint
4. Wait for deployment completion
5. Run health checks

### Phase 7: Finalization (10-20 seconds)
1. Get scoring URI and swagger URI
2. Update database record:
   - status: "active"
   - azure_model_version
   - scoring_uri
   - swagger_uri
   - deployed_at timestamp
3. Cleanup temporary files
4. Log completion

### Error Handling
- Any failure updates database:
  - status: "failed"
  - error_message: exception details
- User sees error in deployments page
- Temporary files cleaned up
- Logged for debugging

## 🔒 Security Features

### Row Level Security (RLS)
- All deployment queries filtered by `auth.uid() = user_id`
- Users can only see their own deployments
- Enforced at database level

### Service Role Key
- Backend uses service role key
- Bypasses RLS for system operations
- Never exposed to frontend

### Azure Credentials
- Stored as environment variables
- Not committed to git (in .gitignore)
- Only accessible by backend container

### API Key Authentication
- Azure endpoints use key-based auth
- Keys managed in Azure portal
- Rotatable for security

## 📈 Performance Considerations

### Background Processing
- Deployments run asynchronously
- User doesn't wait for 5-10 minute process
- Status updates polled by frontend

### Database Indexing
- Indexes on user_id, status, endpoint_name
- Fast queries for filtered lists
- Efficient stats aggregation

### Caching Opportunities
- Stats can be cached (1-5 minutes)
- Deployment list can be cached
- Invalidate on create/delete

### Resource Management
- Temporary model files cleaned up
- Docker container memory limits
- Azure compute auto-scaling

## 🐛 Troubleshooting

### Issue: Azure packages not found
**Solution:**
```powershell
# Rebuild Docker image
docker build -t mlopt:v3 --no-cache .
```

### Issue: Authentication failed
**Solution:**
```powershell
# Verify environment variables
docker exec mlopt-server env | grep AZURE

# Check service principal permissions
# Must have Contributor role on workspace
```

### Issue: Deployment stuck in "Deploying"
**Solution:**
1. Check Azure ML studio for deployment status
2. Look at Docker logs: `docker logs mlopt-server`
3. Verify quota availability in Azure
4. Check workspace endpoint limits

### Issue: Azure Quota Exceeded
**Error:** `Not enough quota available for Standard_DS2_v2`

**Quick Fix (Already Applied):**
- Default changed to `Standard_DS1_v2` (1 core instead of 2)
- Uses less quota, perfect for testing

**Full Solution:**
1. See `AZURE_QUOTA_TROUBLESHOOTING.md` for complete guide
2. Option A: Use DS1_v2 (1 core) - No action needed
3. Option B: Request quota increase from Azure Portal
4. Option C: Delete unused deployments to free quota
5. Option D: Use different Azure region

**Steps to Request Quota Increase:**
1. Go to Azure Portal → Search "Quotas"
2. Filter: Provider=`Microsoft.MachineLearningServices`, Region=`East US 2`
3. Find: `Standard DSv2 Family vCPUs`
4. Request increase to 12-20 cores
5. Wait 1-3 business days

### Issue: Model conversion fails
**Solution:**
- Ensure model is sklearn-compatible
- Check PyCaret model type
- Verify joblib can load pickle file
- Test locally: `joblib.load('model.pkl')`

### Issue: RLS policy blocks query
**Solution:**
```sql
-- Check policies
SELECT * FROM pg_policies WHERE tablename = 'deployments';

-- Verify user_id matches auth.uid()
SELECT auth.uid(), user_id FROM deployments;
```

## 🚀 Future Enhancements

### Short-term (Next Sprint)
- [ ] Deployment logs viewer
- [ ] Cost estimation before deploy
- [ ] Batch deployment (multiple models)
- [ ] Deployment templates

### Mid-term (Next Month)
- [ ] Auto-scaling configuration
- [ ] Custom environment variables
- [ ] Deployment versioning
- [ ] Rollback functionality
- [ ] A/B testing support

### Long-term (Next Quarter)
- [ ] Multi-cloud support (AWS SageMaker, GCP Vertex AI)
- [ ] Kubernetes deployment option
- [ ] Edge deployment (ONNX export)
- [ ] Real-time monitoring dashboard
- [ ] Cost analytics and optimization
- [ ] Model performance tracking

## 📝 API Reference

### POST /deployments/deploy
**Request:**
```json
{
  "model_id": "uuid",
  "endpoint_name": "my-endpoint",  // optional
  "instance_type": "Standard_DS2_v2",
  "instance_count": 1,
  "description": "Production deployment"
}
```

**Response:**
```json
{
  "deployment_id": "uuid",
  "status": "deploying",
  "message": "Deployment started. This may take 5-10 minutes.",
  "endpoint_name": "endpoint-20241108..."
}
```

### GET /deployments/list
**Query Params:**
- `user_id` (required)
- `status` (optional): deploying, active, failed

**Response:**
```json
{
  "deployments": [
    {
      "id": "uuid",
      "deployment_name": "deploy-20241108...",
      "endpoint_name": "endpoint-20241108...",
      "status": "active",
      "scoring_uri": "https://...",
      "trained_models": {
        "model_name": "My Model",
        "algorithm": "Random Forest"
      },
      ...
    }
  ],
  "total": 5
}
```

### GET /deployments/stats
**Query Params:**
- `user_id` (required)

**Response:**
```json
{
  "total_deployments": 10,
  "active_deployments": 7,
  "deploying_count": 2,
  "failed_deployments": 1,
  "last_deployment_date": "2024-11-08T10:30:00Z"
}
```

### DELETE /deployments/{id}
**Query Params:**
- `user_id` (required)

**Response:**
```json
{
  "message": "Deployment deleted successfully"
}
```

## ✅ Testing Checklist

- [ ] Database schema applied successfully
- [ ] Docker image builds with Azure packages
- [ ] Container runs with environment variables
- [ ] Frontend pages load without errors
- [ ] Deploy button appears on models page
- [ ] DeploymentDialog opens and closes
- [ ] Form validation works
- [ ] Deployment starts (status: deploying)
- [ ] Deployments page shows new deployment
- [ ] Stats cards display correct counts
- [ ] Search filters deployments
- [ ] Status filter works
- [ ] Deployment completes (status: active)
- [ ] Scoring URI is displayed
- [ ] Swagger link opens Azure docs
- [ ] Delete functionality works
- [ ] User isolation enforced
- [ ] Error handling shows messages

## 📚 Documentation Links

- [Azure ML Python SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-ml-readme)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Supabase RLS Policies](https://supabase.com/docs/guides/auth/row-level-security)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [Next.js App Router](https://nextjs.org/docs/app)

---

**Implementation Status:** ✅ Complete
**Last Updated:** 2024-11-08
**Version:** 1.0.0
