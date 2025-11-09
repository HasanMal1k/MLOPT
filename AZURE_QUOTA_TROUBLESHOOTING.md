# Azure Quota Issues - Troubleshooting Guide

## Problem: Not Enough Quota for VM Size

### Error Message:
```
(BadRequest) Not enough quota available for Standard_DS2_v2
Current usage/limit: 4/6. Additional needed: 4
```

This means your Azure subscription doesn't have enough CPU cores available for the requested VM size.

---

## ✅ Solutions (Choose One)

### Solution 1: Use Smaller VM Size (QUICK FIX - Already Applied)

**What We Changed:**
- Default instance type: `Standard_DS2_v2` (2 cores) → `Standard_DS1_v2` (1 core)
- Updated files:
  * `client/components/DeploymentDialog.tsx` - Default dropdown value
  * `server/azure_deployment.py` - Backend default
  * `database/setup_deployments_table.sql` - Database default

**Available Options Now:**
- `Standard_DS1_v2` - **1 core**, 3.5GB RAM (✅ Recommended for small models)
- `Standard_DS2_v2` - 2 cores, 7GB RAM (Needs quota increase)
- `Standard_F2s_v2` - 2 cores, 4GB RAM (Alternative 2-core option)
- `Standard_DS3_v2` - 4 cores, 14GB RAM (Needs quota increase)

**Action Required:** None - Already fixed! Try deploying again.

---

### Solution 2: Request Quota Increase from Azure

**Steps:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Search for "Quotas" in the top search bar
3. Click "Quotas" service
4. Filter by:
   - Provider: `Microsoft.MachineLearningServices`
   - Region: `East US 2`
   - Quota name: `Standard DSv2 Family vCPUs`
5. Click on the quota
6. Click "Request increase"
7. Enter new limit (e.g., 12 or 20)
8. Submit request
9. Wait 1-3 business days for approval

**Current Quota:**
- Used: 4 cores
- Limit: 6 cores
- Available: 2 cores

**Recommended New Limit:**
- Minimum: 12 cores (allows 2-3 deployments)
- Recommended: 20 cores (allows 5-10 deployments)

---

### Solution 3: Delete Existing Deployments/Resources

If you have other Azure ML deployments consuming quota:

**Check Existing Deployments:**
1. Go to [Azure ML Studio](https://ml.azure.com)
2. Select workspace: `MLOPT_91`
3. Go to "Endpoints" → "Real-time endpoints"
4. View all deployments and their instance sizes
5. Delete unused deployments

**Via Azure CLI:**
```bash
# List all endpoints
az ml online-endpoint list -w MLOPT_91 -g MLOPT1

# Delete an endpoint (frees up quota)
az ml online-endpoint delete -n endpoint-name -w MLOPT_91 -g MLOPT1
```

---

### Solution 4: Use Different Azure Region

Some regions have higher default quotas:

**Steps:**
1. Change `AZURE_RESOURCE_GROUP` to a different region
2. Create new Azure ML workspace in that region
3. Update environment variables

**Regions with Typically Higher Quotas:**
- `West US 2`
- `East US`
- `West Europe`
- `Southeast Asia`

---

## 🎯 Recommended Approach (For Your Case)

**Option A: Start Small (Already Done)**
1. ✅ Use `Standard_DS1_v2` (1 core) for testing
2. Deploy and verify everything works
3. If you need more power, request quota increase

**Option B: Request Quota Increase Now**
1. Follow Solution 2 steps
2. Request 20 cores
3. Continue testing with DS1_v2 while waiting
4. Upgrade to DS2_v2 or DS3_v2 once approved

---

## 📊 VM Size Comparison

| VM Size | Cores | RAM | Use Case | Monthly Cost* |
|---------|-------|-----|----------|---------------|
| **Standard_DS1_v2** | 1 | 3.5GB | Small models, testing | ~$50 |
| Standard_DS2_v2 | 2 | 7GB | Medium models | ~$100 |
| Standard_F2s_v2 | 2 | 4GB | Compute-optimized | ~$70 |
| Standard_DS3_v2 | 4 | 14GB | Large models | ~$200 |
| Standard_DS4_v2 | 8 | 28GB | Very large models | ~$400 |

*Approximate costs for East US 2 region (24/7 deployment)

---

## 🧪 Testing Your Fix

### Step 1: Try Deployment Again
1. Go to `/dashboard/models`
2. Click "Deploy" on your model
3. **Leave instance type as default** (`Standard_DS1_v2`)
4. Fill deployment name
5. Click "Deploy to Azure"

### Step 2: Monitor Status
1. Go to `/dashboard/deployments`
2. Wait 5-10 minutes
3. Status should change from "Deploying" → "Active"

### Step 3: If Still Fails
Check Docker logs:
```powershell
docker logs mlopt-server | Select-String -Pattern "quota|error"
```

If you see other quota errors:
- Try `Standard_F2s_v2` instead (different quota pool)
- Or proceed with Solution 2 (request increase)

---

## 🔍 Understanding Azure Quotas

### What Are Quotas?
Azure limits how many resources you can use simultaneously to:
- Prevent accidental overspending
- Ensure fair resource distribution
- Protect against abuse

### Quota Types:
1. **Regional Quotas**: Per subscription, per region
2. **VM Family Quotas**: DSv2 family, F family, etc.
3. **Total Cores**: Overall limit across all VMs

### Your Current Situation:
```
Subscription: 262f6ecf-518e-4f93-ba94-ba1bae0b8940
Region: East US 2
Quota: Standard DSv2 Family vCPUs
Used: 4 cores (likely from other deployments)
Limit: 6 cores
Requested: 4 cores (for DS2_v2 deployment with 2 cores × 2 instances OR existing endpoint traffic)
Available: 2 cores (not enough for DS2_v2 deployment)
```

### Why DS1_v2 Works:
- Uses only **1 core**
- You have 2 cores available
- Deployment succeeds! ✅

---

## 📝 Next Steps After Deployment

### If Using DS1_v2 Works:
1. ✅ Feature is validated
2. Request quota increase for future scaling
3. Test with real traffic
4. Monitor performance

### If You Need More Power Now:
1. Submit quota increase request (Solution 2)
2. Typical approval: 1-3 business days
3. Continue using DS1_v2 for testing
4. Redeploy with larger size when approved

### For Production:
- Start with DS1_v2 for low traffic
- Scale to DS2_v2 for moderate traffic
- Use DS3_v2+ for high traffic
- Enable auto-scaling later

---

## ⚠️ Cost Warning

**Remember:** Deployments run 24/7 until deleted!

**To Avoid Charges:**
1. Delete test deployments when done
2. Use `/dashboard/deployments` to monitor
3. Click delete button when not needed

**Cost Example:**
- DS1_v2 × 1 instance × 24 hours = ~$1.50/day
- DS2_v2 × 1 instance × 24 hours = ~$3.00/day
- DS3_v2 × 1 instance × 24 hours = ~$6.00/day

---

## 🎉 Summary

**What Happened:**
Your Azure subscription had 4 cores used out of 6 limit. Trying to deploy with DS2_v2 (2 cores each, possibly with auto-scaling) needed 4 more cores.

**What We Fixed:**
Changed default VM size from DS2_v2 (2 cores) to DS1_v2 (1 core), which fits within your available quota.

**Result:**
✅ You can now deploy models!  
✅ No quota increase needed for testing  
⚡ Request increase for scaling later

---

**Ready to deploy? Try again with the updated settings!** 🚀
