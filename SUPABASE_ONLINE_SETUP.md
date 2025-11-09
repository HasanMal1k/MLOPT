# 🚀 Supabase Online Setup for Payments

## Step-by-Step Guide

### 1. Go to Supabase Table Editor

1. Open your Supabase project dashboard
2. Click **"Table Editor"** in the left sidebar

---

## 2. Create `payment_intents` Table

Click **"New Table"** and configure:

**Table Name:** `payment_intents`

**Columns:**

| Column Name | Type | Default Value | Extra |
|------------|------|---------------|-------|
| `id` | uuid | `gen_random_uuid()` | Primary Key ✓ |
| `user_id` | uuid | (none) | Foreign Key → auth.users |
| `plan_id` | text | (none) | - |
| `amount` | numeric | (none) | - |
| `currency` | text | `'USD'` | - |
| `status` | text | `'pending'` | - |
| `payment_token` | text | (none) | Nullable ✓ |
| `created_at` | timestamptz | `now()` | - |
| `paid_at` | timestamptz | (none) | Nullable ✓ |

**Enable RLS:** ✓ Check this box

Click **Save**

---

## 3. Create `subscriptions` Table

Click **"New Table"** again:

**Table Name:** `subscriptions`

**Columns:**

| Column Name | Type | Default Value | Extra |
|------------|------|---------------|-------|
| `id` | uuid | `gen_random_uuid()` | Primary Key ✓ |
| `user_id` | uuid | (none) | Foreign Key → auth.users, Unique ✓ |
| `plan_id` | text | (none) | - |
| `status` | text | `'active'` | - |
| `current_period_start` | timestamptz | (none) | - |
| `current_period_end` | timestamptz | (none) | - |
| `cancel_at_period_end` | boolean | `false` | - |
| `created_at` | timestamptz | `now()` | - |
| `updated_at` | timestamptz | `now()` | - |

**Enable RLS:** ✓ Check this box

Click **Save**

---

## 4. Add RLS Policies

### For `payment_intents` table:

1. Click on `payment_intents` table
2. Click **"RLS"** or **"Policies"** button
3. Click **"New Policy"**

**Policy 1: Select (Read)**
- Policy Name: `Users can view own payment intents`
- Allowed operation: `SELECT`
- Policy definition:
```sql
auth.uid() = user_id
```

**Policy 2: Insert (Create)**
- Policy Name: `Users can create own payment intents`
- Allowed operation: `INSERT`
- WITH CHECK:
```sql
auth.uid() = user_id
```

**Policy 3: Update**
- Policy Name: `Users can update own payment intents`
- Allowed operation: `UPDATE`
- USING:
```sql
auth.uid() = user_id
```

---

### For `subscriptions` table:

1. Click on `subscriptions` table
2. Click **"RLS"** or **"Policies"** button
3. Click **"New Policy"**

**Policy 1: Select (Read)**
- Policy Name: `Users can view own subscription`
- Allowed operation: `SELECT`
- Policy definition:
```sql
auth.uid() = user_id
```

**Policy 2: Insert**
- Policy Name: `Users can create own subscription`
- Allowed operation: `INSERT`
- WITH CHECK:
```sql
auth.uid() = user_id
```

**Policy 3: Update**
- Policy Name: `Users can update own subscription`
- Allowed operation: `UPDATE`
- USING:
```sql
auth.uid() = user_id
```

---

## 5. Add Trigger for Auto-Update (Optional)

This auto-updates the `updated_at` field in subscriptions.

1. Go to **SQL Editor**
2. Click **"New Query"**
3. Paste this:

```sql
-- Create function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS update_subscriptions_updated_at ON subscriptions;
CREATE TRIGGER update_subscriptions_updated_at
  BEFORE UPDATE ON subscriptions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
```

4. Click **Run**

---

## ✅ That's It!

Your database is ready. Now:

1. Visit: `http://localhost:3000/dashboard/pricing`
2. Click **"Subscribe Now"**
3. Use test card: `4111 1111 1111 1111`
4. Complete the payment!

---

## 🔍 Verify Setup

Go to **Table Editor** and you should see:
- ✅ `payment_intents` table (with RLS enabled)
- ✅ `subscriptions` table (with RLS enabled)
- ✅ Both tables have 3 policies each

---

## 🎯 Quick Test

After creating tables, test by:
1. Going to pricing page
2. Logging in (important!)
3. Clicking subscribe
4. Completing test payment
5. Check `payment_intents` table - should see new row
6. Check `subscriptions` table - should see your subscription

---

## 🆘 Troubleshooting

**Can't see tables?**
- Refresh the Table Editor page
- Make sure you clicked "Save" after creating each table

**RLS Error?**
- Make sure all 6 policies are created (3 per table)
- Verify "Enable RLS" is checked on both tables
- Make sure you're logged in when testing

**Foreign key error?**
- When creating `user_id` column, select "Foreign Key"
- Choose `auth.users` → `id`

---

**Need help?** Check if tables appear in your Table Editor sidebar!
