# 🚀 2Checkout Integration Guide - Real Payments

## ✅ What You Have Now

**Complete 2Checkout (Verifone) integration:**
- ✅ Hosted checkout (easiest - redirect to 2Checkout page)
- ✅ Works in Pakistan and 200+ countries
- ✅ Accepts Visa, Mastercard, PayPal, and more
- ✅ Production-ready code

---

## 📋 Setup Steps (30 minutes)

### Step 1: Create 2Checkout Account

1. **Go to 2Checkout:**
   ```
   https://www.2checkout.com/signup/
   ```

2. **Sign Up:**
   - Choose "Merchant Account"
   - Enter business information
   - Select "Individual" or "Company"
   - Complete registration

3. **Verify Email:**
   - Check your email
   - Click verification link

---

### Step 2: Complete Account Verification

2Checkout needs these documents:

**Required:**
- ✅ Government-issued ID (Passport/CNIC)
- ✅ Proof of address (utility bill/bank statement)
- ✅ Business information (if company)
- ✅ Website URL (can be localhost for testing)

**Upload Process:**
1. Login to 2Checkout dashboard
2. Go to **"Account" → "Company Information"**
3. Upload required documents
4. Wait 1-3 business days for approval

---

### Step 3: Get API Credentials

Once approved:

1. **Login to 2Checkout Dashboard:**
   ```
   https://secure.2checkout.com/cpanel/
   ```

2. **Get Seller ID:**
   - Dashboard → **"Account"**
   - Copy your **Seller ID** (e.g., `901457471`)

3. **Get Publishable Key:**
   - Dashboard → **"Integrations" → "API"**
   - Enable **"API Access"**
   - Copy **"Publishable Key"**

4. **Get Secret Key:**
   - Same location as Publishable Key
   - Copy **"Secret Key"** (keep this private!)

---

### Step 4: Configure Environment Variables

Create/update `.env.local` in your `client/` folder:

```env
# Supabase (you already have these)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key

# 2Checkout Configuration
NEXT_PUBLIC_2CO_SELLER_ID=901457471
NEXT_PUBLIC_2CO_PUBLISHABLE_KEY=your_publishable_key_here
TWO_CHECKOUT_SECRET_KEY=your_secret_key_here

# Environment (sandbox or production)
NEXT_PUBLIC_2CO_ENVIRONMENT=sandbox

# Site URL
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

**For Production:**
```env
NEXT_PUBLIC_2CO_ENVIRONMENT=production
NEXT_PUBLIC_SITE_URL=https://yourdomain.com
```

---

### Step 5: Test in Sandbox Mode

2Checkout provides sandbox environment for testing:

1. **Set Environment:**
   ```env
   NEXT_PUBLIC_2CO_ENVIRONMENT=sandbox
   ```

2. **Test Cards:**
   ```
   Visa: 4111111111111111
   Mastercard: 5555555555554444
   Amex: 378282246310005
   
   CVV: Any 3 digits (e.g., 123)
   Expiry: Any future date (e.g., 12/25)
   ```

3. **Test Flow:**
   - Go to: `http://localhost:3000/dashboard/pricing`
   - Click "Subscribe Now"
   - Enter test card details
   - Complete payment
   - Check Supabase for payment record

---

### Step 6: Go Live

When ready for real payments:

1. **Switch to Production:**
   ```env
   NEXT_PUBLIC_2CO_ENVIRONMENT=production
   ```

2. **Update Site URL:**
   ```env
   NEXT_PUBLIC_SITE_URL=https://mlopt.com
   ```

3. **Deploy to Production:**
   ```bash
   # Build and deploy
   pnpm build
   # Deploy to Vercel/Netlify
   ```

4. **Test with Real Card:**
   - Use your real card
   - Make small payment ($0.50 or $1)
   - Verify in 2Checkout dashboard
   - Refund the test payment

---

## 💰 Pricing & Fees

### 2Checkout Fees:
- **Standard:** 3.5% + $0.35 per transaction
- **No monthly fees**
- **No setup fees**

### Payout Schedule:
- **First payout:** 30 days after first sale
- **After first payout:** Weekly or monthly
- **Minimum payout:** $100 (can be adjusted)

---

## 🌍 Supported Countries

### Can Sell From (Merchant):
- ✅ Pakistan
- ✅ India
- ✅ United States
- ✅ United Kingdom
- ✅ Most countries (200+)

### Can Accept From (Customer):
- ✅ Worldwide
- ✅ 200+ countries
- ✅ 87 payment methods
- ✅ 26 currencies

---

## 🔐 Security & Compliance

### Already Handled by 2Checkout:
- ✅ PCI DSS Level 1 compliant
- ✅ 3D Secure authentication
- ✅ Fraud detection
- ✅ Chargeback protection
- ✅ SSL encryption

### Your Responsibility:
- ✅ Keep secret keys private
- ✅ Use HTTPS in production
- ✅ Never commit keys to git

---

## 📊 Integration Types

You have **both** integrated:

### 1. Hosted Checkout (Simplest - Current)
```typescript
// User clicks Subscribe → Redirects to 2Checkout page
handleSimplePayment() // Already implemented
```

**Pros:**
- ✅ Easiest to implement
- ✅ 2Checkout handles everything
- ✅ No PCI compliance needed
- ✅ Professional checkout page

**Cons:**
- ⚠️ Leaves your site (redirect)

---

### 2. Inline Checkout (Advanced - Available)
```typescript
// User stays on your site, card form embedded
handlePlanSelect() // Already implemented
```

**Pros:**
- ✅ Never leave your site
- ✅ Better UX
- ✅ Custom styling

**Cons:**
- ⚠️ Slightly more complex
- ⚠️ Need card form UI

---

## 🔧 Dashboard Features

In 2Checkout Dashboard you can:

1. **View Transactions:**
   - Dashboard → "Sales"
   - See all payments
   - Refund if needed

2. **Manage Subscriptions:**
   - Dashboard → "Subscriptions"
   - Cancel/modify subscriptions
   - View recurring billing

3. **Reports:**
   - Dashboard → "Reports"
   - Revenue reports
   - Export to CSV

4. **Webhooks (Advanced):**
   - Dashboard → "Integrations" → "Webhooks"
   - Get notified of payment events
   - Auto-update your database

---

## 🚨 Important Notes

### Testing:
- ✅ Always test in **sandbox** mode first
- ✅ Use test cards (real cards won't work in sandbox)
- ✅ Verify database updates

### Production:
- ✅ Switch to **production** environment
- ✅ Use real credentials
- ✅ Test with small amount first
- ✅ Monitor for errors

### Bank Account:
- 🏦 Add bank account in dashboard for payouts
- 🇵🇰 For Pakistan: Use local bank account
- 💵 Choose currency (USD, PKR, etc.)

---

## 📝 Complete Checklist

### Setup Phase:
- [ ] Create 2Checkout account
- [ ] Verify email
- [ ] Upload documents (ID, address proof)
- [ ] Wait for approval (1-3 days)
- [ ] Get Seller ID
- [ ] Get Publishable Key
- [ ] Get Secret Key
- [ ] Add keys to `.env.local`
- [ ] Add bank account for payouts

### Testing Phase:
- [ ] Set environment to `sandbox`
- [ ] Run SQL script in Supabase
- [ ] Start local server
- [ ] Test with test card
- [ ] Verify payment in database
- [ ] Verify payment in 2Checkout dashboard
- [ ] Test refund

### Production Phase:
- [ ] Switch to `production` environment
- [ ] Deploy to production server
- [ ] Update site URL
- [ ] Test with real card (small amount)
- [ ] Verify payout settings
- [ ] Monitor first transactions
- [ ] Set up customer support

---

## 🆘 Troubleshooting

### "Seller ID not found"
- Check you copied correct Seller ID
- Verify environment (sandbox vs production)
- Re-enter credentials

### "Payment declined"
- In sandbox: Use test cards only
- In production: Check card is valid
- Check 2Checkout account is active

### "No checkout URL"
- Verify `NEXT_PUBLIC_2CO_SELLER_ID` is set
- Check environment variables loaded
- Restart dev server

### Payment successful but no subscription:
- Check Supabase connection
- Verify RLS policies
- Check API route `/api/payments/verify`

---

## 📚 Resources

**2Checkout Documentation:**
- Main Docs: https://www.2checkout.com/documentation/
- API Reference: https://knowledgecenter.2checkout.com/
- Test Cards: https://knowledgecenter.2checkout.com/Documentation/05Card_details_for_Testing

**Support:**
- Email: support@2checkout.com
- Phone: +1-888-690-2555
- Live Chat: In dashboard

**Dashboard:**
- Login: https://secure.2checkout.com/cpanel/
- Sandbox: https://sandbox.2checkout.com/cpanel/

---

## 🎯 Quick Start Commands

### 1. Database Setup
```sql
-- Run in Supabase SQL Editor
-- Copy from: database/setup_payments_online.sql
```

### 2. Environment Setup
```bash
# Create .env.local in client/ folder
cd client
notepad .env.local
```

### 3. Start Development
```bash
cd client
pnpm run dev
```

### 4. Test Payment
```
URL: http://localhost:3000/dashboard/pricing
Card: 4111111111111111
CVV: 123
Expiry: 12/25
```

---

## ✅ What Works Right Now

Even without 2Checkout account:
- ✅ Full UI is ready
- ✅ Payment flow implemented
- ✅ Database integration complete
- ✅ Error handling included
- ✅ Success/failure messages

**Just add credentials and you're live!**

---

## 🚀 Going Live (Pakistan Specific)

### For Pakistani Merchants:

1. **Bank Account:**
   - Add Pakistani bank account in 2Checkout
   - Accept USD or PKR
   - 2Checkout converts automatically

2. **Taxes:**
   - Register for NTN (if business)
   - Declare income from 2Checkout
   - Keep transaction records

3. **Withdrawals:**
   - Payouts to Pakistani banks work
   - Usually takes 2-3 business days
   - Minimum $100 payout

4. **Customer Support:**
   - Provide support email
   - Handle refund requests
   - Respond within 24 hours

---

## 💡 Pro Tips

1. **Start Small:**
   - Test with friends/family
   - Offer discount for early users
   - Collect feedback

2. **Monitor Closely:**
   - Check dashboard daily (first month)
   - Watch for failed payments
   - Track refund requests

3. **Customer Experience:**
   - Clear pricing
   - Easy cancellation
   - Fast support response

4. **Security:**
   - Never share secret keys
   - Use environment variables
   - Enable 2FA on 2Checkout account

---

## 🎉 You're Ready!

Your payment system is **production-ready**. Just:

1. Sign up for 2Checkout → 10 minutes
2. Get verified → 1-3 days
3. Add credentials → 2 minutes
4. Start accepting payments! 🚀

**Total time to live: ~3 days** (mostly waiting for verification)

For now, the UI and flow are perfect for your portfolio. When you want real payments, follow this guide! 💳
