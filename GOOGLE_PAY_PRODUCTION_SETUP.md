# 🔧 Google Pay Setup - Complete Guide

## 📋 Current Status: TEST Mode (No Setup Needed)

You're currently in **TEST mode** which requires:
- ✅ Zero configuration
- ✅ No Google account
- ✅ No merchant verification
- ✅ Works immediately

---

## 🚀 Phase 1: TEST Mode (Current - No Setup)

### What You Have Now:
```typescript
environment="TEST"  // In your pricing page
merchantId: 'BCR2DN4T4F4AQRCX'  // Test merchant ID
```

### Test Cards:
- **Visa**: `4111 1111 1111 1111`
- **Mastercard**: `5555 5555 5555 4444`
- **Amex**: `3782 822463 10005`

### Limitations:
- ❌ No real money
- ❌ Only test cards work
- ✅ Perfect for portfolio/demo
- ✅ Shows complete payment flow

---

## 💼 Phase 2: PRODUCTION Setup (When Ready)

### Prerequisites:
- ✅ Business entity (Company/Sole Proprietor)
- ✅ Business bank account
- ✅ Tax ID number
- ✅ Website with SSL (HTTPS)
- ✅ Privacy policy & Terms of service

---

## 🔑 Step-by-Step: Google Pay Production Setup

### Step 1: Apply for Google Pay Business Account

1. **Go to Google Pay Business Console**
   ```
   https://pay.google.com/business/console
   ```

2. **Sign in** with your Google Business account

3. **Click** "Get Started" or "Apply Now"

4. **Fill Business Information:**
   - Business name
   - Business address
   - Tax ID (EIN in US)
   - Business type
   - Website URL
   - Estimated transaction volume

5. **Upload Documents:**
   - Business license
   - Bank account details
   - ID proof (passport/driver's license)
   - Address proof

6. **Wait for Approval** (1-7 days typically)

---

### Step 2: Configure Payment Gateway

Google Pay needs a **payment processor** to actually move money. Choose one:

#### Option A: Stripe (Recommended - Global)
```
1. Go to: https://stripe.com
2. Sign up for business account
3. Complete verification (KYC)
4. Get API keys from Dashboard
5. Enable Google Pay in Stripe Dashboard
```

**Stripe Configuration:**
```typescript
// Update your pricing page
tokenizationSpecification: {
  type: 'PAYMENT_GATEWAY',
  parameters: {
    gateway: 'stripe',
    'stripe:version': '2018-10-31',
    'stripe:publishableKey': 'pk_live_...'  // Your real Stripe key
  }
}
```

#### Option B: Razorpay (Best for India/Pakistan)
```
1. Go to: https://razorpay.com
2. Sign up and complete KYC
3. Get API keys
4. Enable Google Pay
```

**Razorpay Configuration:**
```typescript
tokenizationSpecification: {
  type: 'PAYMENT_GATEWAY',
  parameters: {
    gateway: 'razorpay',
    gatewayMerchantId: 'your_razorpay_id'
  }
}
```

#### Option C: PayPal
```
1. Go to: https://developer.paypal.com
2. Create business account
3. Get API credentials
4. Enable Google Pay
```

---

### Step 3: Get Google Merchant ID

1. **In Google Pay Console** (after approval)
2. Go to **"Settings"** → **"Business Profile"**
3. Copy your **Merchant ID** (looks like: `12345678901234567890`)
4. Update your code:

```typescript
// In pricing page
merchantInfo: {
  merchantId: '12345678901234567890',  // Your REAL merchant ID
  merchantName: 'MLOPT'
}
```

---

### Step 4: Switch to Production

Update your pricing page (`app/dashboard/pricing/page.tsx`):

```typescript
// Change this line:
environment="PRODUCTION"  // Was "TEST"

// And use your real merchant ID:
merchantId: '12345678901234567890'  // Your real ID from Google

// And real gateway credentials:
parameters: {
  gateway: 'stripe',  // or 'razorpay'
  gatewayMerchantId: 'your_real_gateway_id'
}
```

---

### Step 5: Domain Verification

1. **In Google Pay Console**
2. Go to **"Settings"** → **"Domains"**
3. **Add** your production domain:
   ```
   https://mlopt.com
   ```
4. **Verify ownership** (add meta tag or DNS record)
5. Google will send verification instructions

---

### Step 6: Test Production Mode

1. **Use real card** (small amount like $0.01)
2. **Verify** payment goes through
3. **Check** money in your bank account
4. **Refund** the test payment
5. **Monitor** for any errors

---

## 💳 Payment Flow - Production

```
User clicks Subscribe
    ↓
Google Pay sheet appears
    ↓
User enters REAL card
    ↓
Payment sent to Gateway (Stripe/Razorpay)
    ↓
Gateway charges the card
    ↓
Money moves to YOUR bank account
    ↓
Your database updated
    ↓
User gets access
```

---

## 🌍 Geographic Requirements

### ✅ Supported Countries (for receiving payments):

**Stripe:**
- 🇺🇸 United States
- 🇬🇧 United Kingdom
- 🇪🇺 Most EU countries
- 🇦🇪 UAE
- 🇸🇬 Singapore
- 🇦🇺 Australia
- 🇨🇦 Canada
- [Full list: https://stripe.com/global]

**Razorpay:**
- 🇮🇳 India
- 🇲🇾 Malaysia

**PayPal:**
- 200+ countries including Pakistan

### 🇵🇰 For Pakistan Specifically:

**Option 1: Use PayPal**
- Register business with PayPal
- Link Pakistani bank account
- Receive payments globally

**Option 2: Use Foreign Entity**
- Register company in UAE/Singapore
- Open business bank account there
- Use Stripe with that entity
- Transfer to Pakistan via wire transfer

**Option 3: Use Payoneer**
- Get Payoneer account
- Receive payments from Stripe to Payoneer
- Transfer to Pakistani bank

---

## 💰 Fees (Production)

### Google Pay:
- **Fee**: FREE (No transaction fees from Google)
- You only pay your payment gateway fees

### Stripe:
- **Standard**: 2.9% + $0.30 per transaction
- **International**: +1.5% for non-US cards

### Razorpay:
- **India**: 2% per transaction
- **International**: 3%

### PayPal:
- **Standard**: 2.9% + $0.30
- **International**: 4.4% + fixed fee

---

## 🔐 Security Requirements (Production)

### Your Website Must Have:
1. ✅ **SSL Certificate** (HTTPS)
2. ✅ **Privacy Policy** page
3. ✅ **Terms of Service** page
4. ✅ **Refund Policy** page
5. ✅ **Contact Information** page
6. ✅ **Secure server** (not localhost)

### Compliance:
- **PCI DSS**: Handled by payment gateway (Stripe/Razorpay)
- **GDPR**: If serving EU customers
- **Local laws**: Pakistan Electronic Transactions Ordinance

---

## 📊 Environment Comparison

| Feature | TEST Mode | PRODUCTION Mode |
|---------|-----------|-----------------|
| **Setup Time** | 0 minutes | 3-7 days |
| **Cost** | Free | Gateway fees |
| **Real Money** | ❌ No | ✅ Yes |
| **Test Cards** | ✅ Yes | ❌ No |
| **Google Approval** | Not needed | Required |
| **Payment Gateway** | Not needed | Required |
| **Bank Account** | Not needed | Required |
| **Business Entity** | Not needed | Required |
| **Perfect For** | Demo/Portfolio | Real business |

---

## 🎯 Recommended Path

### Now (Learning/Portfolio):
✅ Stay in TEST mode
- Show payment flow
- Demo to potential employers
- Perfect for portfolio

### When You're Ready to Launch:
1. **Month 1**: Apply for Google Pay merchant account
2. **Month 2**: Set up payment gateway (Stripe/Razorpay)
3. **Month 3**: Complete verification
4. **Month 4**: Switch to production

---

## 🚀 Quick Checklist for Going Live

- [ ] Business registered
- [ ] Bank account opened
- [ ] Tax ID obtained
- [ ] Google Pay merchant account approved
- [ ] Payment gateway account active (Stripe/Razorpay)
- [ ] Domain verified
- [ ] SSL certificate installed
- [ ] Legal pages created (Privacy, Terms, Refund)
- [ ] Production keys added to environment variables
- [ ] Test transaction completed successfully
- [ ] Refund tested
- [ ] Monitoring/logging set up

---

## 💡 Pro Tips

### For Demo/Portfolio (Now):
- ✅ Keep TEST mode
- ✅ Show the complete flow
- ✅ Mention "TEST mode" in demos
- ✅ Use test cards
- ✅ Highlight the architecture

### For Production (Later):
- ✅ Start with small transactions
- ✅ Monitor closely for first month
- ✅ Set up fraud detection
- ✅ Have customer support ready
- ✅ Keep detailed transaction logs

---

## 🆘 Common Issues

### Google Pay Not Showing:
- Browser must be Chrome/Edge
- Must have card saved in Google account (for production)
- Domain must be verified (for production)

### Payment Declining:
- Check gateway is active
- Verify API keys are correct
- Ensure sufficient funds in test gateway
- Check card is supported

### Money Not Appearing:
- Gateway takes 2-7 days to deposit
- Check gateway dashboard first
- Verify bank account is linked
- Look for holds or verification requirements

---

## 📚 Resources

**Google Pay:**
- Business Console: https://pay.google.com/business/console
- Documentation: https://developers.google.com/pay/api
- Test cards: https://developers.google.com/pay/api/android/guides/resources/test-card-suite

**Payment Gateways:**
- Stripe: https://stripe.com/docs
- Razorpay: https://razorpay.com/docs
- PayPal: https://developer.paypal.com/home

**Legal:**
- Privacy Policy Generator: https://www.privacypolicygenerator.info
- Terms Generator: https://www.termsandconditionsgenerator.com

---

## ✅ Summary

**Right Now:**
- You're in TEST mode ✅
- Zero setup needed ✅
- Works immediately ✅
- Perfect for demos ✅

**For Production:**
- Apply to Google Pay Business Console
- Set up payment gateway (Stripe/Razorpay/PayPal)
- Get merchant verification
- Switch environment to "PRODUCTION"
- Use real API keys
- Start accepting real payments!

---

**Questions?** Stay in TEST mode until you're ready to launch! 🚀
