# 🚀 Quick Start - Payment Integration

## ⚡ 3-Minute Setup

### 1. Run This SQL (Copy & Paste to Supabase)

1. Go to https://supabase.com/dashboard
2. Click your project
3. Click "SQL Editor" (left sidebar)
4. Click "New Query"
5. Copy EVERYTHING from `database/migrations/create_payment_tables.sql`
6. Paste and click "Run"

✅ Done? You'll see "Success. No rows returned"

### 2. Test It

```powershell
# Start your app (if not running)
cd client
npm run dev
```

Then visit: http://localhost:3000/dashboard/pricing

### 3. Make a Test Payment

1. Click "Subscribe Now" on any plan
2. Click the Google Pay button
3. Use card: `4111 1111 1111 1111`
4. Expiry: `12/25`, CVV: `123`
5. Done! 🎉

---

## ✅ What You Just Built

- ✅ Real Google Pay integration
- ✅ Secure payment database
- ✅ Subscription management
- ✅ Production-ready architecture

---

## 🎥 Show It Off

Perfect for your portfolio/demo:
1. Browse beautiful pricing page
2. Click subscribe
3. Official Google Pay checkout
4. Auto-redirect after payment
5. View active subscription

---

## 💡 It's in TEST Mode

- No real money charged
- Use test cards
- Safe to demo anywhere

---

## 📚 Need More?

Read the full guide: `PAYMENT_SETUP.md`

---

## 🆘 Not Working?

### Google Pay button not showing?
- Use Chrome or Edge browser
- Check browser console for errors

### Database error?
- Re-run the SQL migration
- Make sure you're logged in

### Still stuck?
- Check `PAYMENT_SETUP.md` troubleshooting section
- Review Supabase logs
