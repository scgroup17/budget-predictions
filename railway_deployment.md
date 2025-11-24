# Railway Deployment Guide for ML Flask Service

## Step 1: Create Railway Account

1. Go to https://railway.app/
2. Click "Login" and sign up with GitHub
3. ✓ Free tier: $5 credit/month (enough for testing)
4. ✓ Paid tier: $5/month + usage (recommended for production)

## Step 2: Prepare Your Project

Create a new directory `ml-service/` with these files:

### File: `requirements.txt`
```
flask==3.0.0
flask-restx==1.3.0
flask-cors==4.0.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
supabase==2.0.3
requests==2.31.0
gunicorn==21.2.0
```

### File: `Procfile`
```
web: gunicorn app:app --timeout 300 --workers 2
```

### File: `runtime.txt`
```
python-3.11.6
```

### File: `app.py`
```python
# Use the Flask service code from the previous artifact
```

### File: `.gitignore`
```
__pycache__/
*.pyc
*.pkl
*.json
.env
venv/
```

## Step 3: Deploy to Railway

### Option A: GitHub Deployment (Recommended)

1. **Create GitHub repo:**
   ```bash
   cd ml-service
   git init
   git add .
   git commit -m "Initial ML service"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/ml-service.git
   git push -u origin main
   ```

2. **Deploy on Railway:**
   - Go to Railway dashboard
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your `ml-service` repo
   - Railway will auto-detect Python and deploy

3. **Set Environment Variables:**
   In Railway dashboard → Variables tab, add:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
   ATTOM_API_KEY=your_attom_key_here
   PORT=5000
   ```

4. **Get your service URL:**
   - Railway will assign a URL like: `https://ml-service-production-xxxx.up.railway.app`
   - Copy this URL - you'll need it for Supabase edge functions

### Option B: Railway CLI Deployment

1. **Install Railway CLI:**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login:**
   ```bash
   railway login
   ```

3. **Initialize project:**
   ```bash
   cd ml-service
   railway init
   ```

4. **Deploy:**
   ```bash
   railway up
   ```

5. **Set environment variables:**
   ```bash
   railway variables set SUPABASE_URL=https://...
   railway variables set SUPABASE_SERVICE_ROLE_KEY=...
   railway variables set ATTOM_API_KEY=...
   ```

## Step 4: Upload Initial Models to Supabase Storage

You need to upload your existing model files before the Flask service can load them.

### Using Supabase Dashboard:

1. Go to your Supabase project → Storage
2. Create a new bucket called `ml-models` (private)
3. Create folder structure: `models/v1/`
4. Upload these files:
   - `budget_models_enhanced.pkl`
   - `label_encoders_enhanced.pkl`
   - `model_performance_enhanced.json`

### Using Python Script:

```python
from supabase import create_client

supabase = create_client(
    "https://your-project.supabase.co",
    "your_service_role_key"
)

# Create bucket
supabase.storage.create_bucket('ml-models', {'public': False})

# Upload files
with open('budget_models_enhanced.pkl', 'rb') as f:
    supabase.storage.from_('ml-models').upload(
        'models/v1/budget_models_enhanced.pkl',
        f.read()
    )

with open('label_encoders_enhanced.pkl', 'rb') as f:
    supabase.storage.from_('ml-models').upload(
        'models/v1/label_encoders_enhanced.pkl',
        f.read()
    )

with open('model_performance_enhanced.json', 'rb') as f:
    supabase.storage.from_('ml-models').upload(
        'models/v1/model_performance_enhanced.json',
        f.read()
    )

print("✓ Models uploaded to Supabase Storage")
```

## Step 5: Test Your Deployment

### Test Health Endpoint:
```bash
curl https://your-railway-url.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": 65,
  "model_version": "v1",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Test Prediction Endpoint:
```bash
curl -X POST https://your-railway-url.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "Plumbing Fixtures",
    "features": {
      "arv": 450000,
      "property_type": "SFR",
      "zip_code": "33178",
      "project_year": 2024,
      "building_size": 2049,
      "bedrooms": 3,
      "bathrooms": 2.5,
      "year_built": 2006
    }
  }'
```

Expected response:
```json
{
  "predicted_amount": 8500.42,
  "model_used": "Random Forest",
  "model_tier": "A",
  "model_r2": 0.5242,
  "model_mape": 76.24,
  "model_rmse": 5621.03,
  "model_version": "v1"
}
```

## Step 6: Configure Supabase Edge Functions

Now that your Python service is deployed, update your Supabase project:

Add this secret to Supabase:
```bash
# In Supabase dashboard → Project Settings → Edge Functions → Secrets
PYTHON_PREDICTION_API_URL=https://your-railway-url.up.railway.app
```

## Troubleshooting

### Models not loading:
- Check Railway logs: `railway logs`
- Verify Supabase Storage has files in `ml-models/models/v1/`
- Verify environment variables are set correctly

### Timeout errors:
- Increase timeout in `Procfile`: `--timeout 600`
- For retraining, consider using Railway's background workers

### Out of memory:
- Upgrade Railway plan (free tier: 512MB RAM)
- Or optimize model file sizes (compress with joblib)

## Monitoring & Logs

**View logs:**
```bash
railway logs
```

**Monitor requests:**
- Railway dashboard → Metrics tab
- Shows CPU, memory, request count

**Set up alerts:**
- Railway → Settings → Notifications
- Get email when service is down

## Cost Estimation

**Railway Pricing:**
- Free tier: $5 credit/month
- Paid tier: $5/month + $0.000463/GB-hour

**Estimated monthly cost for ML service:**
- Small (512MB RAM, <1000 requests/month): $5-8/month
- Medium (1GB RAM, <10000 requests/month): $10-15/month
- Large (2GB RAM, unlimited requests): $20-30/month

**Start with free tier**, upgrade if needed.

## Next Steps

Once your Python service is deployed:
1. ✅ Test all endpoints
2. ✅ Create Supabase edge functions (next artifact)
3. ✅ Run database migrations
4. ✅ Update React UI components
5. ✅ Test end-to-end flow
