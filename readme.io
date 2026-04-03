# Insurance Charges Prediction — ML + Cloud Backend

## Project Structure

```
insurance_project/
├── insurance.csv          ← dataset
├── train.py               ← trains all models, saves the best one
├── app/
│   └── main.py            ← FastAPI backend (all endpoints)
├── model/                 ← auto-created by train.py
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.json
│   └── metadata.json
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## API Endpoints

| Method | Route              | What it does                            |
|--------|--------------------|-----------------------------------------|
| GET    | /                  | Welcome message                         |
| GET    | /health            | Model name, R², MAE — live status       |
| GET    | /model/info        | Full comparison of all trained models   |
| POST   | /predict           | Predict charge for one patient          |
| POST   | /predict/batch     | Predict for up to 100 patients at once  |
| GET    | /predict/example   | Sample prediction (instant test)        |
| GET    | /stats/dataset     | Dataset statistics                      |

---

## Step-by-Step: Run Locally

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train.py
```
This trains 5 models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting),
picks the best by cross-validation R², and saves all artifacts into `model/`.

### Step 3 — Start the API server
```bash
uvicorn app.main:app --reload
```

### Step 4 — Test it
Open your browser at:
- http://localhost:8000/docs        ← Interactive Swagger UI (great for exhibition)
- http://localhost:8000/predict/example  ← Quick prediction test
- http://localhost:8000/health      ← Server status

### Step 5 — Make a prediction (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "bmi": 28.5,
    "children": 2,
    "sex": "male",
    "smoker": "no",
    "region": "southeast"
  }'
```

Expected response:
```json
{
  "predicted_charge": 7842.15,
  "predicted_charge_str": "$7,842.15",
  "model_used": "Gradient Boosting",
  "model_r2": 0.8544,
  "bmi_category": "Overweight",
  "risk_category": "Low",
  "confidence_range": { "low": 6666.03, "high": 9018.27 }
}
```

---

## Step-by-Step: Deploy with Docker

### Step 1 — Build the image
```bash
docker build -t insurance-api .
```

### Step 2 — Run the container
```bash
docker run -p 8000:8000 insurance-api
```

### Step 3 — Test
```bash
curl http://localhost:8000/health
```

---

## Step-by-Step: Deploy to the Cloud

### Option A — Google Cloud Run (recommended, free tier)
```bash
# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/insurance-api

# 2. Deploy to Cloud Run
gcloud run deploy insurance-api \
  --image gcr.io/YOUR_PROJECT_ID/insurance-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# 3. You get a public HTTPS URL like:
# https://insurance-api-xxxx-uc.a.run.app
```

### Option B — AWS EC2
```bash
# On EC2 instance (after SSH in):
sudo apt update && sudo apt install -y docker.io
sudo docker build -t insurance-api .
sudo docker run -d -p 80:8000 insurance-api
# API available at http://YOUR_EC2_PUBLIC_IP/docs
```

### Option C — Render.com (easiest, no CLI needed)
1. Push this project to GitHub
2. Go to render.com → New → Web Service
3. Connect your GitHub repo
4. Set: Build Command = `pip install -r requirements.txt && python train.py`
5. Set: Start Command  = `uvicorn app.main:app --host 0.0.0.0 --port 8000`
6. Deploy — get a free public URL instantly

---

## Batch Prediction Example

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"age": 25, "bmi": 22.0, "children": 0, "sex": "female", "smoker": "no",  "region": "northeast"},
      {"age": 45, "bmi": 33.0, "children": 2, "sex": "male",   "smoker": "yes", "region": "southeast"},
      {"age": 60, "bmi": 40.0, "children": 3, "sex": "male",   "smoker": "yes", "region": "southwest"}
    ]
  }'
```

---

## Models Trained

| Model              | Trained on   | Notes                            |
|--------------------|-------------|----------------------------------|
| Linear Regression  | Scaled data  | Baseline from notebook           |
| Ridge Regression   | Scaled data  | GridSearchCV for best alpha      |
| Lasso Regression   | Scaled data  | GridSearchCV for best alpha      |
| Random Forest      | Raw data     | 200 trees, max_depth=10          |
| Gradient Boosting  | Raw data     | 200 estimators, lr=0.05          |

Best model is auto-selected by 5-fold cross-validated R².