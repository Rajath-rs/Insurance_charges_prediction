"""
Insurance Charges Prediction - Model Training Script
Run this once to train all models and save the best one.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading and preprocessing data...")
print("=" * 50)

df = pd.read_csv("insurance.csv")
print(f"Dataset shape: {df.shape}")

# Drop duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"After dropping duplicates: {df.shape}")

# Log-transform target (as done in the notebook)
df["log_charges"] = np.log(df["charges"])

# One-hot encode categoricals
df_encoded = pd.get_dummies(df.drop(columns=["charges", "log_charges"]), drop_first=True).astype(int)
df_encoded["log_charges"] = df["log_charges"].values

print(f"Features after encoding: {list(df_encoded.columns)}")

# ─────────────────────────────────────────────
# 2. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────
feature_cols = [c for c in df_encoded.columns if c != "log_charges"]
X = df_encoded[feature_cols]
y = df_encoded["log_charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 3. SCALING (only numerical columns)
# ─────────────────────────────────────────────
num_cols = ["age", "bmi", "children"]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

# ─────────────────────────────────────────────
# 4. TRAIN MULTIPLE MODELS & COMPARE
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Training and comparing models...")
print("=" * 50)

def evaluate(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    y_pred  = model.predict(Xte)
    ytr_pred = model.predict(Xtr)
    r2      = r2_score(yte, y_pred)
    mae     = mean_absolute_error(np.exp(yte), np.exp(y_pred))
    rmse    = np.sqrt(mean_squared_error(yte, y_pred))
    cv      = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2").mean()
    print(f"\n{name}")
    print(f"  Train R²     : {r2_score(ytr, ytr_pred):.4f}")
    print(f"  Test  R²     : {r2:.4f}")
    print(f"  CV    R²     : {cv:.4f}")
    print(f"  MAE  ($)     : {mae:.2f}")
    print(f"  RMSE (log)   : {rmse:.4f}")
    return {"name": name, "model": model, "test_r2": r2, "cv_r2": cv, "mae": mae, "rmse": rmse}

results = []

# Linear Regression
results.append(evaluate("Linear Regression", LinearRegression(),
                         X_train_scaled, X_test_scaled, y_train, y_test))

# Ridge
ridge_grid = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10, 50, 100]},
                          cv=5, scoring="r2")
ridge_grid.fit(X_train_scaled, y_train)
print(f"\nBest Ridge alpha: {ridge_grid.best_params_['alpha']}")
results.append(evaluate("Ridge Regression", Ridge(alpha=ridge_grid.best_params_["alpha"]),
                         X_train_scaled, X_test_scaled, y_train, y_test))

# Lasso
lasso_grid = GridSearchCV(Lasso(), {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
                          cv=5, scoring="r2")
lasso_grid.fit(X_train_scaled, y_train)
print(f"\nBest Lasso alpha: {lasso_grid.best_params_['alpha']}")
results.append(evaluate("Lasso Regression", Lasso(alpha=lasso_grid.best_params_["alpha"]),
                         X_train_scaled, X_test_scaled, y_train, y_test))

# Random Forest
results.append(evaluate("Random Forest",
                         RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                         X_train, X_test, y_train, y_test))

# Gradient Boosting
results.append(evaluate("Gradient Boosting",
                         GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=4, random_state=42),
                         X_train, X_test, y_train, y_test))

# ─────────────────────────────────────────────
# 5. PICK BEST MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Selecting best model...")
print("=" * 50)

best = max(results, key=lambda r: r["cv_r2"])
print(f"Best model: {best['name']}  (CV R² = {best['cv_r2']:.4f})")

best_model   = best["model"]
# Determine whether the best model needed scaling
needs_scaling = best["name"] in ("Linear Regression", "Ridge Regression", "Lasso Regression")

# ─────────────────────────────────────────────
# 6. SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Saving model artifacts...")
print("=" * 50)

os.makedirs("model", exist_ok=True)

# Save model
with open("model/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature column order (critical for correct prediction)
with open("model/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# Save model metadata
metadata = {
    "best_model_name"  : best["name"],
    "needs_scaling"    : needs_scaling,
    "test_r2"          : round(best["test_r2"], 4),
    "cv_r2"            : round(best["cv_r2"], 4),
    "mae_usd"          : round(best["mae"], 2),
    "num_cols"         : num_cols,
    "feature_cols"     : feature_cols,
    "all_model_results": [
        {k: v for k, v in r.items() if k != "model"}
        for r in results
    ]
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved: model/best_model.pkl")
print("Saved: model/scaler.pkl")
print("Saved: model/feature_cols.json")
print("Saved: model/metadata.json")
print("\nTraining complete!")
print(f"Best Model : {best['name']}")
print(f"Test R²    : {best['test_r2']:.4f}")
print(f"MAE ($)    : {best['mae']:.2f}")