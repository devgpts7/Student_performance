"""
Model Training Pipeline — portable, relative paths
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "study_hours", "attendance_pct", "sleep_hours", "prev_marks",
    "assignment_completion", "distraction_score", "subject_difficulty",
    "coaching_type", "revision_freq"
]
TARGET = "final_marks"

# ── Relative paths (portable across machines / cloud) ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../utils/
_ROOT = os.path.dirname(_HERE)                               # .../student_predictor/
MODEL_DIR = os.path.join(_ROOT, "models")
DATA_PATH  = os.path.join(_ROOT, "data", "student_data.csv")
os.makedirs(MODEL_DIR, exist_ok=True)


def ensure_data():
    """Generate dataset if not present."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        _generate_dataset(DATA_PATH)


def _generate_dataset(path, n=2000):
    np.random.seed(42)
    study_hours          = np.random.uniform(0, 12, n)
    attendance           = np.random.uniform(30, 100, n)
    sleep_hours          = np.random.uniform(3, 10, n)
    prev_marks           = np.random.uniform(20, 95, n)
    assignment_completion= np.random.uniform(0, 100, n)
    distraction_score    = np.random.uniform(0, 10, n)
    subject_difficulty   = np.random.randint(1, 6, n)
    coaching_type        = np.random.choice([0, 1], n)
    revision_freq        = np.random.randint(0, 8, n)

    marks = (
        prev_marks * 0.35 + study_hours * 2.8 + attendance * 0.22
        + sleep_hours * 2.1 - distraction_score * 2.5
        + assignment_completion * 0.12 - subject_difficulty * 1.8
        + coaching_type * 3.5 + revision_freq * 1.2
        + np.random.normal(0, 5, n)
    )
    marks = np.clip(marks, 0, 100)

    df = pd.DataFrame({
        "study_hours": study_hours, "attendance_pct": attendance,
        "sleep_hours": sleep_hours, "prev_marks": prev_marks,
        "assignment_completion": assignment_completion,
        "distraction_score": distraction_score,
        "subject_difficulty": subject_difficulty,
        "coaching_type": coaching_type,
        "revision_freq": revision_freq, "final_marks": marks
    })
    df.to_csv(path, index=False)
    print(f"[data] Generated {n} rows → {path}")


def load_and_preprocess(path=None):
    if path is None:
        path = DATA_PATH
    ensure_data()
    df = pd.read_csv(path).dropna()
    df["study_hours"]           = df["study_hours"].clip(0, 16)
    df["attendance_pct"]        = df["attendance_pct"].clip(0, 100)
    df["sleep_hours"]           = df["sleep_hours"].clip(2, 12)
    df["assignment_completion"] = df["assignment_completion"].clip(0, 100)
    df["distraction_score"]     = df["distraction_score"].clip(0, 10)
    return df


def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_r2  = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
    metrics = {
        "name": name,
        "r2":   round(r2_score(y_test, y_pred), 4),
        "cv_r2":round(cv_r2, 4),
        "mae":  round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "residual_std": round(np.std(y_test - y_pred), 4),
    }
    return model, metrics, y_pred


def train_all_models(df):
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    models_cfg = {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Random Forest":      RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
        "XGBoost":            xgb.XGBRegressor(n_estimators=300, learning_rate=0.07, max_depth=6,
                                                subsample=0.8, random_state=42, verbosity=0),
    }

    all_metrics, trained_models = [], {}
    for name, mdl in models_cfg.items():
        scaled = name in ["Linear Regression", "Ridge Regression"]
        Xtr = X_train_sc if scaled else X_train.values
        Xte = X_test_sc  if scaled else X_test.values
        fitted, metrics, _ = evaluate_model(mdl, Xtr, Xte, y_train.values, y_test.values, name)
        all_metrics.append(metrics)
        trained_models[name] = fitted
        print(f"  ✓ {name}: R²={metrics['r2']}  MAE={metrics['mae']}  RMSE={metrics['rmse']}")

    best      = max(all_metrics, key=lambda x: x["cv_r2"])
    best_name = best["name"]
    best_mdl  = trained_models[best_name]
    print(f"\n  🏆 Best: {best_name}  CV-R²={best['cv_r2']}")

    # SHAP
    scaled_best = best_name in ["Linear Regression", "Ridge Regression"]
    X_shap      = X_test_sc if scaled_best else X_test.values
    try:
        if best_name in ["XGBoost", "Gradient Boosting", "Random Forest"]:
            explainer = shap.TreeExplainer(best_mdl)
        else:
            explainer = shap.LinearExplainer(best_mdl, X_train_sc if scaled_best else X_train.values)
        shap_values = explainer.shap_values(X_shap[:200])
    except Exception as e:
        print(f"  [SHAP skipped: {e}]")
        explainer, shap_values = None, None

    version = _next_version()
    artifacts = {
        "version": version, "timestamp": datetime.now().isoformat(),
        "best_model_name": best_name, "scaler": scaler,
        "best_model": best_mdl, "all_models": trained_models,
        "all_metrics": all_metrics, "best_metrics": best,
        "feature_cols": FEATURE_COLS,
        "explainer": explainer, "shap_values_sample": shap_values,
        "X_test_df_sample": X_test.iloc[:200],
        "y_test_sample": y_test.values[:200],
        "residual_std": best["residual_std"],
    }

    save_path = os.path.join(MODEL_DIR, f"artifacts_{version}.pkl")
    joblib.dump(artifacts, save_path)

    vfile = os.path.join(MODEL_DIR, "latest_version.json")
    existing = _all_versions()
    if version not in existing:
        existing.append(version)
    with open(vfile, "w") as f:
        json.dump({"latest": version, "all_versions": existing}, f)

    print(f"  ✓ Saved → {save_path}")
    return artifacts


def _next_version():
    versions = _all_versions()
    nums = [int(v.replace("v", "")) for v in versions if v.startswith("v")]
    return f"v{max(nums)+1}" if nums else "v1"


def _all_versions():
    vfile = os.path.join(MODEL_DIR, "latest_version.json")
    if not os.path.exists(vfile):
        return []
    try:
        with open(vfile) as f:
            content = f.read().strip()
        return json.loads(content).get("all_versions", []) if content else []
    except Exception:
        return []


def load_artifacts(version=None):
    vfile = os.path.join(MODEL_DIR, "latest_version.json")
    if version is None:
        if not os.path.exists(vfile):
            return None
        try:
            with open(vfile) as f:
                version = json.loads(f.read().strip())["latest"]
        except Exception:
            return None
    path = os.path.join(MODEL_DIR, f"artifacts_{version}.pkl")
    return joblib.load(path) if os.path.exists(path) else None


def get_all_versions_list():
    return _all_versions()


if __name__ == "__main__":
    df = load_and_preprocess()
    print(f"Data: {len(df)} rows")
    train_all_models(df)
