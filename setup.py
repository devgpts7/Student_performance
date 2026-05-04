#!/usr/bin/env python3
"""
Startup script - ensures dataset + model artifacts exist before app launch.
Run this once or include in build step.
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

DATA_PATH = os.path.join(BASE, "data", "student_data.csv")
MODEL_PATH = os.path.join(BASE, "models", "latest_version.json")


def ensure_dataset():
    if not os.path.exists(DATA_PATH):
        print("📊 Generating dataset...")
        exec(open(os.path.join(BASE, "data", "generate_dataset.py")).read())
        print("✓ Dataset ready")
    else:
        print(f"✓ Dataset found: {DATA_PATH}")


def ensure_models():
    if not os.path.exists(MODEL_PATH):
        print("🤖 Training models (first run)...")
        from utils.model_trainer import load_and_preprocess, train_all_models
        df = load_and_preprocess(DATA_PATH)
        arts = train_all_models(df)
        print(f"✓ Models trained — best: {arts['best_model_name']} R²={arts['best_metrics']['r2']}")
    else:
        print("✓ Models found")


if __name__ == "__main__":
    ensure_dataset()
    ensure_models()
    print("\n🚀 Ready! Run: streamlit run app.py")
