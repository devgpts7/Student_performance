# 🎓 GradeIQ — Student Performance Predictor

> **Production-grade ML web app** that predicts student exam scores with confidence intervals, SHAP explanations, and personalized recommendations.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Live Demo

👉 **[Try it live →](https://your-app.streamlit.app)**

---

## 📸 Screenshots

| Prediction Dashboard | SHAP Explanation | What-If Simulator |
|:---:|:---:|:---:|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## ✨ Features

### 🔮 Core Prediction Engine
- **5 ML models compared**: Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
- **Auto best-model selection** via 5-fold CV R² score
- **Confidence Interval**: e.g., `72 – 78` marks (95% CI)
- **Pass/Fail Probability** using residual distribution
- **Risk Badge**: 🟢 Low / 🟡 Medium / 🔴 High

### 🎛️ What-If Simulator
- Live sliders → instant prediction update
- Factor Impact Analysis: "+1hr study = +3.2 marks"

### 🔍 SHAP Explainability
- Per-prediction SHAP waterfall chart
- Which factors push/pull your score
- Signed importance values per feature

### 📊 EDA Insights
- Attendance buckets vs marks
- Sleep hours vs performance scatter
- Distraction score impact
- Full correlation heatmap

### 💡 Recommendations Engine
- Rule-based + model-informed suggestions
- `sleep < 6h → −4.2% estimated impact`
- `+1 hr study → +2.8–3.5% expected gain`

### 📁 Bulk CSV Upload
- Upload any CSV with feature columns
- Download results with CI + Risk badges
- Histogram of class performance distribution

### 🏅 Model Versioning
- Retrain → new version saved (v1, v2, ...)
- Side-by-side version comparison
- All metrics tracked: R², MAE, RMSE, CV R²

### 🕐 Session History
- All predictions saved in session
- Trend chart over multiple runs

---

## 📊 Model Performance

| Model | R² | CV R² | MAE | RMSE |
|---|---|---|---|---|
| **Linear Regression** ✅ | 0.911 | 0.917 | 4.06 | 5.01 |
| Ridge Regression | 0.910 | 0.916 | 4.06 | 5.01 |
| XGBoost | 0.858 | 0.861 | 5.08 | 6.30 |
| Gradient Boosting | 0.854 | 0.857 | 5.09 | 6.40 |
| Random Forest | 0.801 | 0.802 | 5.85 | 7.47 |

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/gradeiq-predictor
cd gradeiq-predictor
pip install -r requirements.txt

# Generate dataset + train models
python data/generate_dataset.py
python utils/model_trainer.py

# Launch app
streamlit run app.py
```

---

## 📁 Project Structure

```
gradeiq-predictor/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── .streamlit/
│   └── config.toml           # Dark theme config
├── data/
│   ├── generate_dataset.py   # Synthetic data generator (2000 samples)
│   └── student_data.csv      # Training dataset
├── models/
│   ├── artifacts_v1.pkl      # Trained models + scaler + SHAP
│   └── latest_version.json   # Version registry
└── utils/
    ├── model_trainer.py      # Training pipeline + versioning
    └── predictor.py          # Inference + CI + recommendations
```

---

## 🔬 Input Features

| Feature | Range | Description |
|---|---|---|
| `study_hours` | 0–12 | Daily study hours |
| `attendance_pct` | 30–100 | Class attendance percentage |
| `sleep_hours` | 3–10 | Nightly sleep duration |
| `prev_marks` | 0–100 | Previous exam score |
| `assignment_completion` | 0–100 | % assignments submitted |
| `distraction_score` | 0–10 | Phone/social media distraction |
| `subject_difficulty` | 1–5 | Perceived difficulty |
| `coaching_type` | 0/1 | Self-study vs coaching |
| `revision_freq` | 0–7 | Weekly revision sessions |

---

## 🚢 Deployment

### Streamlit Cloud (Recommended)
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Deploy!

### Render
```bash
# render.yaml provided
render up
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🤝 Contributing

PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

*Built with ❤️ using Streamlit, Scikit-learn, XGBoost, SHAP, and Plotly*
