"""
🎓 GradeIQ — Student Performance Predictor
Production Streamlit App — portable, cloud-ready
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import json, os, sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Portable imports ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.model_trainer import (
    load_artifacts, train_all_models, load_and_preprocess,
    MODEL_DIR, DATA_PATH, FEATURE_COLS, get_all_versions_list, ensure_data
)
from utils.predictor import (
    predict_with_confidence, pass_fail_probability, get_risk_level,
    generate_recommendations, FEATURE_LABELS
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GradeIQ — Student Performance Predictor",
    page_icon="🎓", layout="wide", initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.main { background: #0a0e1a; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2a3a; }
.hero-card {
    background: linear-gradient(135deg,#1a1f35 0%,#0d1117 100%);
    border: 1px solid #2d3748; border-radius: 16px; padding: 2rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#6366f1,#8b5cf6,#a78bfa);
}
.metric-card {
    background:#111827; border:1px solid #1f2937; border-radius:12px;
    padding:1.2rem 1.5rem; text-align:center;
}
.metric-value { font-size:2.2rem; font-weight:700; font-family:'JetBrains Mono',monospace; }
.metric-label { font-size:0.78rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.3rem; }
.rec-card { background:#111827; border-left:4px solid #6366f1; border-radius:8px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
.rec-card.high { border-left-color:#ef4444; }
.rec-card.medium { border-left-color:#f59e0b; }
.rec-card.low { border-left-color:#22c55e; }
.rec-header { font-weight:600; font-size:0.95rem; color:#e2e8f0; margin-bottom:0.3rem; }
.rec-body { font-size:0.85rem; color:#94a3b8; }
.rec-gain { font-size:0.78rem; color:#6366f1; font-weight:600; margin-top:0.3rem; font-family:'JetBrains Mono',monospace; }
.section-header { font-size:1.1rem; font-weight:700; color:#c7d2fe;
    border-bottom:1px solid #1e2a3a; padding-bottom:0.5rem;
    margin-bottom:1rem; text-transform:uppercase; letter-spacing:0.08em; }
.stButton>button { background:linear-gradient(135deg,#6366f1,#8b5cf6); color:white;
    border:none; border-radius:8px; font-weight:600; }
h1,h2,h3 { color:#e2e8f0 !important; }
.stSelectbox label,.stSlider label,.stRadio label { color:#94a3b8 !important; font-size:0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Auto-train on first run ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model(version=None):
    """Load existing artifacts or train fresh if not found."""
    arts = load_artifacts(version)
    if arts is None:
        ensure_data()
        df = load_and_preprocess()
        arts = train_all_models(df)
    return arts


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 GradeIQ")
    st.markdown("<p style='color:#6b7280;font-size:0.8rem;margin-top:-0.5rem;'>Student Performance Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "🔮 Predict", "🎛️ What-If Simulator", "📊 EDA Insights",
        "📁 Bulk CSV", "🏅 Model Compare", "🕐 History"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Model Version**")
    all_versions = get_all_versions_list()
    if all_versions:
        sel_ver = st.selectbox("Select Version", all_versions, index=len(all_versions)-1, label_visibility="collapsed")
    else:
        sel_ver = None
        st.caption("Model training on startup…")

    if st.button("🔄 Retrain Model"):
        st.cache_resource.clear()
        with st.spinner("Training all models (≈30 sec)…"):
            ensure_data()
            df = load_and_preprocess()
            arts = train_all_models(df)
        st.success(f"✓ New version: {arts['version']}")
        st.rerun()

    st.divider()
    st.caption("Built with Streamlit + XGBoost + SHAP")


# ── Load model (auto-trains if missing) ───────────────────────────────────────
with st.spinner("🔄 Loading model… (first run auto-trains, ~30 sec)"):
    arts = load_or_train_model(sel_ver)

model        = arts["best_model"]
scaler       = arts["scaler"]
best_name    = arts["best_model_name"]
residual_std = arts["residual_std"]
use_scaled   = best_name in ["Linear Regression", "Ridge Regression"]
all_metrics  = arts["all_metrics"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_input_form(prefix="main"):
    with st.expander("📥 Enter Student Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            study_hours = st.slider("📚 Study Hours/Day", 0.0, 12.0, 5.0, 0.5, key=f"{prefix}_study")
            attendance  = st.slider("🏫 Attendance %", 30.0, 100.0, 75.0, 1.0, key=f"{prefix}_att")
            sleep_hours = st.slider("😴 Sleep Hours", 3.0, 10.0, 7.0, 0.5, key=f"{prefix}_sleep")
        with c2:
            prev_marks = st.slider("📈 Previous Marks", 0.0, 100.0, 60.0, 1.0, key=f"{prefix}_prev")
            assignment = st.slider("📝 Assignment Completion %", 0.0, 100.0, 80.0, 1.0, key=f"{prefix}_assign")
            distraction= st.slider("📱 Distraction Score (0=focused, 10=max)", 0.0, 10.0, 4.0, 0.5, key=f"{prefix}_dist")
        with c3:
            difficulty = st.select_slider("⚡ Subject Difficulty", [1,2,3,4,5], value=3, key=f"{prefix}_diff")
            coaching   = st.radio("🎓 Study Mode", ["Self Study","Coaching"], key=f"{prefix}_coach", horizontal=True)
            revision   = st.slider("🔄 Weekly Revision Frequency", 0, 7, 3, key=f"{prefix}_rev")
    return {
        "study_hours": study_hours, "attendance_pct": attendance,
        "sleep_hours": sleep_hours, "prev_marks": prev_marks,
        "assignment_completion": assignment, "distraction_score": distraction,
        "subject_difficulty": difficulty,
        "coaching_type": 1 if coaching == "Coaching" else 0,
        "revision_freq": revision,
    }


def render_prediction_output(inputs, show_shap=True):
    pred, lower, upper = predict_with_confidence(model, scaler, inputs, residual_std, best_name, use_scaled)
    prob_pass = pass_fail_probability(pred, residual_std)
    risk_level, risk_color, risk_emoji = get_risk_level(pred)
    recs = generate_recommendations(inputs, pred)

    # Hero card
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown(f"<p style='color:#94a3b8;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>Predicted Score</p>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='font-size:3.5rem;margin:0;font-family:JetBrains Mono,monospace;color:#a78bfa;'>{pred:.1f}<span style='font-size:1.5rem;color:#6b7280;'>/100</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;font-size:0.9rem;margin-top:0.3rem;'>95% Confidence Interval: <span style='color:#c7d2fe;font-family:JetBrains Mono,monospace;font-weight:600;'>{lower:.1f} – {upper:.1f}</span></p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#6366f1;">{pred:.1f}</div><div class="metric-label">Predicted Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#22c55e;">{lower:.0f}–{upper:.0f}</div><div class="metric-label">Confidence Band</div></div>', unsafe_allow_html=True)
    with c3:
        clr = "#22c55e" if prob_pass > 70 else "#f59e0b" if prob_pass > 40 else "#ef4444"
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{clr};">{prob_pass}%</div><div class="metric-label">Pass Probability</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div style="font-size:1.5rem;">{risk_emoji}</div><div class="metric-value" style="color:{risk_color};font-size:1.4rem;">{risk_level}</div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)

    # SHAP
    if show_shap and arts.get("explainer") is not None:
        st.markdown('<div class="section-header" style="margin-top:2rem;">🔍 SHAP Feature Impact</div>', unsafe_allow_html=True)
        try:
            X_inp  = np.array([[inputs[f] for f in FEATURE_COLS]])
            X_proc = scaler.transform(X_inp) if use_scaled else X_inp
            shap_vals = arts["explainer"].shap_values(X_proc)[0]
            shap_df = pd.DataFrame({
                "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLS],
                "SHAP Value": shap_vals,
                "Abs": np.abs(shap_vals)
            }).sort_values("Abs", ascending=True)
            colors = ["#ef4444" if v < 0 else "#6366f1" for v in shap_df["SHAP Value"]]
            fig = go.Figure(go.Bar(
                x=shap_df["SHAP Value"], y=shap_df["Feature"], orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}" for v in shap_df["SHAP Value"]], textposition="outside",
            ))
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                              font=dict(color="#94a3b8", size=12),
                              xaxis=dict(title="Impact on Score", gridcolor="#1f2937", zerolinecolor="#374151"),
                              yaxis=dict(gridcolor="#1f2937"),
                              height=350, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔴 Red = pulls score down  |  🔵 Blue = pushes score up")
        except Exception as e:
            st.info(f"SHAP not available for this model config: {e}")

    # Recommendations
    st.markdown('<div class="section-header" style="margin-top:1.5rem;">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    for rec in recs:
        cls = rec["priority"].lower()
        st.markdown(f"""
        <div class="rec-card {cls}">
            <div class="rec-header">{rec['icon']} {rec['factor']}</div>
            <div class="rec-body">{rec['message']}</div>
            <div class="rec-gain">Impact: {rec['gain']}</div>
        </div>""", unsafe_allow_html=True)

    # History
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "predicted": round(pred,1), "ci_low": round(lower,1), "ci_high": round(upper,1),
        "pass_prob": prob_pass, "risk": risk_level,
        "model": best_name, "version": arts["version"], "inputs": inputs.copy()
    }
    if not st.session_state.history or st.session_state.history[-1]["inputs"] != inputs:
        st.session_state.history.append(entry)
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── PREDICT ──────────────────────────────────────────────────────────────────
if "Predict" in page:
    st.markdown("# 🔮 Grade Prediction")
    st.markdown(f"<p style='color:#6b7280;'>Model: <b style='color:#a78bfa;'>{best_name}</b> · Version: <b style='color:#6366f1;'>{arts['version']}</b> · R²: <b style='color:#22c55e;'>{arts['best_metrics']['r2']}</b></p>", unsafe_allow_html=True)
    inputs = render_input_form("predict")
    if st.button("🚀 Predict My Score", use_container_width=True):
        with st.spinner("Computing prediction + SHAP…"):
            render_prediction_output(inputs, show_shap=True)


# ── WHAT-IF ───────────────────────────────────────────────────────────────────
elif "What-If" in page:
    st.markdown("# 🎛️ What-If Simulator")
    st.markdown("<p style='color:#6b7280;'>Adjust sliders → see live prediction update instantly.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Adjust Parameters")
        study_h = st.slider("Study Hours/Day", 0.0, 12.0, 5.0, 0.5, key="wi_study")
        att     = st.slider("Attendance %", 30.0, 100.0, 75.0, 1.0, key="wi_att")
        sleep   = st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.5, key="wi_sleep")
        prev    = st.slider("Previous Marks", 0.0, 100.0, 60.0, 1.0, key="wi_prev")
        assign  = st.slider("Assignment Completion %", 0.0, 100.0, 80.0, 1.0, key="wi_assign")
        dist    = st.slider("Distraction Score", 0.0, 10.0, 4.0, 0.5, key="wi_dist")
        diff    = st.select_slider("Subject Difficulty", [1,2,3,4,5], value=3, key="wi_diff")
        coach   = st.radio("Study Mode", ["Self Study","Coaching"], key="wi_coach", horizontal=True)
        rev     = st.slider("Weekly Revision", 0, 7, 3, key="wi_rev")

    base = {
        "study_hours": study_h, "attendance_pct": att, "sleep_hours": sleep,
        "prev_marks": prev, "assignment_completion": assign, "distraction_score": dist,
        "subject_difficulty": diff, "coaching_type": 1 if coach=="Coaching" else 0, "revision_freq": rev
    }
    pred, lower, upper = predict_with_confidence(model, scaler, base, residual_std, best_name, use_scaled)
    prob = pass_fail_probability(pred, residual_std)
    risk, rclr, remo = get_risk_level(pred)

    with c2:
        st.markdown("### Live Result")
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:16px;padding:2rem;text-align:center;">
            <div style="font-size:4rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#a78bfa;">{pred:.1f}</div>
            <div style="color:#64748b;margin-top:0.3rem;">/ 100 marks</div>
            <div style="color:#c7d2fe;margin-top:1rem;font-size:1.1rem;">CI: {lower:.1f} – {upper:.1f}</div>
            <div style="margin-top:1rem;">
                <span style="background:{rclr}22;color:{rclr};border:1px solid {rclr};border-radius:6px;padding:0.3rem 0.8rem;font-weight:600;">{remo} {risk} RISK</span>
            </div>
            <div style="margin-top:1rem;color:#6b7280;">Pass Probability: <span style="color:#22c55e;font-weight:700;">{prob}%</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### If You Change…")
        improvements = {
            "+1hr Study":           {"study_hours": min(study_h+1,12)},
            "+1hr Sleep":           {"sleep_hours": min(sleep+1,10)},
            "Attendance +10%":      {"attendance_pct": min(att+10,100)},
            "Distraction −2pts":   {"distraction_score": max(dist-2,0)},
            "Revision +2/wk":       {"revision_freq": min(rev+2,7)},
            "100% Assignments":     {"assignment_completion": 100.0},
        }
        rows = []
        for label, changes in improvements.items():
            mod = base.copy(); mod.update(changes)
            np2, _, _ = predict_with_confidence(model, scaler, mod, residual_std, best_name, use_scaled)
            rows.append({"Change": label, "Score Impact": round(np2-pred,2)})
        delta_df = pd.DataFrame(rows).sort_values("Score Impact", ascending=True)
        colors_d = ["#6366f1" if v>0 else "#ef4444" for v in delta_df["Score Impact"]]
        fig2 = go.Figure(go.Bar(
            x=delta_df["Score Impact"], y=delta_df["Change"], orientation="h",
            marker_color=colors_d,
            text=[f"{v:+.1f} pts" for v in delta_df["Score Impact"]], textposition="outside"
        ))
        fig2.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#111827",
                           font=dict(color="#94a3b8",size=12),
                           xaxis=dict(title="Score Change", gridcolor="#1f2937", zerolinecolor="#374151"),
                           height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)


# ── EDA ───────────────────────────────────────────────────────────────────────
elif "EDA" in page:
    st.markdown("# 📊 EDA Insights")
    df_raw = pd.read_csv(DATA_PATH)
    tab1, tab2, tab3, tab4 = st.tabs(["Attendance vs Marks","Sleep vs Performance","Distraction Impact","Correlations"])

    with tab1:
        df_raw["att_bucket"] = pd.cut(df_raw["attendance_pct"],
            bins=[0,50,60,75,85,100], labels=["<50%","50–60%","60–75%","75–85%","85%+"])
        grp = df_raw.groupby("att_bucket", observed=False)["final_marks"].agg(["mean","std"]).reset_index()
        fig = go.Figure(go.Bar(
            x=grp["att_bucket"].astype(str), y=grp["mean"].round(1),
            error_y=dict(type="data", array=grp["std"].round(1)),
            marker_color=["#ef4444","#f97316","#f59e0b","#22c55e","#6366f1"],
        ))
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#111827",
                          font=dict(color="#94a3b8"), height=350,
                          title="Attendance Bucket vs Average Marks",
                          xaxis_title="Attendance %", yaxis_title="Average Marks")
        st.plotly_chart(fig, use_container_width=True)
        st.info("📌 85%+ attendance students score ~18 marks higher than <50% group.")

    with tab2:
        fig = px.scatter(df_raw, x="sleep_hours", y="final_marks", color="study_hours",
                         trendline="lowess", opacity=0.5, color_continuous_scale="Viridis",
                         title="Sleep Hours vs Final Marks (coloured by Study Hours)")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#111827",
                          font=dict(color="#94a3b8"), height=400)
        st.plotly_chart(fig, use_container_width=True)
        corr = df_raw["sleep_hours"].corr(df_raw["final_marks"])
        st.info(f"📌 Sleep–Marks Pearson r = {corr:.3f}. Sweet spot: 6.5–8 hours.")

    with tab3:
        fig = px.scatter(df_raw, x="distraction_score", y="final_marks", color="attendance_pct",
                         trendline="ols", opacity=0.5, color_continuous_scale="RdYlGn",
                         title="Distraction Score vs Marks")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#111827",
                          font=dict(color="#94a3b8"), height=400)
        st.plotly_chart(fig, use_container_width=True)
        corr2 = df_raw["distraction_score"].corr(df_raw["final_marks"])
        st.info(f"📌 Distraction–Marks r = {corr2:.3f}. Every +1 distraction → ~−2.5 marks.")

    with tab4:
        corr_matrix = df_raw[FEATURE_COLS + ["final_marks"]].corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu",
                        title="Feature Correlation Heatmap", aspect="auto")
        fig.update_layout(paper_bgcolor="#0d1117", font=dict(color="#94a3b8"), height=500)
        st.plotly_chart(fig, use_container_width=True)
        top = corr_matrix["final_marks"].drop("final_marks").abs().sort_values(ascending=False)
        st.markdown("**Top Drivers of Final Marks:**")
        for feat, val in top.items():
            arrow = "▲" if corr_matrix["final_marks"][feat]>0 else "▼"
            st.markdown(f"- `{FEATURE_LABELS.get(feat,feat)}`: **{val:.3f}** {arrow}")


# ── BULK CSV ──────────────────────────────────────────────────────────────────
elif "Bulk" in page:
    st.markdown("# 📁 Bulk CSV Prediction")
    template = pd.DataFrame([{f: 5.0 if "hours" in f or f=="revision_freq" else
                                 75.0 if "pct" in f else 60.0 if f=="prev_marks" else
                                 80.0 if "assignment" in f else 4.0 if "distraction" in f else
                                 3 if "difficulty" in f else 0 for f in FEATURE_COLS}])
    st.download_button("📥 Download Template CSV", template.to_csv(index=False), "template.csv","text/csv")

    uploaded = st.file_uploader("Upload Student CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        missing = [f for f in FEATURE_COLS if f not in df_up.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            results = []
            for _, row in df_up.iterrows():
                inp = {f: row[f] for f in FEATURE_COLS}
                p,lo,hi = predict_with_confidence(model,scaler,inp,residual_std,best_name,use_scaled)
                pp = pass_fail_probability(p, residual_std)
                rl,_,_ = get_risk_level(p)
                results.append({**inp,"Predicted_Marks":round(p,1),"CI_Low":round(lo,1),
                                 "CI_High":round(hi,1),"Pass_Probability_%":pp,"Risk_Level":rl})
            res_df = pd.DataFrame(results)
            st.success(f"✓ Processed {len(res_df)} students")
            col1,col2,col3 = st.columns(3)
            col1.metric("Avg Predicted Marks", f"{res_df['Predicted_Marks'].mean():.1f}")
            col2.metric("High Risk Students",  f"{(res_df['Risk_Level']=='HIGH').sum()}")
            col3.metric("Avg Pass Probability",f"{res_df['Pass_Probability_%'].mean():.1f}%")
            fig = px.histogram(res_df, x="Predicted_Marks", nbins=20, color="Risk_Level",
                               color_discrete_map={"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#ef4444"},
                               title="Class Distribution of Predicted Marks")
            fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#111827", font=dict(color="#94a3b8"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(res_df, use_container_width=True)
            st.download_button("📥 Download Results", res_df.to_csv(index=False),"predictions.csv","text/csv")


# ── MODEL COMPARE ─────────────────────────────────────────────────────────────
elif "Model" in page:
    st.markdown("# 🏅 Model Comparison")
    metrics_df = pd.DataFrame(all_metrics)
    best_row   = arts["best_metrics"]["name"]
    fig = make_subplots(rows=1,cols=3,subplot_titles=["R² (higher=better)","MAE (lower=better)","RMSE (lower=better)"])
    colors_m = ["#6366f1" if n==best_row else "#374151" for n in metrics_df["name"]]
    fig.add_bar(x=metrics_df["name"],y=metrics_df["r2"],  marker_color=colors_m,name="R²",  row=1,col=1)
    fig.add_bar(x=metrics_df["name"],y=metrics_df["mae"], marker_color=colors_m,name="MAE", row=1,col=2)
    fig.add_bar(x=metrics_df["name"],y=metrics_df["rmse"],marker_color=colors_m,name="RMSE",row=1,col=3)
    fig.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#111827",font=dict(color="#94a3b8"),
                      height=380,showlegend=False,title=f"Best: {best_row} (highlighted in purple)")
    st.plotly_chart(fig, use_container_width=True)
    display = metrics_df[["name","r2","cv_r2","mae","rmse"]].copy()
    display.columns = ["Model","R²","CV R²","MAE","RMSE"]
    display["Selected"] = display["Model"].apply(lambda x: "✅" if x==best_row else "")
    st.dataframe(display, use_container_width=True, hide_index=True)

    versions = get_all_versions_list()
    if len(versions) > 1:
        st.markdown("### Version Comparison")
        v1c, v2c = st.columns(2)
        with v1c: v1 = st.selectbox("Version A", versions, key="vc_v1")
        with v2c: v2 = st.selectbox("Version B", versions, index=min(1,len(versions)-1), key="vc_v2")
        if st.button("Compare"):
            a1, a2 = load_artifacts(v1), load_artifacts(v2)
            if a1 and a2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{v1} — {a1['best_model_name']}**")
                    m1 = a1["best_metrics"]
                    st.metric("R²",m1["r2"]); st.metric("MAE",m1["mae"]); st.metric("RMSE",m1["rmse"])
                with col2:
                    st.markdown(f"**{v2} — {a2['best_model_name']}**")
                    m2 = a2["best_metrics"]
                    st.metric("R²",m2["r2"],delta=round(m2["r2"]-m1["r2"],4))
                    st.metric("MAE",m2["mae"],delta=round(m2["mae"]-m1["mae"],4))
                    st.metric("RMSE",m2["rmse"],delta=round(m2["rmse"]-m1["rmse"],4))


# ── HISTORY ───────────────────────────────────────────────────────────────────
elif "History" in page:
    st.markdown("# 🕐 Prediction History")
    if not st.session_state.history:
        st.info("No predictions yet. Go to **Predict** page to get started!")
    else:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []; st.rerun()
        hist_df = pd.DataFrame([{
            "Time":h["timestamp"],"Predicted":h["predicted"],
            "CI":f"{h['ci_low']}–{h['ci_high']}","Pass %":h["pass_prob"],
            "Risk":h["risk"],"Model":h["model"],"Version":h["version"]
        } for h in reversed(st.session_state.history)])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_scatter(
            x=list(range(len(st.session_state.history))),
            y=[h["predicted"] for h in st.session_state.history],
            mode="lines+markers", line=dict(color="#6366f1",width=2),
            marker=dict(size=8,color="#a78bfa"), fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"
        )
        fig.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#111827",font=dict(color="#94a3b8"),
                          xaxis_title="Prediction #",yaxis_title="Predicted Marks",height=300)
        st.plotly_chart(fig, use_container_width=True)
