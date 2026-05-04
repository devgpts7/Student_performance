"""
Prediction utilities: confidence intervals, pass/fail prob, risk badges, recommendations
"""
import numpy as np
import pandas as pd
from scipy import stats


FEATURE_COLS = [
    "study_hours", "attendance_pct", "sleep_hours", "prev_marks",
    "assignment_completion", "distraction_score", "subject_difficulty",
    "coaching_type", "revision_freq"
]

FEATURE_LABELS = {
    "study_hours": "Study Hours/Day",
    "attendance_pct": "Attendance %",
    "sleep_hours": "Sleep Hours",
    "prev_marks": "Previous Marks",
    "assignment_completion": "Assignment Completion %",
    "distraction_score": "Distraction Score",
    "subject_difficulty": "Subject Difficulty",
    "coaching_type": "Coaching Type",
    "revision_freq": "Weekly Revision Frequency",
}


def predict_with_confidence(model, scaler, input_dict, residual_std, best_model_name, use_scaled=True, confidence=0.95):
    """Returns predicted marks with confidence interval."""
    X = np.array([[input_dict[f] for f in FEATURE_COLS]])
    if use_scaled:
        X_proc = scaler.transform(X)
    else:
        X_proc = X
    pred = float(model.predict(X_proc)[0])
    pred = np.clip(pred, 0, 100)

    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * residual_std
    lower = np.clip(pred - margin, 0, 100)
    upper = np.clip(pred + margin, 0, 100)
    return pred, lower, upper


def pass_fail_probability(pred, residual_std, pass_threshold=40):
    """Probability of scoring >= pass_threshold using normal distribution."""
    z = (pass_threshold - pred) / residual_std
    prob_pass = 1 - stats.norm.cdf(z)
    return round(prob_pass * 100, 1)


def get_risk_level(pred):
    if pred >= 70:
        return "LOW", "#22c55e", "🟢"
    elif pred >= 50:
        return "MEDIUM", "#f59e0b", "🟡"
    else:
        return "HIGH", "#ef4444", "🔴"


def generate_recommendations(input_dict, pred):
    recs = []
    
    if input_dict["sleep_hours"] < 6:
        deficit = 6 - input_dict["sleep_hours"]
        impact = round(deficit * 2.1, 1)
        recs.append({
            "icon": "😴",
            "factor": "Sleep Deficit",
            "message": f"Sleep < 6h → estimated −{impact}% impact. Aim for 7–8h for peak cognitive performance.",
            "priority": "HIGH",
            "gain": f"−{impact}% current impact"
        })

    if input_dict["study_hours"] < 4:
        gain_low = round((5 - input_dict["study_hours"]) * 2.5, 1)
        gain_high = round((5 - input_dict["study_hours"]) * 3.2, 1)
        recs.append({
            "icon": "📚",
            "factor": "Low Study Hours",
            "message": f"+1 hr study/day → +{gain_low}–{gain_high}% expected gain. Structured study beats marathon sessions.",
            "priority": "HIGH",
            "gain": f"+{gain_low}–{gain_high}% potential"
        })

    if input_dict["attendance_pct"] < 75:
        recs.append({
            "icon": "🏫",
            "factor": "Low Attendance",
            "message": f"Attendance at {input_dict['attendance_pct']:.0f}% — below 75% threshold. Each class missed = missed context + notes.",
            "priority": "HIGH",
            "gain": "Critical — affects eligibility"
        })

    if input_dict["distraction_score"] > 6:
        impact = round((input_dict["distraction_score"] - 4) * 2.5, 1)
        recs.append({
            "icon": "📱",
            "factor": "High Distraction",
            "message": f"Distraction score {input_dict['distraction_score']}/10 costs ~{impact}% marks. Try Pomodoro (25+5 min) + phone-free study zones.",
            "priority": "MEDIUM",
            "gain": f"−{impact}% current drag"
        })

    if input_dict["assignment_completion"] < 70:
        recs.append({
            "icon": "📝",
            "factor": "Incomplete Assignments",
            "message": f"Only {input_dict['assignment_completion']:.0f}% assignments done. Assignments reinforce concepts and often carry internal marks.",
            "priority": "MEDIUM",
            "gain": "+3–6% potential"
        })

    if input_dict["revision_freq"] < 2:
        recs.append({
            "icon": "🔄",
            "factor": "Low Revision",
            "message": "Revising <2x/week = rapid forgetting (Ebbinghaus curve). Daily 20-min reviews outperform weekly cramming.",
            "priority": "MEDIUM",
            "gain": "+2–4% retention boost"
        })

    if input_dict["coaching_type"] == 0 and pred < 55:
        recs.append({
            "icon": "🎓",
            "factor": "Consider Coaching",
            "message": "Self-study is powerful but structured coaching provides accountability + doubt clearing for sub-55 scorers.",
            "priority": "LOW",
            "gain": "+3–5% structured guidance"
        })

    if not recs:
        recs.append({
            "icon": "⭐",
            "factor": "Strong Profile",
            "message": "Your study profile is excellent! Maintain consistency and focus on weak subjects for peak performance.",
            "priority": "LOW",
            "gain": "Consistency = key"
        })

    return recs


def whatif_delta(base_input, changed_key, changed_value, model, scaler, use_scaled, residual_std):
    """Compute predicted score change for what-if analysis."""
    new_input = base_input.copy()
    new_input[changed_key] = changed_value
    base_pred, _, _ = predict_with_confidence(model, scaler, base_input, residual_std, "", use_scaled)
    new_pred, _, _ = predict_with_confidence(model, scaler, new_input, residual_std, "", use_scaled)
    return round(new_pred - base_pred, 2)
