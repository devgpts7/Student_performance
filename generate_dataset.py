import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

study_hours = np.random.uniform(0, 12, n)
attendance = np.random.uniform(30, 100, n)
sleep_hours = np.random.uniform(3, 10, n)
prev_marks = np.random.uniform(20, 95, n)
assignment_completion = np.random.uniform(0, 100, n)
distraction_score = np.random.uniform(0, 10, n)
subject_difficulty = np.random.randint(1, 6, n)
coaching_type = np.random.choice([0, 1], n)  # 0=self, 1=coaching
revision_freq = np.random.randint(0, 8, n)

marks = (
    prev_marks * 0.35
    + study_hours * 2.8
    + attendance * 0.22
    + sleep_hours * 2.1
    - distraction_score * 2.5
    + assignment_completion * 0.12
    - subject_difficulty * 1.8
    + coaching_type * 3.5
    + revision_freq * 1.2
    + np.random.normal(0, 5, n)
)
marks = np.clip(marks, 0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance_pct": attendance,
    "sleep_hours": sleep_hours,
    "prev_marks": prev_marks,
    "assignment_completion": assignment_completion,
    "distraction_score": distraction_score,
    "subject_difficulty": subject_difficulty,
    "coaching_type": coaching_type,
    "revision_freq": revision_freq,
    "final_marks": marks
})

df.to_csv("/home/claude/student_predictor/data/student_data.csv", index=False)
print(f"Dataset created: {len(df)} rows")
print(df.describe())
