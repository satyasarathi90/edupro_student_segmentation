from __future__ import annotations

import pandas as pd


def load_teacher_csv(df_teacher: pd.DataFrame) -> pd.DataFrame:
    """Validate and lightly normalize teacher data.

    Currently the project doesn't use teacher data for feature engineering,
    but we keep a stable place for future teacher-aware features.
    """

    # Normalize basic dtypes if present
    if "TeacherID" in df_teacher.columns:
        df_teacher["TeacherID"] = pd.to_numeric(df_teacher["TeacherID"], errors="coerce").astype("Int64")

    # Ensure string columns are string
    for c in ["TeacherName", "Gender", "Expertise"]:
        if c in df_teacher.columns:
            df_teacher[c] = df_teacher[c].astype(str)

    # Numeric columns
    for c in ["Age", "YearsOfExperience", "TeacherRating"]:
        if c in df_teacher.columns:
            df_teacher[c] = pd.to_numeric(df_teacher[c], errors="coerce")

    return df_teacher

