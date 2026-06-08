from __future__ import annotations

import numpy as np
import pandas as pd


def _learning_depth_index(levels: pd.Series, course_level_to_depth: dict[str, int]) -> float:
    """Return (beginner_ratio) / (advanced_ratio+eps) style index.

    If you want a different mapping, adjust course_level_to_depth.
    """
    counts = levels.value_counts(dropna=True)
    beginner = counts.get("Beginner", 0)
    advanced = counts.get("Advanced", 0)
    eps = 1e-6
    return float((beginner + eps) / (advanced + eps))


def build_learner_features(df_users: pd.DataFrame, df_courses: pd.DataFrame, df_tx: pd.DataFrame) -> pd.DataFrame:
    """Build learner-level profiles using Users/Courses/Transactions.

    Engineered features follow the task description.
    """
    # Join tx with course metadata
    tx = df_tx.merge(df_users[["UserID", "Age", "Gender"]], on="UserID", how="left")
    tx = tx.merge(
        df_courses[["CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating"]],
        on="CourseID",
        how="left",
    )

    # Aggregate spend and counts
    user_courses = (
        tx.groupby(["UserID"])  # learner grain
        .agg(
            total_courses_enrolled=("CourseID", "nunique"),
            avg_spending_per_learner=("Amount", "mean"),
        )
        .reset_index()
    )

    # Avg courses per category (diversity baseline)
    cat_counts = tx.groupby(["UserID", "CourseCategory"])["CourseID"].nunique().reset_index(name="cat_courses")
    avg_courses_per_category = cat_counts.groupby("UserID")["cat_courses"].mean().reset_index(name="avg_courses_per_category")

    # Preferred category (most frequent category)
    preferred_cat = (
        tx.groupby(["UserID", "CourseCategory"])["CourseID"].nunique().reset_index(name="n")
        .sort_values(["UserID", "n"], ascending=[True, False])
        .groupby("UserID").head(1)[["UserID", "CourseCategory"]]
        .rename(columns={"CourseCategory": "preferred_course_category"})
    )

    # Preferred level
    preferred_level = (
        tx.groupby(["UserID", "CourseLevel"])["CourseID"].nunique().reset_index(name="n")
        .sort_values(["UserID", "n"], ascending=[True, False])
        .groupby("UserID").head(1)[["UserID", "CourseLevel"]]
        .rename(columns={"CourseLevel": "preferred_course_level"})
    )

    # Avg enrolled course rating
    avg_rating = tx.groupby("UserID")["CourseRating"].mean().reset_index(name="avg_enrolled_course_rating")

    # Diversity score = number of categories explored
    diversity = tx.groupby("UserID")["CourseCategory"].nunique().reset_index(name="diversity_score")

    # Learning depth index: beginner vs advanced ratio based on CourseLevel
    # (Assumes CourseLevel values include Beginner/Advanced; otherwise it still produces ratio via categories)
    depth = (
        tx.groupby("UserID")[["CourseLevel"]]
        .agg(lambda s: _learning_depth_index(s, {"Beginner": 0, "Advanced": 1}))
        .reset_index()
        .rename(columns={"CourseLevel": "learning_depth_index"})
    )

    # Base demographics
    demo = df_users[["UserID", "Age", "Gender"]].copy()

    # Merge all features
    out = demo.merge(user_courses, on="UserID", how="left")
    out = out.merge(avg_courses_per_category, on="UserID", how="left")
    out = out.merge(preferred_cat, on="UserID", how="left")
    out = out.merge(preferred_level, on="UserID", how="left")
    out = out.merge(avg_rating, on="UserID", how="left")
    out = out.merge(diversity, on="UserID", how="left")
    out = out.merge(depth, on="UserID", how="left")

    # Fill missing engineered numeric values
    for c in [
        "total_courses_enrolled",
        "avg_courses_per_category",
        "avg_spending_per_learner",
        "avg_enrolled_course_rating",
        "diversity_score",
        "learning_depth_index",
    ]:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    # Drop rows that lack core identifier
    out = out.dropna(subset=["UserID"]).copy()
    # Keep UserID as string-coded id (e.g., U00003).
    out["UserID"] = out["UserID"].astype(str)

    return out


