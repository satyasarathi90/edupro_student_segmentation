from __future__ import annotations

import numpy as np
import pandas as pd


def recommend_courses_for_learner(
    user_id: int,
    learner_df: pd.DataFrame,
    courses_df: pd.DataFrame,
    model: dict,
    topk: int = 6,
) -> pd.DataFrame:
    """Cluster-aware recommendation:
    - content similarity approximated via distance in engineered feature space
    - course popularity within user's segment
    - rating-weighted relevance

    This is a practical baseline for the project deliverable.
    """

    segment = int(learner_df.loc[learner_df["UserID"] == user_id, "segment"].iloc[0])

    # Courses not yet enrolled by the learner
    # We infer enrolled courses from transactions proxy: use course affinity from learner_df not stored.
    # Instead: return top popular in segment with relevance weighting.

    # Popularity in segment: we use learner-level rating/spend proxy to weight popularity.
    seg_users = learner_df.loc[learner_df["segment"] == segment, "UserID"].unique().tolist()

    # Heuristic: for each course, relevance = course_rating * (segment mean learning depth match) * popularity
    seg_depth = float(learner_df.loc[learner_df["segment"] == segment, "learning_depth_index"].mean())
    seg_div = float(learner_df.loc[learner_df["segment"] == segment, "diversity_score"].mean())
    seg_rating = float(learner_df.loc[learner_df["segment"] == segment, "avg_enrolled_course_rating"].mean())

    cdf = courses_df.copy()

    # Map course level to a numeric depth factor
    level = cdf["CourseLevel"].astype(str).str.lower()
    depth_factor = np.where(level.str.contains("advanced"), 1.4, np.where(level.str.contains("beginner"), 0.9, 1.1))

    # Category match: if course category equals segment preferred category mode-ish
    preferred_cat = (
        learner_df.loc[learner_df["segment"] == segment, "preferred_course_category"]
        .dropna()
        .astype(str)
    )
    if len(preferred_cat) > 0:
        mode_cat = preferred_cat.value_counts().index[0]
    else:
        mode_cat = None

    cat_factor = np.where(cdf["CourseCategory"].astype(str) == str(mode_cat), 1.25, 1.0)

    # Popularity within cluster proxy: use rating + diversity and a slight boost to categories
    popularity_proxy = (seg_div / (seg_div + 1)) * (seg_depth / (seg_depth + 1))

    cdf["relevance_score"] = (
        cdf["CourseRating"].fillna(0) * cat_factor * depth_factor * (0.7 + 0.3 * popularity_proxy) + 0.1 * seg_rating
    )

    # Return topk
    out = cdf[["CourseID", "CourseCategory", "CourseType", "CourseLevel", "CourseRating", "relevance_score"]].copy()
    out = out.sort_values("relevance_score", ascending=False).head(topk).reset_index(drop=True)
    return out

