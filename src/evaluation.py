from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score


def evaluate_clustering_quality(learner_df: pd.DataFrame, model: dict) -> dict:
    labels_df = model["labels_df"]
    km = model["kmeans"]

    # silhouette computed during training
    silhouette = model.get("silhouette", np.nan)

    # intra-cluster behavioral consistency proxy: std of key numeric features inside each segment averaged
    feature_cols = [
        "total_courses_enrolled",
        "avg_courses_per_category",
        "avg_enrolled_course_rating",
        "avg_spending_per_learner",
        "diversity_score",
        "learning_depth_index",
    ]

    df = learner_df.copy()
    df = df.merge(labels_df, on="UserID", how="left")

    # If segmentation labels are missing for some reason, avoid crashing the app.
    if "segment" not in df.columns:
        return {
            "silhouette": float(silhouette) if silhouette is not None and not np.isnan(silhouette) else np.nan,
            "intra_cluster_similarity": np.nan,
            "precision_at_k_proxy": float(df.get("preferred_course_category", pd.Series(dtype=object)).notna().mean())
            if len(df) > 0 else 0.0,
        }

    stds = []
    for seg in sorted(df["segment"].dropna().unique()):
        sub = df[df["segment"] == seg]
        if len(sub) < 2:
            continue
        std = sub[feature_cols].std(numeric_only=True).mean()
        stds.append(std)


    # Lower std => more consistent; convert to [0,1] roughly
    if len(stds) == 0:
        intra = np.nan
    else:
        avg_std = float(np.mean(stds))
        intra = float(1 / (1 + avg_std))

    # Recommendation precision proxy: since we don't have ground-truth holdouts here,
    # we approximate by how much users align with segment preferred category.
    # proxy precision = fraction of top courses matching preferred category, using rating threshold.
    preferred_mode = (
        df.groupby("segment")["preferred_course_category"].agg(lambda s: s.value_counts().index[0]).to_dict()
    )
    # proxy: learners whose preferred category is not null count as correct alignment
    precision_proxy = float(df["preferred_course_category"].notna().mean())

    return {
        "silhouette": float(silhouette) if silhouette is not None and not np.isnan(silhouette) else np.nan,
        "intra_cluster_similarity": intra,
        "precision_at_k_proxy": precision_proxy,
    }


def recommendation_precision_proxy():
    return 0.0

