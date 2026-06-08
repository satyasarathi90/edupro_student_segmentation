from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def _prepare_feature_matrix(learner_df: pd.DataFrame):
    # Numerical features
    numeric_cols = [
        "Age",
        "total_courses_enrolled",
        "avg_courses_per_category",
        "avg_enrolled_course_rating",
        "avg_spending_per_learner",
        "diversity_score",
        "learning_depth_index",
    ]

    # Categorical features
    cat_cols = ["Gender", "preferred_course_category", "preferred_course_level"]

    learner_df = learner_df.copy()

    # Ensure columns exist
    for c in numeric_cols:
        if c not in learner_df.columns:
            learner_df[c] = 0
    for c in cat_cols:
        if c not in learner_df.columns:
            learner_df[c] = "Unknown"

    # Force numeric conversion (prevents strings causing StandardScaler crash)
    for c in numeric_cols:
        learner_df[c] = pd.to_numeric(learner_df[c], errors="coerce")

    # Replace inf with NaN then fill NaN with 0
    learner_df[numeric_cols] = learner_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    learner_df[numeric_cols] = learner_df[numeric_cols].fillna(0)

    # Categorical safe handling
    learner_df[cat_cols] = learner_df[cat_cols].fillna("Unknown").astype(str)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X = pre.fit_transform(learner_df)
    return pre, X



def segment_learners(learner_df: pd.DataFrame, k: int = 4):
    """Cluster learners and return model artifacts."""
    learner_df = learner_df.copy()

    # Hard guard: StandardScaler (via pipeline) fails on empty datasets
    if learner_df.empty or len(learner_df) == 0:
        return {
            "preprocessor": None,
            "kmeans": None,
            "labels_df": pd.DataFrame({"UserID": [], "segment": []}),
            "silhouette": np.nan,
            "pca": None,
            "projection_df": None,
        }

    pre, X = _prepare_feature_matrix(learner_df)

    # If all rows became invalid, stop early
    if getattr(X, "shape", (0, 0))[0] == 0:
        return {
            "preprocessor": None,
            "kmeans": None,
            "labels_df": pd.DataFrame({"UserID": learner_df.get("UserID", []).values if "UserID" in learner_df.columns else [], "segment": []}),
            "silhouette": np.nan,
            "pca": None,
            "projection_df": None,
        }

    k = int(k)
    if k < 2:
        k = 2
    if X.shape[0] < k:
        k = max(2, int(X.shape[0]))

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)


    # Silhouette (only if enough samples)
    sil = np.nan
    if X.shape[0] >= k + 1 and k > 1:
        sil = float(silhouette_score(X, labels))

    # Projection for visualization
    pca = None
    projection_df = None
    try:
        if X.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=42)
            proj = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
            projection_df = pd.DataFrame({
                "UserID": learner_df["UserID"].values,
                "pc1": proj[:, 0],
                "pc2": proj[:, 1],
                "segment": labels,
            })
    except Exception:
        pca = None

    labels_df = pd.DataFrame({"UserID": learner_df["UserID"].values, "segment": labels.astype(int)})

    # Also attach labels onto learner_df so callers that expect in-place behavior work.
    learner_df = learner_df.merge(labels_df, on="UserID", how="left", validate="one_to_one")

    return {
        "preprocessor": pre,
        "kmeans": km,
        "labels_df": labels_df,
        "silhouette": sil,
        "pca": pca,
        "projection_df": projection_df,
    }

