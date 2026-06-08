from __future__ import annotations

import os

# joblib is currently not used in the app; keeping it commented avoids unused warnings.
#import joblib


import numpy as np
import numpy as np
import pandas as pd
import streamlit as st

# Robust Plotly import: if Plotly fails for any reason, fall back to tables.
try:
    import plotly.express as px  # type: ignore
except Exception:  # pragma: no cover
    px = None



# Ensure local imports work when running `streamlit run app.py`
import sys

ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_edupro_dataset
from src.feature_engineering import build_learner_features
from src.segmentation import segment_learners
from src.recommendation import recommend_courses_for_learner
from src.evaluation import evaluate_clustering_quality


st.set_page_config(page_title="EduPro • Student Segmentation & Personalized Recommendations", layout="wide")

st.title("EduPro • Student Segmentation & Personalized Course Recommendation System")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset upload (optional)
st.sidebar.header("Data")
uploaded_users = st.sidebar.file_uploader("Users CSV", type=["csv"], key="users")
uploaded_courses = st.sidebar.file_uploader("Courses CSV", type=["csv"], key="courses")
uploaded_transactions = st.sidebar.file_uploader("Transactions CSV", type=["csv"], key="transactions")

# The app expects `data/transaction.csv` (singular) based on this repository's naming.
# Keep UI text generic so users can upload either file name; loader will use uploaded file if provided.
uploaded_teachers = st.sidebar.file_uploader("Teacher CSV", type=["csv"], key="teachers")

# Defaults
default_users = os.path.join(DATA_DIR, "users.csv")
default_courses = os.path.join(DATA_DIR, "courses_cleaned.csv")
default_transactions = os.path.join(DATA_DIR, "transaction.csv")
default_teachers = os.path.join(DATA_DIR, "teacher.csv")

if uploaded_users is None and (not os.path.exists(default_users)):
    st.info("Add `data/users.csv` or upload it in the sidebar.")
if uploaded_courses is None and (not os.path.exists(default_courses)):
    st.info("Add `data/courses_cleaned.csv` or upload it in the sidebar.")
if uploaded_transactions is None and (not os.path.exists(default_transactions)):
    st.info("Add `data/transactions.csv` or upload it in the sidebar.")
# Teachers are optional for now.


run_button = st.sidebar.button("Train & Generate Recommendations", type="primary")

st.sidebar.divider()

K_DEFAULT = st.sidebar.slider("K clusters", 2, 10, 4)
TOPK_DEFAULT = st.sidebar.slider("Recommendations per learner (top-k)", 3, 15, 6)

if not run_button:
    st.stop()

# Load dataset
with st.spinner("Loading dataset..."):
    # The loader handles UploadedFile -> temp CSV
    df_users, df_courses, df_tx, df_teacher = load_edupro_dataset(
        data_dir=DATA_DIR,
        uploaded_users=uploaded_users,
        uploaded_courses=uploaded_courses,
        uploaded_transactions=uploaded_transactions,
        uploaded_teachers=uploaded_teachers,
    )

# Teacher preview (helps confirm teacher.csv is loaded correctly)
st.sidebar.subheader("Teacher CSV Preview")
if df_teacher is None or df_teacher.empty:
    st.sidebar.caption("No teacher data loaded.")
else:
    st.sidebar.dataframe(df_teacher.head(10), use_container_width=True)



# Feature engineering
with st.spinner("Building learner profiles & engineered features..."):
    learner_df = build_learner_features(df_users=df_users, df_courses=df_courses, df_tx=df_tx)

# Segmentation
with st.spinner("Segmenting learners (clustering)..."):
    model = segment_learners(learner_df, k=int(K_DEFAULT))
    learner_df = learner_df.merge(
        model["labels_df"], on="UserID", how="left", validate="one_to_one"
    )

# Evaluation
with st.spinner("Evaluating cluster quality / recommendation proxy metrics..."):
    metrics = evaluate_clustering_quality(learner_df, model)

st.subheader("Cluster Overview")

labels_df = model["labels_df"].copy()
labels_df = labels_df.merge(
    learner_df[["UserID", "total_courses_enrolled", "diversity_score"]],
    on="UserID",
    how="left",
)

# 1) Segment size distribution
segment_counts = (

    labels_df.groupby("segment", as_index=False)["UserID"]
    .count()
    .rename(columns={"UserID": "learners"})
)
if px is None:
    st.dataframe(segment_counts)
else:
    st.plotly_chart(
        px.bar(
            segment_counts,
            x="segment",
            y="learners",
            color="segment",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title="Learners per Segment",
        ),
        use_container_width=True,
    )

# 2) Segment comparison in graph format (feature means by segment)
# Use only numeric features that are present.
segment_feature_candidates = [
    "total_courses_enrolled",
    "avg_courses_per_category",
    "avg_enrolled_course_rating",
    "avg_spending_per_learner",
    "diversity_score",
    "learning_depth_index",
]
segment_feature_cols = [
    c for c in segment_feature_candidates if c in learner_df.columns
]

if len(segment_feature_cols) > 0:
    seg_feature_means = (
        learner_df.groupby("segment")[segment_feature_cols]
        .mean()
        .reset_index()
    )
    seg_feature_means_long = seg_feature_means.melt(
        id_vars=["segment"],
        var_name="feature",
        value_name="mean_value",
    )

    if px is None:
        st.dataframe(seg_feature_means)
    else:
        # Heatmap gives a compact overview.
        heat = px.density_heatmap(
            seg_feature_means_long,
            x="feature",
            y="segment",
            z="mean_value",
            histfunc="avg",
            color_continuous_scale="Viridis",
            title="Segment Feature Means (Heatmap)",
        )
        heat.update_layout(xaxis_title="Feature", yaxis_title="Segment")
        st.plotly_chart(heat, use_container_width=True)


# (No extra plotly rendering here; chart is already rendered above when px exists)




st.subheader("Cluster Quality")
cols = st.columns(3)
cols[0].metric("Silhouette (selected K)", f"{metrics['silhouette']:.3f}")
cols[1].metric("Intra-cluster similarity", f"{metrics['intra_cluster_similarity']:.3f}")
cols[2].metric("Recommendation precision@k proxy", f"{metrics['precision_at_k_proxy']:.3f}")

# Visualization: 2D projection using PCA stored in model
# If plotly is not available, fall back to a dataframe preview.
if model.get("pca") is not None:
    proj = model["projection_df"].copy()
    if px is not None:
        fig = px.scatter(
            proj,
            x="pc1",
            y="pc2",
            color="segment",
            hover_data=["UserID"],
        )
        fig.update_traces(marker=dict(size=8, opacity=0.85))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(proj.head(200))


st.divider()

st.subheader("Learner Profile Explorer")

all_user_ids = sorted(learner_df["UserID"].unique().tolist())
selected_user = st.selectbox("Select a learner (UserID)", all_user_ids)

user_row = learner_df.loc[learner_df["UserID"] == selected_user].iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric("Assigned Segment", int(user_row["segment"]))
c2.metric("Total Courses Enrolled", int(user_row["total_courses_enrolled"]))
c3.metric("Diversity Score", float(user_row["diversity_score"]))

st.markdown("### Engineered Features")
feature_cols = [
    "total_courses_enrolled",
    "avg_courses_per_category",
    "preferred_course_category",
    "preferred_course_level",
    "avg_enrolled_course_rating",
    "avg_spending_per_learner",
    "diversity_score",
    "learning_depth_index",
]

feat_table = pd.DataFrame({
    "feature": feature_cols,
    "value": [user_row.get(c, np.nan) for c in feature_cols],
})
st.dataframe(feat_table)

st.divider()

st.subheader("Personalized Course Recommendations")

topk = int(TOPK_DEFAULT)
recs = recommend_courses_for_learner(
    user_id=selected_user,
    learner_df=learner_df,
    courses_df=df_courses,
    model=model,
    topk=topk,
)

st.dataframe(recs)

st.markdown("### Recommended Learning Paths (simple ordered list)")
# Sort by relevance score descending
ordered = recs.sort_values("relevance_score", ascending=False).reset_index(drop=True)
st.dataframe(ordered[["CourseID", "CourseCategory", "CourseLevel", "relevance_score"]])

st.divider()

st.subheader("Segment Comparison")
seg_stats = learner_df.groupby("segment")[
    [
        "total_courses_enrolled",
        "avg_courses_per_category",
        "avg_spending_per_learner",
        "diversity_score",
        "learning_depth_index",
        "avg_enrolled_course_rating",
    ]
].agg(["mean", "median"]).round(3)

st.dataframe(seg_stats)


# Recommendation precision proxy (optional display)
# computed already in evaluation


