# Executive Summary — EduPro Learner Segmentation & Personalized Recommendations

## Purpose
EduPro serves learners with distinct goals and learning behaviors. Generic recommendations reduce engagement and course completion. This project builds a personalization engine that:
- segments learners into meaningful profiles,
- assigns each learner to a segment,
- recommends courses aligned to the learner’s preferences and engagement patterns.

## Approach
1. **Learner profiling**: derive engagement, preference, and behavioral features from Users, Courses, and Transactions.
2. **Segmentation**: apply unsupervised clustering (K-Means) to group similar learners.
3. **Recommendation**: score courses using segment-aware signals:
   - course rating quality,
   - category/level compatibility,
   - segment learning depth and diversity.

## Outputs
- Interactive Streamlit dashboard for educators and product stakeholders.
- Segment analytics (cluster sizes + engineered feature comparisons).
- Personalized top-k course recommendations per selected learner.

## Expected Benefits
- Higher relevance of recommendations.
- Improved learner engagement and retention.
- Better course discovery and long-term platform loyalty.

## Notes for Deployment
The system is dataset-driven. Stakeholder performance measurement should be refined using explicit holdout experiments (recommendation relevance at future enrollment time).
