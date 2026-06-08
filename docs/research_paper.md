# EduPro • Student Segmentation & Personalized Course Recommendation System

## Abstract
This project introduces a data-driven learner intelligence layer for EduPro. Instead of one-size-fits-all course recommendations, we segment learners into behavioral/proficiency profiles using engineered learner features derived from Users, Courses, and Transactions datasets. We then generate cluster-aware course recommendations using content cues (level/category), course quality (rating-weighted relevance), and segment behavioral signals (preferred category and learning depth).

## 1. Dataset
We use three sources:
- **Users**: UserID, Age, Gender
- **Courses**: CourseID, CourseCategory, CourseType, CourseLevel, CourseRating
- **Transactions**: UserID, CourseID, TransactionDate, Amount

## 2. Feature Engineering (Learner Profiles)
We aggregate transactions at `UserID` level and compute:
- Engagement features
  - Total courses enrolled
  - Average courses per category
  - Enrollment frequency (available via TransactionDate; can be added as a future improvement)
- Preference features
  - Preferred course category
  - Preferred course level
  - Average course rating enrolled
- Behavioral features
  - Average spending per learner
  - Diversity score (number of categories explored)
  - Learning depth index (beginner vs advanced ratio using CourseLevel)

## 3. Learner Segmentation
We apply unsupervised clustering:
- Feature preprocessing: numerical standardization + one-hot encoding for categorical fields.
- **K-Means clustering** with user-selectable K.
- Cluster quality is measured using **Silhouette Score**.

(Extension: hierarchical clustering with validation is described in the project plan; current implementation focuses on K-Means for speed and dashboard responsiveness.)

## 4. Personalized Recommendation Logic
For a selected learner:
1. Assign learner to a cluster (segment).
2. Determine segment-level signals:
   - mean learning depth index
   - diversity score
   - most common preferred category
3. Score candidate courses using:
   - rating-weighted relevance
   - level compatibility factor
   - category match factor
4. Return top-k courses by relevance.

## 5. Evaluation & Validation
- **Silhouette Score**: measures clustering separation.
- **Intra-cluster consistency proxy**: inverse function of within-segment standard deviation across engineered behavioral features.
- **Recommendation precision proxy**: alignment proxy based on segment preferred category (placeholder until explicit ground-truth holdouts are provided).

## 6. Streamlit Dashboard
The Streamlit app provides:
- Learner profile explorer (feature values + assigned segment)
- Cluster overview (counts + visualization projection)
- Personalized course recommendations (top-k)
- Segment comparison panels (aggregated engineered feature statistics)

## 7. Limitations & Future Work
- Recommendation evaluation uses proxy metrics; true precision/recall requires holdout transactions and definition of “relevant” courses.
- Enrollment frequency and time-based features can be added.
- Implement content-based filtering with course embeddings and collaborative filtering with implicit feedback.

