from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import pandas as pd

from .config import USERS, COURSES, TRANSACTIONS, TEACHERS





def _read_csv(path: str) -> pd.DataFrame:
    # Use UTF-8-sig to tolerate BOM in header cells.
    return pd.read_csv(path, encoding="utf-8-sig")


def _write_uploaded_to_temp(uploaded_file, suffix: str) -> str:
    # UploadedFile from Streamlit has .getvalue()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return tmp_path


def _require_columns(df: pd.DataFrame, required: dict[str, str], name: str) -> None:
    missing = []
    for k, col in required.items():
        if col not in df.columns:
            missing.append(f"{k} -> {col}")
    if missing:
        raise ValueError(f"{name} CSV is missing required columns: {missing}. Available: {list(df.columns)}")


def load_edupro_dataset(
    data_dir: str,
    uploaded_users=None,
    uploaded_courses=None,
    uploaded_transactions=None,
    uploaded_teachers=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Users/Courses/Transactions (+ optional Teachers) CSVs.

    Streamlit uploads are optionally written to temp files.
    """

    users_path = None
    courses_path = None
    tx_path = None
    teachers_path = None


    default_users = os.path.join(data_dir, "users.csv")
    default_courses = os.path.join(data_dir, "courses.csv")
    default_transactions = os.path.join(data_dir, "transactions.csv")
    # Repository ship defaults:
    # - app.py uses `courses_cleaned.csv` and `transaction.csv`
    # Keep compatibility by falling back to those filenames.
    alt_courses = os.path.join(data_dir, "courses_cleaned.csv")
    alt_transactions = os.path.join(data_dir, "transaction.csv")
    default_teachers = os.path.join(data_dir, "teacher.csv")


    if uploaded_users is not None:
        users_path = _write_uploaded_to_temp(uploaded_users, suffix="_users.csv")
    elif os.path.exists(default_users):
        users_path = default_users

    if uploaded_courses is not None:
        courses_path = _write_uploaded_to_temp(uploaded_courses, suffix="_courses.csv")
    elif os.path.exists(default_courses):
        courses_path = default_courses
    elif os.path.exists(alt_courses):
        courses_path = alt_courses

    if uploaded_transactions is not None:
        tx_path = _write_uploaded_to_temp(uploaded_transactions, suffix="_transactions.csv")
    elif os.path.exists(default_transactions):
        tx_path = default_transactions
    elif os.path.exists(alt_transactions):
        tx_path = alt_transactions

    if uploaded_teachers is not None:
        teachers_path = _write_uploaded_to_temp(uploaded_teachers, suffix="_teachers.csv")
    elif os.path.exists(default_teachers):
        teachers_path = default_teachers

    if users_path is None or courses_path is None or tx_path is None:
        raise FileNotFoundError(
            "Missing dataset CSVs. Add `data/users.csv`, `data/courses.csv`, `data/transactions.csv` or upload them in the sidebar."
        )


    df_users = _read_csv(users_path)
    df_courses = _read_csv(courses_path)
    df_tx = _read_csv(tx_path)

    # Teachers are optional: only validate if we actually loaded a non-empty teachers CSV.
    if teachers_path is not None:
        df_teacher = _read_csv(teachers_path)
    else:
        df_teacher = pd.DataFrame(columns=list(TEACHERS.values()))

    _require_columns(df_users, USERS, "Users")
    _require_columns(df_courses, COURSES, "Courses")
    _require_columns(df_tx, TRANSACTIONS, "Transactions")

    # Fix: some runs can end up with an empty DF while still having teachers_path;
    # in that case we should not require teacher columns.
    if df_teacher is not None and not df_teacher.empty:
        _require_columns(df_teacher, TEACHERS, "Teachers")



    # Normalize dtypes (lightweight)
    # IDs in this dataset are string-coded (e.g., U00003, CR00016, TT00001).
    # Avoid coercing them to numeric, otherwise we end up with nulls and empty joins.
    df_users["UserID"] = df_users["UserID"].astype(str)
    df_courses["CourseID"] = df_courses["CourseID"].astype(str)
    df_tx["TransactionID"] = df_tx["TransactionID"].astype(str)
    df_tx["UserID"] = df_tx["UserID"].astype(str)
    df_tx["CourseID"] = df_tx["CourseID"].astype(str)
    if "TeacherID" in df_tx.columns:
        df_tx["TeacherID"] = df_tx["TeacherID"].astype(str)


    # Dataset uses day-month-year like "25-10-2025".
    df_tx["TransactionDate"] = pd.to_datetime(df_tx["TransactionDate"], errors="coerce", dayfirst=True)
    df_tx["Amount"] = pd.to_numeric(df_tx["Amount"], errors="coerce")


    df_tx = df_tx.dropna(subset=["UserID", "CourseID", "TransactionDate"]).copy()

    # Ensure we have the required numeric columns with safe coercion.
    if "Amount" in df_tx.columns:
        df_tx["Amount"] = pd.to_numeric(df_tx["Amount"], errors="coerce").fillna(0.0)

    # Teacher normalization (optional but ensures consistent dtypes)
    try:
        from .teacher_utils import load_teacher_csv  # local import to avoid circulars
        if df_teacher is not None and not df_teacher.empty:
            df_teacher = load_teacher_csv(df_teacher)
    except Exception:
        # Keep app running even if teacher normalization fails.
        pass

    return df_users, df_courses, df_tx, df_teacher



