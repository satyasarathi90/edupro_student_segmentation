from __future__ import annotations

# Column names expected in the three CSVs.
# These match the task's dataset fields; update if your CSV uses different headers.
USERS = {
    "UserID": "UserID",
    "UserName": "UserName",
    "Age": "Age",
    "Gender": "Gender",
    "Email": "Email", # (note: CSV may include UTF-8 BOM; loader strips it)
}

COURSES = {
    "CourseID": "CourseID",
    "CourseName": "CourseName",
    "CourseCategory": "CourseCategory",
    "CourseType": "CourseType",
    "CourseLevel": "CourseLevel",
    "CoursePrice": "CoursePrice",
    "CourseDuration": "CourseDuration",
    "CourseRating": "CourseRating",
}

TRANSACTIONS = {
    "TransactionID": "TransactionID",
    "UserID": "UserID",
    "CourseID": "CourseID",
    "TransactionDate": "TransactionDate",
    "Amount": "Amount",
    "PaymentMethod": "PaymentMethod",
    "TeacherID": "TeacherID",
}

TEACHERS = {
    "TeacherID": "TeacherID",
    "TeacherName": "TeacherName",
    "Age": "Age",
    "Gender": "Gender",
    "Expertise": "Expertise",
    "YearsOfExperience": "YearsOfExperience",
    "TeacherRating": "TeacherRating",
}



