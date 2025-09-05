SYSTEM = """You are a careful information extraction assistant.
Only produce data that you are confident about.
Follow the schema exactly."""

USER_TEMPLATE = """Extract a Task from the text below.

Rules:
- priority is of ["low","medium","high","urgent"]
- tags are at most 8, lowercase, single tokens if possible
- deadline is ISO-8601 (YYYY-MM-DD) if clearly stated, else null
- confidence is 0..1 calibrated; 0.5 = unsure, 0.9 = very sure

Text:
---
{TEXT}
---"""
