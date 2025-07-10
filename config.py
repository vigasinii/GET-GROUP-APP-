# config.py - Reads all sensitive data from Streamlit secrets

import streamlit as st

# Firebase Configuration
FIREBASE_WEB_API_KEY = st.secrets["FIREBASE_WEB_API_KEY"]
FIREBASE_DB_URL = st.secrets["FIREBASE_DB_URL"]

# Gemini AI Configuration
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Admin Registration Code
ADMIN_REGISTRATION_CODE = st.secrets["ADMIN_REGISTRATION_CODE"]

# Application Configuration
APP_CONFIG = st.secrets["APP_CONFIG"]
DEFAULT_USERS = st.secrets["DEFAULT_USERS"]

# Constants
ASSET_CATEGORIES = [
    "Laptop", "Desktop", "Server", "Printer", "Monitor",
    "Network Equipment", "Mobile Device", "Storage Device", "Other"
]

TICKET_CATEGORIES = ["Technical", "Hardware", "Access", "Network", "General"]
PRIORITY_LEVELS = ["Low", "Medium", "High"]
TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
ASSET_STATUSES = ["Active", "Inactive", "Maintenance", "Retired"]
USER_STATUSES = ["Active", "Inactive"]
PROJECTS = ["Dubai Operations", "Identity Management", "Project Alpha", "General"]
REGIONS = ["UAE", "Singapore", "India", "KSA"]

# Email Config
EMAIL_CONFIG = st.secrets["EMAIL_CONFIG"]

# Feature Flags
FEATURE_FLAGS = st.secrets["FEATURE_FLAGS"]

# Security Config
SECURITY_CONFIG = st.secrets["SECURITY_CONFIG"]

# SLA Configuration
SLA_CONFIG = st.secrets["SLA_CONFIG"]


# Helper Functions
def get_safe_email(email: str) -> str:
    return email.replace('@', '_at_').replace('.', '_dot_')

def get_email_from_safe(safe_email: str) -> str:
    return safe_email.replace('_at_', '@').replace('_dot_', '.')

def validate_config():
    """Basic validation to ensure critical secrets are loaded"""
    errors = []
    if not FIREBASE_WEB_API_KEY:
        errors.append("Missing FIREBASE_WEB_API_KEY")
    if not FIREBASE_DB_URL:
        errors.append("Missing FIREBASE_DB_URL")
    if not GEMINI_API_KEY:
        errors.append("Missing GEMINI_API_KEY")
    if not ADMIN_REGISTRATION_CODE:
        errors.append("Missing ADMIN_REGISTRATION_CODE")
    return errors
