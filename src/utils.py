"""
utils.py
─────────────────────────────────────────────────────────────
Shared utility functions for the Student Depression Detection system.
"""

import os
import pandas as pd
import numpy as np


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')


def format_feature_name(name: str) -> str:
    """Convert snake_case column name to human-readable label."""
    mapping = {
        'age':                         'Age',
        'gender':                      'Gender',
        'cgpa':                        'CGPA',
        'academic_pressure':           'Academic Pressure',
        'study_satisfaction':          'Study Satisfaction',
        'work_study_hours':            'Work/Study Hours',
        'financial_stress':            'Financial Stress',
        'sleep_duration':              'Sleep Duration',
        'dietary_habits':              'Dietary Habits',
        'have_you_ever_had_suicidal_thoughts': 'Suicidal Thoughts',
        'suicidal_thoughts':           'Suicidal Thoughts',
        'family_history_of_mental_illness': 'Family History',
        'family_history':              'Family History',
    }
    return mapping.get(name, name.replace('_', ' ').title())


def get_plot_path(filename: str) -> str:
    return os.path.join(PLOTS_DIR, filename)


def risk_category(prob_dep: float) -> tuple[str, str]:
    """
    Classify depression risk level.
    Returns (category, emoji).
    """
    if prob_dep < 25:
        return 'Low Risk',      '🟢'
    elif prob_dep < 50:
        return 'Moderate Risk', '🟡'
    elif prob_dep < 75:
        return 'High Risk',     '🟠'
    else:
        return 'Very High Risk','🔴'
