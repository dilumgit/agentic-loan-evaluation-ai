"""
Custom Styling and Theming for Bank Loan Decision Support Interface.

Research Title:
An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation
Using Machine Learning Predictions
"""

import streamlit as st


def apply_custom_styles():
    """Inject custom CSS for clean, professional banking UI presentation."""
    custom_css = """
    <style>
        /* Main Container Spacing */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Title & Subtitle */
        .main-header {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            font-size: 2.1rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4b5563;
            margin-bottom: 1.2rem;
            line-height: 1.4;
        }

        /* Advisory Disclaimer Banner */
        .advisory-banner {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 0.85rem 1.2rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            font-size: 0.92rem;
            color: #1e40af;
            line-height: 1.45;
        }

        /* Section Card Box */
        .section-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        .section-card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.75rem;
            border-bottom: 2px solid #f3f4f6;
            padding-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Recommendation Badges */
        .badge-approve {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            padding: 0.4rem 0.9rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }
        .badge-reject {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
            padding: 0.4rem 0.9rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }
        .badge-manual {
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fde68a;
            padding: 0.4rem 0.9rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }
        .badge-info {
            background-color: #f3f4f6;
            color: #374151;
            border: 1px solid #d1d5db;
            padding: 0.4rem 0.9rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }

        /* XAI Factor Badges */
        .tag-positive {
            background-color: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.86rem;
            font-weight: 500;
            margin: 0.2rem 0;
            display: block;
        }
        .tag-negative {
            background-color: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-size: 0.86rem;
            font-weight: 500;
            margin: 0.2rem 0;
            display: block;
        }

        /* Audit metadata block */
        .audit-box {
            font-family: 'Consolas', 'Courier New', monospace;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            color: #334155;
            line-height: 1.5;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
