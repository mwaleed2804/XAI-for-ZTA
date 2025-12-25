# XAI-for-ZTA

## Overview
This project implements an **Explainable AI (XAI) system for policy justification in a Zero Trust Architecture (ZTA)** as an academic Proof of Concept and is not intended as a production-ready Zero Trust system.
The goal is to ensure that **access control decisions made by machine learning models are transparent, interpretable, and auditable**, rather than acting as black-box decisions.

The system predicts access decisions and provides **human-understandable explanations** to justify why a policy decision was allowed or denied.

---

## Problem Statement
Traditional Zero Trust systems often rely on ML-based risk scoring models that:
- Act as black boxes
- Do not explain *why* access was granted or denied
- Create trust, compliance, and audit challenges

This project addresses these issues by integrating **Explainable AI techniques** to justify access policies clearly and transparently.

---

## Solution Approach
- Train a machine learning model on Zero Trust–style access data
- Use **Explainable AI (XAI)** techniques to explain predictions
- Provide **policy justification** using feature-level explanations
- Log decisions for **audit and accountability**

---

## System Architecture
- **Frontend:** Streamlit (Interactive UI)
- **Backend:** Python
- **ML Models:** XGBoost / Scikit-learn
- **Explainability:** SHAP / feature importance
- **Data Handling:** Pandas, NumPy
- **Logging:** Decision & audit logs

---

## Features
- Real-time access decision prediction
- Explainable AI–based policy justification
- Feature contribution visualization
- Audit logging for transparency
- User-friendly Streamlit interface

---

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- XGBoost
- SHAP
- Pandas
- NumPy

## Dataset
The project uses a synthetic Zero Trust–style dataset simulating user trust levels,
device posture, and access risk attributes to ensure privacy and reproducibility.

## Explainability Output
- Feature importance scores for each access decision
- SHAP-based explanations for allow/deny outcomes
- Transparent justification aligned with Zero Trust principles

## Future Scope
- Integration with real IAM or SIEM systems
- Real-time access decision streaming
- Advanced XAI dashboards for security analysts
- Deployment using cloud-native services
