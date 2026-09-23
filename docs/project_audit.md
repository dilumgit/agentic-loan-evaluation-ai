# Comprehensive Project Repository Audit & Architecture Freeze

## 1. Executive Summary & Audit Context
- **Research Title:** *“An Explainable Policy-Aware Agentic AI Decision Support System for Bank Loan Evaluation Using Machine Learning Predictions”*
- **Audit Purpose:** Perform a thorough, non-destructive audit and inventory of all repository files following the completion of data synthesis optimization (Gaussian Copula), final dataset preprocessing (Dataset v2), final Random Forest baseline training, and final XGBoost proposed model evaluation.
- **Architectural Principle:** Strictly preserve historical research artifacts (e.g. early CTGAN experiments, diagnostic scripts, legacy model binaries) in a structured `archive/` hierarchy to ensure complete scientific reproducibility, while isolating active authoritative files for downstream Explainable AI (SHAP/LIME) and Multi-Agent decision support.

---

## 2. Complete Repository Audit & Status Inventory

| File / Folder Path | Status | Reason / Technical Role | Action Taken |
|---|---|---|---|
| **Root & Config** | | | |
| `README.md` | **ACTIVE / REQUIRED** | Project overview, provenance, phase tracking, and directory schema. | Updated with authoritative architecture |
| `requirements.txt` | **ACTIVE / REQUIRED** | Python dependencies (`pandas`, `scikit-learn`, `xgboost`, `sdv`, etc.). | Preserved |
| `app.py` | **REVIEW** | Future Streamlit user interface placeholder. | Preserved untouched for future UI phase |
| **Data Layer (`data/`)** | | | |
| `data/raw/google_form_responses_500.csv` | **ACTIVE / REQUIRED** | The original, unchanged 500 empirical survey questionnaire responses. | Preserved as immutable seed dataset |
| `data/synthetic/loan_evaluation_dataset_3000_v2.csv` | **ACTIVE / REQUIRED** | **PRIMARY AUTHORITATIVE DATASET** (500 Seed + 2,500 Gaussian Copula synthetic). | Preserved as active primary dataset |
| `data/processed/X_train.csv` | **ACTIVE / REQUIRED** | Preprocessed training features matrix ($2,400 \times 55$). | Preserved as authoritative training features |
| `data/processed/X_test.csv` | **ACTIVE / REQUIRED** | Preprocessed holdout test features matrix ($600 \times 55$). | Preserved as authoritative holdout features |
| `data/processed/y_train.csv` | **ACTIVE / REQUIRED** | Encoded training target labels ($2,400 \times 1$). | Preserved as authoritative training targets |
| `data/processed/y_test.csv` | **ACTIVE / REQUIRED** | Encoded holdout test target labels ($600 \times 1$). | Preserved as authoritative holdout targets |
| `data/processed/preprocessor.joblib` | **ACTIVE / REQUIRED** | Serialized `ColumnTransformer` fitted strictly on `X_train`. | Preserved as active preprocessor artifact |
| `data/processed/feature_names.json` | **ACTIVE / REQUIRED** | Canonical 55 transformed feature names list. | Preserved as active feature schema |
| `data/preprocess_dataset.py` | **ACTIVE / REQUIRED** | Authoritative dataset v2 preprocessing pipeline script. | Preserved as active preprocessing pipeline |
| `data/validate_dataset_v2.py` | **ACTIVE / REQUIRED** | Dataset v2 validation and model-readiness verification script. | Preserved as active validation script |
| `data/sdv_compare_synthesizers.py` | **ACTIVE / REQUIRED** | Multi-synthesizer benchmark & candidate generation script. | Preserved as active synthesis benchmark |
| `data/validate_seed_dataset.py` | **ACTIVE / REQUIRED** | Empirical 500-record seed data validator script. | Preserved as active seed validator |
| `data/dataset_schema.md` | **ACTIVE / REQUIRED** | Complete 17-column dataset schema documentation. | Preserved as active documentation |
| `data/preprocessing_config.md` | **ACTIVE / REQUIRED** | Preprocessing hierarchy and categorical tier documentation. | Preserved as active documentation |
| **Models Layer (`models/`)** | | | |
| `models/baseline/train_final_random_forest.py` | **ACTIVE / REQUIRED** | Authoritative Random Forest baseline training & evaluation pipeline. | Preserved as active training pipeline |
| `models/baseline/final_random_forest.joblib` | **ACTIVE / REQUIRED** | **FINAL AUTHORITATIVE BASELINE MODEL** artifact. | Preserved as active baseline binary |
| `models/xgboost/train_final_xgboost.py` | **ACTIVE / REQUIRED** | Authoritative XGBoost proposed model training & evaluation pipeline. | Preserved as active training pipeline |
| `models/xgboost/final_xgboost.joblib` | **ACTIVE / REQUIRED** | **FINAL AUTHORITATIVE PROPOSED MODEL** artifact. | Preserved as active proposed model binary |
| **Evaluation Layer (`evaluation/`)** | | | |
| `evaluation/final_dataset_v2_summary.json` | **ACTIVE / REQUIRED** | Dataset v2 summary, partition distributions, and feature counts. | Preserved as active evaluation summary |
| `evaluation/final_model_comparison.csv` | **ACTIVE / REQUIRED** | Final comparative benchmark table (Random Forest vs. XGBoost). | Preserved as active comparison table |
| `evaluation/final_random_forest_results.json` | **ACTIVE / REQUIRED** | Final Random Forest evaluation metrics JSON. | Preserved as active results artifact |
| `evaluation/final_random_forest_confusion_matrix.csv` | **ACTIVE / REQUIRED** | Final Random Forest confusion matrix table ($N=600$). | Preserved as active confusion matrix |
| `evaluation/final_random_forest_predictions.csv` | **ACTIVE / REQUIRED** | Final Random Forest holdout sample predictions & probabilities. | Preserved as active prediction artifact |
| `evaluation/final_random_forest_cross_validation.csv` | **ACTIVE / REQUIRED** | Final Random Forest 5-Fold Stratified CV metrics table. | Preserved as active CV artifact |
| `evaluation/final_random_forest_feature_importance.csv` | **ACTIVE / REQUIRED** | Final Random Forest native Gini feature importances. | Preserved as active feature ranking |
| `evaluation/final_random_forest_classification_report.json` | **ACTIVE / REQUIRED** | Final Random Forest detailed classification report JSON. | Preserved as active classification report |
| `evaluation/final_random_forest_roc_curve.png` | **ACTIVE / REQUIRED** | Final Random Forest ROC Curve figure. | Preserved as active figure |
| `evaluation/final_random_forest_precision_recall_curve.png` | **ACTIVE / REQUIRED** | Final Random Forest Precision-Recall Curve figure. | Preserved as active figure |
| `evaluation/final_xgboost_results.json` | **ACTIVE / REQUIRED** | Final XGBoost proposed model evaluation metrics JSON. | Preserved as active results artifact |
| `evaluation/final_xgboost_confusion_matrix.csv` | **ACTIVE / REQUIRED** | Final XGBoost confusion matrix table ($N=600$). | Preserved as active confusion matrix |
| `evaluation/final_xgboost_predictions.csv` | **ACTIVE / REQUIRED** | Final XGBoost holdout sample predictions & probabilities. | Preserved as active prediction artifact |
| `evaluation/final_xgboost_cross_validation.csv` | **ACTIVE / REQUIRED** | Final XGBoost 5-Fold Stratified CV metrics table. | Preserved as active CV artifact |
| `evaluation/final_xgboost_feature_importance.csv` | **ACTIVE / REQUIRED** | Final XGBoost native Gain feature importances. | Preserved as active feature ranking |
| `evaluation/final_xgboost_classification_report.json` | **ACTIVE / REQUIRED** | Final XGBoost detailed classification report JSON. | Preserved as active classification report |
| `evaluation/final_xgboost_roc_curve.png` | **ACTIVE / REQUIRED** | Final XGBoost ROC Curve figure. | Preserved as active figure |
| `evaluation/final_xgboost_precision_recall_curve.png` | **ACTIVE / REQUIRED** | Final XGBoost Precision-Recall Curve figure. | Preserved as active figure |
| `evaluation/synthesizer_selection.csv` | **ACTIVE / REQUIRED** | Multi-synthesizer ranking & selection matrix. | Preserved as active synthesis evidence |
| `evaluation/synthesizer_distribution_comparison.csv` | **ACTIVE / REQUIRED** | Category-level marginal distribution comparison table. | Preserved as active distribution evidence |
| **Documentation Layer (`docs/`)** | | | |
| `docs/project_audit.md` | **ACTIVE / REQUIRED** | Complete repository audit and inventory table. | Active documentation artifact |
| `docs/project_cleanup_report.md` | **ACTIVE / REQUIRED** | Architecture freeze and cleanup report. | Active documentation artifact |
| `docs/synthetic_data_improvement.md` | **ACTIVE / REQUIRED** | Synthesizer benchmark and dataset v2 generation report. | Active documentation artifact |
| `docs/final_preprocessing_v2.md` | **ACTIVE / REQUIRED** | Final preprocessing protocol and feature specification report. | Active documentation artifact |
| `docs/final_random_forest_baseline.md` | **ACTIVE / REQUIRED** | Final Random Forest baseline evaluation report (v2). | Active documentation artifact |
| `docs/final_xgboost_model.md` | **ACTIVE / REQUIRED** | Final XGBoost proposed model evaluation report (v2). | Active documentation artifact |
| `docs/final_model_comparison.md` | **ACTIVE / REQUIRED** | Final comparative evaluation & model selection report. | Active documentation artifact |
| `docs/model_diagnostic_analysis.md` | **ACTIVE / REQUIRED** | Diagnostic analysis report identifying CTGAN signal dilution. | Active diagnostic report |
| **Archived Research Evidence (`archive/`)** | | | |
| `archive/datasets/loan_evaluation_dataset_3000.csv` | **LEGACY** | Previous CTGAN 3,000-record combined dataset. | Moved to `archive/datasets/` |
| `archive/datasets/loan_evaluation_synthetic_2000_up_to_100m.csv` | **LEGACY** | Initial synthetic 2,000-record dataset. | Moved to `archive/datasets/` |
| `archive/datasets/sdv_generated_2500.csv` | **LEGACY** | Previous CTGAN 2,500 synthetic records. | Moved to `archive/datasets/` |
| `archive/datasets/sdv_model.pkl` | **LEGACY** | Previous fitted CTGAN synthesizer model binary. | Moved to `archive/datasets/` |
| `archive/datasets/sdv_metadata.json` | **LEGACY** | Previous SDV metadata specification. | Moved to `archive/datasets/` |
| `archive/datasets/sdv_augment.py` | **LEGACY** | Initial CTGAN augmentation script. | Moved to `archive/datasets/` |
| `archive/datasets/sdv_validation.py` | **LEGACY** | Initial synthetic data validation script. | Moved to `archive/datasets/` |
| `archive/datasets/inspect_dataset.py` | **LEGACY** | Initial dataset exploration script. | Moved to `archive/datasets/` |
| `archive/datasets/candidates/` | **LEGACY** | Evaluated 2,500-record candidate datasets (`ctgan`, `gaussian_copula`, `tvae`). | Moved to `archive/datasets/candidates/` |
| `archive/models/baseline/random_forest.joblib` | **LEGACY** | Previous CTGAN-trained Random Forest model binary. | Moved to `archive/models/baseline/` |
| `archive/models/baseline/random_forest.py` | **LEGACY** | Previous Random Forest training script. | Moved to `archive/models/baseline/` |
| `archive/models/xgboost/xgboost_model.joblib` | **LEGACY** | Previous CTGAN-trained XGBoost model binary. | Moved to `archive/models/xgboost/` |
| `archive/models/xgboost/xgboost_model.py` | **LEGACY** | Previous XGBoost training script. | Moved to `archive/models/xgboost/` |
| `archive/evaluation/random_forest_results.json` | **LEGACY** | Previous CTGAN Random Forest metrics JSON. | Moved to `archive/evaluation/` |
| `archive/evaluation/random_forest_confusion_matrix.csv` | **LEGACY** | Previous CTGAN Random Forest confusion matrix. | Moved to `archive/evaluation/` |
| `archive/evaluation/random_forest_predictions.csv` | **LEGACY** | Previous CTGAN Random Forest predictions. | Moved to `archive/evaluation/` |
| `archive/evaluation/random_forest_feature_importance.csv` | **LEGACY** | Previous CTGAN Random Forest feature importance. | Moved to `archive/evaluation/` |
| `archive/evaluation/evaluate_baseline.py` | **LEGACY** | Previous baseline validation script. | Moved to `archive/evaluation/` |
| `archive/evaluation/xgboost_results.json` | **LEGACY** | Previous CTGAN XGBoost metrics JSON. | Moved to `archive/evaluation/` |
| `archive/evaluation/xgboost_confusion_matrix.csv` | **LEGACY** | Previous CTGAN XGBoost confusion matrix. | Moved to `archive/evaluation/` |
| `archive/evaluation/xgboost_predictions.csv` | **LEGACY** | Previous CTGAN XGBoost predictions. | Moved to `archive/evaluation/` |
| `archive/evaluation/xgboost_feature_importance.csv` | **LEGACY** | Previous CTGAN XGBoost feature importance. | Moved to `archive/evaluation/` |
| `archive/evaluation/model_comparison.csv` | **LEGACY** | Previous CTGAN model comparison table. | Moved to `archive/evaluation/` |
| `archive/evaluation/compare_models.py` | **LEGACY** | Previous model comparison script. | Moved to `archive/evaluation/` |
| `archive/evaluation/original_vs_sdv_distribution.csv` | **LEGACY** | Previous Step 6 CTGAN distribution comparison table. | Moved to `archive/evaluation/` |
| `archive/evaluation/feature_target_association.csv` | **LEGACY** | Previous Step 6 CTGAN association table. | Moved to `archive/evaluation/` |
| `archive/evaluation/cross_validation_results.csv` | **LEGACY** | Previous Step 6 CTGAN cross-validation table. | Moved to `archive/evaluation/` |
| `archive/evaluation/prediction_probability_analysis.csv` | **LEGACY** | Previous Step 6 CTGAN probability analysis. | Moved to `archive/evaluation/` |
| `archive/evaluation/roc_curve.png` | **LEGACY** | Previous Step 6 CTGAN ROC curve. | Moved to `archive/evaluation/` |
| `archive/evaluation/precision_recall_curve.png` | **LEGACY** | Previous Step 6 CTGAN Precision-Recall curve. | Moved to `archive/evaluation/` |
| `archive/evaluation/run_diagnostic_analysis.py` | **LEGACY** | Previous Step 6 diagnostic execution script. | Moved to `archive/evaluation/` |
| `archive/docs/random_forest_baseline.md` | **LEGACY** | Previous CTGAN Random Forest report. | Moved to `archive/docs/` |
| `archive/docs/xgboost_model.md` | **LEGACY** | Previous CTGAN XGBoost report. | Moved to `archive/docs/` |
| `archive/docs/sdv_augmentation.md` | **LEGACY** | Previous initial CTGAN augmentation report. | Moved to `archive/docs/` |
| **Future Components (Untouched)** | | | |
| `explainability/` | **REVIEW** | Future SHAP & LIME explainer modules. | Untouched / Reserved for Step 11 |
| `agents/` | **REVIEW** | Future Multi-Agent orchestration module. | Untouched / Reserved for Step 13 |
| `rag/` | **REVIEW** | Future FAISS vector store & policy retrieval module. | Untouched / Reserved for Step 12 |
| `ui/` | **REVIEW** | Future Streamlit user interface module. | Untouched / Reserved for Step 14 |

---

## 3. Summary Statistics of Project Audit
- **Active / Authoritative Code & Data Files:** `32` files
- **Archived Historical Research Artifacts:** `34` files
- **Temporary Cache / Scratch Files Removed:** `0` (Clean workspace, no dangling caches)
- **Downstream Modules Preserved for Future Phases:** `explainability/`, `rag/`, `agents/`, `ui/`
