# Documentation consistency update — 15 September 2026

This update synchronized the repository documentation with the current evaluation workflow.

Changes:

- corrected `VALIDATION_REPORT.md` so it no longer claims that model selection uses the final test set;
- clarified that the packaged `models/model_metadata.json` describes a historical fixed-split baseline artifact;
- synchronized `PROJECT_SUMMARY.txt` with training-only 5-fold cross-validation and untouched holdout evaluation;
- added `doc/README.md` as a documentation source-of-truth guide;
- added `doc/CURRENT_REPORT.md` as the current synchronized technical/business report;
- added `doc/REPORT_STATUS.md` to distinguish current documentation from historical imported DOCX/PDF artifacts;
- added `doc/FINAL_CHECKLIST.md` to make project assumptions and evaluation requirements explicit;
- retained passenger-count questions as not applicable because the supplied dataset contains no genuine `passenger_count` field;
- retained supplied `distance_km` as the source distance feature and Haversine distance for reference/audit only.

No passenger labels were fabricated and no historical metrics were relabeled as cross-validation results.
