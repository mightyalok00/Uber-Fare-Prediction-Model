# `.github/` — Repository Automation

This folder contains GitHub-specific project automation.

## `workflows/python-checks.yml`

The workflow is split by responsibility:

- **Python 3.12 job** — authoritative training/reproducibility check.
- **Python 3.14.7 job** — committed-model loading and Streamlit compatibility check.

The workflow intentionally does not auto-commit regenerated model files. Training outputs are validated in temporary CI directories so repository history stays explicit and reviewable.

## Why two Python versions?

Python 3.12 is used for stable training reproducibility. Python 3.14.7 is used to prove the committed saved model and Streamlit app can load and run in the deployment environment.
