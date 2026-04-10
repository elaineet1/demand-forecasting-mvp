# GitHub Backup Notes

This project is safe to back up to a **private** GitHub repository, but it is best to keep the repository lightweight.

## Recommended To Keep In GitHub

- `app.py`
- `pages/`
- `src/`
- `README.md`
- `requirements.txt`
- `NARRATIVE_COPILOT_PROMPTS.md`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml.example`
- `sample_data/`
- `data/simulated/`
- `data/singapore_2021_holidays_attached_format.xlsx`

## Recommended To Exclude From GitHub

These folders are large and are better kept in cloud storage or local backup:

- `data/Inventory/`
- `data/Sales/`
- `Sample/`
- `venv/`
- `output/`
- `outputs/`
- `.streamlit/secrets.toml`

## Why

- Keeps the repo small and faster to clone
- Avoids uploading personal or raw working data unnecessarily
- Reduces the chance of exposing sensitive data
- Makes recovery on a new laptop much easier

## Suggested Backup Strategy

1. Push the lightweight app code to a **private** GitHub repo.
2. Keep large raw data in a separate cloud folder such as Google Drive, OneDrive, Dropbox, or an external drive.
3. Keep your OpenAI key only in `.streamlit/secrets.toml` locally or in a password manager, not in GitHub.
4. If you need a demo dataset in GitHub, keep only `data/simulated/` and `sample_data/`.
