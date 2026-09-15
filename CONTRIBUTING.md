# Contributing to STTL

Thanks for helping improve the pipeline. Small, focused pull requests are the
easiest to review and validate.

## Local setup

Use Python 3.11 for the same environment used by continuous integration:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest -v test_pdf_pipeline.py
```

Install Tesseract and the optional Surya runtime only when exercising OCR
fallbacks locally. The unit suite uses synthetic PDFs and does not require a
private sample document, a model download, or service credentials.

## Pull requests

- Explain the user-visible change and its trade-offs.
- Add or update focused tests for changed routing, parsing, or evaluation logic.
- Run the unit suite before opening the pull request.
- Keep generated outputs out of the diff.

## Data handling

Do not commit source PDFs, screenshots, OCR output, raw provider responses,
benchmark logs, ground truth derived from restricted documents, credentials, or
machine-specific paths. If a fixture is necessary, make it synthetic and keep
it minimal. Any proposed public sample must have documented redistribution
rights and privacy review before it is added.

## Reporting issues

Please provide a sanitized, minimal reproduction. Do not paste document
contents, API keys, signed URLs, or personal data into public issues. See
[SECURITY.md](SECURITY.md) for security-sensitive reports.
