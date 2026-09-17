# Reproducible CPU image for the compact STTL pipeline. Set INSTALL_SURYA=1
# when the optional accuracy-first Hindi route is required.
FROM python:3.11-slim

ARG INSTALL_SURYA=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt requirements.lock requirements-ocr.txt ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.lock \
    && python -m pip install --no-deps --no-build-isolation . \
    && if [ "$INSTALL_SURYA" = "1" ]; then python -m pip install -r requirements-ocr.txt; fi

COPY . .

ENTRYPOINT ["python", "pdf_pipeline.py"]
