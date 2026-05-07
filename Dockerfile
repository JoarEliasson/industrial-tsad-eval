FROM python:3.12-slim

ARG INSTALL_EXTRAS=dev,datasets,acquisition,profile,torch

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY docs ./docs
COPY examples ./examples

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[${INSTALL_EXTRAS}]"

CMD ["itse", "--help"]
