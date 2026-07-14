# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/python -m pip install .


FROM python:3.13-slim AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    SIGNALATTICE_DATA_DIR=/app/data

RUN groupadd --gid 10001 signalattice \
    && useradd --uid 10001 --gid signalattice --create-home --shell /usr/sbin/nologin signalattice

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=signalattice:signalattice configs ./configs
COPY --chown=signalattice:signalattice scripts ./scripts
COPY --chown=signalattice:signalattice data ./data

RUN mkdir -p /app/reports/figures /app/experiments /app/models \
    && chown -R signalattice:signalattice /app

USER signalattice:signalattice

ENTRYPOINT ["signalattice"]
CMD ["--help"]
