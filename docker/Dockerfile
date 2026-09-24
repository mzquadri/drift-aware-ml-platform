# Two stages so the runtime image carries no compiler and no build cache.
FROM python:3.11-slim AS build

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --prefix=/install .

FROM python:3.11-slim AS runtime

# Runs as a non-root user. A model server has no reason to be root, and the
# first thing a reviewer checks in a container is whether it is.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=build /install /usr/local
COPY --chown=app:app src ./src

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8000

USER app
EXPOSE 8000

# Liveness only. Readiness is /ready, and the orchestrator is what should be
# asking that one, because it is the thing that decides on routing.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "mlplatform.service:app", "--host", "0.0.0.0", "--port", "8000"]
