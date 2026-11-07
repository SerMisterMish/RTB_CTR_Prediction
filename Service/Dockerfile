# ---------------------------------------------------------------------------
# Production Multi-Stage / Lightweight Dockerfile for CTR RTB Service
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and pre-trained artifacts
COPY src/ /app/src/
COPY artifacts/ /app/artifacts/

# Switch to non-privileged user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container healthcheck using built-in Python urllib w/o curl/apt-get)
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
