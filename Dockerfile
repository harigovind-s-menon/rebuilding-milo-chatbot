# Stage 1: build with poetry, compile dependencies
FROM python:3.13-slim AS build

# Install system deps required for building wheels and general use
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libpq-dev gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (bundle installer)
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.2.1

# Set workdir
WORKDIR /app

# Copy only dependency files first for caching
COPY pyproject.toml poetry.lock* /app/

# Configure poetry to not create virtualenvs inside container
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --no-dev

# Copy source
COPY . /app

# Stage 2: small runtime image
FROM python:3.13-slim

# Create non-root user (optional)
RUN useradd --create-home appuser
WORKDIR /app

# Copy virtualenv / site-packages from build stage
COPY --from=build /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy app files
COPY --chown=appuser:appuser . /app

# Expose port and switch to non-root user
EXPOSE 8000
USER appuser

# Set simple entrypoint (use uvicorn by default)
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]