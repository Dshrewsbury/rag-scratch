# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13.0
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.

# Install system dependencies
# A lot of these dependecies are for llama-cpp, which currently isnt 
# implemented but will be shortly
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopenblas-dev \
    ninja-build \
    build-essential \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*
    
# Copy only requirements first for better layer caching
COPY app/requirements.txt /app/requirements.txt
#COPY .env .env


# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install pytest pytest-cov

# Create necessary directories with proper permissions
RUN mkdir -p /app/assets/embeddings/recursive /app/assets/embeddings/semantic /app/assets/models /app/assets/chat_history /app/assets/documents \
    && chown -R appuser:appuser /app 

# Copy application code
COPY --chown=appuser:appuser . /app/

# Switch to the non-privileged user to run the application.
USER appuser
# Expose the port that the application listens on.
EXPOSE 8000



# Run the application.
CMD ["uvicorn", "app.api:app", "--host=0.0.0.0", "--port=8000"]
