FROM python:3.12-slim

LABEL maintainer="priya"
LABEL description="Ideological Dynamics Simulator"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY ideasim/ ideasim/
COPY tests/ tests/

# Install the package with dev dependencies (includes pytest for test runs)
RUN pip install --no-cache-dir ".[dev]"

# Default output directory for results
RUN mkdir -p /app/output

ENTRYPOINT ["ideasim"]
CMD ["simulate", "-o", "/app/output/results.json"]
