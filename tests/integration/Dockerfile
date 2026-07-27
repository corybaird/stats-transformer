FROM python:3.12-slim

# Install system dependencies and R
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    gfortran \
    libgsl-dev \
    cmake \
    libnlopt-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Install R packages
RUN Rscript -e 'options(warn=2); install.packages(c("vars", "svars", "jsonlite", "lmtest", "sandwich", "AER", "plm", "tsDyn"), repos="https://cloud.r-project.org/")'

WORKDIR /app

# Set environment variable to indicate R is available for tests
ENV R_AVAILABLE=1

CMD ["uv", "run", "python", "-m", "pytest", "tests/integration/", "-v"]
