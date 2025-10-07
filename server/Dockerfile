# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps + uv (non-interactive)
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential gcc \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && rm -rf /var/lib/apt/lists/*

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# ✅ Check uv version
RUN uv --version

# Copy dependencies
COPY requirements.txt .

# Install dependencies using uv with unsafe-best-match strategy
RUN uv pip install --system --index-strategy unsafe-best-match -r requirements.txt

# Copy app source
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]