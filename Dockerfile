FROM python:3.11-slim

# Prevent Python from writing pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY models.py .
COPY interpreter.py .
COPY guardrails.py .
COPY optimizer.py .
COPY validator.py .

# Expose the service port
EXPOSE 8000

# Run the application — no secrets baked in
# GEMINI_API_KEY must be provided at runtime via -e or --env-file
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
