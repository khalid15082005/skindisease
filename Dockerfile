FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_flask.txt .

# Upgrade pip & install deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements_flask.txt

# Copy project files
COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
