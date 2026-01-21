FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# libgomp1 is needed for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set PYTHONPATH to include src if needed
ENV PYTHONPATH=/app/src

# Expose the port for MLflow serving
EXPOSE 5001

# Default command: Serve the model
# We assume the model is at models/final_model relative to the build context
CMD ["mlflow", "models", "serve", "-m", "models/final_model", "-p", "5000", "-h", "0.0.0.0", "--no-conda"]
