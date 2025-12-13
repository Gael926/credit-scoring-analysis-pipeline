FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Directory for the model
WORKDIR /app

# Note: The model should be mounted or copied. 
# Here we assume the user mounts the project or copies the model after training.
# Usage: docker run -v $(pwd)/models/final_model:/model -p 5000:5000 credit-scoring-serving
# Or if you want to bake it in:
# COPY models/final_model /app/model

# We'll use a generic entrypoint or CMD that expects /model mount
ENV MLFLOW_Model_URI=/model

# Default command: Serve the model mounted at /model
CMD ["mlflow", "models", "serve", "-m", "/model", "-p", "5000", "-h", "0.0.0.0", "--no-conda"]
