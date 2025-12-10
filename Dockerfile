# 1. BASE IMAGE: Use a lightweight Python base image
FROM python:3.9-slim

# 2. SET ENVIRONMENT: Set a working directory inside the container
WORKDIR /app

# 3. SETUP ENVIRONMENT VARIABLES
# Used by config.py to correctly set PROJECT_ROOT to /app
ENV IN_DOCKER=true

# 4. INSTALL DEPENDENCIES: Copy requirements and install them.
# Create a requirements.txt file with: torch, transformers, pandas, numpy, scikit-learn, flask
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. SETUP PROJECT STRUCTURE: Create necessary directories
RUN mkdir -p /app/src/DistilBERT_base
RUN mkdir -p /app/input


# 6. COPY APPLICATION FILES: Copy all required code and assets
# Copy the entire source package
COPY src/DistilBERT_base /app/src

# Copy the trained model file (MUST be trained first!)
COPY model.bin /app/model.bin 

# Copy necessary data file (only if needed by the API, but good practice)
COPY input/imdb.csv /app/input/imdb.csv

# 7. EXPOSE PORT: Inform Docker that the container listens on this port
EXPOSE 5000

# 8. RUN COMMAND: Define the command to start the application
# We use the correct module execution syntax: python -m package.module
CMD ["python", "-m", "src.DistilBERT_base.api"]