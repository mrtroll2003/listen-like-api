# Start with an official Python base image (choose a specific version)
# Debian-based images like 'slim-bullseye' are common and have 'apt-get'
FROM python:3.12-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install ffmpeg and other system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
# It's often better to combine update, install, and clean in one RUN layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential cmake \
    # Add any other system deps here, like 'build-essential' if needed by pip packages
    # Clean up apt caches to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt ./
# Install Python dependencies
# --no-cache-dir reduces image size slightly
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy the rest of your application code into the container
# Ensure paths match your project structure (api folder, vercel.json if needed inside)
COPY ./api ./api
# Copy vercel.json if needed by the container runtime itself (usually not)
# COPY vercel.json .

# Expose the port FastAPI/Uvicorn will run on (often 8000)
# Note: Vercel often handles port mapping automatically, but it's good practice.
EXPOSE 8000

# Command to run your application using Uvicorn
# Vercel expects the app to listen on 0.0.0.0 and will map the port.
# The entrypoint file is now relative to WORKDIR (/app)
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]