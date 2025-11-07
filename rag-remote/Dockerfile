FROM python:3.11-slim

# Set working directory
WORKDIR /allycat

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    netcat-traditional \
    procps \
    wget \
    tree  \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt


# Copy project files
COPY . .
RUN chmod +x ./docker-startup.sh

# Create a non-root user and switch to it
# RUN adduser --disabled-password --gecos '' appuser
# RUN chown -R appuser:appuser /app  
# USER appuser

# Expose the port for webapp
EXPOSE 8080  

# CMD ["python", "app_flask.py"]
# CMD ["./docker-startup.sh"]
ENTRYPOINT ["./docker-startup.sh"]
