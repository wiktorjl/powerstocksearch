# Use an official Python runtime as a parent image
FROM python:3.12

# Set environment variables for best practices
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app
# Install system dependencies, including network tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    iputils-ping \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*
 
# Install system dependencies if needed (e.g., for psycopg2).
# psycopg2-binary usually bundles its dependencies, so this might not be needed.
# Uncomment and add packages if build errors occur related to missing libraries.
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code, static files, templates, and SQL files
# Ensure all necessary parts of the application are included.
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY sql/ ./sql/
# config.py is inside src, so it's already copied by `COPY src/ ./src/`

# Make port 5000 available to the world outside this container (the port Flask runs on)
EXPOSE 5000

# Define the command to run the app using Gunicorn (a production WSGI server)
# Bind to 0.0.0.0 to accept connections from any IP.
# Adjust the number of workers based on expected load and server resources.
# The format is 'module:variable' where 'module' is the Python path and 'variable' is the Flask app instance.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "src.web.flaskapp:app"]