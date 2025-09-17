# Base image with Python
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your whole project into the container
COPY . .

# Default command: run your main script
CMD ["python", "mini_project_1.py"]
