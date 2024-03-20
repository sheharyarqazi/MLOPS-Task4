# Use the official Python image as base
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /Task 4

# Copy requirements.txt file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from your local repository into the container
COPY . .

# Expose port 5000 to allow communication with Flask app
EXPOSE 5000

# Command to run your Flask application
CMD ["python", "app.py"]
