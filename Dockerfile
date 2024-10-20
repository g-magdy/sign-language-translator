# Use an official Python runtime as a parent image
FROM python:3.12.6

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for Git, Git LFS, and OpenCV
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1-mesa-glx \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Clone your GitHub repository inside the container
RUN git clone https://github.com/yourusername/your-repo.git /app

# Change directory to /app
WORKDIR /app

# Pull Git LFS files
RUN git lfs pull

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
