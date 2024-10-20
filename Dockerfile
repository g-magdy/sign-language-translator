# Use an official Python runtime as a parent image
FROM python:3.12.6

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies for OpenCV and Git LFS in one command
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install git-lfs and pull large files
RUN git lfs install

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
