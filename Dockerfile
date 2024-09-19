# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir torch==1.8.0 tqdm numpy pandas

# Run preprocess.py when the container launches
CMD ["python", "preprocess.py"]
