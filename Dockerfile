# Use a slim Python image for reduced size and faster builds
FROM python:3.11-slim
# Set the working directory inside the container
WORKDIR /app
# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application code
COPY app.py .
COPY model.joblib .
# Expose the port the API will run on (FastAPI default 8000)
EXPOSE 8000
# Command to run the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
