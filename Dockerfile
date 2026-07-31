FROM python:3.10-slim

# Install system dependencies required by OpenCV (which ultralytics needs)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (Hugging Face Spaces strict requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy requirements first to leverage Docker caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose the standard HF Spaces port
EXPOSE 7860

# Run the Flask app
CMD ["python", "webapp/app.py"]
