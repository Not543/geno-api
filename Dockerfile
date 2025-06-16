FROM python:3.10-slim

WORKDIR /app

# 🛠️ Install required system libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 🗂️ Copy all files
COPY . .

# 📦 Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 🌐 Expose port (optional)
EXPOSE 7860

# 🚀 Run your app
CMD ["python", "app.py"]
