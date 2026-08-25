FROM python:3.10-slim

# libgl1/libglib2.0-0: required by opencv-python-headless at import time
# even though it's the "headless" build (used for OCR image preprocessing).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "main.py"]
