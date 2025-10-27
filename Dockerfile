# === الأساس ===
FROM python:3.9-slim

# === مجلد العمل ===
WORKDIR /app

# === تثبيت المكتبات النظامية المهمة لـ OpenCV و TensorFlow ===
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# === نسخ ملف المتطلبات ===
COPY requirements.txt .

# === تثبيت بايثون packages ===
RUN pip install --no-cache-dir -r requirements.txt

# === نسخ المشروع بالكامل ===
COPY . .

# === نسخ مجلد الموديل داخل الـ image (مهم جدًا) ===
COPY models/ /app/models/

# === إنشاء المجلدات اللازمة داخل /app (لضمان المسارات الصحيحة) ===
RUN mkdir -p /app/static/uploads /app/static/masks /app/templates /app/models

# === المنفذ ===
EXPOSE 8080

# === أمر التشغيل ===
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
