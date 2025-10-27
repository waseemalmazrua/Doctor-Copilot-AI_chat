FROM python:3.9-slim

# تعيين مجلد العمل
WORKDIR /app

# تثبيت الاعتماديات النظامية
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات أولاً (لتحسين caching)
COPY requirements.txt .

# تثبيت بايثون packages
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY . .

# إنشاء المجلدات اللازمة
RUN mkdir -p static/uploads static/masks templates model

# تعيين port
EXPOSE 8080

# أمر التشغيل - استخدم هذا
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
