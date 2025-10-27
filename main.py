from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import uuid
import io
import base64
import json

app = FastAPI(title="Dr. Copilot", version="2.0")

# 🔧 إعدادات من ملف .env
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

print("🎯 Using configuration from .env file")
print(f"🔑 API Key: {'✅' if api_key and api_key.startswith('sk-') else '❌'}")

# 📁 إنشاء المجلدات
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/masks", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# 🧠 تعريف دوال القياس المفقودة
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def iou(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    union = tf.keras.backend.sum(y_true, -1) + tf.keras.backend.sum(y_pred, -1) - intersection
    return (intersection + smooth) / (union + smooth)


# 🤖 إعداد OpenAI
openai_client = None
if api_key and api_key.startswith('sk-'):
    try:
        openai_client = OpenAI(api_key=api_key)
        # اختبار الاتصال
        openai_client.models.list()
        print("✅ OpenAI client configured successfully!")
    except Exception as e:
        print(f"❌ OpenAI client failed: {e}")
        openai_client = None
else:
    print("❌ OpenAI not configured properly")

# 🧠 تحميل النموذج
model = None
try:
    model_path = "model/my_model_waseem_finetuned_50.keras"
    if os.path.exists(model_path):
        # تحميل مع دوال القياس المخصصة
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'dice_coef': dice_coef,
                'iou': iou
            },
            compile=False
        )
        print("✅ Model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
    else:
        print(f"❌ Model file not found: {model_path}")
        model_dir = "model"
        if os.path.exists(model_dir):
            available_models = os.listdir(model_dir)
            print(f"📁 Available models: {available_models}")
except Exception as e:
    print(f"❌ Model failed to load: {e}")


# 🆕 🔥 أضف هذه الدالة هنا - إرسال تلقائي للمساعد AI
async def analyze_with_ai_assistant(image_path: str, analysis_results: dict):
    """إرسال الصورة ونتائج التحليل تلقائياً للمساعد AI"""
    try:
        if not openai_client:
            print("❌ OpenAI client not available")
            return "AI assistant not available"

        print(f"🔄 Sending image to AI assistant: {image_path}")

        # ✅ أرسل base64 بدون data URL header
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')

        print("✅ Image converted to base64")

        # بناء رسالة مفصلة
        analysis_context = f"""
        Neural Segmentation Analysis has detected the following:
        - Detection Percentage: {analysis_results['detection_percentage']}% of scan area
        - Status: {analysis_results['status']}
        - Color Indicator: {analysis_results['color']}

        Please provide a comprehensive educational analysis of this medical image:

        1. Describe the anatomical structures visible
        2. Explain any notable features or patterns
        3. Provide educational context about what these findings might represent
        4. Suggest what a medical professional would consider
        5. Emphasize the importance of clinical correlation

        Focus on descriptive, educational insights rather than definitive diagnoses.
        """

        messages = [
            {
                "role": "system",
                "content": """You are Dr. Copilot, a medical imaging education assistant. 
                Provide detailed, descriptive analysis of medical images for educational purposes.
                Focus on anatomical description, feature explanation, and clinical context.

                ✅ WHAT YOU CAN AND SHOULD DO:
                - Describe visible anatomical structures in detail
                - Explain imaging features, textures, and patterns
                - Identify normal vs. abnormal appearances
                - Provide educational insights about medical imaging
                - Suggest what medical professionals look for in such scans
                - Explain the clinical significance of various findings
                - Guide users on appropriate next steps

                🚫 REMINDERS:
                - You are an educational assistant, not a diagnostic tool
                - Always recommend professional medical consultation
                - Focus on descriptive analysis rather than definitive conclusions
                - Use clear, accessible language for medical education"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_context},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

        print("🔄 Calling OpenAI API...")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ AI assistant error: {e}")
        return f"AI analysis failed: {str(e)}"


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "openai_configured": openai_client is not None
    })


@app.get("/api/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model is not None,
        "openai_configured": openai_client is not None,
        "config": {
            "api_key_configured": bool(api_key and api_key.startswith('sk-'))
        }
    })


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """تحليل الصور - مع إرسال تلقائي للمساعد AI"""
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse({"success": False, "error": "يرجى رفع ملف صورة"})

        content = await file.read()
        if len(content) == 0:
            return JSONResponse({"success": False, "error": "الملف فارغ"})

        # معالجة الصورة
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # حفظ الصورة الأصلية
        unique_id = str(uuid.uuid4())[:8]
        original_filename = f"original_{unique_id}.png"
        original_path = f"static/uploads/{original_filename}"
        image.save(original_path)

        # 🔥 متغيرات التحليل
        detection_percent = 0
        status = "Not Analyzed"
        color = "#007bff"
        overlay_image = image
        mask_resized = Image.new("L", image.size, 0)

        if model:
            try:
                # معالجة الصورة للنموذج (256x256)
                processed = image.resize((256, 256))
                img_array = np.array(processed) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # التنبؤ
                prediction = model.predict(img_array, verbose=0)

                # استخراج الماسك من التنبؤ
                if len(prediction.shape) == 4:
                    pred_mask = prediction[0, ..., 0]
                else:
                    pred_mask = prediction[0]

                print(f"📊 Prediction range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
                print(f"📈 Prediction mean: {pred_mask.mean():.3f}")

                # العتبة الذكية
                threshold = 0.3
                binary_mask = (pred_mask > threshold).astype(np.uint8)

                # حساب نسبة الاكتشاف
                mask_area = np.sum(binary_mask)
                total_area = binary_mask.size
                detection_percent = (mask_area / total_area) * 100

                print(f"🎯 Using threshold: {threshold}")
                print(f"📦 Mask pixels: {mask_area}/{total_area} ({detection_percent:.2f}%)")

                # 🔥 إصلاح الـ overlay - بدون cv2
                # إنشاء القناع الملون
                # 🔥 إصلاح الـ overlay (نسخة تعمل على Cloud Run و localhost)
                mask_resized = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(image.size)
                binary_mask_resized = np.array(mask_resized) > 0

                # تحويل الصورة الأصلية إلى مصفوفة NumPy
                original_array = np.array(image.convert("RGB"))

                # تحويل الألوان إلى BGR لتجنب مشكلة الألوان في بيئات headless
                original_bgr = original_array[..., ::-1]

                # إنشاء ماسك ملون باللون الأحمر
                red_mask = np.zeros_like(original_bgr)
                red_mask[binary_mask_resized] = [0, 0, 255]  # أحمر (BGR)

                # دمج الماسك مع الصورة الأصلية
                alpha = 0.6
                overlay_bgr = (original_bgr * (1 - alpha) + red_mask * alpha).astype(np.uint8)

                # إعادة التحويل إلى RGB للعرض الصحيح في المتصفح
                overlay_rgb = overlay_bgr[..., ::-1]
                overlay_image = Image.fromarray(overlay_rgb)

                # تحديد حالة الاكتشاف
                if detection_percent < 2.0:
                    status = "Normal"
                    color = "#00ff00"
                elif detection_percent < 10.0:
                    status = "Minor Findings"
                    color = "#ffff00"
                else:
                    status = "Significant Findings"
                    color = "#ff0000"

            except Exception as model_error:
                print(f"❌ Model prediction error: {model_error}")
                detection_percent = 0
                status = "Analysis Failed"
                color = "#ff0000"

        # 🔥 🆕 الجديد: إرسال تلقائي للمساعد AI
        ai_analysis = ""
        if openai_client:
            analysis_results = {
                "detection_percentage": detection_percent,
                "status": status,
                "color": color
            }
            try:
                print("🚀 Starting AI assistant analysis...")
                ai_analysis = await analyze_with_ai_assistant(original_path, analysis_results)
                print("🎉 AI analysis completed successfully!")
            except Exception as ai_error:
                print(f"❌ AI analysis failed: {ai_error}")
                ai_analysis = f"AI analysis unavailable: {str(ai_error)}"
        else:
            ai_analysis = "🔶 AI assistant not available - using neural segmentation only"

         # ✅ حفظ مؤقت في /tmp
        overlay_path = f"/tmp/overlay_{unique_id}.png"
        overlay_image.save(overlay_path)

        # ✅ تحويل الصورة الناتجة إلى Base64 مباشرة
        with open(overlay_path, "rb") as img_file:
            encoded_overlay = base64.b64encode(img_file.read()).decode("utf-8")

        # ✅ نفس الشيء لو تبي الأصلية (اختياري)
        with open(original_path, "rb") as img_file:
            encoded_original = base64.b64encode(img_file.read()).decode("utf-8")

        # ✅ إرجاع النتيجة كـ JSON يحتوي صور Base64
        return JSONResponse({
            "success": True,
            "analysis": {
                "detection_percentage": round(detection_percent, 2),
                "status": status,
                "color": color
            },
            "ai_response": ai_analysis,
            "original_image_base64": encoded_original,
            "overlay_image_base64": encoded_overlay,
            "model_used": "Neural Segmentation + AI Assistant" if model else "Demo"
        })

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/chat")
async def chat_with_ai(request: dict):
    """Chat endpoint for AI conversation with image support"""
    try:
        message = request.get("message", "")
        image_url = request.get("image_url", "")  # 🔥 هذا يأتي كـ base64 كامل من Frontend

        if not message and not image_url:
            return JSONResponse({"success": False, "error": "Message or image is required"})

        # If OpenAI is not configured, use demo response
        if not openai_client:
            return JSONResponse({
                "success": True,
                "response": "Hello! I'm your medical AI assistant. Currently in demo mode.",
                "demo_mode": True
            })

        # 🔥 استخدام GPT-4 Vision أو GPT-4o لمشاهدة الصور
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are Dr. Copilot, an advanced medical AI imaging assistant. Your role is to provide detailed, educational analysis of medical images.

✅ WHAT YOU CAN AND SHOULD DO:
- Describe visible anatomical structures in detail
- Explain imaging features, textures, and patterns
- Identify normal vs. abnormal appearances
- Provide educational insights about medical imaging
- Suggest what medical professionals look for in such scans
- Explain the clinical significance of various findings
- Guide users on appropriate next steps

📋 SPECIFIC ANALYSIS FRAMEWORK:
1. **Image Description**: Describe what you see - anatomy, structures, regions
2. **Feature Analysis**: Note densities, intensities, textures, contrasts  
3. **Pattern Recognition**: Identify any notable patterns or symmetries
4. **Educational Context**: Explain what these features mean in medical terms
5. **Professional Guidance**: Emphasize consultation with healthcare providers

🚫 REMINDERS:
- You are an educational assistant, not a diagnostic tool
- Always recommend professional medical consultation
- Focus on descriptive analysis rather than definitive conclusions
- Use clear, accessible language for medical education

Be thorough, descriptive, and educational in your analysis."""
                }
            ]

            # 🔥 🔥 🔥 معالجة الصورة بشكل صحيح
            if image_url:
                # ✅ الإصلاح: إذا image_url هو base64 كامل (مثل: data:image/png;base64,xxxx)
                # لا تضيف header مرة أخرى - أرسله كما هو
                final_image_url = image_url

                message_content = [
                    {"type": "text",
                     "text": message or "Please provide a detailed educational analysis of this medical image, describing what you observe and explaining the features in medical context."},
                    {"type": "image_url", "image_url": {"url": final_image_url}}
                ]
            else:
                # إذا مافي صورة، استخدم prompt مختلف
                enhanced_message = message
                if "image" in message.lower() or "scan" in message.lower() or "mri" in message.lower() or "x-ray" in message.lower():
                    enhanced_message = f"{message} - Please describe what medical professionals typically look for in such images and what various findings might indicate."

                message_content = enhanced_message

            messages.append({
                "role": "user",
                "content": message_content
            })

            print(f"📤 Sending to OpenAI - Message: {message}, Has Image: {bool(image_url)}")

            response = openai_client.chat.completions.create(
                model="gpt-4o",  # ✅ تأكد أنه gpt-4o
                messages=messages,
                max_tokens=1500,  # زود الـ tokens عشان ردود أطول
                temperature=0.7
            )

            response_text = response.choices[0].message.content
            print("✅ OpenAI response received successfully")

            return JSONResponse({
                "success": True,
                "response": response_text,
                "demo_mode": False
            })

        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            # 🔥 Fallback بسيط بدون محاولة إرسال الصورة مرة أخرى
            try:
                # إذا فشل بسبب الصورة، حاول بدونها
                fallback_message = message or "Please analyze this medical image"

                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are Dr. Copilot, a medical AI assistant. Provide helpful medical information and educational insights."
                        },
                        {
                            "role": "user",
                            "content": f"{fallback_message} (Note: Detailed image analysis is currently unavailable - please describe what you see for general medical guidance)"
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
                return JSONResponse({
                    "success": True,
                    "response": response_text,
                    "demo_mode": False,
                    "fallback": True
                })
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                return JSONResponse({
                    "success": True,
                    "response": "I'm here to help with medical questions. Please describe the image or symptoms in detail for educational guidance. Always consult healthcare professionals for medical concerns.",
                    "demo_mode": True
                })

    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🩺 Dr. Copilot - نظام متكامل")
    print("=" * 60)
    print(f"📊 النموذج: {'✅ محمل' if model else '❌ غير متاح'}")
    print(f"🤖 OpenAI API: {'✅ مفعل' if openai_client else '❌ معطل'}")
    print(f"🔑 API Key: {'✅ مضبوط' if api_key and api_key.startswith('sk-') else '❌ مفقود'}")
    print("=" * 60)
    print("🌐 الخادم يعمل على: http://localhost:8001")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
