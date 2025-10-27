# 🏥 Doctor Copilot - Medical AI Assistant

A comprehensive medical AI assistant that combines MRI image analysis with intelligent chat capabilities using OpenAI's GPT models and workflows.

## ✨ Features

- **MRI Image Analysis**: Upload and analyze medical images using a fine-tuned TensorFlow model
- **AI Chat Assistant**: Intelligent medical consultation powered by OpenAI GPT
- **Workflow Integration**: Support for OpenAI Workflows for advanced AI processing
- **Risk Assessment**: Automatic risk level classification (low, medium, high)
- **Bilingual Support**: Arabic and English interface
- **Real-time Analysis**: Fast image processing with visual results

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script
python setup.py
```

### 2. Configure API Keys

Edit the `.env` file and add your credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_WORKFLOW_ID=your_workflow_id_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

The application will be available at: `http://localhost:8000`

## 📋 Requirements

- Python 3.8+
- OpenAI API Key
- TensorFlow model file (`model/my_model_waseem_finetuned_50.keras`)

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_WORKFLOW_ID` | OpenAI Workflow ID (optional) | None |
| `MODEL_PATH` | Path to TensorFlow model | `model/my_model_waseem_finetuned_50.keras` |
| `DEBUG` | Enable debug mode | `True` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MAX_FILE_SIZE` | Maximum file size in bytes | `10485760` (10MB) |
| `MAX_TOKENS` | Maximum tokens for chat | `500` |
| `TEMPERATURE` | Chat temperature | `0.7` |

## 🏗️ Project Structure

```
Doctor-Copilot-2025/
├── main.py                 # FastAPI application
├── setup.py               # Setup script
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── model/
│   └── my_model_waseem_finetuned_50.keras
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── uploads/           # Uploaded images
│   ├── masks/            # Generated masks
│   └── js/
│       └── app.js        # Frontend JavaScript
└── README.md
```

## 🔌 API Endpoints

### Health Check
```
GET /api/health
```
Returns system status and configuration.

### Image Analysis
```
POST /api/analyze
Content-Type: multipart/form-data
```
Upload an image for MRI analysis.

**Response:**
```json
{
  "success": true,
  "analysis": {
    "lesion_percentage": 12.5,
    "severity": "يحتاج مراجعة",
    "risk_level": "medium",
    "recommendation": "مراجعة طبية خلال أسبوع"
  },
  "original_image": "/static/uploads/original_abc123.png",
  "mask_image": "/static/masks/mask_abc123.png"
}
```

### Chat
```
POST /api/chat
Content-Type: application/json
```
Send a message to the AI assistant.

**Request:**
```json
{
  "message": "What are the symptoms of diabetes?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Diabetes symptoms include...",
  "workflow_used": false
}
```

## 🤖 OpenAI Workflow Integration

The application supports OpenAI Workflows for advanced AI processing:

1. Set your `OPENAI_WORKFLOW_ID` in the `.env` file
2. The chat endpoint will automatically use your workflow
3. Falls back to regular GPT-3.5-turbo if workflow fails

## 🎨 Frontend Features

- **Drag & Drop**: Upload images by dragging and dropping
- **Real-time Analysis**: Live progress indicators
- **Risk Visualization**: Color-coded risk levels
- **Responsive Design**: Works on desktop and mobile
- **Arabic RTL Support**: Full right-to-left language support

## 🔍 Troubleshooting

### Common Issues

1. **Model not loading**
   - Check if the model file exists at the specified path
   - Verify TensorFlow compatibility

2. **OpenAI API errors**
   - Verify your API key is correct
   - Check your OpenAI account credits

3. **File upload issues**
   - Ensure file size is under 10MB
   - Check file format (PNG, JPG, JPEG)

### Health Check

Visit `/api/health` to check system status:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "openai_available": true,
  "workflow_configured": true,
  "model_path": "model/my_model_waseem_finetuned_50.keras",
  "api_key_configured": true
}
```

## 📝 License

This project is for educational and research purposes. Please ensure compliance with medical regulations and OpenAI's usage policies.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation
- Ensure all dependencies are installed correctly

---

**⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. Always consult with qualified medical professionals for actual medical diagnosis and treatment.
