// Application State
let currentFile = null;
let chatImageFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Starting application initialization...');
    initializeApp();
    initializeChatSystem();
});

function initializeApp() {
    setupEventListeners();
    updateSystemStatus();
}

function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const sendButton = document.getElementById('sendMessage');
    const chatInput = document.getElementById('chatInput');

    // File Input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Upload Area Interactions
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // Action Buttons
    analyzeBtn.addEventListener('click', analyzeImage);
    clearBtn.addEventListener('click', clearSelection);

    // Chat Events
    sendButton.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });
}

function handleFileSelection(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showNotification('File size must be less than 10MB', 'error');
        return;
    }

    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('uploadArea').innerHTML = `
            <div class="image-preview">
                <img src="${e.target.result}" class="preview-image" alt="Preview">
                <p style="margin-top: 15px; color: white;">${file.name}</p>
            </div>
        `;

        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('clearBtn').style.display = 'inline-flex';
    };
    reader.readAsDataURL(file);
}

function clearSelection() {
    currentFile = null;
    document.getElementById('uploadArea').innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
        </div>
        <div class="upload-text">
            <h3>Upload MRI Scan</h3>
            <p>Drag & drop or click to browse</p>
            <p>Supported: PNG, JPG, JPEG | Max: 10MB</p>
        </div>
    `;
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('clearBtn').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'none';
}

async function analyzeImage() {
    if (!currentFile) {
        alert('Please select an image first');
        return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('resultsContainer');

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    loading.style.display = 'block';
    resultsContainer.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log('📦 API Response:', result); // 🔥 للت debugging

        if (result.success) {
            document.getElementById('originalImage').src = `data:image/png;base64,${result.original_image_base64}`;
            document.getElementById('overlayImage').src = `data:image/png;base64,${result.overlay_image_base64}`;
            resultsContainer.style.display = 'block';
            showNotification('Analysis completed successfully', 'success');

            // 🔥 🆕 الجديد: عرض رد المساعد AI تلقائياً في الشات
            if (result.ai_response && result.ai_response.trim() !== '') {
                console.log('🤖 AI Response received:', result.ai_response);
                addMessageToChat('bot', result.ai_response);
            } else {
                console.log('❌ No AI response found');
                addMessageToChat('bot', 'Neural analysis completed. No AI insights available.');
            }

        } else {
            throw new Error(result.error);
        }

    } catch (error) {
        console.error('Analysis error:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
    }
}
// 🔥 NEW Chat System with Image Upload
function initializeChatSystem() {
    console.log('🔧 Initializing chat system...');
    setupChatUpload(); // إضافة رفع الصور للشات
    document.getElementById('chatInput').focus();
}

// 🔥 NEW: Chat Image Upload Functionality
function setupChatUpload() {
    const chatImageUpload = document.getElementById('chatImageUpload');
    const chatImagePreview = document.getElementById('chatImagePreview');

    chatImageUpload.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            handleChatImageUpload(file);
        }
    });
}

function handleChatImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file', 'error');
        return;
    }

    chatImageFile = file;

    const reader = new FileReader();
    reader.onload = function(e) {
        const chatImagePreview = document.getElementById('chatImagePreview');
        chatImagePreview.innerHTML = `
            <div class="chat-image-preview">
                <img src="${e.target.result}" style="width: 50px; height: 50px; border-radius: 5px; object-fit: cover;">
                <span style="flex: 1; font-size: 0.9rem; color: white;">${file.name}</span>
                <button onclick="clearChatImage()" style="background: #ff4444; border: none; border-radius: 50%; width: 25px; height: 25px; color: white; cursor: pointer;">×</button>
            </div>
        `;
        chatImagePreview.style.display = 'block';

        // 🔥 إرسال الصورة مباشرة للـ AI مع رسالة
        sendImageToAI(file, "Analyze this medical image and describe what you see");
    };
    reader.readAsDataURL(file);
}

function clearChatImage() {
    document.getElementById('chatImageUpload').value = '';
    document.getElementById('chatImagePreview').style.display = 'none';
    document.getElementById('chatImagePreview').innerHTML = '';
    chatImageFile = null;
}

// 🔥 NEW: Send image to AI for analysis
async function sendImageToAI(file, message = "Analyze this medical image") {
    const chatMessages = document.getElementById('chatMessages');

    // Add user message with image
    const reader = new FileReader();
    reader.onload = function(e) {
        addMessageToChat('user', message, e.target.result);
    };
    reader.readAsDataURL(file);

    // Show typing indicator
    const typingIndicator = addTypingIndicator();

    try {
        // Convert image to Base64
        const base64Image = await fileToBase64(file);

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                image_url: base64Image  // 🔥 إرسال الصورة كـ Base64
            })
        });

        const result = await response.json();

        // Remove typing indicator
        typingIndicator.remove();

        if (result.success) {
            addMessageToChat('bot', result.response);
        } else {
            addMessageToChat('bot', 'Sorry, I could not analyze the image. Please try again.');
        }

    } catch (error) {
        typingIndicator.remove();
        addMessageToChat('bot', 'Error analyzing image. Please check your connection.');
    }
}

// 🔥 Convert file to Base64
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

async function sendChatMessage() {
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    const message = chatInput.value.trim();

    // إذا فيه صورة في الشات، أرسلها للتحليل
    if (chatImageFile) {
        await sendImageToAI(chatImageFile, message || "Analyze this medical image");
        clearChatImage();
        chatInput.value = '';
        return;
    }

    if (!message) return;

    // Add user message to chat
    addMessageToChat('user', message);
    chatInput.value = '';

    // Show typing indicator
    const typingIndicator = addTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        const result = await response.json();

        // Remove typing indicator
        typingIndicator.remove();

        if (result.success) {
            addMessageToChat('bot', result.response);
        } else {
            addMessageToChat('bot', 'Sorry, I encountered an error. Please try again.');
        }

    } catch (error) {
        typingIndicator.remove();
        addMessageToChat('bot', 'Connection error. Please check your internet connection.');
    }

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// تحديث دالة addMessageToChat لدعم الصور
// تحديث دالة addMessageToChat لدعم الصور
function addMessageToChat(sender, message, imageUrl = null, imageCaption = '') {
    const chatMessages = document.getElementById('chatMessages');

    // 🔥 تحقق إذا العنصر موجود
    if (!chatMessages) {
        console.error('❌ chatMessages element not found!');
        return;
    }

    const messageClass = sender === 'user' ? 'user-message' : 'bot-message';
    const senderName = sender === 'user' ? 'You' : 'Dr. Copilot';

    const messageElement = document.createElement('div');
    messageElement.className = `message ${messageClass}`;

    let contentHTML = `<strong>${senderName}:</strong>`;
    if (message) contentHTML += ` ${message.replace(/\n/g, '<br>')}`;
    if (imageUrl) {
        contentHTML += `<br><img src="${imageUrl}" class="chat-image" alt="${imageCaption}">`;
        if (imageCaption) contentHTML += `<div style="font-size: 0.8rem; opacity: 0.7; margin-top: 5px;">${imageCaption}</div>`;
    }

    messageElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas ${sender === 'user' ? 'fa-user' : 'fa-robot'}"></i>
        </div>
        <div class="message-content">
            ${contentHTML}
        </div>
    `;

    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    console.log(`✅ Message added from ${senderName}:`, message.substring(0, 50) + '...');
}

function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingElement = document.createElement('div');
    typingElement.className = 'message bot-message typing-indicator';
    typingElement.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <strong>Dr. Copilot:</strong>
            <span class="typing-dots">
                <span>.</span><span>.</span><span>.</span>
            </span>
        </div>
    `;

    chatMessages.appendChild(typingElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return typingElement;
}

// System Status
async function updateSystemStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        const statusText = document.getElementById('statusText');

        // 🔥 اضف هذا التحقق قبل استخدام statusText
        if (!statusText) {
            console.log('statusText element not found yet');
            return;
        }

        if (data.status === 'healthy') {
            let statusMessage = 'System Online';
            if (data.model_loaded) statusMessage += ' • AI Model Ready';
            if (data.openai_configured) statusMessage += ' • Assistant Ready';
            statusText.textContent = statusMessage;
        } else {
            statusText.textContent = 'System Limited';
        }
    } catch (error) {
        const statusText = document.getElementById('statusText');
        // 🔥 تحقق هنا أيضاً
        if (statusText) {
            statusText.textContent = 'Connection Error';
        }
    }
}

// Notification System
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? '#ff4444' : '#00ff00'};
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        font-weight: 600;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 4000);
}

// Add CSS animations if not already in HTML
if (!document.querySelector('#dynamic-styles')) {
    const style = document.createElement('style');
    style.id = 'dynamic-styles';
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        .typing-dots span {
            animation: typing 1.4s infinite;
        }

        @keyframes typing {
            0%, 60%, 100% { opacity: 0.3; }
            30% { opacity: 1; }
        }
    `;
    document.head.appendChild(style);
}
