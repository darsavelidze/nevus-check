// Main JavaScript for Nevus Analyzer

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const results = document.getElementById('results');

let selectedFile = null;

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File selection
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
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
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showError('Пожалуйста, выберите изображение');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        results.style.display = 'none';
        error.style.display = 'none';
    };

    reader.readAsDataURL(file);
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    loading.style.display = 'block';
    results.style.display = 'none';
    error.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Ошибка при анализе');
        }
    } catch (err) {
        showError('Ошибка подключения к серверу');
        console.error(err);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    document.getElementById('resultClass').textContent = data.predicted_class.toUpperCase();
    document.getElementById('resultDescription').textContent = data.predicted_description;
    document.getElementById('resultConfidence').textContent = 
        `Уверенность: ${data.confidence_percentage}`;

    const allPredictionsDiv = document.getElementById('allPredictions');
    allPredictionsDiv.innerHTML = '';

    data.all_predictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-info">
                <div class="prediction-name">${pred.class.toUpperCase()}</div>
                <div class="prediction-desc">${pred.description}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${pred.probability * 100}%"></div>
                </div>
            </div>
            <div class="prediction-probability">${pred.percentage}</div>
        `;
        allPredictionsDiv.appendChild(item);
    });

    results.style.display = 'block';
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
}
