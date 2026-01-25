"""
Flask routes for the web application
"""

from flask import Blueprint, render_template, request, jsonify, current_app
from app.model_loader import get_model
from app.utils import allowed_file, preprocess_image, save_uploaded_file

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Get model instance
        model = get_model()
        
        if not model.is_loaded():
            return jsonify({'error': 'Модель не загружена'}), 500
        
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Недопустимый формат файла. Используйте JPG или PNG'}), 400
        
        # Save file
        filepath = save_uploaded_file(file, current_app.config['UPLOAD_FOLDER'])
        
        # Preprocess image
        img_array = preprocess_image(filepath)
        
        # Make prediction
        result = model.predict(img_array)
        result['success'] = True
        
        # Convert absolute path to relative URL path for web display
        relative_path = filepath.replace('\\', '/').replace(current_app.static_folder.replace('\\', '/'), '/static')
        result['image_path'] = relative_path
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Ошибка при обработке: {str(e)}'}), 500

@main_bp.route('/health')
def health():
    """Health check endpoint"""
    try:
        model = get_model()
        return jsonify({
            'status': 'ok',
            'model_loaded': model.is_loaded()
        })
    except:
        return jsonify({
            'status': 'error',
            'model_loaded': False
        }), 500
