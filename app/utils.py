"""
Utility functions for image processing and validation
"""

import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(75, 75)):
    """
    Preprocess image for model inference
    
    Args:
        image_path: Path to image file
        target_size: Target size tuple (height, width)
    
    Returns:
        Preprocessed image array ready for model input
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def save_uploaded_file(file, upload_folder):
    """
    Safely save uploaded file
    
    Args:
        file: FileStorage object from Flask
        upload_folder: Directory to save file
    
    Returns:
        Path to saved file
    """
    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filepath
