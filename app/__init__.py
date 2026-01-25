"""
Nevus Analyzer Web Application
Flask app for skin lesion classification
"""

from flask import Flask
import os

def create_app():
    """Application factory"""
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
