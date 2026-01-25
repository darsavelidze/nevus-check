"""
Model loader and inference utilities
"""

import os
import json
import numpy as np
from tensorflow import keras


class SkinLesionModel:
    """Wrapper for skin lesion classification model"""
    
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    CLASS_DESCRIPTIONS = {
        'akiec': 'Актинический кератоз / Болезнь Боуэна',
        'bcc': 'Базально-клеточная карцинома',
        'bkl': 'Доброкачественный кератоз',
        'df': 'Дерматофиброма',
        'mel': 'Меланома',
        'nv': 'Меланоцитарный невус',
        'vasc': 'Сосудистое поражение'
    }
    
    def __init__(self, model_dir='models'):
        """Initialize and load model"""
        self.model_dir = model_dir
        self.model = None
        self.load_model()
    
    def build_model_architecture(self, input_shape=(50, 50, 3), num_classes=7):
        """Build model architecture from hyperparameters"""
        hparams_path = os.path.join(self.model_dir, 'best_hparams.json')
        
        if not os.path.exists(hparams_path):
            raise FileNotFoundError(f"Hyperparameters file not found: {hparams_path}")
        
        with open(hparams_path, 'r') as f:
            hparams = json.load(f)
        
        filters = hparams['filters']
        dense_units = hparams['dense_units']
        dropout_rate = hparams['dropout']
        
        layers = [keras.layers.Input(shape=input_shape)]
        
        # Convolutional blocks
        for f in filters:
            layers.append(keras.layers.Conv2D(f, (3, 3), activation='relu', padding='same'))
            layers.append(keras.layers.BatchNormalization())
            layers.append(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        layers.append(keras.layers.Flatten())
        
        # Dense layers
        for units in dense_units:
            layers.append(keras.layers.Dense(units, activation='relu'))
            layers.append(keras.layers.BatchNormalization())
            layers.append(keras.layers.Dropout(dropout_rate))
        
        layers.append(keras.layers.Dense(num_classes, activation='softmax'))
        
        model = keras.Sequential(layers)
        return model
    
    def load_model(self):
        """Load trained model weights"""
        print("Loading model...")
        
        try:
            print("Building model architecture from hyperparameters...")
            self.model = self.build_model_architecture()
            
            # Try loading weights from available files
            weight_files = [
                'skin_lesion_cnn_paper_final_weights.h5',
                'skin_lesion_classifier_final_weights.h5',
                'skin_lesion_classifier_best_weights.h5'
            ]
            
            loaded = False
            for weight_file in weight_files:
                weight_path = os.path.join(self.model_dir, weight_file)
                if os.path.exists(weight_path):
                    print(f"Loading weights from {weight_file}...")
                    self.model.load_weights(weight_path)
                    print(f"✓ Model successfully loaded!")
                    loaded = True
                    break
            
            if not loaded:
                raise FileNotFoundError("No weight files found in models directory")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict(self, image_array):
        """Make prediction on preprocessed image array"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        predictions = self.model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        # Format all predictions
        results = []
        for i, class_name in enumerate(self.CLASS_NAMES):
            results.append({
                'class': class_name,
                'description': self.CLASS_DESCRIPTIONS[class_name],
                'probability': float(predictions[0][i]),
                'percentage': f"{float(predictions[0][i]) * 100:.2f}%"
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'predicted_class': predicted_class,
            'predicted_description': self.CLASS_DESCRIPTIONS[predicted_class],
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'all_predictions': results
        }
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None


# Global model instance
_model_instance = None

def get_model():
    """Get singleton model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = SkinLesionModel()
    return _model_instance
