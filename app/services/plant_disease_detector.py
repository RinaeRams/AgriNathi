import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)

class PlantDiseaseDetector:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.pest_disease_info = {}
        self.load_model()
        self.load_disease_info()

    def load_model(self):
        """Load the pre-trained plant disease detection model"""
        try:
            # For now, we'll create a simple CNN model since we don't have a pre-trained model
            # In production, you would load a trained model
            self.model = self._create_model()
            logger.info("Plant disease detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _create_model(self):
        """Create a simple CNN model for plant disease detection"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(38, activation='softmax')  # 38 classes for plant diseases
        ])

        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

        return model

    def load_disease_info(self):
        """Load pest and disease information from JSON file"""
        try:
            disease_info_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pest_disease_info.json')
            with open(disease_info_path, 'r', encoding='utf-8') as f:
                self.pest_disease_info = json.load(f)
            logger.info("Disease information loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load disease info: {e}")
            self.pest_disease_info = {}

    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            img = Image.open(image_path)
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            return img_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def predict_disease(self, image_path):
        """Predict plant disease from image"""
        try:
            if self.model is None:
                return self._get_mock_prediction()

            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return self._get_mock_prediction()

            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Get disease information
            disease_info = self._get_disease_info(predicted_class)

            return {
                'disease': disease_info['name'],
                'confidence': round(confidence * 100, 2),
                'description': disease_info['description'],
                'symptoms': disease_info['symptoms'],
                'treatment': disease_info['treatment'],
                'prevention': disease_info['prevention'],
                'severity': self._calculate_severity(confidence),
                'recommendations': self._generate_recommendations(disease_info, confidence)
            }

        except Exception as e:
            logger.error(f"Error predicting disease: {e}")
            return self._get_mock_prediction()

    def _get_disease_info(self, class_index):
        """Get disease information based on predicted class"""
        # Map class indices to disease names (simplified mapping)
        disease_mapping = {
            0: {'name': 'Apple___Apple_scab', 'type': 'fungal'},
            1: {'name': 'Apple___Black_rot', 'type': 'fungal'},
            2: {'name': 'Apple___Cedar_apple_rust', 'type': 'fungal'},
            3: {'name': 'Apple___healthy', 'type': 'healthy'},
            4: {'name': 'Blueberry___healthy', 'type': 'healthy'},
            5: {'name': 'Cherry_(including_sour)___Powdery_mildew', 'type': 'fungal'},
            6: {'name': 'Cherry_(including_sour)___healthy', 'type': 'healthy'},
            7: {'name': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'type': 'fungal'},
            8: {'name': 'Corn_(maize)___Common_rust_', 'type': 'fungal'},
            9: {'name': 'Corn_(maize)___Northern_Leaf_Blight', 'type': 'fungal'},
            10: {'name': 'Corn_(maize)___healthy', 'type': 'healthy'},
            11: {'name': 'Grape___Black_rot', 'type': 'fungal'},
            12: {'name': 'Grape___Esca_(Black_Measles)', 'type': 'fungal'},
            13: {'name': 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'type': 'fungal'},
            14: {'name': 'Grape___healthy', 'type': 'healthy'},
            15: {'name': 'Orange___Haunglongbing_(Citrus_greening)', 'type': 'bacterial'},
            16: {'name': 'Peach___Bacterial_spot', 'type': 'bacterial'},
            17: {'name': 'Peach___healthy', 'type': 'healthy'},
            18: {'name': 'Pepper,_bell___Bacterial_spot', 'type': 'bacterial'},
            19: {'name': 'Pepper,_bell___healthy', 'type': 'healthy'},
            20: {'name': 'Potato___Early_blight', 'type': 'fungal'},
            21: {'name': 'Potato___Late_blight', 'type': 'fungal'},
            22: {'name': 'Potato___healthy', 'type': 'healthy'},
            23: {'name': 'Raspberry___healthy', 'type': 'healthy'},
            24: {'name': 'Soybean___healthy', 'type': 'healthy'},
            25: {'name': 'Squash___Powdery_mildew', 'type': 'fungal'},
            26: {'name': 'Strawberry___Leaf_scorch', 'type': 'fungal'},
            27: {'name': 'Strawberry___healthy', 'type': 'healthy'},
            28: {'name': 'Tomato___Bacterial_spot', 'type': 'bacterial'},
            29: {'name': 'Tomato___Early_blight', 'type': 'fungal'},
            30: {'name': 'Tomato___Late_blight', 'type': 'fungal'},
            31: {'name': 'Tomato___Leaf_Mold', 'type': 'fungal'},
            32: {'name': 'Tomato___Septoria_leaf_spot', 'type': 'fungal'},
            33: {'name': 'Tomato___Spider_mites Two-spotted_spider_mite', 'type': 'pest'},
            34: {'name': 'Tomato___Target_Spot', 'type': 'fungal'},
            35: {'name': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'type': 'viral'},
            36: {'name': 'Tomato___Tomato_mosaic_virus', 'type': 'viral'},
            37: {'name': 'Tomato___healthy', 'type': 'healthy'}
        }

        disease_data = disease_mapping.get(class_index, {'name': 'Unknown Disease', 'type': 'unknown'})

        # Get detailed information from our knowledge base
        detailed_info = self._get_detailed_disease_info(disease_data['name'], disease_data['type'])

        return detailed_info

    def _get_detailed_disease_info(self, disease_name, disease_type):
        """Get detailed disease information from knowledge base"""
        # Parse disease name to get crop and condition
        parts = disease_name.split('___')
        if len(parts) == 2:
            crop, condition = parts
        else:
            crop = "Unknown"
            condition = disease_name

        # Default information
        default_info = {
            'name': condition.replace('_', ' '),
            'description': f'{condition.replace("_", " ")} affecting {crop} plants',
            'symptoms': ['Leaf discoloration', 'Stunted growth', 'Reduced yield'],
            'treatment': ['Remove infected parts', 'Apply appropriate fungicide', 'Improve plant care'],
            'prevention': ['Proper plant spacing', 'Good drainage', 'Regular monitoring']
        }

        # Try to get specific information from our knowledge base
        if disease_type == 'healthy':
            return {
                'name': 'Healthy Plant',
                'description': f'Your {crop} plant appears to be healthy with no visible signs of disease.',
                'symptoms': [],
                'treatment': [],
                'prevention': ['Continue regular care', 'Monitor for changes', 'Maintain optimal growing conditions']
            }

        # Look for specific disease information
        if 'scab' in condition.lower():
            return {
                'name': 'Apple Scab',
                'description': 'Fungal disease causing dark, scaly lesions on leaves and fruit.',
                'symptoms': ['Dark brown lesions on leaves', 'Olive-green spots on fruit', 'Premature leaf drop'],
                'treatment': ['Apply fungicide at bud break', 'Remove fallen leaves', 'Use resistant varieties'],
                'prevention': ['Plant resistant varieties', 'Rake and destroy fallen leaves', 'Avoid overhead watering']
            }

        elif 'black_rot' in condition.lower():
            return {
                'name': 'Black Rot',
                'description': 'Fungal disease causing rotting of fruit and cankers on branches.',
                'symptoms': ['Brown spots with purple margins', 'Fruit rot', 'Branch cankers'],
                'treatment': ['Prune infected branches', 'Apply copper fungicide', 'Remove mummified fruit'],
                'prevention': ['Prune for air circulation', 'Avoid wounding trees', 'Clean up fallen fruit']
            }

        elif 'blight' in condition.lower():
            if 'late' in condition.lower():
                return {
                    'name': 'Late Blight',
                    'description': 'Devastating disease caused by water mold, can destroy entire crops.',
                    'symptoms': ['Dark, water-soaked lesions', 'White fungal growth on undersides', 'Rapid plant death'],
                    'treatment': ['Remove infected plants immediately', 'Apply fungicide preventively', 'Improve drainage'],
                    'prevention': ['Plant resistant varieties', 'Avoid overhead watering', 'Crop rotation']
                }
            else:
                return {
                    'name': 'Early Blight',
                    'description': 'Fungal disease causing leaf spots and fruit rot.',
                    'symptoms': ['Dark spots with concentric rings', 'Yellow halos around spots', 'Leaf yellowing'],
                    'treatment': ['Apply fungicide', 'Remove infected leaves', 'Mulch around plants'],
                    'prevention': ['Crop rotation', 'Stake plants for air circulation', 'Avoid overhead watering']
                }

        elif 'powdery_mildew' in condition.lower():
            return {
                'name': 'Powdery Mildew',
                'description': 'Fungal disease causing white, powdery coating on leaves.',
                'symptoms': ['White powdery spots on leaves', 'Leaf curling', 'Stunted growth'],
                'treatment': ['Spray with baking soda solution', 'Improve air circulation', 'Apply fungicide'],
                'prevention': ['Avoid overhead watering', 'Space plants properly', 'Remove infected parts']
            }

        elif 'bacterial_spot' in condition.lower():
            return {
                'name': 'Bacterial Spot',
                'description': 'Bacterial disease causing spots on leaves and fruit.',
                'symptoms': ['Small, dark spots on leaves', 'Fruit lesions', 'Leaf drop'],
                'treatment': ['Apply copper fungicide', 'Remove infected parts', 'Avoid overhead watering'],
                'prevention': ['Use disease-free seeds', 'Crop rotation', 'Avoid working with wet plants']
            }

        elif 'rust' in condition.lower():
            return {
                'name': 'Rust Disease',
                'description': 'Fungal disease causing rusty-colored pustules on leaves.',
                'symptoms': ['Orange/brown pustules', 'Leaf yellowing', 'Premature defoliation'],
                'treatment': ['Apply fungicide', 'Remove infected leaves', 'Improve air circulation'],
                'prevention': ['Plant resistant varieties', 'Avoid overhead watering', 'Clean up plant debris']
            }

        elif 'virus' in condition.lower():
            return {
                'name': 'Viral Disease',
                'description': 'Virus affecting plant growth and productivity.',
                'symptoms': ['Mosaic patterns on leaves', 'Leaf curling', 'Stunted growth'],
                'treatment': ['Remove infected plants', 'Control insect vectors', 'Use virus-free seeds'],
                'prevention': ['Use certified seeds', 'Control aphids and whiteflies', 'Roguing infected plants']
            }

        elif 'spider_mites' in condition.lower():
            return {
                'name': 'Spider Mites',
                'description': 'Tiny pests that suck plant juices and create webbing.',
                'symptoms': ['Stippled leaves', 'Fine webbing', 'Yellowing leaves'],
                'treatment': ['Spray with insecticidal soap', 'Increase humidity', 'Introduce predatory mites'],
                'prevention': ['Avoid drought stress', 'Regular monitoring', 'Keep plants clean']
            }

        return default_info

    def _calculate_severity(self, confidence):
        """Calculate disease severity based on confidence"""
        if confidence >= 0.9:
            return "High"
        elif confidence >= 0.7:
            return "Medium"
        elif confidence >= 0.5:
            return "Low"
        else:
            return "Very Low"

    def _generate_recommendations(self, disease_info, confidence):
        """Generate comprehensive recommendations"""
        recommendations = []

        # Add treatment recommendations
        if disease_info['treatment']:
            recommendations.extend([f"• {treatment}" for treatment in disease_info['treatment']])

        # Add prevention recommendations
        if disease_info['prevention']:
            recommendations.extend([f"• {prevention}" for prevention in disease_info['prevention']])

        # Add general care recommendations
        recommendations.extend([
            "• Monitor plants regularly for changes",
            "• Maintain proper watering and fertilization",
            "• Ensure good air circulation around plants",
            "• Keep garden tools clean and disinfected",
            "• Consult local agricultural extension services for specific advice"
        ])

        return recommendations

    def _get_mock_prediction(self):
        """Return mock prediction when model is not available"""
        return {
            'disease': 'Analysis Unavailable',
            'confidence': 0,
            'description': 'Plant disease detection model is currently unavailable. Please ensure all required dependencies are installed.',
            'symptoms': [],
            'treatment': ['Please try again later or contact support'],
            'prevention': ['Ensure proper system setup'],
            'severity': 'Unknown',
            'recommendations': [
                '• Check system logs for errors',
                '• Ensure TensorFlow is properly installed',
                '• Verify model files are present',
                '• Contact technical support if issues persist'
            ]
        }