import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, measure, segmentation
from skimage.feature import canny
from skimage.measure import label, regionprops
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class AdvancedImageAnalyzer:
    def __init__(self):
        self.disease_knowledge = self._load_disease_knowledge()
        logger.info("Advanced Image Analyzer initialized with OpenCV and scikit-image")

    def _load_disease_knowledge(self) -> Dict:
        """Load disease knowledge base for image analysis"""
        try:
            with open('data/pest_disease_info.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load disease knowledge: {e}")
            return {}

    def analyze_plant_image(self, image_path: str) -> Dict[str, Any]:
        """Comprehensive plant image analysis using OpenCV and computer vision"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Unable to load image'}

            # Convert to RGB for analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Perform comprehensive analysis
            analysis_results = {
                'basic_info': self._get_basic_image_info(image, image_path),
                'color_analysis': self._analyze_colors(image_rgb),
                'texture_analysis': self._analyze_texture(image),
                'shape_analysis': self._analyze_shapes(image_rgb),
                'segmentation': self._segment_image(image_rgb),
                'disease_detection': self._detect_visual_symptoms(image_rgb),
                'health_assessment': self._assess_plant_health(image_rgb),
                'recommendations': []
            }

            # Generate comprehensive recommendations
            analysis_results['recommendations'] = self._generate_comprehensive_recommendations(analysis_results)

            return analysis_results

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {'error': f'Analysis failed: {str(e)}'}

    def _get_basic_image_info(self, image: np.ndarray, image_path: str) -> Dict:
        """Get basic image information"""
        height, width = image.shape[:2]
        file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0

        return {
            'dimensions': f'{width}x{height}',
            'file_size': f'{file_size / 1024:.1f} KB',
            'channels': image.shape[2] if len(image.shape) > 2 else 1,
            'image_type': 'RGB' if len(image.shape) == 3 else 'Grayscale'
        }

    def _analyze_colors(self, image_rgb: np.ndarray) -> Dict:
        """Analyze color distribution and characteristics"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

        # Calculate color statistics
        color_stats = {}
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            channel_data = image_rgb[:, :, i]
            color_stats[channel] = {
                'mean': float(np.mean(channel_data)),
                'std': float(np.std(channel_data)),
                'min': int(np.min(channel_data)),
                'max': int(np.max(channel_data))
            }

        # Analyze green content (plant health indicator)
        green_channel = image_rgb[:, :, 1]
        total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
        green_pixels = np.sum(green_channel > 100)  # Threshold for green
        green_percentage = (green_pixels / total_pixels) * 100

        # Analyze yellow/brown content (disease indicator)
        yellow_brown_mask = (
            (image_rgb[:, :, 0] > 150) &  # Red channel
            (image_rgb[:, :, 1] > 100) &  # Green channel
            (image_rgb[:, :, 2] < 100)    # Blue channel
        )
        yellow_brown_percentage = (np.sum(yellow_brown_mask) / total_pixels) * 100

        # Analyze brown/black spots (severe disease)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        dark_pixels = np.sum(gray < 50)
        dark_percentage = (dark_pixels / total_pixels) * 100

        return {
            'color_statistics': color_stats,
            'green_content': f'{green_percentage:.1f}%',
            'yellow_brown_content': f'{yellow_brown_percentage:.1f}%',
            'dark_spots': f'{dark_percentage:.1f}%',
            'dominant_colors': self._get_dominant_colors(image_rgb),
            'color_health_score': self._calculate_color_health_score(green_percentage, yellow_brown_percentage, dark_percentage)
        }

    def _get_dominant_colors(self, image_rgb: np.ndarray, k: int = 3) -> List[str]:
        """Extract dominant colors from image"""
        try:
            # Reshape image for k-means
            pixels = image_rgb.reshape(-1, 3).astype(np.float32)

            # Use k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert centers to color names
            color_names = []
            for center in centers:
                color_names.append(self._rgb_to_color_name(center))

            return color_names

        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return ['Unable to analyze']

    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB values to color name"""
        r, g, b = rgb

        # Simple color classification
        if g > r and g > b:
            if g > 150:
                return 'Green (Healthy)'
            elif g > 100:
                return 'Yellow-Green (Stressed)'
            else:
                return 'Brown-Green (Diseased)'
        elif r > g and r > b:
            return 'Red/Brown (Disease spots)'
        elif b > g and b > r:
            return 'Blue/Purple (Possible nutrient deficiency)'
        else:
            gray_value = (r + g + b) / 3
            if gray_value < 50:
                return 'Black (Dead tissue)'
            elif gray_value < 100:
                return 'Dark Brown (Severe damage)'
            else:
                return 'Light Brown (Mild damage)'

    def _calculate_color_health_score(self, green_pct: float, yellow_brown_pct: float, dark_pct: float) -> str:
        """Calculate plant health score based on color analysis"""
        if green_pct > 60 and yellow_brown_pct < 10 and dark_pct < 5:
            return 'Excellent (Very healthy plant)'
        elif green_pct > 40 and yellow_brown_pct < 20 and dark_pct < 10:
            return 'Good (Healthy with minor issues)'
        elif green_pct > 20 and yellow_brown_pct < 30 and dark_pct < 20:
            return 'Fair (Moderate health issues)'
        elif green_pct > 10 and yellow_brown_pct < 40 and dark_pct < 30:
            return 'Poor (Significant health problems)'
        else:
            return 'Critical (Severe disease/damage)'

    def _analyze_texture(self, image: np.ndarray) -> Dict:
        """Analyze image texture characteristics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate texture features
            # GLCM (Gray Level Co-occurrence Matrix) features
            glcm = self._calculate_glcm_features(gray)

            # Edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Contrast analysis
            contrast = gray.std()

            # Smoothness analysis
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            smoothness = laplacian.var()

            return {
                'contrast': f'{contrast:.2f}',
                'smoothness': f'{smoothness:.2f}',
                'edge_density': f'{edge_density:.3f}',
                'texture_uniformity': glcm.get('uniformity', 'N/A'),
                'texture_description': self._describe_texture(contrast, smoothness, edge_density)
            }

        except Exception as e:
            logger.error(f"Error analyzing texture: {e}")
            return {'error': 'Texture analysis failed'}

    def _calculate_glcm_features(self, gray: np.ndarray) -> Dict:
        """Calculate GLCM texture features"""
        try:
            # Simple GLCM calculation (simplified version)
            # In production, use skimage.feature.graycomatrix and graycoprops

            # Calculate uniformity (angular second moment)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            uniformity = np.sum(hist ** 2)

            return {
                'uniformity': f'{uniformity[0]:.4f}',
                'energy': f'{np.sqrt(uniformity[0]):.4f}'
            }

        except Exception as e:
            return {'error': str(e)}

    def _describe_texture(self, contrast: float, smoothness: float, edge_density: float) -> str:
        """Describe texture characteristics"""
        if contrast < 20 and smoothness < 100:
            return 'Smooth and uniform (healthy leaf surface)'
        elif contrast > 50 and edge_density > 0.1:
            return 'Rough and irregular (possible disease spots or damage)'
        elif smoothness > 200:
            return 'Very textured (possible fungal growth or pest damage)'
        else:
            return 'Moderate texture (normal leaf surface)'

    def _analyze_shapes(self, image_rgb: np.ndarray) -> Dict:
        """Analyze shapes and morphological features"""
        try:
            # Convert to grayscale and threshold
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour (main leaf/plant part)
                largest_contour = max(contours, key=cv2.contourArea)

                # Calculate shape features
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0

                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                return {
                    'contour_count': len(contours),
                    'main_contour_area': int(area),
                    'perimeter': f'{perimeter:.1f}',
                    'compactness': f'{compactness:.2f}',
                    'aspect_ratio': f'{aspect_ratio:.2f}',
                    'solidity': f'{solidity:.2f}',
                    'shape_description': self._describe_shape(compactness, aspect_ratio, solidity)
                }
            else:
                return {'error': 'No shapes detected in image'}

        except Exception as e:
            logger.error(f"Error analyzing shapes: {e}")
            return {'error': 'Shape analysis failed'}

    def _describe_shape(self, compactness: float, aspect_ratio: float, solidity: float) -> str:
        """Describe shape characteristics"""
        description = []

        if compactness < 15:
            description.append('regular shape')
        else:
            description.append('irregular shape (possible damage)')

        if aspect_ratio > 1.5:
            description.append('elongated')
        elif aspect_ratio < 0.7:
            description.append('wide')
        else:
            description.append('balanced proportions')

        if solidity < 0.9:
            description.append('with indentations or holes (possible pest damage)')

        return ', '.join(description)

    def _segment_image(self, image_rgb: np.ndarray) -> Dict:
        """Segment image into different regions"""
        try:
            # Simple color-based segmentation
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

            # Segment healthy green areas
            green_mask = cv2.inRange(hsv, (35, 50, 50), (80, 255, 255))
            green_pixels = cv2.countNonZero(green_mask)

            # Segment yellow/diseased areas
            yellow_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
            yellow_pixels = cv2.countNonZero(yellow_mask)

            # Segment brown/damaged areas
            brown_mask = cv2.inRange(hsv, (5, 50, 20), (20, 255, 200))
            brown_pixels = cv2.countNonZero(brown_mask)

            total_pixels = image_rgb.shape[0] * image_rgb.shape[1]

            return {
                'healthy_green_region': f'{(green_pixels/total_pixels*100):.1f}%',
                'yellow_diseased_region': f'{(yellow_pixels/total_pixels*100):.1f}%',
                'brown_damaged_region': f'{(brown_pixels/total_pixels*100):.1f}%',
                'segmentation_quality': 'Good' if (green_pixels + yellow_pixels + brown_pixels) > total_pixels * 0.5 else 'Poor'
            }

        except Exception as e:
            logger.error(f"Error segmenting image: {e}")
            return {'error': 'Segmentation failed'}

    def _detect_visual_symptoms(self, image_rgb: np.ndarray) -> Dict:
        """Detect visual symptoms of plant diseases"""
        symptoms = []

        # Analyze color patterns for disease symptoms
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Check for chlorosis (yellowing)
        yellow_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
        yellow_percentage = (cv2.countNonZero(yellow_mask) / (image_rgb.shape[0] * image_rgb.shape[1])) * 100

        if yellow_percentage > 20:
            symptoms.append('Chlorosis (yellowing) - possible nutrient deficiency or disease')

        # Check for necrosis (brown/black spots)
        brown_mask = cv2.inRange(hsv, (5, 50, 20), (20, 255, 200))
        brown_percentage = (cv2.countNonZero(brown_mask) / (image_rgb.shape[0] * image_rgb.shape[1])) * 100

        if brown_percentage > 10:
            symptoms.append('Necrotic spots - tissue death, possible fungal or bacterial infection')

        # Check for powdery coating (fungal growth)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        powdery_regions = np.sum(np.abs(gray - blurred) > 30)
        powdery_percentage = (powdery_regions / (image_rgb.shape[0] * image_rgb.shape[1])) * 100

        if powdery_percentage > 15:
            symptoms.append('Powdery coating - possible powdery mildew fungal infection')

        # Check for irregular patterns (pest damage)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        if edge_density > 0.15:
            symptoms.append('Irregular edges - possible insect feeding damage')

        return {
            'detected_symptoms': symptoms if symptoms else ['No obvious disease symptoms detected'],
            'symptom_severity': self._assess_symptom_severity(yellow_percentage, brown_percentage, powdery_percentage, edge_density),
            'confidence_level': 'High' if len(symptoms) > 2 else 'Medium' if len(symptoms) > 0 else 'Low'
        }

    def _assess_symptom_severity(self, yellow_pct: float, brown_pct: float, powdery_pct: float, edge_density: float) -> str:
        """Assess overall symptom severity"""
        severity_score = (yellow_pct * 0.3 + brown_pct * 0.4 + powdery_pct * 0.2 + edge_density * 100 * 0.1)

        if severity_score < 5:
            return 'Mild - Monitor closely'
        elif severity_score < 15:
            return 'Moderate - Treatment recommended'
        elif severity_score < 30:
            return 'Severe - Immediate action required'
        else:
            return 'Critical - Urgent professional intervention needed'

    def _assess_plant_health(self, image_rgb: np.ndarray) -> Dict:
        """Comprehensive plant health assessment"""
        # Combine multiple analysis results
        color_health = self._analyze_colors(image_rgb)
        texture_info = self._analyze_texture(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        symptoms = self._detect_visual_symptoms(image_rgb)

        # Calculate overall health score (0-100)
        health_score = 100

        # Deduct points based on various factors
        if 'yellow_brown_content' in color_health:
            yellow_pct = float(color_health['yellow_brown_content'].rstrip('%'))
            health_score -= yellow_pct * 0.5

        if 'dark_spots' in color_health:
            dark_pct = float(color_health['dark_spots'].rstrip('%'))
            health_score -= dark_pct * 2

        if 'edge_density' in texture_info:
            edge_pct = float(texture_info['edge_density']) * 100
            health_score -= edge_pct * 0.3

        health_score = max(0, min(100, health_score))

        return {
            'overall_health_score': f'{health_score:.1f}/100',
            'health_category': self._categorize_health(health_score),
            'risk_level': 'Low' if health_score > 70 else 'Medium' if health_score > 40 else 'High',
            'immediate_action_needed': health_score < 50
        }

    def _categorize_health(self, score: float) -> str:
        """Categorize plant health based on score"""
        if score >= 80:
            return 'Excellent Health'
        elif score >= 60:
            return 'Good Health'
        elif score >= 40:
            return 'Fair Health'
        elif score >= 20:
            return 'Poor Health'
        else:
            return 'Critical Condition'

    def _generate_comprehensive_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate comprehensive recommendations based on all analysis"""
        recommendations = []

        # Basic care recommendations
        recommendations.extend([
            "🌱 Maintain proper watering schedule - avoid overwatering",
            "🧪 Test soil pH and nutrient levels regularly",
            "✂️ Prune dead or diseased plant parts immediately",
            "👁️ Monitor plants daily for changes in appearance"
        ])

        # Color-based recommendations
        color_analysis = analysis_results.get('color_analysis', {})
        if 'yellow_brown_content' in color_analysis:
            yellow_pct = float(color_analysis['yellow_brown_content'].rstrip('%'))
            if yellow_pct > 20:
                recommendations.append("🟡 Yellowing detected - check for nutrient deficiencies, especially nitrogen")

        if 'dark_spots' in color_analysis:
            dark_pct = float(color_analysis['dark_spots'].rstrip('%'))
            if dark_pct > 10:
                recommendations.append("⚫ Dark spots present - apply copper-based fungicide and improve air circulation")

        # Symptom-based recommendations
        disease_detection = analysis_results.get('disease_detection', {})
        symptoms = disease_detection.get('detected_symptoms', [])
        if len(symptoms) > 1:
            recommendations.append("🔍 Multiple symptoms detected - consult agricultural extension service")

        # Health-based recommendations
        health_assessment = analysis_results.get('health_assessment', {})
        health_score = float(health_assessment.get('overall_health_score', '100').split('/')[0])

        if health_score < 50:
            recommendations.extend([
                "🚨 Plant health critical - immediate professional consultation recommended",
                "💊 Apply broad-spectrum fungicide and insecticide as preventive measure",
                "🔄 Consider replanting if condition doesn't improve within 7 days"
            ])
        elif health_score < 70:
            recommendations.extend([
                "⚠️ Plant showing stress - improve growing conditions",
                "🌿 Apply organic fertilizers and ensure proper drainage",
                "🛡️ Use protective netting against pests"
            ])

        # Texture and shape recommendations
        texture_analysis = analysis_results.get('texture_analysis', {})
        if 'texture_description' in texture_analysis:
            texture_desc = texture_analysis['texture_description']
            if 'disease spots' in texture_desc or 'damage' in texture_desc:
                recommendations.append("🔸 Irregular texture detected - inspect for pest damage and fungal infections")

        return recommendations

    def get_image_summary(self, image_path: str) -> str:
        """Generate a comprehensive text summary of the image analysis"""
        analysis = self.analyze_plant_image(image_path)

        if 'error' in analysis:
            return f"Analysis failed: {analysis['error']}"

        summary = f"""
🌱 PLANT IMAGE ANALYSIS SUMMARY
================================

📊 BASIC INFORMATION:
- Dimensions: {analysis['basic_info']['dimensions']}
- File Size: {analysis['basic_info']['file_size']}
- Image Type: {analysis['basic_info']['image_type']}

🎨 COLOR ANALYSIS:
- Green Content: {analysis['color_analysis']['green_content']}
- Yellow/Brown Areas: {analysis['color_analysis']['yellow_brown_content']}
- Dark Spots: {analysis['color_analysis']['dark_spots']}
- Health Score: {analysis['color_analysis']['color_health_score']}

🔍 TEXTURE ANALYSIS:
- Contrast: {analysis['texture_analysis']['contrast']}
- Edge Density: {analysis['texture_analysis']['edge_density']}
- Description: {analysis['texture_analysis']['texture_description']}

📐 SHAPE ANALYSIS:
- Contour Count: {analysis['shape_analysis']['contour_count']}
- Main Area: {analysis['shape_analysis']['main_contour_area']} pixels
- Description: {analysis['shape_analysis']['shape_description']}

🩺 DISEASE DETECTION:
- Symptoms: {', '.join(analysis['disease_detection']['detected_symptoms'])}
- Severity: {analysis['disease_detection']['symptom_severity']}
- Confidence: {analysis['disease_detection']['confidence_level']}

❤️ HEALTH ASSESSMENT:
- Overall Score: {analysis['health_assessment']['overall_health_score']}
- Category: {analysis['health_assessment']['health_category']}
- Risk Level: {analysis['health_assessment']['risk_level']}

💡 RECOMMENDATIONS:
{chr(10).join('- ' + rec for rec in analysis['recommendations'][:5])}

================================
Analysis completed using OpenCV and computer vision algorithms.
        """.strip()

        return summary