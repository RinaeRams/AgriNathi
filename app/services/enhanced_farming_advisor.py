import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedFarmingAdvisor:
    def __init__(self):
        self.crop_data = None
        self.farming_knowledge = {}
        self.weather_data = None
        self.load_crop_data()
        self.load_farming_knowledge()

    def load_crop_data(self):
        """Load crop recommendation data from CSV"""
        try:
            crop_file = 'data/Crop_recommendation.csv'
            self.crop_data = pd.read_csv(crop_file)
            logger.info(f"Loaded crop data with {len(self.crop_data)} records")
        except Exception as e:
            logger.error(f"Failed to load crop data: {e}")
            self.crop_data = None

    def load_farming_knowledge(self):
        """Load comprehensive farming knowledge base"""
        try:
            with open('data/pest_disease_info.json', 'r', encoding='utf-8') as f:
                self.farming_knowledge = json.load(f)
            logger.info("Farming knowledge base loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load farming knowledge: {e}")
            self.farming_knowledge = {}

    def analyze_farming_query(self, query: str, language: str = 'en') -> Dict:
        """Analyze farming query and provide comprehensive advice"""
        query_lower = query.lower()

        # Determine query type and extract key information
        query_analysis = self._analyze_query_type(query_lower)

        # Get crop-specific advice if crop is mentioned
        crop_advice = self._get_crop_specific_advice(query_analysis.get('crop'))

        # Get seasonal advice
        seasonal_advice = self._get_seasonal_advice()

        # Get general farming tips
        general_tips = self._get_general_farming_tips(query_analysis.get('topic'))

        # Get market insights
        market_insights = self._get_market_insights(query_analysis.get('crop'))

        # Get weather-based recommendations
        weather_advice = self._get_weather_based_advice()

        # Compile comprehensive response
        response = {
            'query_analysis': query_analysis,
            'crop_advice': crop_advice,
            'seasonal_advice': seasonal_advice,
            'general_tips': general_tips,
            'market_insights': market_insights,
            'weather_advice': weather_advice,
            'recommendations': self._compile_recommendations(
                crop_advice, seasonal_advice, general_tips, weather_advice
            ),
            'follow_up_questions': self._generate_follow_up_questions(query_analysis)
        }

        return response

    def _analyze_query_type(self, query: str) -> Dict:
        """Analyze the type of farming query"""
        analysis = {
            'topic': 'general',
            'crop': None,
            'issue': None,
            'urgency': 'normal',
            'keywords': []
        }

        # Define keywords for different crops
        crop_keywords = {
            'maize': ['maize', 'corn', 'mealie', 'umngqusho'],
            'rice': ['rice', 'irayisi'],
            'wheat': ['wheat', 'inkomn'],
            'potatoes': ['potato', 'potatoes', 'amazambane'],
            'tomatoes': ['tomato', 'tomatoes', 'utamatisi'],
            'beans': ['bean', 'beans', 'imbotyi'],
            'peas': ['pea', 'peas', 'pigeon peas', 'amapine'],
            'chickpea': ['chickpea', 'chickpeas', 'indlubu'],
            'kidneybeans': ['kidney bean', 'kidney beans'],
            'mothbeans': ['moth bean', 'moth beans'],
            'mungbean': ['mung bean', 'mung beans', 'green gram'],
            'blackgram': ['black gram', 'blackgram', 'urd bean'],
            'lentil': ['lentil', 'lentils', 'masoor'],
            'pomegranate': ['pomegranate', 'granate'],
            'banana': ['banana', 'bananas', 'ubhanana'],
            'mango': ['mango', 'mangoes', 'umango'],
            'grapes': ['grape', 'grapes', 'amagilebhisi'],
            'watermelon': ['watermelon', 'watermelons', 'ikhabe'],
            'muskmelon': ['muskmelon', 'muskmelons', 'cantaloupe'],
            'apple': ['apple', 'apples', 'ama-apula'],
            'orange': ['orange', 'oranges', 'ama-orange'],
            'papaya': ['papaya', 'papayas', 'uphapaya'],
            'coconut': ['coconut', 'coconuts', 'ukhukhunathi'],
            'cotton': ['cotton', 'ukotini'],
            'jute': ['jute', 'jute plant'],
            'coffee': ['coffee', 'ikofi']
        }

        # Check for crop mentions
        for crop, keywords in crop_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    analysis['crop'] = crop
                    analysis['keywords'].append(keyword)
                    break
            if analysis['crop']:
                break

        # Determine topic
        if any(word in query for word in ['disease', 'sick', 'yellow', 'spots', 'blight', 'rot', 'fungus', 'bacterial']):
            analysis['topic'] = 'disease'
            analysis['urgency'] = 'high'
        elif any(word in query for word in ['pest', 'insect', 'bug', 'worm', 'aphid', 'beetle', 'mite']):
            analysis['topic'] = 'pest'
            analysis['urgency'] = 'high'
        elif any(word in query for word in ['water', 'irrigation', 'rain', 'drought']):
            analysis['topic'] = 'irrigation'
        elif any(word in query for word in ['fertilizer', 'nutrient', 'soil', 'ph', 'manure']):
            analysis['topic'] = 'nutrition'
        elif any(word in query for word in ['plant', 'seed', 'sow', 'planting']):
            analysis['topic'] = 'planting'
        elif any(word in query for word in ['harvest', 'yield', 'crop']):
            analysis['topic'] = 'harvesting'
        elif any(word in query for word in ['market', 'price', 'sell', 'buy']):
            analysis['topic'] = 'market'
        elif any(word in query for word in ['weather', 'rain', 'temperature', 'climate']):
            analysis['topic'] = 'weather'

        return analysis

    def _get_crop_specific_advice(self, crop: str) -> Dict:
        """Get crop-specific farming advice"""
        if not crop or not self.crop_data:
            return {'general': 'General farming practices apply to most crops.'}

        try:
            # Filter data for the specific crop
            crop_data = self.crop_data[self.crop_data['label'].str.lower() == crop.lower()]

            if crop_data.empty:
                return {'general': f'Information for {crop} is not available in our database.'}

            # Calculate optimal conditions
            optimal_conditions = {
                'nitrogen': crop_data['N'].mean(),
                'phosphorus': crop_data['P'].mean(),
                'potassium': crop_data['K'].mean(),
                'temperature': crop_data['temperature'].mean(),
                'humidity': crop_data['humidity'].mean(),
                'ph': crop_data['ph'].mean(),
                'rainfall': crop_data['rainfall'].mean()
            }

            # Generate advice based on optimal conditions
            advice = {
                'optimal_conditions': optimal_conditions,
                'soil_preparation': self._get_soil_advice(crop, optimal_conditions),
                'watering_schedule': self._get_watering_advice(crop, optimal_conditions),
                'fertilizer_recommendation': self._get_fertilizer_advice(crop, optimal_conditions),
                'pest_management': self._get_pest_advice(crop),
                'harvesting_tips': self._get_harvesting_advice(crop)
            }

            return advice

        except Exception as e:
            logger.error(f"Error getting crop advice: {e}")
            return {'error': 'Unable to retrieve crop-specific advice.'}

    def _get_soil_advice(self, crop: str, conditions: Dict) -> str:
        """Generate soil preparation advice"""
        ph = conditions.get('ph', 7.0)
        if ph < 6.0:
            return f"For {crop}, maintain soil pH around {ph:.1f}. Add lime to increase pH if needed."
        elif ph > 7.5:
            return f"For {crop}, maintain soil pH around {ph:.1f}. Add sulfur to decrease pH if needed."
        else:
            return f"For {crop}, maintain soil pH around {ph:.1f} for optimal growth."

    def _get_watering_advice(self, crop: str, conditions: Dict) -> str:
        """Generate watering schedule advice"""
        rainfall = conditions.get('rainfall', 100)
        humidity = conditions.get('humidity', 50)

        if rainfall > 150:
            return f"{crop} requires high moisture. Ensure {rainfall:.0f}mm rainfall or equivalent irrigation."
        elif rainfall < 80:
            return f"{crop} is drought-tolerant but needs {rainfall:.0f}mm rainfall. Supplement with irrigation during dry periods."
        else:
            return f"{crop} needs moderate watering. Aim for {rainfall:.0f}mm rainfall equivalent."

    def _get_fertilizer_advice(self, crop: str, conditions: Dict) -> str:
        """Generate fertilizer recommendations"""
        n = conditions.get('nitrogen', 50)
        p = conditions.get('phosphorus', 50)
        k = conditions.get('potassium', 50)

        return f"For {crop}, use NPK ratio of {n:.0f}-{p:.0f}-{k:.0f}. Apply nitrogen during vegetative growth, phosphorus at planting, and potassium during fruiting."

    def _get_pest_advice(self, crop: str) -> str:
        """Get pest management advice for specific crop"""
        crop_pests = self.farming_knowledge.get('crop_specific', {}).get(crop, {})

        if crop_pests:
            pests = crop_pests.get('pests', [])
            diseases = crop_pests.get('diseases', [])
            management = crop_pests.get('management', [])

            advice = f"Common pests for {crop}: {', '.join(pests[:3])}. "
            advice += f"Common diseases: {', '.join(diseases[:3])}. "
            advice += f"Management: {', '.join(management)}."

            return advice
        else:
            return f"Monitor {crop} regularly for pests and diseases. Use integrated pest management practices."

    def _get_harvesting_advice(self, crop: str) -> str:
        """Get harvesting tips for specific crop"""
        harvest_tips = {
            'maize': 'Harvest when kernels are fully developed and kernels are hard. Moisture content should be 20-25%.',
            'rice': 'Harvest when 80-85% of grains are straw-colored. Use sickles for manual harvesting.',
            'wheat': 'Harvest when grains are hard and golden yellow. Use combine harvesters for large fields.',
            'potatoes': 'Harvest when vines have died back. Cure potatoes in dark, well-ventilated area for 2 weeks.',
            'tomatoes': 'Harvest when fruits are fully colored and firm. Pick regularly to encourage more production.',
            'beans': 'Harvest when pods are dry and rattle. Use mechanical harvesters for large-scale operations.'
        }

        return harvest_tips.get(crop, f'Harvest {crop} at optimal maturity stage. Consult local agricultural extension for specific timing.')

    def _get_seasonal_advice(self) -> str:
        """Get seasonal farming advice"""
        current_month = datetime.now().month

        if current_month in [9, 10, 11]:  # Spring in Southern Hemisphere
            return "Spring planting season: Prepare soil, plant cool-season crops, apply pre-emergent herbicides."
        elif current_month in [12, 1, 2]:  # Summer
            return "Summer growing season: Monitor irrigation, apply fertilizers, control pests and diseases."
        elif current_month in [3, 4, 5]:  # Autumn
            return "Autumn harvest season: Prepare for harvest, store crops properly, plan for next season."
        else:  # Winter
            return "Winter planning season: Plan crop rotation, maintain equipment, attend farming workshops."

    def _get_general_farming_tips(self, topic: str) -> List[str]:
        """Get general farming tips based on topic"""
        tips = {
            'disease': [
                'Practice crop rotation to break disease cycles',
                'Use disease-resistant varieties when available',
                'Ensure proper plant spacing for air circulation',
                'Avoid working with wet plants to prevent disease spread',
                'Apply fungicides preventively during high-risk periods'
            ],
            'pest': [
                'Monitor fields regularly for early pest detection',
                'Use beneficial insects to control pest populations',
                'Apply pesticides only when necessary and follow label instructions',
                'Rotate different types of pesticides to prevent resistance',
                'Maintain field borders with trap crops'
            ],
            'irrigation': [
                'Water deeply but less frequently to encourage deep root growth',
                'Use drip irrigation to conserve water and reduce disease',
                'Water early in the morning to minimize evaporation',
                'Monitor soil moisture regularly',
                'Adjust irrigation based on weather conditions and crop stage'
            ],
            'nutrition': [
                'Test soil regularly to determine nutrient needs',
                'Apply fertilizers based on soil test recommendations',
                'Use organic matter to improve soil fertility',
                'Apply lime if soil pH is too low',
                'Split fertilizer applications throughout the growing season'
            ]
        }

        return tips.get(topic, [
            'Maintain proper soil health through regular testing and amendments',
            'Practice integrated pest management',
            'Keep detailed records of farming activities',
            'Stay informed about new farming technologies and methods',
            'Join local farming cooperatives for support and knowledge sharing'
        ])

    def _get_market_insights(self, crop: str) -> str:
        """Get market insights for specific crop"""
        if not crop:
            return "Monitor local market prices regularly. Consider joining farming cooperatives for better market access."

        market_tips = {
            'maize': 'Maize prices fluctuate with international markets. Store grain properly to sell during high-price periods.',
            'rice': 'Rice has stable demand. Consider value addition like processing for higher returns.',
            'wheat': 'Wheat prices are influenced by global supply. Consider contract farming for price stability.',
            'potatoes': 'Potatoes have seasonal price variations. Early crop can command premium prices.',
            'tomatoes': 'Tomato prices vary with supply. Consider greenhouse production for off-season sales.'
        }

        return market_tips.get(crop, f"For {crop}, research local market demand and prices. Consider value addition and direct marketing to consumers.")

    def _get_weather_based_advice(self) -> str:
        """Get weather-based farming advice"""
        # This would integrate with actual weather data in production
        return "Monitor weather forecasts for irrigation and pest management decisions. Prepare for extreme weather events."

    def _compile_recommendations(self, crop_advice: Dict, seasonal_advice: str,
                               general_tips: List[str], weather_advice: str) -> List[str]:
        """Compile all recommendations into a comprehensive list"""
        recommendations = []

        # Add crop-specific recommendations
        if crop_advice and 'optimal_conditions' in crop_advice:
            conditions = crop_advice['optimal_conditions']
            recommendations.append(f"🌱 Maintain optimal conditions: pH {conditions['ph']:.1f}, temperature {conditions['temperature']:.1f}°C")

        # Add seasonal advice
        if seasonal_advice:
            recommendations.append(f"📅 {seasonal_advice}")

        # Add general tips
        recommendations.extend([f"💡 {tip}" for tip in general_tips[:3]])

        # Add weather advice
        if weather_advice:
            recommendations.append(f"🌤️ {weather_advice}")

        return recommendations

    def _generate_follow_up_questions(self, query_analysis: Dict) -> List[str]:
        """Generate follow-up questions to gather more information"""
        questions = []

        if not query_analysis.get('crop'):
            questions.append("Which crop are you asking about?")

        if query_analysis.get('topic') == 'disease':
            questions.extend([
                "Can you describe the symptoms you're seeing?",
                "When did you first notice the problem?",
                "Have you applied any treatments already?"
            ])

        elif query_analysis.get('topic') == 'pest':
            questions.extend([
                "What type of pest are you dealing with?",
                "How severe is the infestation?",
                "What crops are affected?"
            ])

        elif query_analysis.get('topic') == 'irrigation':
            questions.extend([
                "What irrigation system are you using?",
                "How often do you water your crops?",
                "What is your water source?"
            ])

        return questions[:3]  # Limit to 3 questions