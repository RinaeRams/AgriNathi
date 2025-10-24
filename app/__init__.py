from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import base64
import io
import sys
import json
from datetime import datetime

# Try to import Vonage (formerly Nexmo), but don't fail if not available
try:
    import vonage
    VONAGE_AVAILABLE = True
    print("Vonage (formerly Nexmo) client available for IVR features.")
except ImportError:
    print("Warning: Vonage not available. IVR features will use basic NCCO responses.")
    VONAGE_AVAILABLE = False

# Create the Flask application instance
app = Flask(__name__)

def create_app():
    """Initialize and configure the Flask application"""
    global app
    
    # Ensure app exists
    if not app:
        app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    app.config['UPLOAD_FOLDER'] = 'data/audio_recordings'

    # Session configuration
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True

    # Initialize session
    # Note: In production, use a proper session store like Redis

    # Default placeholders for services and models (ensure names exist even if imports fail)
    speech_service = None
    translation_service = None
    plant_disease_detector = None
    farming_advisor = None
    image_analyzer = None
    tf_model = None
    tf_labels = None

    # Import the actual voice recognition service
    try:
        # Set up Google Cloud credentials (from env or file)
        from app.utils import setup_google_credentials
        credentials_path = setup_google_credentials()

        # Import services from the local app/services directory
        import sys
        sys.path.append('.')
        from app.services.speech_service import SpeechToTextService
        from app.services.translation_service import TranslationService
        from app.services.plant_disease_detector import PlantDiseaseDetector
        from app.services.enhanced_farming_advisor import EnhancedFarmingAdvisor
        from app.services.advanced_image_analyzer import AdvancedImageAnalyzer

        speech_service = SpeechToTextService()
        translation_service = TranslationService()
        plant_disease_detector = PlantDiseaseDetector()
        farming_advisor = EnhancedFarmingAdvisor()
        image_analyzer = AdvancedImageAnalyzer()
        # Attempt to load a trained Keras model for plant classification + labels
        tf_model = None
        tf_labels = None
        try:
            import tensorflow as _tf
            from tensorflow.keras.models import load_model as _load_model
            import json as _json
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'plant_mobilenetv2.h5')
            labels_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'labels.json')
            if os.path.exists(model_path) and os.path.exists(labels_path):
                try:
                    tf_model = _load_model(model_path)
                    with open(labels_path, 'r') as lf:
                        tf_labels = _json.load(lf)
                    print(f'Loaded TF model from {model_path} with {len(tf_labels)} labels')
                except Exception as e:
                    print(f'Warning: failed to load TF model: {e}')
            else:
                print('No trained TF model found at models/plant_mobilenetv2.h5 (skipping)')
        except Exception as e:
            print(f'TensorFlow not available or failed to import: {e}')
        print("Google Cloud services initialized successfully!")

        class VoiceRecognition:
            def process_audio(self, audio_base64):
                try:
                    # Decode base64 audio
                    import base64
                    audio_data = base64.b64decode(audio_base64)

                    # Save temporary audio file
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_file_path = temp_file.name

                    try:
                        # Transcribe audio using Google Cloud Speech-to-Text
                        transcript = speech_service.transcribe_audio(temp_file_path, language_code='zu-ZA')

                        if not transcript or transcript.strip() == "":
                            transcript = "Speech not detected. Please try speaking louder and clearer."

                        # Translate to English using Google Cloud Translate
                        translation = translation_service.translate_text(transcript, source_lang="zu", target_lang="en")

                        # Generate agricultural advice based on content
                        advice = self._generate_agricultural_advice(transcript.lower())

                        return {
                            'success': True,
                            'transcript': transcript,
                            'translation': translation,
                            'advice': advice
                        }

                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)

                except Exception as e:
                    print(f"Voice processing error: {e}")
                    return {
                        'success': False,
                        'error': f'Failed to process audio: {str(e)}',
                        'transcript': '',
                        'translation': '',
                        'advice': 'Please try recording again.'
                    }

            def _generate_agricultural_advice(self, zulu_text):
                """Generate agricultural advice based on isiZulu keywords"""
                advice_map = {
                    'izitshalo': 'For plant care: Ensure proper watering, use organic fertilizers, and monitor for pests regularly.',
                    'zifo': 'For plant diseases: Remove affected leaves immediately, improve air circulation, and use copper-based fungicides.',
                    'nambuzane': 'For pest control: Use neem oil spray, introduce beneficial insects, and practice proper crop rotation.',
                    'nisela': 'Watering advice: Water deeply but infrequently, early morning is best, avoid wetting leaves to prevent fungal diseases.',
                    'umanyolo': 'Fertilizer guidance: Use balanced NPK fertilizer, apply during growing season, test soil pH first.',
                    'imbewu': 'Seed planting: Plant during correct season, ensure proper spacing, keep soil moist until germination.',
                    'isimo sezulu': 'Weather considerations: Monitor forecasts, protect crops from frost, prepare drainage for heavy rain.',
                    'khuni': 'Maize care: Plant in well-drained soil, fertilize regularly, watch for corn borer and rust diseases.',
                    'utshani': 'Weed control: Use mulching, hand weeding, or organic herbicides. Prevent weed competition for nutrients.',
                    'umhlaba': 'Soil management: Test soil pH regularly, add organic matter, practice conservation tillage.',
                    'isivuno': 'Harvesting: Harvest at correct maturity, use proper tools, store in cool dry place.',
                    'izilwane': 'Livestock care: Provide clean water, balanced feed, regular health checks, proper housing.'
                }

                for keyword, advice in advice_map.items():
                    if keyword in zulu_text:
                        return advice

                return 'General farming advice: Practice sustainable agriculture, monitor your crops regularly, maintain soil health, and seek local extension services for specific guidance.'

        voice_recognition = VoiceRecognition()

    except Exception as e:
        print(f"Could not import services: {e}. Using mock mode.")
        # Fallback to mock if services not available
        class MockVoiceRecognition:
            def process_audio(self, audio_base64):
                return {
                    'success': True,
                    'transcript': 'Sawubona, ngicela usizo ngezitshalo zami',
                    'translation': 'Hello, I need help with my plants',
                    'advice': 'For plant diseases, ensure proper watering and use organic pesticides.'
                }

        voice_recognition = MockVoiceRecognition()

        # Provide simple fallbacks for other services so routes won't crash
        class _MockPlantDetector:
            def predict_disease(self, image_path):
                return {
                    'disease': 'Analysis Unavailable',
                    'confidence': 0,
                    'description': 'Model unavailable in this environment.',
                    'symptoms': [],
                    'treatment': ['Please try again later'],
                    'prevention': [],
                    'severity': 'Unknown',
                    'recommendations': ['Model not installed']
                }
            def _get_mock_prediction(self):
                return self.predict_disease(None)

        class _MockFarmingAdvisor:
            def analyze_farming_query(self, text):
                return {'recommendations': [], 'crop_advice': {}, 'market_insights': ''}

        class _MockService:
            def transcribe_audio(self, path, language_code=None):
                return ''
            def translate_text(self, text, source_lang=None, target_lang=None):
                return text

        plant_disease_detector = _MockPlantDetector()
        farming_advisor = _MockFarmingAdvisor()
        speech_service = _MockService()
        translation_service = _MockService()

        class VoiceRecognition:
            def process_audio(self, audio_base64):
                try:
                    # Decode base64 audio
                    import base64
                    audio_data = base64.b64decode(audio_base64)

                    # Save temporary audio file
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_file_path = temp_file.name

                    try:
                        # Transcribe audio (will use mock if API not available)
                        transcript = speech_service.transcribe_audio(temp_file_path, language_code='zu-ZA')

                        if not transcript:
                            transcript = "Speech not detected. Please try speaking louder and clearer."

                        # Translate to English
                        translation = translation_service.translate_text(transcript, source_lang="zu", target_lang="en")

                        # Generate agricultural advice based on content
                        advice = self._generate_agricultural_advice(transcript.lower())

                        return {
                            'success': True,
                            'transcript': transcript,
                            'translation': translation,
                            'advice': advice
                        }

                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)

                except Exception as e:
                    print(f"Voice processing error: {e}")
                    return {
                        'success': False,
                        'error': 'Failed to process audio. Please try again.',
                        'transcript': '',
                        'translation': '',
                        'advice': 'Please try recording again.'
                    }

            def _generate_agricultural_advice(self, zulu_text):
                """Generate agricultural advice based on isiZulu keywords"""
                advice_map = {
                    'izitshalo': 'For plant care: Ensure proper watering, use organic fertilizers, and monitor for pests.',
                    'zifo': 'For plant diseases: Remove affected leaves, improve air circulation, and use copper-based fungicides.',
                    'nambuzane': 'For pest control: Use neem oil spray, introduce beneficial insects, and practice crop rotation.',
                    'nisela': 'Watering advice: Water deeply but infrequently, early morning is best, avoid wetting leaves.',
                    'umanyolo': 'Fertilizer guidance: Use balanced NPK fertilizer, apply during growing season, test soil first.',
                    'imbewu': 'Seed planting: Plant during right season, ensure proper spacing, keep soil moist until germination.',
                    'isimo sezulu': 'Weather considerations: Monitor forecasts, protect crops from frost, prepare for rain.',
                    'khuni': 'Maize care: Plant in well-drained soil, fertilize regularly, watch for corn borer and rust.',
                    'utshani': 'Weed control: Use mulching, hand weeding, or organic herbicides. Prevent weed competition.'
                }

                for keyword, advice in advice_map.items():
                    if keyword in zulu_text:
                        return advice

                return 'General farming advice: Practice sustainable agriculture, monitor your crops regularly, and maintain soil health.'

        voice_recognition = VoiceRecognition()

    # Helper inference functions: use TF model if available, otherwise fall back to plant_disease_detector
    def model_predict_image(image_path):
        """Return {'label':..., 'confidence':..., 'gradcam_b64': optional} or fallback to plant_disease_detector result."""
        # If TF model available, run prediction + Grad-CAM
        try:
            if tf_model is not None and tf_labels is not None:
                import numpy as _np
                from PIL import Image
                # Preprocess
                img = Image.open(image_path).convert('RGB').resize((224, 224))
                arr = _np.array(img)
                x = _np.expand_dims(arr, axis=0)
                x = _tf.keras.applications.mobilenet_v2.preprocess_input(x.astype(_np.float32))
                preds = tf_model.predict(x)[0]
                idx = int(_np.argmax(preds))
                conf = float(preds[idx])
                label = tf_labels[idx]

                # Grad-CAM
                try:
                    last_conv = None
                    # try to find a last conv layer name
                    for layer in reversed(tf_model.layers):
                        if 'conv' in layer.name:
                            last_conv = layer.name
                            break
                    if last_conv is None:
                        last_conv = tf_model.layers[-3].name

                    grad_model = _tf.keras.models.Model([tf_model.inputs], [tf_model.get_layer(last_conv).output, tf_model.output])
                    import tensorflow as _tf2
                    with _tf2.GradientTape() as tape:
                        conv_outputs, predictions = grad_model(x)
                        loss = predictions[:, idx]
                    grads = tape.gradient(loss, conv_outputs)
                    pooled_grads = _tf2.reduce_mean(grads, axis=(0, 1, 2))
                    conv_outputs = conv_outputs[0]
                    heatmap = conv_outputs @ pooled_grads[..., _np.newaxis]
                    heatmap = _np.squeeze(heatmap)
                    heatmap = _np.maximum(heatmap, 0)
                    heatmap /= (heatmap.max() + 1e-9)

                    # overlay heatmap using OpenCV if available, else use PIL
                    try:
                        import cv2 as _cv2
                        img_full = _cv2.imread(image_path)
                        hm = _np.uint8(255 * _np.clip(_np.expand_dims(_np.array(_tf2.image.resize(_np.expand_dims(heatmap, -1), (img_full.shape[0], img_full.shape[1]))[:, :, 0]), 0), 0, 1))
                        hm = _cv2.applyColorMap(hm[0], _cv2.COLORMAP_JET)
                        overlay = _cv2.addWeighted(img_full, 0.6, hm, 0.4, 0)
                        _, buffer = _cv2.imencode('.jpg', overlay)
                        import base64 as _base64
                        gradcam_b64 = _base64.b64encode(buffer.tobytes()).decode('ascii')
                    except Exception:
                        # fallback PIL overlay
                        try:
                            import io as _io, base64 as _base64
                            from PIL import Image as _Image
                            orig = _Image.open(image_path).convert('RGB')
                            hm_img = _Image.fromarray(_np.uint8(255 * heatmap)).resize(orig.size).convert('L')
                            hm_color = _Image.new('RGBA', orig.size)
                            # colorize via putpalette not straightforward, so blend grayscale as red
                            hm_color = _Image.merge('RGBA', (_Image.new('L', orig.size, 255), hm_img, _Image.new('L', orig.size, 0), hm_img))
                            overlay = _Image.blend(orig.convert('RGBA'), hm_color, alpha=0.4).convert('RGB')
                            buf = _io.BytesIO()
                            overlay.save(buf, format='JPEG')
                            gradcam_b64 = _base64.b64encode(buf.getvalue()).decode('ascii')
                        except Exception:
                            gradcam_b64 = None
                except Exception:
                    gradcam_b64 = None

                return {'label': label, 'confidence': conf, 'gradcam_b64': gradcam_b64}

            # fallback to detector
            pd = plant_disease_detector.predict_disease(image_path)
            return {'label': pd.get('disease', 'unknown'), 'confidence': pd.get('confidence', 0), 'gradcam_b64': None}
        except Exception as e:
            print(f'Model prediction error: {e}')
            pd = plant_disease_detector._get_mock_prediction() if hasattr(plant_disease_detector, '_get_mock_prediction') else {'disease': 'unknown', 'confidence': 0}
            return {'label': pd.get('disease', 'unknown'), 'confidence': pd.get('confidence', 0), 'gradcam_b64': None}
    

    # Simple user storage (in production, use a database)
    users_db = {}

    # Helper functions
    def login_required(f):
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function

    def load_users():
        """Load users from file (simple persistence)"""
        try:
            with open('data/users.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_users(users):
        """Save users to file"""
        os.makedirs('data', exist_ok=True)
        with open('data/users.json', 'w') as f:
            json.dump(users, f, indent=2)

    # Load users on startup
    users_db = load_users()

    # Authentication Routes
    @app.route('/login')
    def login():
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('login.html')

    @app.route('/login', methods=['POST'])
    def login_post():
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        # Check if user exists and password matches
        user = users_db.get(email)
        if user and user['password'] == password:
            session['user_id'] = user['id']
            session['user_name'] = user['firstName'] + ' ' + user['lastName']
            session['user_email'] = email
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    @app.route('/register')
    def register():
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('register.html')

    @app.route('/register', methods=['POST'])
    def register_post():
        data = request.get_json()

        # Validate required fields
        required_fields = ['firstName', 'lastName', 'email', 'password', 'confirmPassword', 'phone', 'location']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({'success': False, 'message': f'{field} is required'}), 400

        email = data['email'].strip().lower()
        password = data['password']
        confirm_password = data['confirmPassword']

        # Check if passwords match
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'}), 400

        # Check password length
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters long'}), 400

        # Check if user already exists
        if email in users_db:
            return jsonify({'success': False, 'message': 'Email already registered'}), 409

        # Create new user
        user_id = str(len(users_db) + 1)
        user_data = {
            'id': user_id,
            'firstName': data['firstName'].strip(),
            'lastName': data['lastName'].strip(),
            'email': email,
            'password': password,  # In production, hash this password!
            'phone': data['phone'].strip(),
            'location': data['location'],
            'farmSize': data.get('farmSize', ''),
            'registrationDate': datetime.now().isoformat(),
            'lastLogin': None
        }

        users_db[email] = user_data
        save_users(users_db)

        # Don't auto-login after registration - redirect to login page
        return jsonify({'success': True, 'message': 'Registration successful'})

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    # Protected Routes
    @app.route('/')
    @login_required
    def index():
        return render_template('index.html')

    @app.route('/voice-recognition')
    @login_required
    def voice_recognition_page():
        return render_template('voice_recognition.html')

    @app.route('/weather')
    @login_required
    def weather():
        return render_template('weather.html')

    @app.route('/plant-scan')
    @login_required
    def plant_scan():
        return render_template('plant_scan.html')

    @app.route('/api/analyze-plant', methods=['POST'])
    @login_required
    def analyze_plant():
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400

            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image selected'}), 400

            # Accept any uploaded file and validate that it's an image by attempting to open with Pillow.
            # This avoids relying on filename extensions and allows more image formats.
            try:
                from PIL import Image
                # Attempt to open/verify the uploaded file stream as an image
                file.stream.seek(0)
                img = Image.open(file.stream)
                img.verify()  # Verify will raise if not a valid image
                # After verify, seek back to beginning so the file can be saved
                file.stream.seek(0)
            except Exception:
                return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400

            # Save uploaded file temporarily (preserve original extension if present)
            import tempfile
            import os
            _, ext = os.path.splitext(file.filename or '')
            # If no extension, default to .img to preserve a suffix
            suffix = ext if ext else '.img'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name


            try:
                # Also store uploaded images into a collected-images dataset for future use
                try:
                    import shutil, random
                    collected_dir = os.path.join('data', 'collected_images')
                    os.makedirs(collected_dir, exist_ok=True)
                    # create a safe filename with timestamp to avoid collisions
                    import time
                    safe_name = f"upload_{int(time.time())}{os.path.splitext(file.filename or '')[1]}"
                    dest_path = os.path.join(collected_dir, safe_name)
                    shutil.copy(temp_file_path, dest_path)
                except Exception as e:
                    print(f"Warning: could not copy uploaded file to collected_images: {e}")

                # Attempt a lightweight similarity search against a sampled subset of the plant dataset
                similar_results = []
                try:
                    from PIL import Image
                    import numpy as _np

                    def _histogram_vector(img_path, bins=8, size=(256, 256)):
                        with Image.open(img_path) as im:
                            im = im.convert('RGB')
                            im = im.resize(size)
                            arr = _np.array(im)
                        # compute per-channel histogram and normalize
                        h = []
                        for c in range(3):
                            channel = arr[:, :, c]
                            hist, _ = _np.histogram(channel, bins=bins, range=(0, 256))
                            h.append(hist.astype(_np.float32) / (hist.sum() + 1e-9))
                        return _np.concatenate(h)

                    uploaded_vec = _histogram_vector(temp_file_path)

                    # find candidate files (sample to limit compute)
                    dataset_root = os.path.join('data', 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)', 'train')
                    candidates = []
                    if os.path.isdir(dataset_root):
                        for root_dir, _, files in os.walk(dataset_root):
                            for fn in files:
                                if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif')):
                                    candidates.append(os.path.join(root_dir, fn))

                    max_samples = 300
                    if candidates:
                        if len(candidates) > max_samples:
                            candidates = random.sample(candidates, max_samples)

                        scored = []
                        for cpath in candidates:
                            try:
                                vec = _histogram_vector(cpath)
                                # histogram intersection as similarity
                                intersect = float(_np.minimum(uploaded_vec, vec).sum())
                                scored.append((intersect, cpath))
                            except Exception:
                                continue

                        scored.sort(reverse=True, key=lambda x: x[0])
                        # return top 3 similar images (relative paths)
                        for score, path_sim in scored[:3]:
                            # derive class from folder name (parent of file)
                            cls = os.path.basename(os.path.dirname(path_sim))
                            similar_results.append({'path': path_sim.replace('\\', '/'), 'class': cls, 'score': round(score, 4)})
                except Exception as e:
                    print(f"Similarity search skipped: {e}")

                # Analyze the plant image
                result = plant_disease_detector.predict_disease(temp_file_path)

                response = {
                    'success': True,
                    'disease': result['disease'],
                    'confidence': result['confidence'],
                    'description': result['description'],
                    'symptoms': result['symptoms'],
                    'treatment': result['treatment'],
                    'prevention': result['prevention'],
                    'severity': result['severity'],
                    'recommendations': result['recommendations']
                }

                if similar_results:
                    response['similar_dataset_images'] = similar_results

                return jsonify(response)

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            print(f"Plant analysis error: {e}")
            return jsonify({
                'error': 'Failed to analyze image. Please try again.',
                'details': str(e)
            }), 500

    @app.route('/voice-assistant')
    @login_required
    def voice_assistant():
        return render_template('voice_assistant.html')

    @app.route('/voice-query', methods=['POST'])
    def voice_query():
        try:
            data = request.get_json()
            if not data or 'audio' not in data:
                return jsonify({'error': 'No audio data provided'}), 400

            # Process the audio data
            audio_base64 = data['audio']
            result = voice_recognition.process_audio(audio_base64)

            # If successful, enhance with AI farming advisor
            if result.get('success') and result.get('translation'):
                try:
                    enhanced_advice = farming_advisor.analyze_farming_query(result['translation'])
                    result['enhanced_advice'] = enhanced_advice['recommendations'][:3]
                    result['crop_info'] = enhanced_advice.get('crop_advice', {})
                    result['market_insights'] = enhanced_advice.get('market_insights', '')
                except Exception as e:
                    print(f"Enhanced advice error: {e}")
                    result['enhanced_advice'] = []

            return jsonify(result)
        except Exception as e:
            print(f"Error processing voice query: {e}")
            return jsonify({'error': 'Internal server error'}), 500

    @app.route('/test-voice')
    def test_voice():
        return jsonify({
            'message': 'Voice recognition system is active and ready.',
            'success': True
        })

    # IVR Routes for Nexmo/Vonage Integration
    @app.route('/ivr/voice', methods=['GET', 'POST'])
    def ivr_voice():
        """Handle incoming voice calls for AgriNathi IVR system using Nexmo/Vonage"""
        try:
            # For Nexmo/Vonage, we need to return NCCO (Nexmo Call Control Object)
            ncco = [
                {
                    "action": "talk",
                    "language": "en-US",
                    "style": "0",
                    "premium": False,
                    "text": "Welcome to AgriNathi, your agricultural voice assistant. Sawubona, wamukelekile ku-AgriNathi, umsizi wakho wezolimo ngezwi."
                },
                {
                    "action": "talk",
                    "language": "en-US",
                    "style": "0",
                    "premium": False,
                    "text": "Press 1 for weather information. Cindezela u-1 ukuze uthole ulwazi ngesimo sezulu."
                },
                {
                    "action": "talk",
                    "language": "en-US",
                    "style": "0",
                    "premium": False,
                    "text": "Press 2 for farming advice. Cindezela u-2 ukuze uthole izeluleko zezolimo."
                },
                {
                    "action": "talk",
                    "language": "en-US",
                    "style": "0",
                    "premium": False,
                    "text": "Press 3 for market prices. Cindezela u-3 ukuze uthole amanani emakethe."
                },
                {
                    "action": "talk",
                    "language": "en-US",
                    "style": "0",
                    "premium": False,
                    "text": "Press 4 to speak with an expert. Cindezela u-4 ukuze ukhulume nochwepheshe."
                },
                {
                    "action": "input",
                    "eventUrl": [f"{request.host_url.rstrip('/')}/ivr/menu"],
                    "type": ["dtmf"],
                    "dtmf": {
                        "maxDigits": 1,
                        "submitOnHash": False
                    }
                }
            ]
            return jsonify(ncco)
        except Exception as e:
            print(f"IVR voice error: {e}")
            return jsonify([{
                "action": "talk",
                "text": "Sorry, there was an error with the voice system. Please try again later."
            }])

    @app.route('/ivr/menu', methods=['POST'])
    def ivr_menu():
        """Handle IVR menu selections for Nexmo/Vonage"""
        try:
            data = request.get_json()
            dtmf_digits = data.get('dtmf', {}).get('digits', '')

            ncco = []

            if dtmf_digits == '1':
                # Weather information
                ncco = [
                    {
                        "action": "talk",
                        "text": "Getting weather information for Johannesburg. The current temperature is 24 degrees Celsius with partly cloudy conditions. Agricultural advice: Good conditions for fieldwork today."
                    },
                    {
                        "action": "talk",
                        "text": "To return to the main menu, press any key."
                    },
                    {
                        "action": "input",
                        "eventUrl": [f"{request.host_url.rstrip('/')}/ivr/voice"],
                        "type": ["dtmf"],
                        "dtmf": {
                            "maxDigits": 1,
                            "submitOnHash": False
                        }
                    }
                ]

            elif dtmf_digits == '2':
                # Farming advice
                ncco = [
                    {
                        "action": "talk",
                        "text": "Here is some farming advice for maize cultivation. Ensure proper irrigation and monitor for pests regularly. Apply organic fertilizers during the growing season."
                    },
                    {
                        "action": "talk",
                        "text": "To return to the main menu, press any key."
                    },
                    {
                        "action": "input",
                        "eventUrl": [f"{request.host_url.rstrip('/')}/ivr/voice"],
                        "type": ["dtmf"],
                        "dtmf": {
                            "maxDigits": 1,
                            "submitOnHash": False
                        }
                    }
                ]

            elif dtmf_digits == '3':
                # Market prices
                ncco = [
                    {
                        "action": "talk",
                        "text": "Current market prices in Johannesburg. Maize is selling at 4,500 rand per ton. Tomatoes are at 12,000 rand per ton."
                    },
                    {
                        "action": "talk",
                        "text": "To return to the main menu, press any key."
                    },
                    {
                        "action": "input",
                        "eventUrl": [f"{request.host_url.rstrip('/')}/ivr/voice"],
                        "type": ["dtmf"],
                        "dtmf": {
                            "maxDigits": 1,
                            "submitOnHash": False
                        }
                    }
                ]

            elif dtmf_digits == '4':
                # Connect to expert
                ncco = [
                    {
                        "action": "talk",
                        "text": "Connecting you to an agricultural expert. Please hold while we connect your call."
                    },
                    {
                        "action": "connect",
                        "endpoint": [
                            {
                                "type": "phone",
                                "number": "27716669966"  # Replace with actual expert number
                            }
                        ]
                    }
                ]

            else:
                # Invalid option or timeout
                ncco = [
                    {
                        "action": "talk",
                        "text": "Invalid option selected. Returning to main menu."
                    },
                    {
                        "action": "input",
                        "eventUrl": [f"{request.host_url.rstrip('/')}/ivr/voice"],
                        "type": ["dtmf"],
                        "dtmf": {
                            "maxDigits": 1,
                            "submitOnHash": False
                        }
                    }
                ]

            return jsonify(ncco)

        except Exception as e:
            print(f"IVR menu error: {e}")
            return jsonify([{
                "action": "talk",
                "text": "Sorry, there was an error processing your request. Please try again."
            }])

    @app.route('/ivr/status', methods=['GET'])
    def ivr_status():
        """Check IVR system status"""
        return jsonify({
            'status': 'active',
            'provider': 'vonage',
            'vonage_available': VONAGE_AVAILABLE,
            'message': 'AgriNathi IVR system is operational with Vonage',
            'features': ['weather', 'farming_advice', 'market_prices', 'expert_connect'],
            'api_endpoint': 'https://api.nexmo.com/v1/calls',
            'supported_languages': ['en-US', 'zu-ZA']
        })

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)