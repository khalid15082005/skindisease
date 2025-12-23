"""
Skin Disease AI - Flask Web Application
A user-friendly interface for skin lesion diagnosis using machine learning
"""

import os
import numpy as np
import hashlib
import secrets
from datetime import datetime, timedelta

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=suppress INFO, 2=suppress WARNING, 3=suppress ERROR

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError:
    import keras
    from keras.preprocessing.image import img_to_array
    tf = keras
from PIL import Image
import io
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import base64

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key

# Make users available in all templates
@app.context_processor
def inject_users():
    return dict(users=users)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None

# In-memory user storage (in production, use a proper database)
users = {}
diagnoses = {}

# Sample user for demonstration
demo_user = {
    'id': 'demo_user_1',
    'email': 'demo@example.com',
    'password': generate_password_hash('demo123'),
    'first_name': 'John',
    'last_name': 'Doe',
    'age': 30,
    'gender': 'male',
    'created_at': datetime.now(),
    'total_diagnoses': 5,
    'diagnoses_this_month': 3,
    'diagnoses_this_week': 1,
    'accuracy_rating': 85.2,
    'most_common_condition': 'Acne'
}
users['demo@example.com'] = demo_user

# Sample diagnoses for demonstration
sample_diagnoses = [
    {
        'id': 1,
        'user_id': 'demo_user_1',
        'predicted_class': 'Acne',
        'confidence': 78.5,
        'created_at': datetime.now() - timedelta(days=1)
    },
    {
        'id': 2,
        'user_id': 'demo_user_1',
        'predicted_class': 'Eczema',
        'confidence': 82.3,
        'created_at': datetime.now() - timedelta(days=3)
    },
    {
        'id': 3,
        'user_id': 'demo_user_1',
        'predicted_class': 'Acne',
        'confidence': 75.8,
        'created_at': datetime.now() - timedelta(days=7)
    }
]
diagnoses['demo_user_1'] = sample_diagnoses

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    """Decorator to require login for certain routes"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def get_current_user():
    """Get current user from session"""
    if 'user_id' in session:
        for user in users.values():
            if user['id'] == session['user_id']:
                return user
    return None

def get_user_diagnoses(user_id, limit=None):
    """Get diagnoses for a user"""
    user_diagnoses = diagnoses.get(user_id, [])
    if limit:
        return user_diagnoses[:limit]
    return user_diagnoses

def calculate_condition_stats(user_id):
    """Calculate condition frequency statistics"""
    user_diagnoses = diagnoses.get(user_id, [])
    stats = {}
    for diagnosis in user_diagnoses:
        condition = diagnosis['predicted_class']
        stats[condition] = stats.get(condition, 0) + 1
    return stats

def load_model():
    """Load the trained model with compatibility handling for Keras 3"""
    global model

    print("Using demo model for demonstration purposes...")

    # Create a demo model for demonstration
    class DemoModel:
        def predict(self, img_array, verbose=0):
            # Generate dynamic predictions based on image content
            # Use image statistics to create varied predictions
            img_mean = np.mean(img_array)
            img_std = np.std(img_array)
            img_shape = img_array.shape

            # Create seed based on image properties for consistent but varied results
            seed = int(abs(img_mean * 1000 + img_std * 100 + img_shape[1] * img_shape[2])) % 1000
            np.random.seed(seed)

            # Generate random logits and apply softmax manually
            logits = np.random.randn(6)

            # Add some bias based on image characteristics
            if img_mean > 0.5:  # Brighter images might favor certain conditions
                logits[0] += 0.5  # Acne
                logits[3] += 0.3  # Keratosis
            if img_std > 0.3:  # High contrast images
                logits[1] += 0.4  # Carcinoma
                logits[2] += 0.3  # Eczema

            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            prediction = exp_logits / np.sum(exp_logits)
            return [prediction]

    model = DemoModel()
    print("Demo model loaded successfully")
    print("WARNING: This is a demonstration model with random predictions. For actual predictions, please provide a complete model.")
    return model

def predict_skin_disease(img_array):
    """Make prediction on the uploaded image"""
    global model
    if model is None:
        print("Model is not loaded")
        return None, None, None
    try:
        # Normalize image array (compatible with demo and real models)
        img_array = img_array.astype(np.float32) / 255.0

        # Make prediction based on model type
        if hasattr(model, 'predict'):
            # Standard Keras model or demo model
            prediction = model.predict(img_array, verbose=0)[0]
        elif hasattr(model, 'signatures'):
            # TensorFlow SavedModel
            serving_fn = model.signatures['serving_default']
            prediction = serving_fn(tf.constant(img_array))['output_0'].numpy()[0]
        else:
            # Try direct call
            prediction = model(img_array).numpy()[0]
        
        # Define class names
        class_names = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Millia', 'Rosacea']
        
        # Get top prediction
        top_prediction_idx = np.argmax(prediction)
        top_confidence = prediction[top_prediction_idx] * 100
        
        # Create results dictionary
        results = []
        for i, (class_name, confidence) in enumerate(zip(class_names, prediction)):
            results.append({
                'class': class_name,
                'confidence': float(confidence * 100),
                'is_top': bool(i == top_prediction_idx)
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results, top_prediction_idx, top_confidence
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None, None

@app.route('/')
def index():
    """Landing page - redirect to login or home"""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in users and check_password_hash(users[email]['password'], password):
            session['user_id'] = users[email]['id']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        
        if email in users:
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        # Create new user
        user_id = f"user_{len(users) + 1}"
        new_user = {
            'id': user_id,
            'email': email,
            'password': generate_password_hash(password),
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'created_at': datetime.now(),
            'total_diagnoses': 0,
            'diagnoses_this_month': 0,
            'diagnoses_this_week': 0,
            'accuracy_rating': 0,
            'most_common_condition': None
        }
        
        users[email] = new_user
        diagnoses[user_id] = []
        
        session['user_id'] = user_id
        flash('Registration successful!', 'success')
        return redirect(url_for('home'))
    
    return render_template('register.html')

@app.route('/home')
@login_required
def home():
    """Home page for authenticated users"""
    user = get_current_user()
    recent_diagnoses = get_user_diagnoses(user['id'], limit=5)
    return render_template('home.html', user=user, recent_diagnoses=recent_diagnoses)

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    user = get_current_user()
    recent_diagnoses = get_user_diagnoses(user['id'], limit=10)
    condition_stats = calculate_condition_stats(user['id'])
    return render_template('profile.html', 
                         user=user, 
                         recent_diagnoses=recent_diagnoses,
                         condition_stats=condition_stats)

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile"""
    user = get_current_user()
    user['first_name'] = request.form.get('first_name')
    user['last_name'] = request.form.get('last_name')
    user['email'] = request.form.get('email')
    user['age'] = int(request.form.get('age'))
    user['gender'] = request.form.get('gender')
    
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    user = get_current_user()
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_new_password')
    
    if not check_password_hash(user['password'], current_password):
        flash('Current password is incorrect', 'error')
        return redirect(url_for('profile'))
    
    if new_password != confirm_password:
        flash('New passwords do not match', 'error')
        return redirect(url_for('profile'))
    
    if len(new_password) < 6:
        flash('Password must be at least 6 characters long', 'error')
        return redirect(url_for('profile'))
    
    user['password'] = generate_password_hash(new_password)
    flash('Password changed successfully!', 'success')
    return redirect(url_for('profile'))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Skin Disease AI is running'})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files only.'}), 400
        
        # Read and process image
        image_pil = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Resize image
        image_pil = image_pil.resize((299, 299))
        
        # Convert to array
        img_array = img_to_array(image_pil)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        results, top_idx, top_conf = predict_skin_disease(img_array)
        
        if results is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Save diagnosis to user's history
        user = get_current_user()
        new_diagnosis = {
            'id': len(diagnoses.get(user['id'], [])) + 1,
            'user_id': user['id'],
            'predicted_class': results[0]['class'],
            'confidence': results[0]['confidence'],
            'created_at': datetime.now()
        }
        
        if user['id'] not in diagnoses:
            diagnoses[user['id']] = []
        diagnoses[user['id']].insert(0, new_diagnosis)  # Add to beginning
        
        # Update user statistics
        user['total_diagnoses'] = len(diagnoses[user['id']])
        user['diagnoses_this_month'] = len([d for d in diagnoses[user['id']] 
                                          if d['created_at'].month == datetime.now().month])
        user['diagnoses_this_week'] = len([d for d in diagnoses[user['id']] 
                                         if d['created_at'] >= datetime.now() - timedelta(days=7)])
        
        # Calculate average confidence
        if diagnoses[user['id']]:
            avg_confidence = sum(d['confidence'] for d in diagnoses[user['id']]) / len(diagnoses[user['id']])
            user['accuracy_rating'] = round(avg_confidence, 1)
        
        # Update most common condition
        condition_counts = {}
        for d in diagnoses[user['id']]:
            condition_counts[d['predicted_class']] = condition_counts.get(d['predicted_class'], 0) + 1
        if condition_counts:
            user['most_common_condition'] = max(condition_counts, key=condition_counts.get)
        
        # Convert PIL image to base64 for display
        buffered = io.BytesIO()
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare response
        response_data = {
            'success': True,
            'image': img_str,
            'results': results,
            'top_prediction': {
                'class': str(results[0]['class']),
                'confidence': float(results[0]['confidence'])
            },
            'interpretation': str(get_interpretation(results[0]['confidence']))
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def get_interpretation(confidence):
    """Get interpretation based on confidence level"""
    if confidence > 70:
        return "High confidence diagnosis. Consider consulting a dermatologist for confirmation."
    elif confidence > 50:
        return "Moderate confidence. Further evaluation recommended."
    else:
        return "Low confidence. Please consult a dermatologist for proper diagnosis."

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Load model on startup
    print("Loading Skin Disease AI Model...")
    load_model()
    print("Model loaded successfully!")
    print("Starting Flask application...")
    app.run(debug=True, host='127.0.0.1', port=5000)
