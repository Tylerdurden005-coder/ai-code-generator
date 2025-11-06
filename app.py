"""
AI CodeGen Pro - BACKEND FROM SCRATCH
VERSION: 1.0.2 (Scratch Substitute)

This version is designed to run with NO external dependencies
other than Flask and Flask-CORS.

It does NOT import any of the generator files (ml_generator.py, etc.)
and instead uses "Dummy" generators. This is to ensure the
backend server can run even if libraries like TensorFlow, PyTorch,
or scikit-learn are not installed.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size
app.config['JSON_SORT_KEYS'] = False

logger.info("="*50)
logger.info("Starting in 'Scratch Substitute' mode.")
logger.info("All generators are Dummies.")
logger.info("This server will run, but code generation will return errors.")
logger.info("="*50)

# --- SCRATCH SUBSTITUTE GENERATORS ---
# These classes are defined directly here and have NO dependencies.

class DummyGenerator:
    """
    This is a "scratch substitute" generator.
    Its only purpose is to exist so the app can run.
    It returns a helpful error message when called.
    """
    def __init__(self, name="Unknown"):
        self.name = name
        logger.warning(f"Using DUMMY generator for {self.name}")

    def generate(self, project_name, description, options, **kwargs):
        error_message = (
            f"Generator '{self.name}' is a 'scratch substitute'. "
            "The backend is running, but the real generation "
            "libraries (e.g., tensorflow, scikit-learn) "
            "are not installed or not imported."
        )
        logger.error(f"Generate called on DUMMY generator: {self.name}")
        return {
            'main_code': f"# ERROR: {error_message}",
            'utils_code': f"# ERROR: {error_message}",
            'requirements': f"# ERROR: {error_message}",
            'readme': f"# ERROR: {error_message}",
            'error': error_message
        }

# --- Validator and Preprocessor Substitutes ---

def validate_request(data):
    """
    Dummy validator. In this scratch build, we'll
    just check for the most basic fields.
    """
    logger.warning("Using DUMMY validator.")
    if not data.get('project_name'):
        return False, "Project name is missing"
    if not data.get('description'):
        return False, "Description is missing"
    if not data.get('category'):
        return False, "Category is missing"
    return True, None  # Always validate true in dummy mode

def preprocess_description(desc):
    """
    Dummy preprocessor. Just returns the description as-is.
    """
    logger.warning("Using DUMMY preprocessor.")
    return desc  # Return as-is in dummy mode


# Initialize generators with our Dummy/Scratch versions
generators = {
    'ml': DummyGenerator("ML"),
    'dl': DummyGenerator("DL"),
    'nlp': DummyGenerator("NLP"),
    'rl': DummyGenerator("RL"),
    'dsa': DummyGenerator("DSA"),
    'web': DummyGenerator("Web")
}

# ========================================
# Health Check Endpoint
# ========================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Check API health status"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.2-SCRATCH',
        'timestamp': datetime.now().isoformat(),
        'available_categories': list(generators.keys()),
        'mode': 'scratch_substitute'
    }), 200


# ========================================
# Main Code Generation Endpoint
# ========================================
@app.route('/api/generate', methods=['POST'])
def generate_code():
    """
    Generate code based on user requirements.
    In scratch mode, this will always return a
    JSON response with an error message.
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            logger.warning("Request received with no data")
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate request
        is_valid, error_message = validate_request(data)
        if not is_valid:
            logger.warning(f"Invalid request: {error_message}")
            return jsonify({'error': error_message}), 400
        
        # Extract parameters
        project_name = data.get('project_name')
        description = data.get('description')
        category = data.get('category', 'ml')
        
        logger.info(f"Attempting code generation for: {project_name}, category: {category}")
        
        # Preprocess description
        processed_description = preprocess_description(description)
        
        # Select appropriate generator
        generator = generators.get(category)
        if not generator:
            logger.error(f"Invalid category specified: {category}")
            return jsonify({'error': f'Invalid category: {category}'}), 400
        
        # Generate code (will return the dummy error message)
        result = generator.generate(
            project_name=project_name,
            description=processed_description,
            options=data
        )
        
        # Add metadata
        result['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'category': category,
            'project_name': project_name,
            'mode': 'scratch_substitute'
        }
        
        logger.info(f"Dummy response sent for: {project_name}")
        
        # We return 200 OK because the API call *worked*.
        # The frontend will see the 'error' key inside the JSON.
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in /api/generate: {str(e)}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# ========================================
# Other Endpoints (stubs)
# ========================================
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models for each category"""
    models = {
        'ml': ['Model A', 'Model B'],
        'dl': ['CNN', 'RNN'],
        'nlp': ['BERT', 'GPT'],
        'rl': ['Q-Learning', 'DQN']
    }
    return jsonify(models), 200

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get available quick start templates"""
    templates = [
        {
            'id': 'image-classifier',
            'name': 'Image Classifier',
            'description': 'CNN-based image classification',
            'category': 'dl'
        },
        {
            'id': 'sentiment-analysis',
            'name': 'Sentiment Analysis',
            'description': 'Text sentiment detection',
            'category': 'nlp'
        }
    ]
    return jsonify(templates), 200

@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    return jsonify({'error': 'Analysis is disabled in scratch mode'}), 400

# ========================================
# Error Handlers
# ========================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

# ========================================
# Main Entry Point
# ========================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True') == 'True'
    
    logger.info(f"Starting AI CodeGen Pro SCRATCH Backend on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Access health check at: http://localhost:{port}/api/health")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )