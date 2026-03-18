"""
🏺 Hieroglyph NLP Pipeline - Flask API Server
Serves the NLP pipeline via REST API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
from models import HieroglyphNLPPipeline

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Global pipeline instance (initialized once on startup)
pipeline = None

# ═══════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════

@app.before_request
def init_pipeline():
    """Initialize pipeline on first request"""
    global pipeline
    if pipeline is None:
        print('\n🚀 Initializing NLP Pipeline on first request...')
        pipeline = HieroglyphNLPPipeline()
        print('✅ Pipeline initialized and ready!\n')


# ═══════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.route('/api/decipher', methods=['POST'])
def decipher():
    """
    POST /api/decipher
    Body: { "codes": ["G17", "N35", "D21"] }
    Response: Full NLP pipeline results
    """
    try:
        data = request.get_json()
        
        if not data or 'codes' not in data:
            return jsonify({'error': 'Missing "codes" field'}), 400
        
        codes = data.get('codes', [])
        
        if not isinstance(codes, list) or len(codes) == 0:
            return jsonify({'error': 'codes must be a non-empty list'}), 400
        
        # Run pipeline
        result = pipeline.process(codes)
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check API status and model availability"""
    try:
        models_loaded = {
            'gardiner_signs': len(pipeline.gardiner_map) > 0,
            'egyptian_dict': len(pipeline.egypt_dict) > 0,
            'intention_map': len(pipeline.intention_map) > 0,
            'spacy_model': pipeline.nlp_spacy is not None,
            'translation_model': pipeline.trans_model is not None,
            'sentiment_model': pipeline.sent_model is not None,
        }
        
        return jsonify({
            'status': 'ready',
            'models': models_loaded,
            'all_loaded': all(models_loaded.values())
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example Gardiner codes"""
    examples = {
        'simple_word': {
            'codes': ['O1'],
            'description': 'House'
        },
        'name_sun': {
            'codes': ['D21', 'N35', 'N5'],
            'description': 'My name is sun'
        },
        'royal_offering': {
            'codes': ['R4', 'X8', 'A42'],
            'description': 'Offering of the king'
        },
        'sun_god': {
            'codes': ['N5', 'R8', 'F35'],
            'description': 'Sun god'
        },
        'son_of_ra': {
            'codes': ['O4', 'N5'],
            'description': 'Son of Ra'
        }
    }
    return jsonify(examples), 200


@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('frontend', filename)
    except:
        return send_from_directory('frontend', 'index.html')


@app.route('/')
def serve_index():
    """Serve main HTML"""
    return send_from_directory('frontend', 'index.html')


# ═══════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ═══════════════════════════════════════════════════════════════════════
# RUN SERVER
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
    
╔══════════════════════════════════════════════════════════════╗
║                  🏺 HIEROGLYPH NLP PIPELINE                  ║
║                   Ancient Egyptian Translator                ║
║                                                              ║
║  Starting server at: http://localhost:5000                  ║
║  Press Ctrl+C to stop                                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,  # Set to True for development
        use_reloader=False
    )
