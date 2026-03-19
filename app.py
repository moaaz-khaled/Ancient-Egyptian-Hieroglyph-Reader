"""
🏺 Hieroglyph NLP Pipeline - Flask API Server
Local GPU version — Seed-X-PPO-7B (bfloat16/float16)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from models import HieroglyphNLPPipeline
import torch

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)


# ═══════════════════════════════════════════════════════════════════════
# INITIALIZATION — مرة واحدة عند start الـ server
# ═══════════════════════════════════════════════════════════════════════

print('\n🚀 Initializing NLP Pipeline...')
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
    """
    try:
        data = request.get_json()
        if not data or 'codes' not in data:
            return jsonify({'error': 'Missing "codes" field'}), 400

        codes = data.get('codes', [])
        if not isinstance(codes, list) or len(codes) == 0:
            return jsonify({'error': 'codes must be a non-empty list'}), 400

        result = pipeline.process(codes)
        return jsonify({'success': True, 'data': result}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check API status and model availability"""
    try:
        # GPU info
        gpu_info = 'CPU only'
        if torch.cuda.is_available():
            name   = torch.cuda.get_device_name(0)
            used   = torch.cuda.memory_allocated(0) / 1e9
            total  = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_info = f'{name} — {used:.1f}/{total:.1f} GB used'

        models_loaded = {
            'gardiner_signs':    len(pipeline.gardiner_map) > 0,
            'egyptian_dict':     len(pipeline.egypt_dict) > 0,
            'intention_map':     len(pipeline.intention_map) > 0,
            'spacy_model':       pipeline.nlp_spacy is not None,
            'translation_model': pipeline.trans_model is not None,
            'sentiment_model':   pipeline.sent_model is not None,
        }
        return jsonify({
            'status':     'ready',
            'model':      'ByteDance-Seed/Seed-X-PPO-7B (local GPU)',
            'gpu':        gpu_info,
            'models':     models_loaded,
            'all_loaded': all(models_loaded.values()),
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example Gardiner codes"""
    examples = {
        'simple_word':    {'codes': ['O1'],               'description': 'House'},
        'name_sun':       {'codes': ['D21', 'N35', 'N5'], 'description': 'My name is sun'},
        'royal_offering': {'codes': ['R4', 'X8', 'A42'],  'description': 'Offering of the king'},
        'sun_god':        {'codes': ['N5', 'R8', 'F35'],  'description': 'Sun god'},
        'son_of_ra':      {'codes': ['O4', 'N5'],         'description': 'Son of Ra'},
    }
    return jsonify(examples), 200


@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('frontend', filename)
    except:
        try:
            return send_from_directory('Design', filename)
        except:
            return send_from_directory('frontend', 'index.html')


@app.route('/')
def serve_index():
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
    gpu_line = 'CPU only'
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_line = f'{name} ({total:.1f} GB)'

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              🏺  HIEROGLYPH NLP PIPELINE                     ║
║          Translation: Seed-X-PPO-7B (Local GPU)             ║
║                                                              ║
║  Server  : http://localhost:5000                             ║
║  GPU     : {gpu_line:<48}║
║  Model   : Seed-X-PPO-7B (bfloat16/float16)                 ║
║  Sentiment: CPU (سايب VRAM كله لـ Seed-X)                   ║
║                                                              ║
║  أول تشغيل: تحميل ~15GB (مرة واحدة بس)                      ║
║  من بعدها: من الـ cache في ثواني                             ║
╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False,   # False عشان PyTorch + Flask مش thread-safe
    )