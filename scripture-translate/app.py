#!/usr/bin/env python3
"""
Web UI for Scripture Translation System

Provides:
- Real-time verse translation
- Terminology database management
- Human evaluation interface
- Translation progress tracking
- Results visualization
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List
import io

from config import Config
from models.base import ScriptureTranslationModel
from models.terminology import TerminologyDB, TermExtractor
from inference import ScriptureTranslator
from evaluation import ScriptureEvaluator, HumanEvaluationInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global state
model_wrapper = None
translator = None
terminology_db = None
evaluator = None


def initialize_system():
    """Initialize the translation system"""
    global model_wrapper, translator, terminology_db, evaluator
    
    logger.info("Initializing translation system...")
    
    # Load model
    model_wrapper = ScriptureTranslationModel(use_lora=False)
    
    # Load terminology database
    terminology_db = TerminologyDB()
    
    # Create translator
    translator = ScriptureTranslator(
        model=model_wrapper.get_model(),
        tokenizer=model_wrapper.get_tokenizer(),
        terminology_db=terminology_db,
        device=Config.get_device(),
        enforce_consistency=True,
    )
    
    # Create evaluator
    evaluator = ScriptureEvaluator(terminology_db)
    
    logger.info("System initialized")


# ============================================================================
# API ENDPOINTS - TRANSLATION
# ============================================================================

@app.route('/api/translate', methods=['POST'])
def translate():
    """Translate a single verse"""
    try:
        data = request.json
        
        source_text = data.get('text', '')
        source_lang = data.get('source_lang', 'eng_Latn')
        target_lang = data.get('target_lang', 'spa_Latn')
        num_beams = data.get('num_beams', 5)
        
        if not source_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Translate
        result = translator.translate_verse(
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            num_beams=num_beams,
        )
        
        return jsonify({
            'success': True,
            'result': result.to_dict(),
        })
    
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/translate/batch', methods=['POST'])
def translate_batch():
    """Translate multiple verses"""
    try:
        data = request.json
        
        verses = data.get('verses', [])
        source_lang = data.get('source_lang', 'eng_Latn')
        target_lang = data.get('target_lang', 'spa_Latn')
        
        if not verses:
            return jsonify({'error': 'No verses provided'}), 400
        
        # Translate batch
        results = translator.translate_batch(
            verses,
            source_lang=source_lang,
            target_lang=target_lang,
            show_progress=False,
        )
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': [r.to_dict() for r in results],
        })
    
    except Exception as e:
        logger.error(f"Batch translation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - TERMINOLOGY
# ============================================================================

@app.route('/api/terminology/add', methods=['POST'])
def add_term():
    """Add a term to the terminology database"""
    try:
        data = request.json
        
        english_term = data.get('english_term', '')
        target_lang = data.get('target_lang', '')
        target_term = data.get('target_term', '')
        confidence = data.get('confidence', 0.9)
        override = data.get('override', False)
        
        if not all([english_term, target_lang, target_term]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        success = terminology_db.add_term(
            english_term,
            target_lang,
            target_term,
            confidence=confidence,
            override=override,
        )
        
        return jsonify({
            'success': success,
            'message': 'Term added' if success else 'Term conflict (not overridden)',
        })
    
    except Exception as e:
        logger.error(f"Add term error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/terminology/lookup', methods=['GET'])
def lookup_term():
    """Look up a term in the database"""
    try:
        english_term = request.args.get('english_term', '')
        target_lang = request.args.get('target_lang', '')
        
        if not english_term or not target_lang:
            return jsonify({'error': 'Missing parameters'}), 400
        
        result = terminology_db.get_with_confidence(english_term, target_lang)
        
        if result:
            target_term, confidence = result
            return jsonify({
                'success': True,
                'english_term': english_term,
                'target_term': target_term,
                'confidence': confidence,
                'usage_count': terminology_db.get_usage_count(english_term, target_lang),
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Term not found',
            })
    
    except Exception as e:
        logger.error(f"Lookup error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/terminology/extract', methods=['POST'])
def extract_terms():
    """Extract theological terms from text"""
    try:
        data = request.json
        text = data.get('text', '')
        target_lang = data.get('target_lang', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        extractor = TermExtractor(terminology_db)
        terms = extractor.extract_theological_terms(text)
        canonical = extractor.get_canonical_terms(text, target_lang) if target_lang else {}
        
        return jsonify({
            'success': True,
            'terms': list(terms),
            'canonical': canonical,
        })
    
    except Exception as e:
        logger.error(f"Extract error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/terminology/conflicts', methods=['GET'])
def get_conflicts():
    """Get terminology conflicts"""
    try:
        conflicts = terminology_db.get_conflicts()
        
        return jsonify({
            'success': True,
            'conflicts': conflicts,
            'count': len(conflicts),
        })
    
    except Exception as e:
        logger.error(f"Get conflicts error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/terminology/resolve', methods=['POST'])
def resolve_conflict():
    """Resolve a terminology conflict"""
    try:
        data = request.json
        
        english_term = data.get('english_term', '')
        target_lang = data.get('target_lang', '')
        chosen_term = data.get('chosen_term', '')
        
        if not all([english_term, target_lang, chosen_term]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        terminology_db.resolve_conflict(english_term, target_lang, chosen_term)
        
        return jsonify({
            'success': True,
            'message': f'Resolved: {english_term} → {chosen_term}',
        })
    
    except Exception as e:
        logger.error(f"Resolve conflict error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/terminology/stats', methods=['GET'])
def get_stats():
    """Get terminology database statistics"""
    try:
        stats = terminology_db.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats,
        })
    
    except Exception as e:
        logger.error(f"Get stats error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/terminology/export', methods=['GET'])
def export_terminology():
    """Export terminology database"""
    try:
        target_lang = request.args.get('target_lang', '')
        
        if not target_lang:
            return jsonify({'error': 'target_lang required'}), 400
        
        # Create JSON
        terms = terminology_db.get_all_terms_for_language(target_lang)
        
        export_data = {
            'language': target_lang,
            'timestamp': datetime.now().isoformat(),
            'terms': [
                {
                    'english': eng,
                    'translation': tgt,
                    'confidence': conf,
                    'usage': terminology_db.get_usage_count(eng, target_lang),
                }
                for eng, (tgt, conf) in sorted(terms.items())
            ]
        }
        
        # Return as JSON file
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        
        return send_file(
            io.BytesIO(json_str.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=f'terminology_{target_lang}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
    
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/terminology/import', methods=['POST'])
def import_terminology():
    """Import terminology from JSON file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        data = json.load(file)
        
        target_lang = data.get('language', '')
        imported = 0
        
        for term_data in data.get('terms', []):
            if term_data.get('approved', False):
                terminology_db.add_term(
                    term_data['english'],
                    target_lang,
                    term_data['translation'],
                    confidence=0.95,
                    override=True,
                )
                imported += 1
        
        terminology_db.save()
        
        return jsonify({
            'success': True,
            'imported': imported,
        })
    
    except Exception as e:
        logger.error(f"Import error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - EVALUATION
# ============================================================================

@app.route('/api/evaluate/bleu', methods=['POST'])
def evaluate_bleu():
    """Calculate BLEU score"""
    try:
        data = request.json
        hypothesis = data.get('hypothesis', '')
        reference = data.get('reference', '')
        
        if not hypothesis or not reference:
            return jsonify({'error': 'Missing hypothesis or reference'}), 400
        
        bleu = evaluator.compute_bleu(hypothesis, reference)
        
        return jsonify({
            'success': True,
            'bleu': float(bleu),
        })
    
    except Exception as e:
        logger.error(f"BLEU evaluation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate/batch', methods=['POST'])
def evaluate_batch():
    """Evaluate a batch of translations"""
    try:
        data = request.json
        
        hypotheses = data.get('hypotheses', [])
        references = data.get('references', [])
        target_lang = data.get('target_lang', 'spa_Latn')
        
        if not hypotheses or not references:
            return jsonify({'error': 'Missing hypotheses or references'}), 400
        
        if len(hypotheses) != len(references):
            return jsonify({'error': 'Hypothesis and reference counts must match'}), 400
        
        metrics = evaluator.evaluate_batch(hypotheses, references, target_lang)
        
        return jsonify({
            'success': True,
            'metrics': metrics.to_dict(),
        })
    
    except Exception as e:
        logger.error(f"Batch evaluation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - SYSTEM INFO
# ============================================================================

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    try:
        return jsonify({
            'success': True,
            'model': Config.MODEL_NAME,
            'device': Config.get_device(),
            'languages': Config.LANGUAGE_CODES,
            'terminology_stats': terminology_db.get_statistics(),
        })
    
    except Exception as e:
        logger.error(f"System info error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/save', methods=['POST'])
def save_system():
    """Save system state"""
    try:
        terminology_db.save()
        
        return jsonify({
            'success': True,
            'message': 'System saved',
        })
    
    except Exception as e:
        logger.error(f"Save error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WEB PAGES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/translate')
def translate_page():
    """Translation interface"""
    return render_template('translate.html')


@app.route('/terminology')
def terminology_page():
    """Terminology management"""
    return render_template('terminology.html')


@app.route('/evaluate')
def evaluate_page():
    """Evaluation interface"""
    return render_template('evaluate.html')


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Initialize system
    initialize_system()
    
    # Run server
    logger.info("Starting Scripture Translation Web Server...")
    logger.info("Visit: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,  # Avoid reloading model
    )
