#!/usr/bin/env python3
"""Web UI for Scripture Translation System with factory pattern.

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
from typing import Dict, List, Optional
import io

from config import Config
from services.translation_service import TranslationService
from services.terminology_service import TerminologyService
from evaluation import ScriptureEvaluator
from utils.logger import get_logger, configure_logging
from exceptions import ModelNotInitializedError, LanguageNotSupportedError

logger = get_logger(__name__)


def create_app(config_override: Optional[Dict] = None) -> Flask:
    """Flask app factory.

    Args:
        config_override: Optional configuration overrides.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)
    CORS(app)

    # Initialize services once in app context
    app.extensions = {
        'translation_service': TranslationService(),
        'terminology_service': TerminologyService(),
        'evaluator': ScriptureEvaluator(),
    }

    @app.before_request
    def ensure_initialized():
        """Ensure services are initialized before handling requests."""
        try:
            translation_service = app.extensions['translation_service']
            translation_service.initialize()
        except Exception as exc:
            logger.error(f"Service initialization failed: {exc}")
            raise ModelNotInitializedError(
                "Failed to initialize translation system. Check logs for details."
            ) from exc

    # ========================================================================
    # API ENDPOINTS - TRANSLATION
    # ========================================================================

    @app.route('/api/v1/translate', methods=['POST'])
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

            translation_service = app.extensions['translation_service']
            result = translation_service.translate_verse(
                source_text=source_text,
                source_lang=source_lang,
                target_lang=target_lang,
                num_beams=num_beams,
            )

            return jsonify({'success': True, 'result': result.to_dict()})
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/translate/batch', methods=['POST'])
    def translate_batch():
        """Translate multiple verses"""
        try:
            data = request.json
            verses = data.get('verses', [])
            source_lang = data.get('source_lang', 'eng_Latn')
            target_lang = data.get('target_lang', 'spa_Latn')

            if not verses:
                return jsonify({'error': 'No verses provided'}), 400

            translation_service = app.extensions['translation_service']
            results = translation_service.translate_batch(
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

    # ========================================================================
    # API ENDPOINTS - TERMINOLOGY
    # ========================================================================

    @app.route('/api/v1/terminology/add', methods=['POST'])
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

            terminology_service = app.extensions['terminology_service']
            success = terminology_service.add_term(
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
        except LanguageNotSupportedError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Add term error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/terminology/lookup', methods=['GET'])
    def lookup_term():
        """Look up a term in the database"""
        try:
            english_term = request.args.get('english_term', '')
            target_lang = request.args.get('target_lang', '')

            if not english_term or not target_lang:
                return jsonify({'error': 'Missing parameters'}), 400

            terminology_service = app.extensions['terminology_service']
            result = terminology_service.get_with_confidence(english_term, target_lang)

            if result:
                target_term, confidence = result
                return jsonify({
                    'success': True,
                    'english_term': english_term,
                    'target_term': target_term,
                    'confidence': confidence,
                })
            else:
                return jsonify({'success': False, 'message': 'Term not found'})
        except LanguageNotSupportedError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Lookup error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/terminology/extract', methods=['POST'])
    def extract_terms():
        """Extract theological terms from text"""
        try:
            data = request.json
            text = data.get('text', '')
            target_lang = data.get('target_lang', '')

            if not text:
                return jsonify({'error': 'No text provided'}), 400

            terminology_service = app.extensions['terminology_service']
            canonical = terminology_service.extract_terms(text, target_lang) if target_lang else {}

            return jsonify({'success': True, 'canonical': canonical})
        except LanguageNotSupportedError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Extract error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/terminology/conflicts', methods=['GET'])
    def get_conflicts():
        """Get terminology conflicts"""
        try:
            terminology_service = app.extensions['terminology_service']
            conflicts = terminology_service.get_conflicts()

            return jsonify({
                'success': True,
                'conflicts': conflicts,
                'count': len(conflicts),
            })
        except Exception as e:
            logger.error(f"Get conflicts error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/terminology/resolve', methods=['POST'])
    def resolve_conflict():
        """Resolve a terminology conflict"""
        try:
            data = request.json
            english_term = data.get('english_term', '')
            target_lang = data.get('target_lang', '')
            chosen_term = data.get('chosen_term', '')

            if not all([english_term, target_lang, chosen_term]):
                return jsonify({'error': 'Missing required fields'}), 400

            terminology_service = app.extensions['terminology_service']
            terminology_service.resolve_conflict(english_term, target_lang, chosen_term)

            return jsonify({
                'success': True,
                'message': f'Resolved: {english_term} → {chosen_term}',
            })
        except LanguageNotSupportedError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Resolve conflict error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/terminology/stats', methods=['GET'])
    def get_stats():
        """Get terminology database statistics"""
        try:
            terminology_service = app.extensions['terminology_service']
            stats = terminology_service.get_statistics()

            return jsonify({'success': True, 'statistics': stats})
        except Exception as e:
            logger.error(f"Get stats error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/terminology/export', methods=['GET'])
    def export_terminology():
        """Export terminology database"""
        try:
            target_lang = request.args.get('target_lang', '')

            if not target_lang:
                return jsonify({'error': 'target_lang required'}), 400

            terminology_service = app.extensions['terminology_service']
            terms = terminology_service.db.get_all_terms_for_language(target_lang)

            export_data = {
                'language': target_lang,
                'timestamp': datetime.now().isoformat(),
                'terms': [
                    {
                        'english': eng,
                        'translation': tgt,
                        'confidence': conf,
                    }
                    for eng, (tgt, conf) in sorted(terms.items())
                ]
            }

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

    # ========================================================================
    # API ENDPOINTS - EVALUATION
    # ========================================================================

    @app.route('/api/v1/evaluate/batch', methods=['POST'])
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

            evaluator = app.extensions['evaluator']
            metrics = evaluator.evaluate_batch(hypotheses, references, target_lang)

            return jsonify({'success': True, 'metrics': metrics.to_dict()})
        except Exception as e:
            logger.error(f"Batch evaluation error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    # ========================================================================
    # API ENDPOINTS - SYSTEM INFO
    # ========================================================================

    @app.route('/api/v1/system/info', methods=['GET'])
    def system_info():
        """Get system information"""
        try:
            terminology_service = app.extensions['terminology_service']
            stats = terminology_service.get_statistics()

            return jsonify({
                'success': True,
                'model': Config.MODEL_NAME,
                'device': Config.get_device(),
                'languages': Config.LANGUAGE_CODES,
                'terminology_stats': stats,
            })
        except Exception as e:
            logger.error(f"System info error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/v1/system/save', methods=['POST'])
    def save_system():
        """Save system state"""
        try:
            terminology_service = app.extensions['terminology_service']
            terminology_service.save()

            return jsonify({'success': True, 'message': 'System saved'})
        except Exception as e:
            logger.error(f"Save error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    # ========================================================================
    # WEB PAGES
    # ========================================================================

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

    # ========================================================================
    # ERROR HANDLERS
    # ========================================================================

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({'error': 'Server error'}), 500

    return app


if __name__ == '__main__':
    Config.ensure_dirs()
    configure_logging()

    logger.info("Starting Scripture Translation Web Server...")
    logger.info("Visit: http://localhost:5000")

    app = create_app()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
    )
