from flask_restx import Resource
from app.config import MODELS, MODEL_PERFORMANCE
from app.models.model_loader import get_model_tier

def register_categories_route(api):
    """Register categories endpoint"""
    
    @api.route('/categories')
    class Categories(Resource):
        @api.doc('list_categories')
        def get(self):
            """List all available budget categories with model info"""
            categories = []
            for cat, info in MODELS.items():
                perf = MODEL_PERFORMANCE.get(cat, {})
                categories.append({
                    'category': cat,
                    'model': info['best_model_name'],
                    'tier': get_model_tier(perf),
                    'r2': round(perf.get('r2', 0), 3),
                    'samples': info.get('training_stats', {}).get('n_samples', 0)
                })
            return sorted(categories, key=lambda x: x['tier'])
