from flask import request
from flask_restx import Resource, Namespace
from datetime import datetime
import app.config as config
from app.config import logger
from app.models.model_loader import get_model_tier
from app.services.prediction_service import predict_budget_item
from app.services.retraining_service import retrain_models

predict_ns = Namespace('predict', description='Prediction operations')
retrain_ns = Namespace('retrain', description='Model retraining operations')
health_ns = Namespace('health', description='Service health checks')

def create_namespaces():
    """Create API namespaces"""
    return predict_ns, retrain_ns, health_ns

def register_routes(schemas):
    """Register all API routes"""
    
    @health_ns.route('')
    class HealthCheck(Resource):
        @health_ns.doc('health_check')
        @health_ns.marshal_with(schemas['health_output'])
        def get(self):
            """Check service health and model status"""
            return {
                'status': 'healthy',
                'models_loaded': len(config.MODELS),
                'model_version': config.CURRENT_MODEL_VERSION,
                'timestamp': datetime.now().isoformat()
            }
    
    @predict_ns.route('')
    class Predict(Resource):
        @predict_ns.doc('predict_budget_item')
        @predict_ns.expect(schemas['predict_input'], validate=True)
        @predict_ns.marshal_with(schemas['predict_output'], code=200)
        @predict_ns.response(400, 'Validation Error', schemas['error_model'])
        @predict_ns.response(404, 'Category Not Found', schemas['error_model'])
        @predict_ns.response(500, 'Prediction Error', schemas['error_model'])
        def post(self):
            """
            Predict expected cost for a budget line item.
            
            **Input:**
            - `category`: The budget category (e.g., "Plumbing Fixtures")
            - `features`: Property and project details (ARV, zip, size, etc.)
            
            **Output:**
            - `predicted_amount`: Expected cost based on similar projects
            - `model_tier`: Reliability indicator (A=high, B=moderate, C=low)
            - Model performance metrics (R², MAPE, RMSE)
            
            **Usage:**
            Compare `predicted_amount` with applicant's requested amount:
            - If request > predicted + (1.96 * RMSE): **HIGH**
            - If request < predicted - (1.96 * RMSE): **LOW**
            - Otherwise: **REASONABLE**
            """
            try:
                data = request.json
                category = data.get('category')
                features = data.get('features', {})
                
                if not category:
                    return {'error': 'Category is required'}, 400
                
                result, error = predict_budget_item(category, features)
                
                if error:
                    return error, 404
                
                return result
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'error': str(e)}, 500
    
    @retrain_ns.route('')
    class Retrain(Resource):
        @retrain_ns.doc('retrain_models')
        @retrain_ns.expect(schemas['retrain_input'], validate=True)
        @retrain_ns.marshal_with(schemas['retrain_output'], code=200)
        @retrain_ns.response(400, 'Validation Error', schemas['error_model'])
        @retrain_ns.response(500, 'Retraining Error', schemas['error_model'])
        def post(self):
            """
            Retrain all models using data from Supabase.
            
            **⚠️ Long-running operation** - Can take several minutes.
            
            **Process:**
            1. Fetch all budgets and items from Supabase (uses environment variables)
            2. Enrich properties via ATTOM API (if not cached)
            3. Train 3 models per category (Ridge, RF, Gradient Boosting)
            4. Select best model based on R², RMSE, MAPE
            5. Save new models to Supabase Storage
            6. Log retraining results
            
            **Triggers:**
            - Manual admin action
            - Future: Monthly cron job
            
            **Note:** Uses SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from environment variables
            """
            try:
                from app.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
                
                if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
                    return {'error': 'Missing Supabase credentials in environment variables'}, 500
                
                data = request.json or {}
                triggered_by = data.get('triggered_by')
                
                result = retrain_models(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, triggered_by)
                return result
                
            except Exception as e:
                logger.error(f"Retraining error: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'error': str(e)}, 500
