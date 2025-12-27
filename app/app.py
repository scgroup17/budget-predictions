"""
ML Flask Service for Budget Predictions
With Swagger/OpenAPI Documentation
Deploy on Railway - Interactive docs at /docs
"""

from flask import Flask
from flask_restx import Api
from flask_cors import CORS
from app.config import logger, PORT
from app.models.model_loader import load_models_from_storage
from app.api.schemas import register_schemas
from app.api.routes import create_namespaces, register_routes
from app.api.categories_route import register_categories_route

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    api = Api(
        app,
        version='1.0',
        title='Budget Prediction ML Service',
        description='''
        Machine Learning service for predicting construction budget costs.
        
        **Features:**
        - Predict expected costs for budget line items
        - Compare applicant amounts to predictions
        - Classify as HIGH / REASONABLE / LOW
        - Retrain models with new data
        
        **Model Tiers:**
        - **Tier A**: R² > 0.4, MAPE < 150% (reliable)
        - **Tier B**: R² 0.2-0.4 (moderate)
        - **Tier C**: R² < 0.2 (low reliability)
        ''',
        doc='/docs'
    )
    
    schemas = register_schemas(api)
    
    predict_ns, retrain_ns, health_ns = create_namespaces()
    
    register_routes(schemas)
    
    api.add_namespace(predict_ns, path='/predict')
    api.add_namespace(retrain_ns, path='/retrain')
    api.add_namespace(health_ns, path='/health')
    
    register_categories_route(api)
    
    if not load_models_from_storage():
        logger.warning("Models not loaded - will retry on first request")
    
    return app

def main():
    """Main entry point"""
    if not load_models_from_storage():
        logger.warning("Models not loaded - will retry on first request")
    
    app = create_app()
    app.run(host='0.0.0.0', port=PORT, debug=False)

if __name__ == '__main__':
    main()
