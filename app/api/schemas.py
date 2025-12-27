from flask_restx import fields

def register_schemas(api):
    """Register all API schemas with the API instance"""
    
    features_model = api.model('PredictionFeatures', {
        'arv': fields.Float(required=True, description='After Repair Value ($)', example=450000),
        'property_type': fields.String(required=False, description='Property type (accepts variations like "Single Family" -> "SFR")', example='SFR'),
        'zip_code': fields.String(required=True, description='5-digit zip code', example='33178'),
        'project_year': fields.Integer(required=False, description='Project year', example=2024),
        'building_size': fields.Integer(required=False, description='Building size in sqft', example=2049),
        'bedrooms': fields.Integer(required=False, description='Number of bedrooms', example=3),
        'bathrooms': fields.Float(required=False, description='Number of bathrooms', example=2.5),
        'year_built': fields.Integer(required=False, description='Year property was built', example=2006)
    })
    
    predict_input = api.model('PredictInput', {
        'category': fields.String(required=True, description='Budget category name', 
                                  example='Plumbing Fixtures (Showers, Bathtubs, Toilets, Vanities)'),
        'features': fields.Nested(features_model, required=True, description='Property and project features')
    })
    
    predict_output = api.model('PredictOutput', {
        'predicted_amount': fields.Float(description='Predicted cost ($)', example=8500.42),
        'model_used': fields.String(description='Model algorithm used', example='Random Forest'),
        'model_tier': fields.String(description='Model reliability tier', example='A', enum=['A', 'B', 'C']),
        'model_r2': fields.Float(description='R² score (0-1)', example=0.5242),
        'model_mape': fields.Float(description='Mean Absolute Percentage Error (%)', example=76.24),
        'model_rmse': fields.Float(description='Root Mean Square Error ($)', example=5621.03),
        'model_version': fields.String(description='Model version', example='v1')
    })
    
    error_model = api.model('ErrorResponse', {
        'error': fields.String(description='Error message'),
        'available_categories': fields.List(fields.String, description='List of valid categories (if category not found)')
    })
    
    health_output = api.model('HealthOutput', {
        'status': fields.String(description='Service status', example='healthy'),
        'models_loaded': fields.Integer(description='Number of category models loaded', example=65),
        'model_version': fields.String(description='Current model version', example='v1'),
        'timestamp': fields.String(description='Current server time', example='2024-01-15T10:30:00')
    })
    
    retrain_input = api.model('RetrainInput', {
        'triggered_by': fields.String(required=False, description='User ID who triggered retraining', example='admin@example.com')
    })
    
    retrain_output = api.model('RetrainOutput', {
        'success': fields.Boolean(description='Whether retraining succeeded'),
        'categories_trained': fields.Integer(description='Number of categories retrained', example=45),
        'new_model_version': fields.String(description='New model version', example='v2'),
        'execution_time_seconds': fields.Integer(description='Total execution time', example=1234),
        'attom_api_calls': fields.Integer(description='Number of ATTOM API calls made', example=50)
    })
    
    return {
        'predict_input': predict_input,
        'predict_output': predict_output,
        'error_model': error_model,
        'health_output': health_output,
        'retrain_input': retrain_input,
        'retrain_output': retrain_output
    }
