"""
ML Flask Service for Budget Predictions
With Swagger/OpenAPI Documentation
Deploy on Railway - Interactive docs at /docs
"""

from flask import Flask, request
from flask_restx import Api, Resource, fields, Namespace
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import logging
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Swagger/OpenAPI with flask-restx
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
    doc='/docs'  # Swagger UI available at /docs
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create namespaces for organization
predict_ns = Namespace('predict', description='Prediction operations')
retrain_ns = Namespace('retrain', description='Model retraining operations')
health_ns = Namespace('health', description='Service health checks')

api.add_namespace(predict_ns, path='/predict')
api.add_namespace(retrain_ns, path='/retrain')
api.add_namespace(health_ns, path='/health')

# Model storage
MODELS = {}
LABEL_ENCODERS = {}
MODEL_PERFORMANCE = {}
CURRENT_MODEL_VERSION = None

# ============================================================================
# SWAGGER SCHEMA DEFINITIONS
# ============================================================================

# Input schema for prediction features
features_model = api.model('PredictionFeatures', {
    'arv': fields.Float(required=True, description='After Repair Value ($)', example=450000),
    'property_type': fields.String(required=False, description='Property type (Single Family = SFR)', example='SFR', 
                                   enum=['SFR', 'Single Family', 'Condo', 'Townhouse', 'Multifamily', 'Land', 'Warehouse', 'Office']),
    'zip_code': fields.String(required=True, description='5-digit zip code', example='33178'),
    'project_year': fields.Integer(required=False, description='Project year', example=2024),
    'building_size': fields.Integer(required=False, description='Building size in sqft', example=2049),
    'bedrooms': fields.Integer(required=False, description='Number of bedrooms', example=3),
    'bathrooms': fields.Float(required=False, description='Number of bathrooms', example=2.5),
    'year_built': fields.Integer(required=False, description='Year property was built', example=2006)
})

# Input schema for prediction request
predict_input = api.model('PredictInput', {
    'category': fields.String(required=True, description='Budget category name', 
                              example='Plumbing Fixtures (Showers, Bathtubs, Toilets, Vanities)'),
    'features': fields.Nested(features_model, required=True, description='Property and project features')
})

# Output schema for prediction response
predict_output = api.model('PredictOutput', {
    'predicted_amount': fields.Float(description='Predicted cost ($)', example=8500.42),
    'model_used': fields.String(description='Model algorithm used', example='Random Forest'),
    'model_tier': fields.String(description='Model reliability tier', example='A', enum=['A', 'B', 'C']),
    'model_r2': fields.Float(description='R² score (0-1)', example=0.5242),
    'model_mape': fields.Float(description='Mean Absolute Percentage Error (%)', example=76.24),
    'model_rmse': fields.Float(description='Root Mean Square Error ($)', example=5621.03),
    'model_version': fields.String(description='Model version', example='v1')
})

# Error response schema
error_model = api.model('ErrorResponse', {
    'error': fields.String(description='Error message'),
    'available_categories': fields.List(fields.String, description='List of valid categories (if category not found)')
})

# Health check response schema
health_output = api.model('HealthOutput', {
    'status': fields.String(description='Service status', example='healthy'),
    'models_loaded': fields.Integer(description='Number of category models loaded', example=65),
    'model_version': fields.String(description='Current model version', example='v1'),
    'timestamp': fields.String(description='Current server time', example='2024-01-15T10:30:00')
})

# Retrain input schema
retrain_input = api.model('RetrainInput', {
    'supabase_url': fields.String(required=True, description='Supabase project URL', 
                                  example='https://your-project.supabase.co'),
    'supabase_key': fields.String(required=True, description='Supabase service role key'),
    'triggered_by': fields.String(required=False, description='User ID who triggered retraining')
})

# Retrain output schema
retrain_output = api.model('RetrainOutput', {
    'success': fields.Boolean(description='Whether retraining succeeded'),
    'categories_trained': fields.Integer(description='Number of categories retrained', example=45),
    'new_model_version': fields.String(description='New model version', example='v2'),
    'execution_time_seconds': fields.Integer(description='Total execution time', example=1234),
    'attom_api_calls': fields.Integer(description='Number of ATTOM API calls made', example=50)
})

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models_from_storage():
    """Load models from Supabase Storage"""
    global MODELS, LABEL_ENCODERS, MODEL_PERFORMANCE, CURRENT_MODEL_VERSION
    
    try:
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials")
            return False
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        version = 'v1'
        CURRENT_MODEL_VERSION = version
        
        logger.info(f"Loading models version {version}...")
        
        models_file = supabase.storage.from_('ml-models').download(f'models/{version}/budget_models_enhanced.pkl')
        encoders_file = supabase.storage.from_('ml-models').download(f'models/{version}/label_encoders_enhanced.pkl')
        perf_file = supabase.storage.from_('ml-models').download(f'models/{version}/model_performance_enhanced.json')
        
        MODELS = pickle.loads(models_file)
        LABEL_ENCODERS = pickle.loads(encoders_file)
        MODEL_PERFORMANCE = json.loads(perf_file.decode('utf-8'))
        
        logger.info(f"✓ Loaded {len(MODELS)} category models")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return False

def get_model_tier(performance):
    """Calculate model tier based on R² and MAPE"""
    r2 = performance.get('r2', 0)
    mape = performance.get('mape', 999)
    
    if r2 > 0.4 and mape < 150:
        return 'A'
    elif 0.2 <= r2 <= 0.4:
        return 'B'
    else:
        return 'C'

def train_simple_average_model(category, category_items, budgets):
    """
    Create a simple average-based model when there are very few samples (<10).
    Uses average cost per ARV ratio from available samples.
    """
    global MODELS, MODEL_PERFORMANCE
    
    try:
        logger.info(f"Creating simple average model for '{category}' with {len(category_items)} samples")
        
        # Calculate average amount and ARV ratio
        valid_samples = []
        for item in category_items:
            budget = next((b for b in budgets if b['id'] == item['budget_id']), None)
            if budget and item.get('amount') and item['amount'] > 0 and budget.get('arv'):
                valid_samples.append({
                    'amount': item['amount'],
                    'arv': budget['arv'],
                    'ratio': item['amount'] / budget['arv']
                })
        
        if len(valid_samples) == 0:
            logger.error(f"No valid samples for '{category}'")
            return False
        
        # Calculate statistics
        avg_amount = np.mean([s['amount'] for s in valid_samples])
        avg_ratio = np.mean([s['ratio'] for s in valid_samples])
        std_ratio = np.std([s['ratio'] for s in valid_samples]) if len(valid_samples) > 1 else avg_ratio * 0.5
        
        # Create a simple predictor class
        class SimpleAveragePredictor:
            def __init__(self, avg_ratio, avg_amount):
                self.avg_ratio = avg_ratio
                self.avg_amount = avg_amount
            
            def predict(self, X):
                # X is a DataFrame with '(ARV) After Repair Value' column
                arvs = X['(ARV) After Repair Value'].values
                # Use ratio if ARV is available, otherwise use average
                predictions = np.where(arvs > 0, arvs * self.avg_ratio, self.avg_amount)
                return predictions
        
        # Save simple model
        MODELS[category] = {
            'best_model': SimpleAveragePredictor(avg_ratio, avg_amount),
            'best_model_name': 'Simple Average',
            'feature_cols': ['(ARV) After Repair Value', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020'],
            'training_stats': {
                'n_samples': len(valid_samples),
                'trained_at': datetime.now().isoformat(),
                'model_type': 'simple_average'
            }
        }
        
        MODEL_PERFORMANCE[category] = {
            'r2': 0.0,  # Simple model doesn't have R²
            'rmse': std_ratio * avg_amount,
            'mape': (std_ratio / avg_ratio * 100) if avg_ratio > 0 else 100
        }
        
        logger.info(f"✓ Created simple model for '{category}': avg=${avg_amount:.0f}, ratio={avg_ratio:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create simple model for '{category}': {str(e)}")
        return False

def train_category_model(category):
    """
    Train a model for a specific category on-the-fly.
    Returns True if successful, False if insufficient data.
    """
    global MODELS, LABEL_ENCODERS, MODEL_PERFORMANCE
    
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
        import requests
        
        supabase_url = os.environ.get('SUPABASE_URL')
        supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials for training")
            return False
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Fetch all data from Supabase
        logger.info(f"Fetching training data for category: {category}")
        budgets = supabase.table('budgets').select('*').execute().data
        items = supabase.table('budget_items').select('*').execute().data
        
        # Filter items for this specific category
        category_items = [item for item in items 
                         if (item.get('user_category') or item.get('ai_category')) == category]
        
        logger.info(f"Found {len(category_items)} samples for '{category}'")
        
        # If very few samples, use simple average model instead of ML
        if len(category_items) < 10:
            logger.info(f"Only {len(category_items)} samples - using simple average model")
            return train_simple_average_model(category, category_items, budgets)
        
        # If 10-50 samples, warn but continue with ML training
        if len(category_items) < 50:
            logger.warning(f"Only {len(category_items)} samples for '{category}' - model may be less accurate")
        
        # Get enrichment data (use cached if available)
        enrichment_data = {}
        for budget in budgets:
            addr = budget.get('address')
            zip_code = budget.get('zip_code')
            if addr and zip_code:
                enrich = supabase.table('property_enrichment').select('*').eq('address', addr).eq('zip_code', zip_code).execute()
                if enrich.data:
                    enrichment_data[f"{addr}_{zip_code}"] = enrich.data[0]
        
        # Prepare training data
        training_data = []
        for item in category_items:
            budget = next((b for b in budgets if b['id'] == item['budget_id']), None)
            if not budget or not item.get('amount') or item['amount'] <= 0:
                continue
            
            enrich = enrichment_data.get(f"{budget.get('address')}_{budget.get('zip_code')}", {})
            
            training_data.append({
                'amount': item['amount'],
                'arv': budget.get('arv', 0),
                'property_type': budget.get('property_type', 'SFR'),
                'zip_code': budget.get('zip_code', '00000'),
                'project_year': budget.get('project_year', 2024),
                'building_size': enrich.get('building_size'),
                'bedrooms': enrich.get('bedrooms'),
                'bathrooms': enrich.get('bathrooms'),
                'year_built': enrich.get('year_built')
            })
        
        # No minimum threshold - train with whatever data we have
        if len(training_data) < 5:
            logger.error(f"After filtering, only {len(training_data)} valid samples - too few to train")
            return False
        
        logger.info(f"Training with {len(training_data)} valid samples")
        
        df = pd.DataFrame(training_data)
        
        # Encode categorical features
        le_prop = LabelEncoder()
        le_zip = LabelEncoder()
        
        df['Property Type_encoded'] = le_prop.fit_transform(df['property_type'].fillna('SFR').astype(str))
        df['Property Zip_encoded'] = le_zip.fit_transform(df['zip_code'].fillna('00000').astype(str))
        df['years_since_2020'] = df['project_year'] - 2020
        
        # Prepare features
        X = df[['arv', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']].copy()
        X.columns = ['(ARV) After Repair Value', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']
        y = df['amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train 3 models and select best
        best_score = -np.inf
        best_model = None
        best_name = None
        best_metrics = {}
        
        for name, model in [
            ('Ridge Regression', Pipeline([('imp', SimpleImputer(strategy='median')), ('reg', Ridge())])),
            ('Random Forest', Pipeline([('imp', SimpleImputer(strategy='median')), ('reg', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))])),
            ('Gradient Boosting', HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=42))
        ]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name
                best_metrics = {
                    'r2': r2,
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
                }
        
        # Save to global MODELS
        MODELS[category] = {
            'best_model': best_model,
            'best_model_name': best_name,
            'feature_cols': list(X.columns),
            'training_stats': {
                'n_samples': len(df),
                'trained_at': datetime.now().isoformat()
            }
        }
        
        MODEL_PERFORMANCE[category] = best_metrics
        
        # Update label encoders
        if 'Property Type' not in LABEL_ENCODERS:
            LABEL_ENCODERS['Property Type'] = le_prop
        if 'Property Zip' not in LABEL_ENCODERS:
            LABEL_ENCODERS['Property Zip'] = le_zip
        
        logger.info(f"✓ Trained {best_name} for '{category}': R²={best_metrics['r2']:.3f}, RMSE=${best_metrics['rmse']:.0f}")
        
        # === IGUAL QUE RETRAIN: Guardar en Storage y DB ===
        try:
            # 1. Actualizar archivos en Storage (IGUAL que Retrain líneas 756-758)
            version = CURRENT_MODEL_VERSION or 'v1'
            
            # Cargar modelos existentes
            try:
                existing_models_file = supabase.storage.from_('ml-models').download(f'models/{version}/budget_models_enhanced.pkl')
                existing_models = pickle.loads(existing_models_file)
            except:
                existing_models = {}
            
            try:
                existing_encoders_file = supabase.storage.from_('ml-models').download(f'models/{version}/label_encoders_enhanced.pkl')
                existing_encoders = pickle.loads(existing_encoders_file)
            except:
                existing_encoders = {}
            
            try:
                existing_perf_file = supabase.storage.from_('ml-models').download(f'models/{version}/model_performance_enhanced.json')
                existing_performance = json.loads(existing_perf_file.decode('utf-8'))
            except:
                existing_performance = {}
            
            # Agregar nuevo modelo a los existentes
            existing_models[category] = MODELS[category]
            existing_performance[category] = {'best_model_name': best_name, **best_metrics}
            existing_encoders['Property Type'] = le_prop
            existing_encoders['Property Zip'] = le_zip
            
            # Guardar de vuelta (SOBRESCRIBE con upsert)
            supabase.storage.from_('ml-models').upload(
                f'models/{version}/budget_models_enhanced.pkl',
                pickle.dumps(existing_models),
                file_options={"upsert": "true"}
            )
            supabase.storage.from_('ml-models').upload(
                f'models/{version}/label_encoders_enhanced.pkl',
                pickle.dumps(existing_encoders),
                file_options={"upsert": "true"}
            )
            supabase.storage.from_('ml-models').upload(
                f'models/{version}/model_performance_enhanced.json',
                json.dumps(existing_performance).encode(),
                file_options={"upsert": "true"}
            )
            
            logger.info(f"✓ Updated Storage with new model for '{category}'")
            
            # 2. Guardar log en DB (IGUAL que Retrain línea 762-771)
            supabase.table('model_retraining_logs').insert({
                'triggered_by': 'on_the_fly_training',
                'total_categories': 1,
                'categories_trained': 1,
                'total_samples': len(df),
                'categories_performance': {category: {'best_model_name': best_name, **best_metrics}},
                'execution_time_seconds': 0,
                'attom_api_calls': 0,
                'new_model_version': version,
                'notes': f'On-the-fly training for: {category}'
            }).execute()
            
            logger.info(f"✓ Logged training to database")
            
        except Exception as storage_error:
            logger.warning(f"Failed to save to storage/db (non-critical): {storage_error}")
            # No fallar si esto falla, el modelo ya está en memoria
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed for '{category}': {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@health_ns.route('')
class HealthCheck(Resource):
    @health_ns.doc('health_check')
    @health_ns.marshal_with(health_output)
    def get(self):
        """Check service health and model status"""
        return {
            'status': 'healthy',
            'models_loaded': len(MODELS),
            'model_version': CURRENT_MODEL_VERSION,
            'timestamp': datetime.now().isoformat()
        }

@predict_ns.route('')
class Predict(Resource):
    @predict_ns.doc('predict_budget_item')
    @predict_ns.expect(predict_input, validate=True)
    @predict_ns.marshal_with(predict_output, code=200)
    @predict_ns.response(400, 'Validation Error', error_model)
    @predict_ns.response(404, 'Category Not Found', error_model)
    @predict_ns.response(500, 'Prediction Error', error_model)
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
            
            # If category not found, try to train it on-the-fly
            if category not in MODELS:
                logger.warning(f"Category '{category}' not found. Attempting on-the-fly training...")
                
                try:
                    # Train model for this specific category
                    trained = train_category_model(category)
                    
                    if trained:
                        logger.info(f"✓ Successfully trained model for '{category}'")
                        # Continue with prediction using newly trained model
                    else:
                        logger.error(f"Failed to train model for '{category}' - insufficient data")
                        return {
                            'error': f'No model available for category: {category}',
                            'reason': 'Insufficient training data (need at least 50 samples)',
                            'available_categories': sorted(list(MODELS.keys()))[:20]
                        }, 404
                        
                except Exception as e:
                    logger.error(f"On-the-fly training failed: {str(e)}")
                    return {
                        'error': f'Failed to train model for category: {category}',
                        'details': str(e)
                    }, 500
            
            # Get model info
            model_info = MODELS[category]
            model = model_info['best_model']
            required_features = model_info['feature_cols']
            
            # Encode categorical features
            encoded_features = {}
            
            if 'Property Type_encoded' in required_features:
                try:
                    prop_type = features.get('property_type', 'SFR')
                    # Normalize property type: "Single Family" -> "SFR"
                    if prop_type == 'Single Family':
                        prop_type = 'SFR'
                    encoded_features['Property Type_encoded'] = LABEL_ENCODERS['Property Type'].transform([prop_type])[0]
                except:
                    encoded_features['Property Type_encoded'] = 0
            
            if 'Property Zip_encoded' in required_features:
                try:
                    zip_code = str(features.get('zip_code', ''))[:5]
                    encoded_features['Property Zip_encoded'] = LABEL_ENCODERS['Property Zip'].transform([zip_code])[0]
                except:
                    encoded_features['Property Zip_encoded'] = 0
            
            # Build feature dictionary
            feature_dict = {
                '(ARV) After Repair Value': features.get('arv', 0),
                'years_since_2020': features.get('project_year', 2024) - 2020,
                'bldgSize': features.get('building_size'),
                'beds': features.get('bedrooms'),
                'bathsTotal': features.get('bathrooms'),
                'yearBuilt': features.get('year_built'),
                **encoded_features
            }
            
            # Add derived features
            if feature_dict['bldgSize'] and feature_dict['bldgSize'] > 0:
                feature_dict['arv_per_sqft'] = feature_dict['(ARV) After Repair Value'] / feature_dict['bldgSize']
            
            if features.get('year_built') and features.get('project_year'):
                feature_dict['age_at_project'] = features['project_year'] - features['year_built']
            
            if feature_dict.get('beds') and feature_dict['beds'] > 0 and feature_dict.get('bathsTotal'):
                feature_dict['baths_per_bed'] = feature_dict['bathsTotal'] / feature_dict['beds']
            
            # Build feature vector
            feature_values = [feature_dict.get(feat, np.nan) for feat in required_features]
            features_df = pd.DataFrame([feature_values], columns=required_features)
            
            # Make prediction
            predicted_amount = float(model.predict(features_df)[0])
            
            # Get model performance
            perf = MODEL_PERFORMANCE.get(category, {})
            
            return {
                'predicted_amount': round(predicted_amount, 2),
                'model_used': model_info['best_model_name'],
                'model_tier': get_model_tier(perf),
                'model_r2': round(perf.get('r2', 0), 4),
                'model_mape': round(perf.get('mape', 0), 2),
                'model_rmse': round(perf.get('rmse', 0), 2),
                'model_version': CURRENT_MODEL_VERSION
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}, 500

@retrain_ns.route('')
class Retrain(Resource):
    @retrain_ns.doc('retrain_models')
    @retrain_ns.expect(retrain_input, validate=True)
    @retrain_ns.marshal_with(retrain_output, code=200)
    @retrain_ns.response(400, 'Validation Error', error_model)
    @retrain_ns.response(500, 'Retraining Error', error_model)
    def post(self):
        """
        Retrain all models using data from Supabase.
        
        **⚠️ Long-running operation** - Can take several minutes.
        
        **Process:**
        1. Fetch all budgets and items from Supabase
        2. Enrich properties via ATTOM API (if not cached)
        3. Train 3 models per category (Ridge, RF, Gradient Boosting)
        4. Select best model based on R², RMSE, MAPE
        5. Save new models to Supabase Storage
        6. Log retraining results
        
        **Triggers:**
        - Manual admin action
        - Future: Monthly cron job
        """
        try:
            start_time = datetime.now()
            data = request.json
            supabase_url = os.environ.get('SUPABASE_URL')
            supabase_key = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
            triggered_by = data.get('triggered_by', 'manual')
            
            if not supabase_url or not supabase_key:
                return {'error': 'Supabase credentials required'}, 400
            
            supabase: Client = create_client(supabase_url, supabase_key)
            
            logger.info("Starting model retraining...")
            
            # Import ML libraries
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
            
            # 1. Fetch data
            logger.info("Fetching training data...")
            budgets = supabase.table('budgets').select('*').execute().data
            items = supabase.table('budget_items').select('*').execute().data
            
            # 2. Check enrichment
            enrichment_data = {}
            attom_calls = 0
            
            logger.info("Checking property enrichment...")
            
            for budget in budgets:
                addr = budget.get('address')
                zip_code = budget.get('zip_code')
                if not addr or not zip_code:
                    continue
                    
                # Check if already enriched in database
                enrich = supabase.table('property_enrichment').select('*').eq('address', addr).eq('zip_code', zip_code).execute()
                
                if enrich.data:
                    # Use cached enrichment
                    enrichment_data[f"{addr}_{zip_code}"] = enrich.data[0]
                    logger.debug(f"Using cached enrichment for {addr}")
                else:
                    # Call Supabase edge function to enrich property
                    logger.info(f"Enriching property: {addr}, {zip_code}")
                    try:
                        enrich_response = requests.post(
                            f"{supabase_url}/functions/v1/enrich-property",
                            headers={
                                'Authorization': f'Bearer {supabase_key}',
                                'Content-Type': 'application/json'
                            },
                            json={
                                'address': addr,
                                'zip_code': zip_code
                            },
                            timeout=30
                        )
                        
                        if enrich_response.ok:
                            enrich_result = enrich_response.json()
                            if enrich_result.get('success'):
                                enrichment_data[f"{addr}_{zip_code}"] = enrich_result.get('enrichment', {})
                                attom_calls += 1
                                logger.info(f"✓ Enriched {addr}")
                            else:
                                logger.warning(f"Enrichment failed for {addr}: {enrich_result.get('error')}")
                        else:
                            logger.warning(f"Enrichment API returned {enrich_response.status_code} for {addr}")
                    
                    except requests.exceptions.Timeout:
                        logger.warning(f"Enrichment timeout for {addr}")
                    except Exception as e:
                        logger.warning(f"Failed to enrich {addr}: {str(e)}")
                    
                    # Add small delay to respect rate limits (200/min = ~0.3s per request)
                    import time
                    time.sleep(0.4)
            
            # 3. Prepare dataset
            training_data = []
            for item in items:
                budget = next((b for b in budgets if b['id'] == item['budget_id']), None)
                if not budget:
                    continue
                
                enrich = enrichment_data.get(f"{budget.get('address')}_{budget.get('zip_code')}", {})
                
                training_data.append({
                    'category': item.get('user_category') or item.get('ai_category'),
                    'amount': item.get('amount'),
                    'arv': budget.get('arv'),
                    'property_type': budget.get('property_type'),
                    'zip_code': budget.get('zip_code'),
                    'project_year': budget.get('project_year', 2024),
                    'building_size': enrich.get('building_size'),
                    'bedrooms': enrich.get('bedrooms'),
                    'bathrooms': enrich.get('bathrooms'),
                    'year_built': enrich.get('year_built')
                })
            
            df = pd.DataFrame(training_data)
            df = df.dropna(subset=['category', 'amount', 'arv'])
            df = df[df['amount'] > 0]
            
            logger.info(f"Training dataset: {len(df)} samples, {df['category'].nunique()} categories")
            
            # 4. Train models
            new_models = {}
            new_performance = {}
            categories_trained = 0
            
            le_prop = LabelEncoder()
            le_zip = LabelEncoder()
            
            df['Property Type_encoded'] = le_prop.fit_transform(df['property_type'].fillna('SFR').astype(str))
            df['Property Zip_encoded'] = le_zip.fit_transform(df['zip_code'].fillna('00000').astype(str))
            df['years_since_2020'] = df['project_year'] - 2020
            
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                if len(cat_df) < 50:
                    continue
                
                X = cat_df[['arv', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']].copy()
                X.columns = ['(ARV) After Repair Value', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']
                y = cat_df['amount']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                best_score = -np.inf
                best_model = None
                best_name = None
                best_metrics = {}
                
                for name, model in [
                    ('Ridge Regression', Pipeline([('imp', SimpleImputer(strategy='median')), ('reg', Ridge())])),
                    ('Random Forest', Pipeline([('imp', SimpleImputer(strategy='median')), ('reg', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))])),
                    ('Gradient Boosting', HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=42))
                ]:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        best_name = name
                        best_metrics = {
                            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                            'mae': float(mean_absolute_error(y_test, y_pred)),
                            'r2': float(r2),
                            'mape': float(mean_absolute_percentage_error(y_test, y_pred) * 100)
                        }
                
                new_models[category] = {
                    'best_model_name': best_name,
                    'best_model': best_model,
                    'feature_cols': list(X.columns),
                    'training_stats': {'n_samples': len(cat_df)}
                }
                new_performance[category] = {'best_model_name': best_name, **best_metrics}
                categories_trained += 1
            
            # 5. Save to storage
            new_version = f"v{int(CURRENT_MODEL_VERSION[1:]) + 1}" if CURRENT_MODEL_VERSION else "v2"
            
            supabase.storage.from_('ml-models').upload(f'models/{new_version}/budget_models_enhanced.pkl', pickle.dumps(new_models))
            supabase.storage.from_('ml-models').upload(f'models/{new_version}/label_encoders_enhanced.pkl', pickle.dumps({'Property Type': le_prop, 'Property Zip': le_zip}))
            supabase.storage.from_('ml-models').upload(f'models/{new_version}/model_performance_enhanced.json', json.dumps(new_performance).encode())
            
            # 6. Log
            exec_time = int((datetime.now() - start_time).total_seconds())
            supabase.table('model_retraining_logs').insert({
                'triggered_by': triggered_by,
                'total_categories': df['category'].nunique(),
                'categories_trained': categories_trained,
                'total_samples': len(df),
                'categories_performance': new_performance,
                'execution_time_seconds': exec_time,
                'attom_api_calls': attom_calls,
                'new_model_version': new_version
            }).execute()
            
            return {
                'success': True,
                'categories_trained': categories_trained,
                'new_model_version': new_version,
                'execution_time_seconds': exec_time,
                'attom_api_calls': attom_calls
            }
            
        except Exception as e:
            logger.error(f"Retraining error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}, 500

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

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    if not load_models_from_storage():
        logger.warning("Models not loaded - will retry on first request")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
