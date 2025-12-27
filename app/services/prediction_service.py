import pandas as pd
import numpy as np
import threading
import app.config as config
from app.config import logger, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from app.models.model_loader import get_model_tier
from supabase import create_client, Client
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pickle

def normalize_property_type(prop_type):
    """Normalize property type variations to standard format"""
    if not prop_type:
        return 'SFR'
    
    property_type_mapping = {
        'Single Family': 'SFR',
        'SingleFamily': 'SFR',
        'Single-Family': 'SFR',
        'single family': 'SFR',
        'Single Family Residence': 'SFR',
        'Multi Family': 'Multifamily',
        'MultiFamily': 'Multifamily',
        'Multi-Family': 'Multifamily',
        'multi family': 'Multifamily',
        'Town House': 'Townhouse',
        'TownHouse': 'Townhouse',
        'town house': 'Townhouse',
        'Condominium': 'Condo'
    }
    
    return property_type_mapping.get(prop_type, prop_type)

def encode_features(features, required_features):
    """Encode categorical features"""
    encoded_features = {}
    
    if 'Property Type_encoded' in required_features:
        try:
            prop_type = normalize_property_type(features.get('property_type', 'SFR'))
            encoded_features['Property Type_encoded'] = config.LABEL_ENCODERS['Property Type'].transform([prop_type])[0]
        except Exception as e:
            logger.warning(f"Failed to encode property_type '{features.get('property_type')}': {e}. Using default.")
            encoded_features['Property Type_encoded'] = 0
    
    if 'Property Zip_encoded' in required_features:
        try:
            zip_code = str(features.get('zip_code', ''))[:5]
            encoded_features['Property Zip_encoded'] = config.LABEL_ENCODERS['Property Zip'].transform([zip_code])[0]
        except:
            encoded_features['Property Zip_encoded'] = 0
    
    return encoded_features

def build_feature_dict(features, encoded_features):
    """Build feature dictionary from input features"""
    feature_dict = {
        '(ARV) After Repair Value': features.get('arv', 0),
        'years_since_2020': features.get('project_year', 2024) - 2020,
        'bldgSize': features.get('building_size'),
        'beds': features.get('bedrooms'),
        'bathsTotal': features.get('bathrooms'),
        'yearBuilt': features.get('year_built'),
        **encoded_features
    }
    
    if feature_dict['bldgSize'] and feature_dict['bldgSize'] > 0:
        feature_dict['arv_per_sqft'] = feature_dict['(ARV) After Repair Value'] / feature_dict['bldgSize']
    
    if features.get('year_built') and features.get('project_year'):
        feature_dict['age_at_project'] = features['project_year'] - features['year_built']
    
    if feature_dict.get('beds') and feature_dict['beds'] > 0 and feature_dict.get('bathsTotal'):
        feature_dict['baths_per_bed'] = feature_dict['bathsTotal'] / feature_dict['beds']
    
    return feature_dict

TRAINING_IN_PROGRESS = set()
TRAINING_LOCK = threading.Lock()

def get_general_model():
    """Get or create GENERAL fallback model"""
    if 'GENERAL' in MODELS:
        return MODELS['GENERAL']
    
    logger.info("Creating GENERAL model on-demand...")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        items = supabase.table('budget_items').select('*').execute().data
        budgets = supabase.table('budgets').select('*').execute().data
        
        training_data = []
        for item in items:
            budget = next((b for b in budgets if b['id'] == item['budget_id']), None)
            if not budget or not item.get('amount'):
                continue
            training_data.append({
                'amount': item['amount'],
                'arv': budget.get('arv'),
                'property_type': budget.get('property_type'),
                'zip_code': budget.get('zip_code'),
                'project_year': budget.get('project_year', 2024)
            })
        
        df = pd.DataFrame(training_data)
        df = df.dropna(subset=['amount', 'arv'])
        df = df[df['amount'] > 0]
        
        if len(df) < 50:
            logger.warning(f"Not enough data for GENERAL model: {len(df)} samples")
            return None
        
        from sklearn.preprocessing import LabelEncoder
        le_prop = LabelEncoder()
        le_zip = LabelEncoder()
        
        df['Property Type_encoded'] = le_prop.fit_transform(df['property_type'].fillna('SFR').astype(str))
        df['Property Zip_encoded'] = le_zip.fit_transform(df['zip_code'].fillna('00000').astype(str))
        df['years_since_2020'] = df['project_year'] - 2020
        
        X = df[['arv', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']].copy()
        X.columns = ['(ARV) After Repair Value', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']
        y = df['amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = Pipeline([('imp', SimpleImputer(strategy='median')), ('reg', Ridge())])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        config.MODELS['GENERAL'] = {
            'best_model_name': 'Ridge Regression',
            'best_model': model,
            'feature_cols': list(X.columns),
            'training_stats': {'n_samples': len(df)}
        }
        config.MODEL_PERFORMANCE['GENERAL'] = {
            'best_model_name': 'Ridge Regression',
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred) * 100)
        }
        
        logger.info(f"✓ GENERAL model created with {len(df)} samples")
        return config.MODELS['GENERAL']
    except Exception as e:
        logger.error(f"Failed to create GENERAL model: {str(e)}")
        return None

def count_category_samples(category):
    """Count samples for a category in budget_items"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        items = supabase.table('budget_items').select('id').or_(f'user_category.eq.{category},ai_category.eq.{category}').execute()
        return len(items.data) if items.data else 0
    except:
        return 0

def train_category_background(category):
    """Train a specific category model in background"""
    with TRAINING_LOCK:
        if category in TRAINING_IN_PROGRESS:
            return
        TRAINING_IN_PROGRESS.add(category)
    
    try:
        logger.info(f"Background training started for category: {category}")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        
        items = supabase.table('budget_items').select('*').or_(f'user_category.eq.{category},ai_category.eq.{category}').execute().data
        budgets = supabase.table('budgets').select('*').execute().data
        
        training_data = []
        for item in items:
            budget = next((b for b in budgets if b['id'] == item['budget_id']), None)
            if not budget or not item.get('amount'):
                continue
            training_data.append({
                'amount': item['amount'],
                'arv': budget.get('arv'),
                'property_type': budget.get('property_type'),
                'zip_code': budget.get('zip_code'),
                'project_year': budget.get('project_year', 2024)
            })
        
        df = pd.DataFrame(training_data)
        df = df.dropna(subset=['amount', 'arv'])
        df = df[df['amount'] > 0]
        
        if len(df) < 20:
            logger.warning(f"Not enough samples for {category}: {len(df)}")
            return
        
        from sklearn.preprocessing import LabelEncoder
        le_prop_train = LabelEncoder()
        le_zip_train = LabelEncoder()
        
        df['Property Type_encoded'] = le_prop_train.fit_transform(df['property_type'].fillna('SFR').astype(str))
        df['Property Zip_encoded'] = le_zip_train.fit_transform(df['zip_code'].fillna('00000').astype(str))
        df['years_since_2020'] = df['project_year'] - 2020
        
        X = df[['arv', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']].copy()
        X.columns = ['(ARV) After Repair Value', 'Property Type_encoded', 'Property Zip_encoded', 'years_since_2020']
        y = df['amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_score = -np.inf
        best_model = None
        best_name = None
        
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
        
        y_pred = best_model.predict(X_test)
        
        config.MODELS[category] = {
            'best_model_name': best_name,
            'best_model': best_model,
            'feature_cols': list(X.columns),
            'training_stats': {'n_samples': len(df)}
        }
        config.MODEL_PERFORMANCE[category] = {
            'best_model_name': best_name,
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred) * 100)
        }
        
        logger.info(f"✓ Model trained for {category}: {best_name}, R²={best_score:.3f}, samples={len(df)}")
    except Exception as e:
        logger.error(f"Failed to train category {category}: {str(e)}")
    finally:
        with TRAINING_LOCK:
            TRAINING_IN_PROGRESS.discard(category)

def predict_budget_item(category, features):
    """Make prediction for a budget item"""
    using_general = False
    perf_key = category
    
    if category not in config.MODELS:
        samples_count = count_category_samples(category)
        logger.info(f"Category '{category}' not found. Samples in DB: {samples_count}")
        
        if samples_count >= 20:
            with TRAINING_LOCK:
                if category not in TRAINING_IN_PROGRESS:
                    thread = threading.Thread(target=train_category_background, args=(category,))
                    thread.daemon = True
                    thread.start()
                    logger.info(f"Triggered background training for '{category}'")
        
        general_model = get_general_model()
        if not general_model:
            return None, {
                'error': f'No model available for category: {category}',
                'available_categories': sorted(list(config.MODELS.keys()))[:20]
            }
        
        model_info = general_model
        using_general = True
        perf_key = 'GENERAL'
        logger.info(f"Using GENERAL model for '{category}'")
    else:
        model_info = config.MODELS[category]
    
    model = model_info['best_model']
    required_features = model_info['feature_cols']
    
    encoded_features = encode_features(features, required_features)
    feature_dict = build_feature_dict(features, encoded_features)
    
    feature_values = [feature_dict.get(feat, np.nan) for feat in required_features]
    features_df = pd.DataFrame([feature_values], columns=required_features)
    
    predicted_amount = float(model.predict(features_df)[0])
    
    perf = config.MODEL_PERFORMANCE.get(perf_key, {})
    
    return {
        'predicted_amount': round(predicted_amount, 2),
        'model_used': model_info['best_model_name'],
        'model_tier': get_model_tier(perf),
        'model_r2': round(perf.get('r2', 0), 4),
        'model_mape': round(perf.get('mape', 0), 2),
        'model_rmse': round(perf.get('rmse', 0), 2),
        'model_version': config.CURRENT_MODEL_VERSION
    }, None
