import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from supabase import create_client, Client
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from app.config import logger, CURRENT_MODEL_VERSION

def fetch_training_data(supabase):
    """Fetch budgets and items from Supabase"""
    logger.info("Fetching training data...")
    budgets = supabase.table('budgets').select('*').execute().data
    items = supabase.table('budget_items').select('*').execute().data
    return budgets, items

def check_enrichment(supabase, budgets):
    """Check property enrichment data"""
    enrichment_data = {}
    attom_calls = 0
    
    for budget in budgets:
        addr = budget.get('address')
        zip_code = budget.get('zip_code')
        if not addr or not zip_code:
            continue
            
        enrich = supabase.table('property_enrichment').select('*').eq('address', addr).eq('zip_code', zip_code).execute()
        if enrich.data:
            enrichment_data[f"{addr}_{zip_code}"] = enrich.data[0]
        else:
            attom_calls += 1
    
    return enrichment_data, attom_calls

def prepare_training_dataset(budgets, items, enrichment_data):
    """Prepare training dataset from raw data"""
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
    return df

def encode_training_features(df):
    """Encode categorical features for training"""
    le_prop = LabelEncoder()
    le_zip = LabelEncoder()
    
    df['Property Type_encoded'] = le_prop.fit_transform(df['property_type'].fillna('SFR').astype(str))
    df['Property Zip_encoded'] = le_zip.fit_transform(df['zip_code'].fillna('00000').astype(str))
    df['years_since_2020'] = df['project_year'] - 2020
    
    return df, le_prop, le_zip

def train_category_models(df):
    """Train models for each category"""
    new_models = {}
    new_performance = {}
    categories_trained = 0
    
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        if len(cat_df) < 20:
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
    
    return new_models, new_performance, categories_trained

def save_models_to_storage(supabase, new_models, label_encoders, new_performance, new_version):
    """Save trained models to Supabase storage"""
    supabase.storage.from_('ml-models').upload(f'models/{new_version}/budget_models_enhanced.pkl', pickle.dumps(new_models))
    supabase.storage.from_('ml-models').upload(f'models/{new_version}/label_encoders_enhanced.pkl', pickle.dumps(label_encoders))
    supabase.storage.from_('ml-models').upload(f'models/{new_version}/model_performance_enhanced.json', json.dumps(new_performance).encode())

def log_retraining(supabase, triggered_by, df, categories_trained, new_performance, exec_time, attom_calls, new_version):
    """Log retraining results to database"""
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

def retrain_models(supabase_url, supabase_key, triggered_by):
    """Main retraining orchestration function"""
    start_time = datetime.now()
    
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("Starting model retraining...")
    
    current_version = CURRENT_MODEL_VERSION
    if not current_version:
        try:
            files = supabase.storage.from_('ml-models').list('models')
            versions = [f['name'] for f in files if f['name'].startswith('v')]
            if versions:
                versions_sorted = sorted(versions, key=lambda x: int(x[1:]))
                current_version = versions_sorted[-1]
                logger.info(f"Detected existing version: {current_version}")
        except Exception as e:
            logger.warning(f"Could not detect existing version: {e}")
    
    budgets, items = fetch_training_data(supabase)
    enrichment_data, attom_calls = check_enrichment(supabase, budgets)
    df = prepare_training_dataset(budgets, items, enrichment_data)
    df, le_prop, le_zip = encode_training_features(df)
    new_models, new_performance, categories_trained = train_category_models(df)
    
    new_version = f"v{int(current_version[1:]) + 1}" if current_version else "v1"
    
    logger.info(f"Trained {categories_trained} new/updated categories")
    
    if categories_trained == 0:
        logger.warning("No categories trained (all have <20 samples). Copying existing models...")
        if current_version:
            try:
                existing_models_file = supabase.storage.from_('ml-models').download(f'models/{current_version}/budget_models_enhanced.pkl')
                existing_encoders_file = supabase.storage.from_('ml-models').download(f'models/{current_version}/label_encoders_enhanced.pkl')
                existing_perf_file = supabase.storage.from_('ml-models').download(f'models/{current_version}/model_performance_enhanced.json')
                
                new_models = pickle.loads(existing_models_file)
                label_encoders = pickle.loads(existing_encoders_file)
                new_performance = json.loads(existing_perf_file.decode('utf-8'))
                
                logger.info(f"✓ Copied {len(new_models)} models from {current_version}")
            except Exception as e:
                logger.error(f"Failed to copy existing models: {e}")
                raise ValueError(f"No new categories trained and failed to copy existing models: {e}")
        else:
            raise ValueError("No categories trained and no existing version to copy from")
    else:
        if current_version:
            try:
                logger.info(f"Loading existing models from {current_version} for incremental update...")
                existing_models_file = supabase.storage.from_('ml-models').download(f'models/{current_version}/budget_models_enhanced.pkl')
                existing_perf_file = supabase.storage.from_('ml-models').download(f'models/{current_version}/model_performance_enhanced.json')
                
                existing_models = pickle.loads(existing_models_file)
                existing_performance = json.loads(existing_perf_file.decode('utf-8'))
                
                new_models = {**existing_models, **new_models}
                new_performance = {**existing_performance, **new_performance}
                
                logger.info(f"✓ Incremental update: {len(existing_models)} existing + {categories_trained} new = {len(new_models)} total")
            except Exception as e:
                logger.warning(f"Could not load existing models for incremental update: {e}. Using only new models.")
    
    label_encoders = {'Property Type': le_prop, 'Property Zip': le_zip}
    save_models_to_storage(supabase, new_models, label_encoders, new_performance, new_version)
    
    exec_time = int((datetime.now() - start_time).total_seconds())
    log_retraining(supabase, triggered_by, df, categories_trained, new_performance, exec_time, attom_calls, new_version)
    
    return {
        'success': True,
        'categories_trained': categories_trained,
        'new_model_version': new_version,
        'execution_time_seconds': exec_time,
        'attom_api_calls': attom_calls
    }
