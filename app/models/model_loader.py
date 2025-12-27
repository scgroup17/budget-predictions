import pickle
import json
from supabase import create_client, Client
from app.config import logger, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
import app.config as config

def load_models_from_storage():
    """Load models from Supabase Storage"""
    try:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            logger.error("Missing Supabase credentials")
            return False
        
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        
        try:
            files = supabase.storage.from_('ml-models').list('models')
            versions = [f['name'] for f in files if f['name'].startswith('v')]
            if versions:
                versions_sorted = sorted(versions, key=lambda x: int(x[1:]))
                version = versions_sorted[-1]
                logger.info(f"Detected latest version: {version}")
            else:
                logger.warning("No versions found in storage, defaulting to v1")
                version = 'v1'
        except Exception as e:
            logger.warning(f"Could not detect version from storage: {e}. Defaulting to v1")
            version = 'v1'
        
        config.CURRENT_MODEL_VERSION = version
        
        logger.info(f"Loading models version {version}...")
        
        models_file = supabase.storage.from_('ml-models').download(f'models/{version}/budget_models_enhanced.pkl')
        encoders_file = supabase.storage.from_('ml-models').download(f'models/{version}/label_encoders_enhanced.pkl')
        perf_file = supabase.storage.from_('ml-models').download(f'models/{version}/model_performance_enhanced.json')
        
        config.MODELS = pickle.loads(models_file)
        config.LABEL_ENCODERS = pickle.loads(encoders_file)
        config.MODEL_PERFORMANCE = json.loads(perf_file.decode('utf-8'))
        
        logger.info(f"✓ Loaded {len(config.MODELS)} category models")
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
