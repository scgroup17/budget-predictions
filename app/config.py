import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
PORT = int(os.environ.get('PORT', 5001))

MODELS = {}
LABEL_ENCODERS = {}
MODEL_PERFORMANCE = {}
CURRENT_MODEL_VERSION = None
