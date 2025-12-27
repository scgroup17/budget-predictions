"""
Entry point for the ML Flask Service
"""

from dotenv import load_dotenv
load_dotenv()

from app.app import create_app
from app.config import PORT

# Expose app for gunicorn
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
