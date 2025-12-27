"""
Entry point for the ML Flask Service
"""

from dotenv import load_dotenv
load_dotenv()

from app.app import main

if __name__ == '__main__':
    main()
