import os
import sys

# Configure Python search paths for Vercel Serverless environment
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
demo_dir = os.path.join(parent_dir, 'demo')

if demo_dir not in sys.path:
    sys.path.insert(0, demo_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the configured Flask application instance
from app import app

# Expose WSGI entrypoint for Vercel
if __name__ == '__main__':
    app.run()
