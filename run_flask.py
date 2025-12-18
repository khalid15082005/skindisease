#!/usr/bin/env python3
"""
Simple script to run the Flask application
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask_app import app

if __name__ == '__main__':
    print("=" * 50)
    print("Skin Disease AI - Flask Application")
    print("=" * 50)
    print("Starting server...")
    print("URL: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        print("Goodbye!")
