#!/usr/bin/env python3
"""
AgriNathi - Agricultural Voice Assistant
Main application runner
"""

from app import app, create_app

def init_app():
    """Initialize the application and register all components"""
    # Initialize the app and register all routes and components
    create_app()
    return app

# Initialize the application
app = init_app()

if __name__ == '__main__':
    print("Starting AgriNathi Agricultural Voice Assistant...")
    print("Server will be available at: http://localhost:5000")
    print("Voice recognition ready for isiZulu input")
    print("Mobile app available for Android/iOS")
    app.run(debug=False, host='0.0.0.0', port=5000)