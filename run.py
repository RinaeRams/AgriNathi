#!/usr/bin/env python3
"""
AgriNathi - Agricultural Voice Assistant
Main application runner
"""

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("Starting AgriNathi Agricultural Voice Assistant...")
    print("Server will be available at: http://localhost:5000")
    print("Voice recognition ready for isiZulu input")
    print("Mobile app available for Android/iOS")
    app.run(debug=True, host='0.0.0.0', port=5000)