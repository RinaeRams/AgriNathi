"""WSGI entry point for AgriNathi"""
from app import app, create_app

# Initialize the application
create_app()

# Export for WSGI
application = app

if __name__ == '__main__':
    application.run()