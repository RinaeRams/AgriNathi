"""WSGI entry point for AgriNathi"""
from run import app

# Export for WSGI
application = app

if __name__ == '__main__':
    application.run()