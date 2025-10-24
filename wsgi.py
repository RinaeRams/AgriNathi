"""WSGI entry point for AgriNathi"""
from run import init_app

# Initialize the application
app = application = init_app()

if __name__ == '__main__':
    application.run()