"""Utility functions for AgriNathi app."""
import os
import json
import tempfile

def setup_google_credentials():
    """Set up Google credentials from environment variable or file.
    
    Returns:
        str: Path to credentials file (either temp file created from env or existing file)
    """
    # First try environment variable
    creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if creds_json:
        try:
            # Parse to validate JSON
            json.loads(creds_json)
            # Write to temp file
            fd, path = tempfile.mkstemp(suffix='.json', prefix='google_creds_')
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(creds_json)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
            print(f"Created temporary credentials file from environment: {path}")
            return path
        except Exception as e:
            print(f"Warning: Could not create credentials from environment: {e}")
    
    # Fallback to file
    credentials_path = os.path.join(os.path.dirname(__file__), '..', 'google-credentials.json')
    if os.path.exists(credentials_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        print(f"Using credentials file: {credentials_path}")
        return credentials_path
    
    print("Warning: No Google credentials found")
    return None