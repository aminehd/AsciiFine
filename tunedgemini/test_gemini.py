#!/usr/bin/env python
"""
A simple script to test the connection to Gemini models on Vertex AI.
"""

import os
import sys
import dotenv
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel
import google.generativeai as genai
# Load environment variables from .env file
dotenv.load_dotenv()

def test_gemini_connection():
    """Test connection to Gemini models on Vertex AI."""
    # Check if credentials are properly set
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    model_name = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-pro")
    
    print("\n=== Gemini Model Connection Test ===\n")
    
    # Check environment variables
    if not credentials_path or not os.path.exists(credentials_path):
        print("❌ Valid credentials not found. Please run test_connection.py first.")
        return False
    
    if not project_id:
        print("❌ GCP_PROJECT_ID environment variable is not set.")
        return False
    
    print(f"✅ Using project: {project_id}")
    print(f"✅ Using location: {location}")
    print(f"✅ Using model: {model_name}")
    
    # Try to initialize Vertex AI and load the Gemini model
    print("\nInitializing Vertex AI...")
    try:
        vertexai.init(project=project_id, location=location)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("The opposite of hot is")
        print(response.text)
    except Exception as e:
        print(f"❌ Failed to list models: {str(e)}")
        return False
        

if __name__ == "__main__":
    if test_gemini_connection():
        print("\n✨ Gemini model test successful! You can now use the model for fine-tuning.")
        sys.exit(0)
    else:
        print("\n❌ Gemini model test failed. Please fix the issues above and try again.")
        sys.exit(1) 