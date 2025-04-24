#!/usr/bin/env python
"""
A simple script to test the connection to Google Cloud and Vertex AI.
"""

import os
import sys
import dotenv
from google.cloud import aiplatform

# Load environment variables from .env file
dotenv.load_dotenv()

def test_connection():
    """Test connection to Google Cloud and Vertex AI."""
    # Check if credentials are properly set
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    
    print("\n=== Google Cloud Connection Test ===\n")
    
    # Check environment variables
    if not credentials_path:
        print("❌ GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print("   Please add it to your .env file or export it directly.")
        return False
    
    if not os.path.exists(credentials_path):
        print(f"❌ Credentials file not found at: {credentials_path}")
        print("   Please check that the path is correct.")
        return False
    
    print(f"✅ Credentials file found at: {credentials_path}")
    
    if not project_id:
        print("❌ GCP_PROJECT_ID environment variable is not set.")
        print("   Please add it to your .env file or export it directly.")
        return False
    
    print(f"✅ Project ID set to: {project_id}")
    print(f"✅ Location set to: {location}")
    
    # Try to initialize the Vertex AI client
    print("\nAttempting to connect to Vertex AI...")
    try:
        aiplatform.init(project=project_id, location=location)
        print("✅ Successfully connected to Vertex AI!")
        
        # Get available models (this will fail if authentication isn't working)
        print("\nFetching available model types...")
        model_types = aiplatform.Model.list(filter="")
        print(f"✅ Found {len(model_types)} model types")
        
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Vertex AI: {str(e)}")
        print("\nPossible issues:")
        print("  - The credentials file may be invalid or expired")
        print("  - The project ID may be incorrect")
        print("  - The Vertex AI API may not be enabled for this project")
        print("  - There might be network connectivity issues")
        print("\nTo enable Vertex AI API, visit:")
        print(f"  https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project={project_id}")
        return False

if __name__ == "__main__":
    if test_connection():
        print("\n✨ Connection test successful! Your project is properly connected to Google Cloud.")
        print("   You can now use the tunedgemini package to fine-tune models and generate ASCII art.")
        sys.exit(0)
    else:
        print("\n❌ Connection test failed. Please fix the issues above and try again.")
        sys.exit(1) 