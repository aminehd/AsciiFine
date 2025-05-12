#!/usr/bin/env python
"""
A simple script to test the connection to Gemini models on Vertex AI.
"""

import os
import dotenv
from google import genai
# import something from tunedgemini 
from tunedgemini.data_loader import  sample_data, load_data
from tunedgemini.fine_tune import fine_tune, get_tuned_model
from tunedgemini.predict_eval import  eval_model, eval_tuned_model

# Load environment variables from .env file
dotenv.load_dotenv()
system_instruct = """
You are a classification service. You will be passed input that represents
a newsgroup post and you must respond with the newsgroup from which the post
originates.
"""
def setup_gemini_client():
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
    # api key is in .env file
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return client


# def ():



if __name__ == "__main__":
    client = setup_gemini_client()
    df_train, df_test = load_data()


    df_baseline_eval = eval_model(client, df_test, "gemini-1.5-flash-001")
    model_id = fine_tune(client, df_train, base_model="gemini-1.5-flash-001")
    tuned_model = get_tuned_model(client, model_id)
    print(f"Done! The model state is: {tuned_model.state.name}")
    # The sampling here is just to minimise your quota usage. If you can, you should
    # evaluate the whole test set with `df_model_eval = df_test.copy()`.
    
    
    df_tuned_eval = eval_tuned_model(client, df_test, model_id)
    
