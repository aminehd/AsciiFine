"""
Utilities for interacting with Vertex AI and Google AI Studio APIs.
"""

import os
import sys
from google.cloud import storage

def check_api_availability(args, vertex_available, genai_available):
    """
    Check if the selected APIs are available based on dependencies and credentials.
    
    Args:
        args: Command-line arguments
        vertex_available: Whether Vertex AI dependencies are installed
        genai_available: Whether Google AI Studio dependencies are installed
    """
    print("Checking API availability...")
    
    # Check if the selected approach is available
    if args.approach in ["vertex", "both"] and not vertex_available:
        print("Warning: Vertex AI dependencies are not installed. Run: pip install google-cloud-aiplatform")
        if args.approach == "vertex":
            sys.exit(1)
    
    if args.approach in ["genai", "both"] and not genai_available:
        print("Warning: Google AI Studio dependencies are not installed. Run: pip install google-generativeai")
        if args.approach == "genai":
            sys.exit(1)
    
    # Check for required credentials
    if args.approach in ["vertex", "both"]:
        if not args.project_id:
            print("Error: --project-id is required for Vertex AI tuning.")
            sys.exit(1)
        
        if not args.bucket_name and args.approach == "vertex":
            print("Error: --bucket-name is required for Vertex AI tuning.")
            sys.exit(1)
    
    if args.approach in ["genai", "both"] and not args.api_key:
        print("Error: --api-key is required for Google AI Studio tuning.")
        sys.exit(1)
    
    print("API availability check completed.")

def initialize_apis(args, vertex_available, genai_available):
    """
    Initialize the required APIs for fine-tuning.
    
    Args:
        args: Command-line arguments
        vertex_available: Whether Vertex AI dependencies are installed
        genai_available: Whether Google AI Studio dependencies are installed
    """
    print("Initializing APIs...")
    
    # Initialize Vertex AI if needed
    if args.approach in ["vertex", "both"] and vertex_available:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=args.project_id, location=args.location)
            print(f"Initialized Vertex AI for project {args.project_id} in {args.location}")
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            if args.approach == "vertex":
                sys.exit(1)
    
    # Initialize Google AI Studio if needed
    if args.approach in ["genai", "both"] and genai_available:
        try:
            import google.generativeai as genai
            genai.configure(api_key=args.api_key)
            print("Initialized Google AI Studio with provided API key")
        except Exception as e:
            print(f"Error initializing Google AI Studio: {e}")
            if args.approach == "genai":
                sys.exit(1)
    
    print("API initialization completed.")

def upload_to_gcs(file_path, args, vertex_available):
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        file_path: Path to the file to upload
        args: Command-line arguments
        vertex_available: Whether Vertex AI dependencies are installed
        
    Returns:
        str: GCS URI of the uploaded file
    """
    print(f"Uploading {file_path} to GCS bucket {args.bucket_name}...")
    
    if not vertex_available:
        print("Error: Vertex AI dependencies are required for GCS upload.")
        sys.exit(1)
    
    try:
        # Initialize storage client
        storage_client = storage.Client(project=args.project_id)
        
        # Get bucket
        bucket = storage_client.bucket(args.bucket_name)
        
        # Prepare destination blob name
        blob_name = f"tuning_data/{os.path.basename(file_path)}"
        blob = bucket.blob(blob_name)
        
        # Upload file
        blob.upload_from_filename(file_path)
        
        # Form the GCS URI
        gcs_uri = f"gs://{args.bucket_name}/{blob_name}"
        print(f"File uploaded successfully to {gcs_uri}")
        
        return gcs_uri
    
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        sys.exit(1) 