"""
Command-line interface utilities for fine-tuning Gemini models.
"""

import argparse
import os

def parse_arguments():
    """
    Parse command-line arguments for fine-tuning.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune Gemini on the 20 Newsgroups dataset")
    
    # General options
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save output files")
    parser.add_argument("--approach", type=str, choices=["vertex", "genai", "both"], default="both",
                        help="Fine-tuning approach to use (default: both)")
    parser.add_argument("--model-size", type=str, choices=["001", "002"], default="001",
                        help="Gemini model size to use (default: 001)")
    parser.add_argument("--test-base-model", action="store_true",
                        help="Test the base model before fine-tuning")
    
    # Vertex AI options
    parser.add_argument("--project-id", type=str, default=None,
                        help="Google Cloud project ID for Vertex AI")
    parser.add_argument("--location", type=str, default="us-central1",
                        help="Google Cloud region for Vertex AI (default: us-central1)")
    parser.add_argument("--bucket-name", type=str, default=None,
                        help="Google Cloud Storage bucket name for Vertex AI")
    parser.add_argument("--training-steps", type=int, default=1000,
                        help="Number of training steps for Vertex AI (default: 1000)")
    
    # Google AI Studio options
    parser.add_argument("--api-key", type=str, default=os.environ.get("GOOGLE_API_KEY"),
                        help="Google AI Studio API key (default: from GOOGLE_API_KEY environment variable)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for Google AI Studio (default: 5)")
    
    # Common fine-tuning options
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for fine-tuning (default: 0.001)")
    
    args = parser.parse_args()
    
    return args

def print_next_steps(vertex_job, genai_job, output_dir):
    """
    Print instructions for next steps after starting fine-tuning jobs.
    
    Args:
        vertex_job: Vertex AI TuningJob object
        genai_job: Google AI Studio TuningJob object
        output_dir: Directory where output files are saved
    """
    print("\n" + "=" * 60)
    print("Fine-tuning Jobs Started")
    print("=" * 60)
    
    # Print information about training data
    print(f"Training data saved in: {output_dir}")
    
    # Print Vertex AI job information
    if vertex_job:
        print("\nVertex AI Fine-tuning:")
        print(f"  Job Name: {vertex_job.resource_name}")
        print("  Check job status in Google Cloud Console:")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={vertex_job.project}")
        print("\n  After completion, you can use the model with:")
        print("  ```python")
        print("  from google.cloud import aiplatform")
        print("  endpoint = model.deploy()")
        print("  response = endpoint.predict(instances=[{\"content\": \"Your text here\"}])")
        print("  ```")
    
    # Print Google AI Studio job information
    if genai_job:
        print("\nGoogle AI Studio Fine-tuning:")
        print(f"  Job ID: {genai_job.id}")
        print("  Check job status in Google AI Studio:")
        print("  https://makersuite.google.com/app/apikey")
        print("\n  After completion, you can use the model with:")
        print("  ```python")
        print("  import google.generativeai as genai")
        print("  genai.configure(api_key='your_api_key')")
        print(f"  model = genai.GenerativeModel('{genai_job.tuned_model_name}')")
        print("  response = model.generate_content(\"Your text here\")")
        print("  ```")
    
    # Print next steps if no jobs were started
    if not vertex_job and not genai_job:
        print("\nNo fine-tuning jobs were started.")
        print("Please check the requirements and try again:")
        print("  - For Vertex AI: pip install google-cloud-aiplatform")
        print("  - For Google AI Studio: pip install google-generativeai")
    
    print("\nFor more information on fine-tuning:")
    print("  - Vertex AI: https://cloud.google.com/vertex-ai/docs/generative-ai/start/tune-models")
    print("  - Google AI Studio: https://ai.google.dev/docs/tune_instructions") 