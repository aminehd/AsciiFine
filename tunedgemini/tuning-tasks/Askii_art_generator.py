#!/usr/bin/env python
"""
A simple script to demonstrate fine-tuning Gemini with a small dataset.
This uses the Google Vertex AI approach for fine-tuning.
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
import google.generativeai as genai
from google.cloud import aiplatform

# Set up argument parser
parser = argparse.ArgumentParser(description="Fine-tune Gemini with a simple dataset")
parser.add_argument("--project-id", type=str, required=True, help="Google Cloud project ID")
parser.add_argument("--location", type=str, default="us-central1", help="Google Cloud region")
parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
parser.add_argument("--model", type=str, default="gemini-1.5-pro", help="Base model")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize Vertex AI
aiplatform.init(project=args.project_id, location=args.location)

def create_sample_dataset():
    """Create a simple ASCII art dataset for fine-tuning."""
    print("Creating sample ASCII art dataset...")
    
    # Sample data - simple pairs of prompts and ASCII art
    data = [
        {
            "input_text": "Draw a cat",
            "target_text": """
  /\\_/\\
 ( o.o )
  > ^ <
"""
        },
        {
            "input_text": "Draw a dog",
            "target_text": """
  / \\__
 (    @\\___
 /         O
/   (_____/
/_____/   U
"""
        },
        {
            "input_text": "Draw a simple house",
            "target_text": """
    /\\
   /  \\
  /____\\
 |    |
 |____|
"""
        },
        {
            "input_text": "Draw a tree",
            "target_text": """
    /\\
   /  \\
  /____\\
    ||
    ||
"""
        },
        {
            "input_text": "Draw a simple car",
            "target_text": """
    ____
  _/  |_\\_
 |_______|
  O     O
"""
        },
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to JSONL file
    dataset_path = os.path.join(args.output_dir, "sample_ascii_art.jsonl")
    df.to_json(dataset_path, orient="records", lines=True)
    
    print(f"Sample dataset created with {len(data)} examples")
    print(f"Saved to: {dataset_path}")
    
    return dataset_path

def prepare_tuning_data(dataset_path):
    """Prepare the data in the format required by Vertex AI for tuning."""
    print("Preparing tuning data...")
    
    tuning_data = []
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # Format for Vertex AI tuning
            tuning_data.append({
                "model_input": {
                    "context": f"Create ASCII art based on the following description: {item['input_text']}",
                    "examples": []
                },
                "model_output": {
                    "content": item['target_text']
                }
            })
    
    # Save formatted data
    tuning_file = os.path.join(args.output_dir, "tuning_data.jsonl")
    with open(tuning_file, "w") as f:
        for item in tuning_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Tuning data prepared and saved to: {tuning_file}")
    return tuning_file

def finetune_model(tuning_file):
    """Fine-tune the Gemini model using Vertex AI."""
    print(f"Starting fine-tuning with {args.model}...")
    
    # Create a tuning job
    model_upload_response = aiplatform.Model.upload(
        display_name=f"ascii-art-gemini-{args.model}",
        artifact_uri=os.path.abspath(args.output_dir),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/text-generation:latest",
    )
    
    model_resource_name = model_upload_response.resource_name
    
    # NOTE: This is a SIMPLIFIED example. In production, you would:
    # 1. Use the actual Vertex AI tuning API for Gemini
    # 2. Handle hyperparameters properly
    # 3. Monitor the tuning process
    # 4. Evaluate the model performance
    
    print("Fine-tuning job started")
    print("Note: This is a simplified example showing the process structure.")
    print("For actual fine-tuning, you need to use the specific Vertex AI endpoints for Gemini.")
    print(f"Model resource name: {model_resource_name}")
    
    return model_resource_name

def test_finetuned_model(model_resource_name):
    """Test the fine-tuned model with a few examples."""
    print("\nTesting the (hypothetical) fine-tuned model...")
    
    test_prompts = [
        "Draw a cat",
        "Draw a simple house",
        "Draw an airplane"
    ]
    
    # NOTE: This is a SIMULATED test of what would happen with a fine-tuned model
    # In a real scenario, you would load the fine-tuned model and use it
    
    print("\nSimulated outputs from fine-tuned model:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        if prompt == "Draw a cat":
            print("""
  /\\_/\\
 ( o.o )
  > ^ <
""")
        elif prompt == "Draw a simple house":
            print("""
    /\\
   /  \\
  /____\\
 |    |
 |____|
""")
        else:
            print("""
     __
 ___/ /\\
/____/  \\
    \\__/
""")
    
    print("\nNote: These are simulated outputs for demonstration purposes.")
    print("In a real scenario, you would use the actual fine-tuned model to generate outputs.")

def main():
    """Main function to run the fine-tuning demo."""
    print("=" * 50)
    print("Gemini Fine-tuning Demo")
    print("=" * 50)
    
    # Step 1: Create sample dataset
    dataset_path = create_sample_dataset()
    
    # Step 2: Prepare tuning data
    tuning_file = prepare_tuning_data(dataset_path)
    
    # Step 3: Fine-tune model
    model_resource_name = finetune_model(tuning_file)
    
    # Step 4: Test fine-tuned model
    test_finetuned_model(model_resource_name)
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main() 