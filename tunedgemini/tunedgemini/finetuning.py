"""
Fine-tuning utilities for Gemini models.
"""

import time
import random
from sklearn.metrics import accuracy_score, classification_report

def test_base_model(df_test, args, genai_available):
    """
    Test the performance of the base model on the test set.
    
    Args:
        df_test: Test DataFrame
        args: Command-line arguments
        genai_available: Whether Google AI Studio dependencies are installed
    """
    if not args.test_base_model:
        return
    
    if not genai_available:
        print("Skipping base model testing: Google AI Studio dependencies not available.")
        return
    
    print("Testing base model performance...")
    
    try:
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=args.api_key)
        
        # Load the base model
        model = genai.GenerativeModel('gemini-1.0-pro')
        
        # Sample a small subset for testing (to save time and API calls)
        sample_size = min(20, len(df_test))
        df_sample = df_test.sample(sample_size, random_state=42)
        
        predictions = []
        actual_labels = []
        
        for i, (_, row) in enumerate(df_sample.iterrows()):
            prompt = f"Classify the following text into one of the 20 newsgroups categories:\n\n{row['text']}"
            
            try:
                response = model.generate_content(prompt)
                prediction = response.text.strip()
                predictions.append(prediction)
                actual_labels.append(row['label'])
                
                # Print progress
                print(f"Tested {i+1}/{sample_size} examples", end="\r")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
            
            except Exception as e:
                print(f"Error generating prediction: {e}")
                predictions.append("error")
                actual_labels.append(row['label'])
        
        print("\nBase model testing completed.")
        
        # Calculate accuracy (exact match)
        exact_match = [1 if p == a else 0 for p, a in zip(predictions, actual_labels)]
        accuracy = sum(exact_match) / len(exact_match)
        
        print(f"Base model accuracy on sample: {accuracy:.2f}")
        
        # Print some examples
        print("\nExample predictions:")
        for i in range(min(5, len(predictions))):
            print(f"Text: {df_sample.iloc[i]['text'][:100]}...")
            print(f"Actual: {actual_labels[i]}")
            print(f"Predicted: {predictions[i]}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error testing base model: {e}")


def finetune_with_vertex(tuning_file_gcs, args, vertex_available):
    """
    Fine-tune a Gemini model using Vertex AI.
    
    Args:
        tuning_file_gcs: GCS URI of the training data file
        args: Command-line arguments
        vertex_available: Whether Vertex AI dependencies are installed
        
    Returns:
        object: Vertex AI TuningJob
    """
    if not vertex_available:
        print("Skipping Vertex AI fine-tuning: dependencies not available.")
        return None
    
    print("Starting fine-tuning with Vertex AI...")
    
    try:
        from google.cloud import aiplatform
        
        # Initialize the platform
        aiplatform.init(project=args.project_id, location=args.location)
        
        # Create a tuning job
        job = aiplatform.TuningJob.create(
            base_model=f"gemini-1.0-pro-{args.model_size}",
            tuning_data=tuning_file_gcs,
            tuned_model_display_name=f"newsgroups-classifier-{int(time.time())}",
            training_steps=args.training_steps,
            learning_rate=args.learning_rate,
            tuning_job_location=args.location
        )
        
        print(f"Vertex AI fine-tuning job started: {job.resource_name}")
        print("Fine-tuning is running in the background and may take several hours.")
        
        return job
    
    except Exception as e:
        print(f"Error starting Vertex AI fine-tuning: {e}")
        return None


def finetune_with_genai(tuning_file, args, genai_available):
    """
    Fine-tune a Gemini model using Google AI Studio.
    
    Args:
        tuning_file: Path to the training data file
        args: Command-line arguments
        genai_available: Whether Google AI Studio dependencies are installed
        
    Returns:
        object: Google AI Studio TuningJob
    """
    if not genai_available:
        print("Skipping Google AI Studio fine-tuning: dependencies not available.")
        return None
    
    print("Starting fine-tuning with Google AI Studio...")
    
    try:
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=args.api_key)
        
        # Create a tuning job
        job = genai.create_tuning_job(
            model=f"gemini-1.0-pro-{args.model_size}",
            training_data=tuning_file,
            tuned_model_name=f"newsgroups-classifier-{int(time.time())}",
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        print(f"Google AI Studio fine-tuning job started: {job.id}")
        print("Fine-tuning is running in the background and may take several hours.")
        
        return job
    
    except Exception as e:
        print(f"Error starting Google AI Studio fine-tuning: {e}")
        return None 