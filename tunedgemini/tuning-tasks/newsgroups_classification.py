#!/usr/bin/env python
"""
Fine-tune Gemini on the 20 Newsgroups dataset for text classification.
source: https://www.kaggle.com/code/markishere/day-4-fine-tuning-a-custom-model
"""

import os
import argparse
from pathlib import Path

# Import local modules
from tunedgemini.data_processing import prepare_dataset, format_for_vertex_tuning, format_for_genai_tuning
from tunedgemini.api_utils import check_api_availability, initialize_apis, upload_to_gcs
from tunedgemini.finetuning import finetune_with_vertex, finetune_with_genai, test_base_model
from tunedgemini.cli import parse_arguments, print_next_steps

# Check API availability
try:
    from google.cloud import aiplatform
    vertex_available = True
except ImportError:
    vertex_available = False

try:
    import google.generativeai as genai
    genai_available = True
except ImportError:
    genai_available = False


def main():
    """Main function to run the fine-tuning pipeline."""
    print("=" * 60)
    print("Gemini Fine-tuning for 20 Newsgroups Classification")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check API availability
    check_api_availability(args, vertex_available, genai_available)
    
    # Initialize APIs
    initialize_apis(args, vertex_available, genai_available)
    
    # Step 1: Prepare the dataset
    df_train, df_test = prepare_dataset(args)
    
    # Step 2: Test base model (optional)
    test_base_model(df_test, args, genai_available)
    
    # Step 3: Format data for the selected approaches
    vertex_tuning_file = None
    genai_tuning_file = None
    
    if args.approach in ["vertex", "both"] and vertex_available and args.project_id:
        vertex_tuning_file = format_for_vertex_tuning(df_train, args.output_dir)
        
    if args.approach in ["genai", "both"] and genai_available and args.api_key:
        genai_tuning_file = format_for_genai_tuning(df_train, args.output_dir)
    
    # Step 4: Upload to GCS if using Vertex AI
    vertex_tuning_file_gcs = None
    if vertex_tuning_file:
        vertex_tuning_file_gcs = upload_to_gcs(vertex_tuning_file, args, vertex_available)
    
    # Step 5: Start fine-tuning with selected approaches
    vertex_job = None
    genai_job = None
    
    if vertex_tuning_file_gcs:
        vertex_job = finetune_with_vertex(vertex_tuning_file_gcs, args, vertex_available)
        
    if genai_tuning_file:
        genai_job = finetune_with_genai(genai_tuning_file, args, genai_available)
    
    # Step 6: Provide instructions for next steps
    print_next_steps(vertex_job, genai_job, args.output_dir)


if __name__ == "__main__":
    main()