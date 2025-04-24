"""
Data processing utilities for the 20 Newsgroups dataset.
"""

import os
import json
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

def prepare_dataset(args):
    """
    Prepare the 20 Newsgroups dataset for fine-tuning.
    
    Args:
        args: Command-line arguments
        
    Returns:
        tuple: (training DataFrame, test DataFrame)
    """
    print("Preparing the 20 Newsgroups dataset...")
    
    # Fetch the 20 Newsgroups dataset
    categories = None  # Use all categories
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                         remove=('headers', 'footers', 'quotes'),
                                         random_state=42)
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': newsgroups_train.data,
        'label': [newsgroups_train.target_names[idx] for idx in newsgroups_train.target]
    })
    
    # Split into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Dataset prepared: {len(df_train)} training examples, {len(df_test)} test examples")
    
    return df_train, df_test

def format_for_vertex_tuning(df_train, output_dir):
    """
    Format the training data for Vertex AI tuning.
    
    Args:
        df_train: Training DataFrame
        output_dir: Directory to save the formatted data
        
    Returns:
        str: Path to the formatted data file
    """
    print("Formatting data for Vertex AI tuning...")
    
    # Create tuning examples in JSONL format required by Vertex AI
    tuning_examples = []
    
    for _, row in df_train.iterrows():
        example = {
            "input_text": f"Classify the following text into one of the 20 newsgroups categories:\n\n{row['text']}",
            "output_text": row['label']
        }
        tuning_examples.append(example)
    
    # Write to JSONL file
    output_file = os.path.join(output_dir, "vertex_tuning_data.jsonl")
    with open(output_file, 'w') as f:
        for example in tuning_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Formatted data saved to {output_file}")
    return output_file

def format_for_genai_tuning(df_train, output_dir):
    """
    Format the training data for Google AI Studio tuning.
    
    Args:
        df_train: Training DataFrame
        output_dir: Directory to save the formatted data
        
    Returns:
        str: Path to the formatted data file
    """
    print("Formatting data for Google AI Studio tuning...")
    
    # Create tuning examples in JSONL format required by Google AI Studio
    tuning_examples = []
    
    for _, row in df_train.iterrows():
        example = {
            "messages": [
                {"role": "user", "content": f"Classify the following text into one of the 20 newsgroups categories:\n\n{row['text']}"},
                {"role": "model", "content": row['label']}
            ]
        }
        tuning_examples.append(example)
    
    # Write to JSONL file
    output_file = os.path.join(output_dir, "genai_tuning_data.jsonl")
    with open(output_file, 'w') as f:
        for example in tuning_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Formatted data saved to {output_file}")
    return output_file 