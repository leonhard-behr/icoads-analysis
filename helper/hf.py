"""huggingface helper functions"""
from datasets import load_from_disk
import os

def load_dataset(dataset_path="./hf_dataset", split=None):
    """
    Load HuggingFace dataset
    Args:
        dataset_path (str): path to saved HF dataset
        split (str): specific split to load (e.g., "3", "4"), if None loads all splits
    Returns:
        Dataset/DatasetDict: HuggingFace dataset
    """
    if os.path.exists(dataset_path):
        print("Loading HuggingFace dataset...")
        dataset = load_from_disk(dataset_path)
        
        if split is not None:
            if isinstance(dataset, dict) and split in dataset:
                print(f"Loading split '{split}'...")
                return dataset[split]
            else:
                print(f"Split '{split}' not found. Available splits: {list(dataset.keys()) if isinstance(dataset, dict) else 'No splits available'}")
                return None
        
        return dataset
    else:
        print("Dataset not found.")
        return None
