"""huggingface helper functions"""
from datasets import load_from_disk, load_dataset as load_dataset_hf
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


# cache dictionary to store loaded datasets
_icoads_cache = {}

def load_icoads_subset(split: str, token: str = None):
    """
    Load the specified ICOADS subset (Group 3–7 or 9).
    Downloads and caches all subsets upon first use.

    Parameters:
        split (str): One of "3", "4", "5", "6", "7", or "9"

    Returns:
        DatasetDict: Hugging Face dataset for the given group
    """
    valid_splits = ["3", "4", "5", "6", "7", "9"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Choose from {valid_splits}.")

    if not _icoads_cache:
        print("Downloading and caching all ICOADS subsets (Groups 3–7, 9)...")
        for group in valid_splits:
            dataset_name = f"leonhard-behr/msg1-enh-icoads-subset-{group}"
            _icoads_cache[group] = load_dataset_hf(dataset_name, token=token)
    
    return _icoads_cache[split]
