# Usage
# import from huggingface
import datasets

# Load the lite configuration of the dataset
raw_dataset = datasets.load_dataset("JanosAudran/financial-reports-sec", "large_lite")

# Load a specific split
raw_dataset = datasets.load_dataset("JanosAudran/financial-reports-sec", "small_full", split="train")