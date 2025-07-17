import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Predefined candidate labels
labels = ["Politics", "Technology", "Health", "Entertainment", "Sports", "Finance", "Science", "World", "Business"]

def classify_article(text):
    try:
        result = classifier(text, candidate_labels=labels)
        top_label = result["labels"][0]
        return top_label
    except Exception as e:
        print(f"Classification error: {e}")
        return "Unknown"
