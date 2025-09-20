# backend/test_model.py
from transformers import pipeline

# Load the updated public model
classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-sst2",
    return_all_scores=True
)

sample = "Scientists uncover a groundbreaking vaccine!"
result = classifier(sample[:512])
print("Prediction:", result)
