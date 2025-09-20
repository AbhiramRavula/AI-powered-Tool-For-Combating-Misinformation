# backend/test_model.py
from transformers import pipeline

# Load the same model you reference in app.py
classifier = pipeline(
    "text-classification",
    model="roberta-base-finetuned-fakenews",
    return_all_scores=True
)

sample = "Breaking: Scientists discover cure for common cold!"
result = classifier(sample[:512])
print("Prediction:", result)
