import json
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the saved test dataset from disk (saved during fine-tuning)
test_dataset = load_from_disk("D:/EEG/src/test_dataset")

# Load the fine-tuned model and tokenizer from disk
model_path = "D:/EEG/src/saved_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Define an inference function that generates output and extracts the predicted label.
def predict_label(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Assume the first line of generated output is the predicted emotion label
    pred_label = output_text.splitlines()[0].strip()
    return pred_label

# Run inference on the test set and collect predictions and true labels
true_labels = []
pred_labels = []

print("Running inference on the test set...")
for sample in test_dataset:
    # Assumes test dataset contains the original "prompt" and "completion" fields.
    prompt = sample["prompt"]
    true_label = sample["completion"].strip()
    pred_label = predict_label(prompt)
    true_labels.append(true_label)
    pred_labels.append(pred_label)

# Compute classification metrics
accuracy = accuracy_score(true_labels, pred_labels)
conf_matrix = confusion_matrix(true_labels, pred_labels)
report = classification_report(true_labels, pred_labels)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
