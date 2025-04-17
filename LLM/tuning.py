import json
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Path to the prompt-completion pairs JSONL file
pairs_file_path = "D:/EEG/src/data/prompt_completion_pairs.jsonl"

# Load the full dataset using the Hugging Face datasets library
dataset = load_dataset("json", data_files={"full": pairs_file_path})["full"]

# Split dataset: 80% training, 20% test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

train_dataset.save_to_disk("D:/EEG/src/train_dataset")
test_dataset.save_to_disk("D:/EEG/src/test_dataset")

print(f"Total training samples before sampling: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# Combine 'prompt' and 'completion' into a single 'text' field
def combine_fields(example):
    example["text"] = example["prompt"].strip() + "\n" + example["completion"].strip()
    return example

train_dataset = train_dataset.map(combine_fields)
test_dataset = test_dataset.map(combine_fields)


# Take 5% of the training data as a sample for fine-tuning
sample_size = max(1, int(0.05 * len(train_dataset)))
train_sample = train_dataset.shuffle(seed=42).select(range(sample_size))
print(f"Using {sample_size} training samples for fine-tuning (5% of training data)")

# Choose the model for fine-tuning (using PyTorch)
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the datasets using the 'text' field, adding a "labels" field so the model can compute loss.
def tokenize_function(example):
    encodings = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

train_sample = train_sample.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Optional: Remove unneeded columns to save memory
columns_to_remove = [col for col in train_sample.column_names if col not in ["input_ids", "attention_mask", "labels"]]
train_sample = train_sample.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)

# Define a data collator for language modeling (which handles padding)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="D:/EEG/src/results",      # Directory for saving model checkpoints and outputs
    num_train_epochs=4,                   # Number of training epochs
    per_device_train_batch_size=4,        # Adjust based on your GPU memory
    save_steps=500,                       # Save a checkpoint every 500 steps
    save_total_limit=2,                   # Only keep the 2 most recent checkpoints
    logging_steps=100,                    # Log every 100 steps
    evaluation_strategy="epoch",          # Evaluate at the end of each epoch
)

# Initialize the Trainer with the train sample and eval datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_sample,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# Start the fine-tuning process
trainer.train()

# Evaluate on the test set
results = trainer.evaluate()
print("Evaluation Results:", results)

# Save the fine-tuned model and tokenizer for future evaluation/deployment
trainer.save_model("D:/EEG/src/saved_model")
tokenizer.save_pretrained("D:/EEG/src/saved_model")
print("Fine-tuned model and tokenizer saved successfully.")
