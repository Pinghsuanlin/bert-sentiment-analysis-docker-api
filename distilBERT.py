# Install necessary libraries if not already installed
# !pip install transformers datasets torch accelerate -U

import pandas as pd
from datasets import Dataset, load_metric
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from transformers import TrainingArguments, Trainer
import torch
import numpy as np

# --- 1. Data Loading and Preparation for Hugging Face ---
print("Step 1: Loading and Preparing Data for DistilBERT...")

# Assuming the IMDB dataset is in a file named 'IMDB_Dataset.csv'
df = pd.read_csv('IMDB_Dataset.csv')
# Sample a smaller set to manage training time, especially on limited hardware
df = df.sample(n=5000, random_state=42).reset_index(drop=True)

# Encode labels: 'positive' -> 1, 'negative' -> 0 (DistilBERT is a binary classification model)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.rename(columns={'review': 'text'}) # Rename column for Hugging Face standard

# Reasoning: Hugging Face 'datasets' is more efficient and standard for handling 
# text data for transformer models. Convert the pandas DataFrame.
hf_dataset = Dataset.from_pandas(df[['text', 'label']])

# Split the Hugging Face dataset
train_test_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# --- 2. Tokenization ---
print("Step 2: Loading Tokenizer and Tokenizing Data...")

# Reasoning: The tokenizer ensures that the input text is converted into tokens 
# (sub-word units) and IDs that the DistilBERT model was pre-trained on. 
# DistilBERT-base-uncased is case-insensitive.
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Function to apply tokenization to the dataset
def tokenize_function(examples):
    # Reasoning: We use truncation=True to handle long reviews (max length is 512 for DistilBERT)
    # and padding=True to ensure all sequences have the same length for batch processing.
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

# Apply tokenization to both datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Select and format only the columns needed for the model (input_ids, attention_mask, label)
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# --- 3. Model Loading ---
print("Step 3: Loading DistilBERT Model...")

# Reasoning: DistilBertForSequenceClassification adds a classification head 
# (a layer of neurons) on top of the pre-trained DistilBERT body, ready for our 
# binary classification task (num_labels=2).
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# --- 4. Defining Training Arguments and Evaluation Metrics ---
print("Step 4: Defining Trainer Configuration...")

# Reasoning: TrainingArguments define the hyper-parameters for the fine-tuning process.
# output_dir: Where model checkpoints and logs are saved.
# num_train_epochs: How many times to loop through the data.
# per_device_train_batch_size: How many examples per training step.
# gradient_accumulation_steps: Useful if VRAM is limited (simulates larger batch size).
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3, 
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Reasoning: Define a function to compute metrics (e.g., Accuracy and F1) 
# during training and evaluation.
def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# --- 5. Trainer Initialization and Training ---
print("Step 5: Initializing and Training the Trainer...")

# Reasoning: The Trainer object abstracts the PyTorch training loop, making 
# fine-tuning transformers much simpler and less error-prone.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)

# Start fine-tuning!
trainer.train()

# --- 6. Final Evaluation ---
print("\nStep 6: Final Evaluation on Test Set")
# Reasoning: The evaluate method runs the model on the held-out evaluation set 
# to get final performance metrics.
results = trainer.evaluate()
print(results)