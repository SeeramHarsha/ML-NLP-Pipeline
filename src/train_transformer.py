import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer, IntervalStrategy
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

def train_transformer_model(input_file, model_output_dir):
    # Load the cleaned dataset
    df = pd.read_csv(input_file)

    # Map labels to integers
    unique_labels = df['label'].unique()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}
    df['label_int'] = df['label'].map(label_to_int)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['reply'], df['label_int'], test_size=0.2, random_state=42
    )

    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples, truncation=True, padding=True, max_length=128)

    train_encodings = tokenize_function(X_train.tolist())
    test_encodings = tokenize_function(X_test.tolist())

    # Create Hugging Face Dataset
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': y_train.tolist()
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': y_test.tolist()
    })

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(unique_labels)
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        # logging_dir='./logs', # Removed for simplicity
        # logging_steps=10, # Removed for simplicity
        # evaluation_strategy=IntervalStrategy.EPOCH, # Removed due to TypeError
        # save_strategy=IntervalStrategy.EPOCH, # Removed due to TypeError
        # load_best_model_at_end=True, # Removed due to TypeError
        # metric_for_best_model="f1", # Removed due to TypeError
        report_to="none" # Disable reporting to services like W&B
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # Save label mappings
    pd.DataFrame([int_to_label]).to_csv(f"{model_output_dir}/label_mapping.csv", index=False)

if __name__ == "__main__":
    train_transformer_model(
        'data/cleaned_emails.csv',
        './models/distilbert_finetuned'
    )