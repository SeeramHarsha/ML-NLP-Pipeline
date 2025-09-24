import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np
import json

def evaluate_baseline_model(input_file, model_path, vectorizer_path):
    df = pd.read_csv(input_file)
    X = df['reply']
    y = df['label']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, f1

def evaluate_transformer_model(input_file, model_dir):
    df = pd.read_csv(input_file)
    
    # Load label mapping
    label_mapping_df = pd.read_csv(f"{model_dir}/label_mapping.csv")
    # Ensure keys are integers
    int_to_label = {int(k): v for k, v in label_mapping_df.iloc[0].to_dict().items()}
    label_to_int = {v: k for k, v in int_to_label.items()}

    df['label_int'] = df['label'].map(label_to_int)

    X = df['reply']
    y = df['label_int']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    predictions = []
    true_labels = []

    for text, label in zip(X_test, y_test):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        pred_label_int = torch.argmax(logits, dim=1).item()
        predictions.append(pred_label_int)
        true_labels.append(label)

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1

if __name__ == "__main__":
    cleaned_data_path = 'data/cleaned_emails.csv'
    baseline_model_path = 'models/baseline_model.joblib'
    tfidf_vectorizer_path = 'models/tfidf_vectorizer.joblib'
    transformer_model_dir = './models/distilbert_finetuned'

    print("Evaluating Baseline Model...")
    baseline_accuracy, baseline_f1 = evaluate_baseline_model(
        cleaned_data_path, baseline_model_path, tfidf_vectorizer_path
    )
    print(f"Baseline Model - Accuracy: {baseline_accuracy:.4f}, F1 Score: {baseline_f1:.4f}")

    print("\nEvaluating Transformer Model...")
    transformer_accuracy, transformer_f1 = evaluate_transformer_model(
        cleaned_data_path, transformer_model_dir
    )
    print(f"Transformer Model - Accuracy: {transformer_accuracy:.4f}, F1 Score: {transformer_f1:.4f}")

    # Compare and decide
    if transformer_f1 > baseline_f1:
        print("\nTransformer model performs better and is recommended for production.")
        best_model = "Transformer"
    else:
        print("\nBaseline model performs better and is recommended for production.")
        best_model = "Baseline"
    
    # Save results
    results = {
        "baseline_model": {
            "accuracy": baseline_accuracy,
            "f1_score": baseline_f1
        },
        "transformer_model": {
            "accuracy": transformer_accuracy,
            "f1_score": transformer_f1
        },
        "best_model_for_production": best_model
    }

    with open('results/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nModel comparison results saved to results/model_comparison.json")