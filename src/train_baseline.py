import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_baseline_model(input_file, model_output_path, vectorizer_output_path):
    # Load the cleaned dataset
    df = pd.read_csv(input_file)

    # Split data into training and testing sets
    X = df['reply']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    # Fit and transform the training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    # Transform the test data
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Baseline Model Accuracy: {accuracy}")
    print(f"Baseline Model F1 Score: {f1}")

    # Save the trained model and vectorizer
    joblib.dump(model, model_output_path)
    joblib.dump(tfidf_vectorizer, vectorizer_output_path)

if __name__ == "__main__":
    train_baseline_model(
        'data/cleaned_emails.csv',
        'models/baseline_model.joblib',
        'models/tfidf_vectorizer.joblib'
    )