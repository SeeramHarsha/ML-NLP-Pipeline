import pandas as pd
import re

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def preprocess_data(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Remove missing values
    df.dropna(inplace=True)
    
    # Clean the text replies
    df['reply'] = df['reply'].apply(clean_text)
    
    # Standardize labels to lowercase
    df['label'] = df['label'].str.lower()
    
    # Save the cleaned data
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data('data/emails.csv', 'data/cleaned_emails.csv')