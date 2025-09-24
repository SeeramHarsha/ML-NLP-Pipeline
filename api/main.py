from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define input and output models for FastAPI
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

# Load the fine-tuned transformer model and tokenizer
model_dir = "./models/distilbert_finetuned"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
model.eval() # Set model to evaluation mode

# Load label mapping
label_mapping_path = os.path.join(model_dir, "label_mapping.csv")
label_mapping_df = pd.read_csv(label_mapping_path)
int_to_label = {int(k): v for k, v in label_mapping_df.iloc[0].to_dict().items()}

@app.get("/")
async def read_root():
    return {"message": "NLP Email Classifier API. Use /predict to get predictions."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    text = request.text

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    
    # Get predicted label and confidence
    predicted_label_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_label_id].item()
    predicted_label = int_to_label[predicted_label_id]

    return PredictionResponse(label=predicted_label, confidence=confidence)

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
