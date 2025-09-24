# Email Sentiment Analysis API

This project implements an NLP pipeline to classify email replies into positive, negative, and neutral categories, and deploys the best-performing model as a FastAPI service.

## Project Structure

The project is organized into the following directories and files:

-   `api/`: Contains the FastAPI application for model deployment.
    -   `main.py`: The main FastAPI application file with the `/predict` endpoint.
-   `data/`: Stores the raw and preprocessed datasets.
    -   `emails.csv`: The original CSV dataset of email replies.
    -   `cleaned_emails.csv`: The preprocessed version of `emails.csv` after cleaning and normalization.
-   `results/`: Stores evaluation results and model comparisons.
    -   `model_comparison.json`: JSON file containing accuracy and F1 scores for both models and the recommended production model.
-   `src/`: Contains Python scripts for data preprocessing, model training, and evaluation.
    -   `preprocess.py`: Script for cleaning and preparing the email dataset.
    -   `train_baseline.py`: Script for training the TF-IDF + Logistic Regression baseline model.
    -   `train_transformer.py`: Script for fine-tuning the DistilBERT transformer model.
    -   `evaluate_models.py`: Script for evaluating and comparing the performance of both models.
-   `README.md`: This file, providing an overview, setup instructions, and execution guide.
-   `requirements.txt`: Lists all Python dependencies required for the project.

## Getting Started

Follow these steps to set up the project, run the ML/NLP pipeline, and deploy the API locally.

### 1. Clone the Repository (if applicable)

If you haven't already, clone this repository to your local machine.

```bash
git clone <repository-url>
cd nlp
```

### 2. Set Up Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Part A – ML/NLP Pipeline: Step-by-Step Execution

Execute the following scripts in the specified order to preprocess data, train models, and evaluate them.

### Step 1: Preprocess Data

This script cleans the raw email data and saves it to `data/cleaned_emails.csv`.

```bash
python src/preprocess.py
```

### Step 2: Train Baseline Model

This script trains a Logistic Regression model using TF-IDF features and saves the model artifacts.

```bash
python src/train_baseline.py
```

### Step 3: Fine-tune Transformer Model

This script fine-tunes a DistilBERT model and saves the model artifacts. This step may take some time depending on your hardware.

```bash
python src/train_transformer.py
```

### Step 4: Evaluate Models

This script evaluates both the baseline and transformer models, compares their performance, and saves the results to `results/model_comparison.json`.

```bash
python src/evaluate_models.py
```

## Part B – Deployment Task: Running the API

After completing the ML/NLP pipeline, you can deploy the best-performing model (DistilBERT) as a FastAPI service.

### Step 1: Run the FastAPI Service

Execute the following command from the project root directory to start the API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be accessible at `http://localhost:8000` or `http://127.0.0.1:8000`.
If you encounter "ERR_ADDRESS_INVALID" or "This site can't be reached", ensure the server is running and try accessing `http://localhost:8000`. Check your terminal for any error messages if the server fails to start.

### Step 2: Make a Prediction Request

You can make a POST request to the `/predict` endpoint with a JSON body containing the `text` of the email reply.

**Request Body Example:**
```json
{
  "text": "Looking forward to the demo!"
}
```

**Example using `curl`:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Looking forward to the demo!\"}"
```

**Response Example:**
```json
{
  "label": "positive",
  "confidence": 0.99
}
```

## Requirements

The `requirements.txt` file lists all Python dependencies required for this project:

```
fastapi
uvicorn
transformers
torch
numpy
datasets
scikit-learn
pandas
