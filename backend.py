# backend.py
import torch
import shap
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model & tokenizer
MODEL_PATH = "./distilbert_femicide_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# SHAP masker & explainer
masker = shap.maskers.Text(tokenizer)
def predict_proba_from_text(texts):
    encoded = tokenizer(list(texts), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

explainer = shap.Explainer(predict_proba_from_text, masker, algorithm="partition")

# FastAPI app
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = [input.text]
    probs = predict_proba_from_text(text)[0]
    shap_values = explainer(text)

    return {
        "probabilities": probs.tolist(),
        "shap_values": shap_values.data[0],  # tokens
        "shap_scores": shap_values.values[0].tolist()  # importance per token
    }
