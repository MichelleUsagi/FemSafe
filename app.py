import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

#  Load Model & Tokenizer 
MODEL_PATH = "./distilbert_femicide_model"

@st.cache_resource
def load_model():
<<<<<<< HEAD
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
=======
    MODEL_DIR = os.path.join(os.getcwd(), "distilbert_femicide_model")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
>>>>>>> f518f29a75fc7e0e92b3842f2c570c7160828d06
    return model, tokenizer

model, tokenizer = load_model()

<<<<<<< HEAD
#  Prediction Function 
=======
# Prediction function
>>>>>>> f518f29a75fc7e0e92b3842f2c570c7160828d06
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class, probs

<<<<<<< HEAD
# Streamlit UI 
st.set_page_config(page_title="Femicide Risk Classifier", page_icon="⚖️", layout="centered")
st.title("⚖️ Femicide Risk Classifier")
st.write("Enter text describing a case scenario, and the model will predict the risk category.")
=======
# Streamlit page config
st.set_page_config(page_title="FemSafe", page_icon="🛡️", layout="centered")
>>>>>>> f518f29a75fc7e0e92b3842f2c570c7160828d06

user_input = st.text_area("Enter case description:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a case description before predicting.")
    else:
        predicted_class, probabilities = predict(user_input)

<<<<<<< HEAD
        # Map index to class name (change this to your labels)
        label_map = {0: "Low Risk", 1: "High Risk"}
        st.subheader(f"Prediction: **{label_map[predicted_class]}**")
        st.write("### Class Probabilities")
        for label_idx, prob in enumerate(probabilities):
            st.write(f"{label_map[label_idx]}: **{prob*100:.2f}%**")
=======
if agree:
    st.subheader("✍️ Enter your message")
    user_text = st.text_area("Type here...", height=150)

    if st.button("Analyze"):
        if user_text.strip():
            result = predict(user_text)
            st.markdown(f"**Result:** {result}")

            if result == "🚨 Not Safe":
                st.error("🚨 Panic mode activated! NGOs have been alerted.")
        else:
            st.warning("Please enter some text before analyzing.")
else:
    st.info("Please agree to the legal terms to proceed.")


>>>>>>> f518f29a75fc7e0e92b3842f2c570c7160828d06
