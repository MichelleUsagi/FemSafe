import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model & tokenizer
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("./distilbert_femicide_model")
    tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_femicide_model")
    return model, tokenizer

model, tokenizer = load_model()

# Predict function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "🚨 Not Safe" if prediction == 1 else "✅ Safe"

# Streamlit UI
st.set_page_config(page_title="FemSafe", page_icon="🛡️", layout="centered")

st.title("🛡️ FemSafe")
st.markdown("**Helping protect women through AI-powered risk detection.**")

# Legal disclaimer
st.subheader("📜 Legal Agreement")
agree = st.checkbox("I understand that any false reporting is subject to legal action.")

if agree:
    st.subheader("✍️ Enter your message")
    user_text = st.text_area("Type here...", height=150)

    if st.button("Analyze"):
        if user_text.strip():
            result = predict(user_text)
            st.markdown(f"**Result:** {result}")

            if result == "🚨 Not Safe":
                st.error("Panic mode activated! NGOs have been alerted.")
        else:
            st.warning("Please enter some text before analyzing.")
else:
    st.info("Please agree to the legal terms to proceed.")
