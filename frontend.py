import requests
import streamlit as st
import numpy as np
import webbrowser

# API endpoint for prediction
API_URL = "http://127.0.0.1:8000/predict"  # Change to your deployed backend URL

# -------------------
# App title
# -------------------
st.set_page_config(page_title="FemSafe", page_icon="🛡️")
st.title("🛡️ FemSafe - Safety Text Analysis & Panic Alert")

# -------------------
# Legal disclaimer
# -------------------
st.subheader("Legal Disclaimer")
st.write("""
By using FemSafe, you agree that the information you provide is truthful.
False reporting may be subject to legal consequences.
""")

accept_terms = st.selectbox(
    "Do you accept these terms?",
    ["No", "Yes - I Agree"]
)

if accept_terms == "Yes - I Agree":
    # -------------------
    # Text input
    # -------------------
    st.subheader("Enter your message")
    user_text = st.text_area("Type your message here:")

    if st.button("Analyze Message"):
        if user_text.strip() == "":
            st.warning("Please enter a message before analysis.")
        else:
            # Send request to backend
            try:
                response = requests.post(API_URL, json={"text": user_text})
                if response.status_code == 200:
                    data = response.json()

                    # Assuming class 0 = Safe, class 1 = Not Safe
                    class_index = int(np.argmax(data["probabilities"]))
                    label = "Safe ✅" if class_index == 0 else "🚨 Not Safe 🚨"

                    st.subheader("Result")
                    st.write(f"**Classification:** {label}")

                    # Panic button if Not Safe
                    if class_index == 1:
                        st.error("Danger detected! Please press the panic button.")
                        if st.button("🚨 PANIC BUTTON 🚨"):
                            # Example: open NGO help page
                            webbrowser.open("https://example-ngo.org/emergency-contact")
                            st.success("NGOs have been alerted!")
                else:
                    st.error("Error from backend. Please try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")
else:
    st.warning("You must accept the terms to use FemSafe.")
