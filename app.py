import os
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Study Assistant")
st.subheader("Interactive Assistant for Python & Machine Learning")

api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API Key not found. Please add your key to environment variables or Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
You are an expert AI and Python Programming Tutor. 
Provide clear, supportive, and concise explanations with code examples where relevant.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about Python or AI/ML..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                contents = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])]
                    ) for m in st.session_state.messages
                ]
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.7,
                    )
                )
                
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {e}")