import os
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional SaaS Dark Theme Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0D0E12;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }
    
    header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }

    /* Top Workspace Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid #1E202C;
        margin-bottom: 28px;
        margin-top: -25px;
    }
    .app-header-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .app-header-sub {
        font-size: 0.85rem;
        color: #8E8EA0;
    }
    .app-status-badge {
        background-color: #1A1C24;
        border: 1px solid #2A2D3D;
        color: #34D399;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #34D399;
        border-radius: 50%;
    }

    /* Sidebar Navigation */
    section[data-testid="stSidebar"] {
        background-color: #111218 !important;
        border-right: 1px solid #1E202C !important;
    }
    
    .sidebar-section-title {
        color: #8E8EA0;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }

    .history-item {
        background-color: #161822;
        border: 1px solid #222534;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #CBD5E1;
        margin-bottom: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* User Message Bubble */
    .user-bubble-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .user-bubble {
        background: linear-gradient(135deg, #9333EA 0%, #6B21A8 100%);
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    /* Chat Input Styling */
    .stChatInput > div {
        background-color: #161822 !important;
        border: 1px solid #2E3245 !important;
        border-radius: 16px !important;
    }
    .stChatInput textarea {
        color: #FFFFFF !important;
    }

    /* Sidebar Buttons */
    div.stButton > button {
        background-color: #161822 !important;
        color: #E2E8F0 !important;
        border: 1px solid #2A2D3D !important;
        border-radius: 10px !important;
        padding: 10px 16px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #222534 !important;
        border-color: #9333EA !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# API Setup
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API Key missing.")
    st.stop()

client = genai.Client(api_key=api_key)

# Locked to the exact model required by your API key
ACTIVE_MODEL = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = """
You are an expert AI & Computer Science Tutor specializing in Python and Machine Learning.
- Be concise, clear, and straight to the point.
- Avoid unnecessary filler text.
- Use proper Markdown formatting (bolding, lists, and code blocks with comments) for technical programming answers.
"""

# Resilient Helper with Retry Loop for High-Demand Spikes
def call_gemini_with_retry(contents, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=ACTIVE_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                )
            )
            return response.text
        except APIError as e:
            if getattr(e, 'code', None) in [429, 503] or "503" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
            raise e
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise e

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_suggestion(prompt_text):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.rerun()

# Sidebar Layout
with st.sidebar:
    if st.button("+ New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'>Chat History</div>", unsafe_allow_html=True)
    
    user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if user_queries:
        for q in user_queries:
            short_q = q if len(q) <= 28 else q[:25] + "..."
            st.markdown(f"<div class='history-item'>{short_q}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #64748B; font-size: 0.85rem;'>No recent conversations.</p>", unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: #1E202C;'><br>", unsafe_allow_html=True)
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Main Workspace Layout
col_left, col_center, col_right = st.columns([1, 6, 1])

with col_center:
    st.markdown("""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-header-title">AI Study Assistant</div>
            <div class="app-header-sub">Python & Machine Learning Engineering Tutor</div>
        </div>
        <div class="app-status-badge">
            <span class="status-dot"></span>
            Gemini 3.8 Flash
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Welcome / Suggested Topics
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.write("Hello! I am your AI Study Assistant. What topic or concept would you like to explore today?")

        st.markdown("<p style='color: #8E8EA0; font-size: 0.85rem; font-weight: 600; margin-top: 20px; margin-bottom: 10px;'>SUGGESTED TOPICS</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Python Lists"):
                send_suggestion("Explain Python List Comprehensions with quick code examples.")
        with col2:
            if st.button("What is CNN?"):
                send_suggestion("Explain CNN architectures in computer vision simply.")
        with col3:
            if st.button("Classification vs Regression"):
                send_suggestion("Explain the key differences between classification and regression.")

    # Render Conversation History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-bubble-container">
                <div class="user-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg['content'])

    # Standard Chat Input Box
    if user_prompt := st.chat_input("Ask a question about Python, ML, or Computer Science..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.rerun()

    # Model Generation
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Thinking..."):
            try:
                contents = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])]
                    ) for m in st.session_state.messages
                ]

                response_text = call_gemini_with_retry(contents)

                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")