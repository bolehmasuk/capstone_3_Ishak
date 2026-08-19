import streamlit as st
from dotenv import load_dotenv
from agent_workflow import build_workflow
from utils import setup_ui, render_sidebar, render_chat_history, handle_user_input

# 1. Inisialisasi Environment & UI
load_dotenv()
setup_ui()

# 2. Persiapan Workflow & State (Cache)
@st.cache_resource
def get_cached_workflow():
    return build_workflow()

if "app_workflow" not in st.session_state:
    st.session_state.app_workflow = get_cached_workflow()
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Eksekusi Komponen Streamlit
selected_provider = render_sidebar()
render_chat_history()
handle_user_input(selected_provider)