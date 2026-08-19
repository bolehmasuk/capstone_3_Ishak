import streamlit as st
from dotenv import load_dotenv
from agent_workflow_2 import build_workflow
from utils import setup_page, check_password, render_header, render_sidebar, render_chat_history, handle_user_input

# 1. Inisialisasi Environment & Pengaturan Halaman Dasar
load_dotenv()
setup_page()

# 2. Sistem Proteksi Autentikasi
if not check_password():
    st.stop()  # Script akan berhenti di sini jika user belum login
    
# 3. Render Header Aplikasi (Hanya tampil setelah login)
render_header()

# 4. Persiapan Workflow & State (Cache)
@st.cache_resource
def get_cached_workflow():
    return build_workflow()

if "app_workflow" not in st.session_state:
    st.session_state.app_workflow = get_cached_workflow()
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Eksekusi Komponen Utama Streamlit
selected_provider = render_sidebar()
render_chat_history()
handle_user_input(selected_provider)