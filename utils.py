import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

def get_clean_text(content):
    if isinstance(content, str): return content
    elif isinstance(content, list):
        return "".join([item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in content])
    return str(content)

def extract_token_usage(msg):
    in_tok, out_tok, tot_tok = 0, 0, 0
    if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
        in_tok = msg.usage_metadata.get('input_tokens', 0)
        out_tok = msg.usage_metadata.get('output_tokens', 0)
        tot_tok = msg.usage_metadata.get('total_tokens', in_tok + out_tok)
    elif hasattr(msg, 'response_metadata') and msg.response_metadata:
        usage = msg.response_metadata.get('token_usage') or msg.response_metadata.get('usage_metadata')
        if isinstance(usage, dict):
            in_tok = usage.get('prompt_tokens', usage.get('prompt_token_count', 0))
            out_tok = usage.get('completion_tokens', usage.get('candidates_token_count', 0))
            tot_tok = usage.get('total_tokens', usage.get('total_token_count', in_tok + out_tok))
        elif hasattr(usage, 'prompt_token_count'):
            in_tok, out_tok = usage.prompt_token_count, getattr(usage, 'candidates_token_count', 0)
            tot_tok = getattr(usage, 'total_token_count', in_tok + out_tok)
    return in_tok, out_tok, tot_tok

def setup_ui():
    st.set_page_config(page_title="Multi-Agent RAG Chatbot", page_icon="🤖", layout="centered")
    st.markdown("<style>.main {background-color: #f9f9f9;} .stChatInput {padding-bottom: 20px;} .stChatMessage {gap: 1rem;}</style>", unsafe_allow_html=True)
    st.title("🤖 Multi-Agent RAG Chatbot")
    st.markdown("Sistem Chatbot Cerdas berbasis **LangGraph Multi-Agent**, **Qdrant Cloud**, dan **RAG** untuk Analisis Data Resume.")
    st.divider()

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Konfigurasi Sistem")
        provider = st.selectbox("Pilih Provider LLM", ("OpenAI",), help="Model AI yang dipakai untuk menghasilkan jawaban teks. (Ke depannya bisa dikembangkan dan ditambahkan provider lain seperti Anthropic, LLaMA, dll.)")
        st.info(f"Model aktif saat ini menggunakan **{provider}**.")
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---\n### 📌 Informasi Aplikasi\n- **Vector DB:** Qdrant Cloud\n- **Embedding:** OpenAI (`text-embedding-3-small`)\n- **Memory:** Min. 3 percakapan\n- **Framework:** LangChain & LangGraph")
    return provider

def render_chat_history():
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            st.markdown(get_clean_text(msg.content))
            if isinstance(msg, AIMessage):
                i, o, t = extract_token_usage(msg)
                if t > 0:
                    with st.expander("📊 Usage Details"): st.code(f"Input Token  : {i}\nOutput Token : {o}\nTotal Token  : {t}")

def handle_user_input(provider):
    if user_input := st.chat_input("Tanyakan sesuatu seputar data resume ...."):
        with st.chat_message("user", avatar="👤"): st.markdown(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        if len(st.session_state.messages) > 6: st.session_state.messages = st.session_state.messages[-6:]
            
        with st.spinner(f"✨ Supervisor sedang menganalisis & merutekan via {provider}..."):
            result = st.session_state.app_workflow.invoke({"messages": st.session_state.messages, "llm_provider": provider})
            bot_res = result["messages"][-1]
            
        clean_ans = get_clean_text(bot_res.content)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(clean_ans)
            i, o, t = extract_token_usage(bot_res)
            if t > 0:
                with st.expander("📊 Usage Details"): st.code(f"Input Token  : {i}\nOutput Token : {o}\nTotal Token  : {t}")
        
        bot_res.content = clean_ans
        st.session_state.messages.append(bot_res)