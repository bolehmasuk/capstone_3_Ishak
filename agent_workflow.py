import os
import operator
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

# PERBAIKAN 1: Tambahkan SystemMessage pada import
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Tambahkan OpenAIEmbeddings di sini
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END

# Muat variabel environment (.env)
load_dotenv()

# --- FUNGSI EMBEDDING BARU ---
def get_embedding_model():
    """Inisialisasi model OpenAI Embeddings secara lokal di file ini"""
    return OpenAIEmbeddings(model="text-embedding-3-small")
# -----------------------------

# 1. Definisikan State LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    llm_provider: str

# 2. LLM Factory
def get_llm(provider_name: str):
    if provider_name.lower() == "openai":
        return ChatOpenAI(model="gpt-5.6-luna", temperature=0.2)
    else:
        # Untuk pengembangan ke depan, kita siapkan apabila butuh LLM model lain, misal Gemini~
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2) 

# 3. Setup Retriever
def get_retriever():
    # Memanggil fungsi embedding yang ada di file ini
    embeddings = get_embedding_model() 
    
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    client = QdrantClient(url=url, api_key=api_key)
    qdrant = QdrantVectorStore(
        client=client,
        collection_name="resume_new",
        embedding=embeddings
    )
    return qdrant.as_retriever(search_kwargs={"k": 10})

# Inisialisasi retriever
retriever = get_retriever()

# 4. Nodes (Agen-agen)
def supervisor_node(state: AgentState):
    """Supervisor merutekan pertanyaan."""
    # PERBAIKAN 2: Gunakan seluruh riwayat pesan, bukan cuma yang terakhir
    messages = state["messages"] 
    llm = get_llm(state["llm_provider"])
    
    # Gunakan SystemMessage untuk instruksi peran
    system_prompt = """Kamu adalah supervisor sistem multi-agent. Tugasmu adalah merutekan pertanyaan pengguna.
    Berdasarkan percakapan berikut, tentukan apakah pesan terakhir pengguna perlu data RAG atau Umum.
    Jika pertanyaan berkaitan dengan Resume, CV, Kandidat, atau data HR, jawab HANYA dengan satu kata: RAG
    Jika pertanyaan berupa sapaan atau umum, jawab HANYA dengan satu kata: GENERAL"""
    
    # Gabungkan instruksi sistem dengan memori percakapan
    prompt_messages = [SystemMessage(content=system_prompt)] + list(messages)
    
    response = llm.invoke(prompt_messages)
    
    content = response.content
    if isinstance(content, list):
        content = "".join([str(item) for item in content])
        
    decision = content.strip().upper()
    
    if "RAG" in decision:
        return {"next_step": "RAG"}
    return {"next_step": "GENERAL"}

def rag_agent_node(state: AgentState):
    """Agen RAG."""
    # PERBAIKAN 3: Gunakan seluruh pesan, tapi pencarian retriever tetap pakai query terakhir
    messages = state["messages"]
    last_message = messages[-1].content 
    llm = get_llm(state["llm_provider"])
    
    docs = retriever.invoke(last_message)
    context = "\n\n".join([d.page_content for d in docs])
    
    if not context:
        return {"messages": [AIMessage(content="Maaf, saya tidak menemukan informasi terkait di database.")]}
    
    # Masukkan konteks DB ke SystemMessage
    system_prompt = f"""Kamu adalah asisten HR. Gunakan informasi konteks di bawah ini untuk menjawab pertanyaan pengguna dengan akurat berdasarkan riwayat percakapan yang ada.
    
    Konteks Database:
    {context}
    """
    
    # Gabungkan konteks dan riwayat pesan sebagai memori AI
    prompt_messages = [SystemMessage(content=system_prompt)] + list(messages)
    
    response = llm.invoke(prompt_messages)
    
    # PERBAIKAN: Kembalikan objek 'response' asli agar usage_metadata tidak hilang
    return {"messages": [response]}

def general_agent_node(state: AgentState):
    """Agen General."""
    # PERBAIKAN 4: Implementasi memori pada agen general
    messages = state["messages"]
    llm = get_llm(state["llm_provider"])
    
    system_prompt = "Kamu adalah asisten AI yang ramah. Jawab pertanyaan umum pengguna dengan santun dan perhatikan riwayat percakapan yang ada."
    
    prompt_messages = [SystemMessage(content=system_prompt)] + list(messages)
    
    response = llm.invoke(prompt_messages)
    
    # PERBAIKAN: Kembalikan objek 'response' asli agar usage_metadata tidak hilang
    return {"messages": [response]}

# 5. Compile LangGraph Workflow
def build_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("RAG", rag_agent_node)
    workflow.add_node("GENERAL", general_agent_node)
    
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_step"],
        {"RAG": "RAG", "GENERAL": "GENERAL"}
    )
    workflow.add_edge("RAG", END)
    workflow.add_edge("GENERAL", END)
    
    return workflow.compile()
