import sys
import os
import time
from dotenv import load_dotenv

# Import LangChain & Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# 1. TAMBAHAN IMPORT UNTUK CHUNKING
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import fungsi loader internal Anda
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.loader import load_and_inspect_csv, convert_to_documents

load_dotenv()

def get_embedding_model():
    """Inisialisasi model OpenAI Embeddings"""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_data_with_delay(collection_name="resume_new"):
    """Ingest data dengan chunking, deteksi dimensi otomatis, dan jeda waktu"""
    print("1. Memuat dataset...")
    df = load_and_inspect_csv("data/Resume.csv")
    if df is None:
        return
    
      
    docs = convert_to_documents(df)
    
    # --- 2. PROSES CHUNKING ---
    print("\n1.5 Melakukan Chunking pada Dokumen...")
    # Anda bisa menyesuaikan chunk_size dan chunk_overlap sesuai kebutuhan
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunked_docs = text_splitter.split_documents(docs)
    print(f"   -> {len(docs)} dokumen awal berhasil dipecah menjadi {len(chunked_docs)} chunks/potongan.")
    # --------------------------
    
    print("\n2. Memuat Model Embedding OpenAI...")
    embeddings = get_embedding_model()
    
    # --- DETEKSI DIMENSI OTOMATIS ---
    print("   -> Mendeteksi ukuran dimensi vektor model OpenAI...")
    sample_vector = embeddings.embed_query("tes dimensi")
    vector_size = len(sample_vector)
    print(f"   -> Dimensi terdeteksi: {vector_size}") 
    # -------------------------------
    
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    client = QdrantClient(url=url, api_key=api_key)
    
    print(f"\n3. Menyiapkan collection '{collection_name}' di Qdrant...")
    if client.collection_exists(collection_name):
        print("   -> Menghapus data lama (Reset)...")
        client.delete_collection(collection_name=collection_name)
        
    print(f"   -> Membuat collection baru dengan ukuran {vector_size}...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Inisialisasi Vector Store
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    
    print("\n4. Mulai Ingestion secara bertahap (Batching)...")
    batch_size = 10 
    
    # PERBAIKAN: Gunakan 'chunked_docs' alih-alih 'docs' asli
    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i:i + batch_size]
        print(f"   -> Meng-ingest chunk {i+1} sampai {min(i+batch_size, len(chunked_docs))}...")
        
        # Masukkan ke Qdrant
        qdrant.add_documents(batch)
        
        # Jeda waktu 2 detik antar batch
        if i + batch_size < len(chunked_docs):
            time.sleep(2) 
            
    print(f"\n✅ Ingestion Selesai! {len(chunked_docs)} potongan data tersimpan dengan aman di Qdrant.")

if __name__ == "__main__":
    ingest_data_with_delay()
