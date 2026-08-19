import sys
import os
import time
from dotenv import load_dotenv

# Import LangChain & Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Import fungsi loader internal Anda
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.loader import load_and_inspect_csv, convert_to_documents

load_dotenv()

def get_embedding_model():
    """Inisialisasi model OpenAI Embeddings"""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_data_with_delay(collection_name="resume_collection"):
    """Ingest data dengan deteksi dimensi otomatis dan jeda waktu"""
    print("1. Memuat dataset...")
    df = load_and_inspect_csv("data/Resume.csv")
    if df is None:
        return
    
    # Ambil 30 data untuk full test MVP
    # df = df.head(30)
    
    docs = convert_to_documents(df)
    
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
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        print(f"   -> Meng-ingest dokumen {i+1} sampai {min(i+batch_size, len(docs))}...")
        
        # Masukkan ke Qdrant
        qdrant.add_documents(batch)
        
        # Jeda waktu 2 detik antar batch
        if i + batch_size < len(docs):
            time.sleep(2) 
            
    print(f"\n✅ Ingestion Selesai! {len(docs)} data tersimpan dengan aman di Qdrant.")

if __name__ == "__main__":
    ingest_data_with_delay()