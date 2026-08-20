import pandas as pd
from langchain_core.documents import Document

def load_and_inspect_csv(file_path: str = "data/Resume.csv"):
    """Step 8: Inspect dataset CSV"""
    print(f"Membaca dataset dari: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("\n=== Informasi Dataset ===")
        print(f"Jumlah baris: {df.shape[0]}")
        print(f"Jumlah kolom: {df.shape[1]}")
        print("\nKolom yang tersedia:")
        print(df.columns.tolist())
        return df
    except Exception as e:
        print(f"❌ Gagal membaca CSV: {e}")
        return None

def convert_to_documents(df: pd.DataFrame):
    """Step 9: Konversi CSV menjadi LangChain Documents"""
    documents = []
    
    for index, row in df.iterrows():
        # Gunakan Resume_str sebagai konten utama untuk di-embed
        # pd.notna memastikan tidak ada error jika ada sel yang kosong (NaN)
        content = str(row["Resume_str"]) if pd.notna(row["Resume_str"]) else ""
        
        # Ambil ID dan Category sebagai metadata. Resume_html diabaikan.
        metadata = {
            "id": str(row["ID"]),
            "category": str(row["Category"])
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    print(f"\n✅ Berhasil mengonversi {len(documents)} baris menjadi LangChain Documents.")
    return documents

if __name__ == "__main__":
    # Pastikan file CSV Anda bernama 'dataset.csv' dan berada di folder 'data/'
    # Jika namanya berbeda (misal: Resume.csv), ganti string di bawah ini
    dataframe = load_and_inspect_csv("data/Resume.csv") 
    
    if dataframe is not None:
        # TIPS MVP: 
        # Dataset resume Kaggle biasanya berisi ribuan baris. 
        # Untuk tahap testing/MVP agar proses embedding ke Qdrant nanti tidak terlalu lama,
        # Anda bisa memotong datanya sementara (misal 100 baris pertama).
        # Hapus '# ' pada dua baris di bawah ini jika ingin memotong data:
        # print("\nMemotong 100 baris pertama untuk testing MVP...")
        # dataframe = dataframe.head(100)
        
        docs = convert_to_documents(dataframe)
        
        # Cek hasil konversi dokumen pertama
        if docs:
            print("\n=== Contoh Dokumen Pertama ===")
            print(f"Metadata : {docs[0].metadata}")
            print(f"Konten   : {docs[0].page_content[:150]}...\n")