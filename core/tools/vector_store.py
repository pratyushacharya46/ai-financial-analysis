import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store():
    """Reads the 10-K, chunks it, and saves it to a local Chroma Vector Database."""
    file_path = "core/data/tsla_10k.txt"
    db_dir = "core/data/chroma_db"
    
    print("🧠 Initializing free local embedding model (HuggingFace)...")
    # We use a fast, completely free local embedding model to avoid API costs
    # It will download a small model (~80MB) the first time you run this.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"📄 Reading document from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {file_path}. Did you run sec_fetcher.py first?")

    print("✂️ Chunking massive text file...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, 
        chunk_overlap=200
    )
    chunks = splitter.create_documents([text])
    print(f"✅ Document successfully split into {len(chunks)} chunks.")
    
    print(f"💾 Building ChromaDB Vector Store at {db_dir}...")
    print("⏳ Please wait, embedding 140 chunks into math vectors (this may take a minute)...")
    
    # Create the vector store and persist it to the local disk
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_dir
    )
    
    print("\n✅ Success! Tesla 10-K has been embedded and saved to your local Vector Database.")
    print(f"Your Researcher Agent can now actively search through the entire document!")

if __name__ == "__main__":
    build_vector_store()