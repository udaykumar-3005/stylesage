import os
import numpy as np
import faiss

# Input: The raw vectors from Phase 2
INPUT_PATH = "artifacts/embeddings.npy"

# Output: The searchable binary index file
OUTPUT_INDEX_PATH = "artifacts/fashion.index"

def build_faiss_index():
    """
    Reads the NumPy embeddings and builds a FAISS Index for fast similarity search.
    """
    
    # LOAD DATA
    print(f"⚡ Loading embeddings from {INPUT_PATH}...")
    embeddings = np.load(INPUT_PATH)
    
    print(f"     Data Shape: {embeddings.shape}")
    print(f"   -> Rows (Images): {embeddings.shape[0]}")
    print(f"   -> Columns (Dimensions): {embeddings.shape[1]}")

    # xPREPARE FOR FAISS
    # FAISS is strict. It requires data to be in float32 format.
    # Even if we saved it as float32 in Phase 2, it's good practice to double-check.
    if embeddings.dtype != 'float32':
        print("Converting data to float32 for FAISS compatibility...")
        embeddings = embeddings.astype('float32')

    print("Normalizing vectors (L2) to ensure Dot Product acts as Cosine Similarity...")
    faiss.normalize_L2(embeddings)

    # Get the dimension size (d = 512 for CLIP)
    d = embeddings.shape[1]

    # 4️⃣ BUILD THE INDEX
    # We use IndexFlatIP (Inner Product)
    # Why? Because our vectors are normalized. 
    # Normalized Vectors + Inner Product = Cosine Similarity (The Gold Standard)
    print("Building FAISS Index (IndexFlatIP)...")
    
    index = faiss.IndexFlatIP(d)
    
    # Add the vectors to the index
    index.add(embeddings)

    # 5️⃣ SAVE TO DISK
    print(f"Index built successfully with {index.ntotal} vectors.")
    print("Saving index to disk...")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(OUTPUT_INDEX_PATH), exist_ok=True)
    
    faiss.write_index(index, OUTPUT_INDEX_PATH)
    print(f"Done! Index saved at: {OUTPUT_INDEX_PATH}")


if __name__ == "__main__":
    try:
        build_faiss_index()
    except Exception as e:
        print(f"Critical Error: {e}")