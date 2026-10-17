import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm  # Library for progress bars (pip install tqdm)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

# -------------------------------------------------------------------
# 📌 CONFIGURATION
# -------------------------------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "artifacts" / "clean_metadata.pkl"
IMAGE_FOLDER = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "embeddings.npy"
BATCH_SIZE = 32
MODEL_ID = "openai/clip-vit-base-patch32"

# -------------------------------------------------------------------
# 📌 DATASET CLASS
# Handles loading images and crashes gracefully if an image is bad
# -------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Open image and ensure RGB (handles PNGs with transparency)
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)
            

# -------------------------------------------------------------------
# 📌 MAIN EMBEDDING FUNCTION
# -------------------------------------------------------------------
def generate_image_embeddings(metadata_path, image_folder, output_path, batch_size=32):
    
    # 1️⃣ Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # 2️⃣ Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Metadata file not found at: {metadata_path}")
    
    print(f"📖 Loading metadata from {metadata_path}...")
    df = pd.read_pickle(metadata_path)

    image_paths = []
    valid_indices = []
    for idx, fname in enumerate(df["filename"]):
        full_path = os.path.join(image_folder, fname)
        if os.path.exists(full_path):
            image_paths.append(full_path)
            valid_indices.append(idx)
    print(f"✅ Valid images found: {len(image_paths)}")
    print(f"❌ Missing images skipped: {len(df) - len(image_paths)}")


    # 3️⃣ Load CLIP Model & Processor
    print(f"🧠 Loading CLIP model ({MODEL_ID})...")
    model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # 4️⃣ Prepare DataLoader
    # We use num_workers=2 to load data in parallel while GPU processes
    dataset = ImageDataset(image_paths, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0
    )

    # 5️⃣ Extraction Loop
    print("⚡ Starting feature extraction...")
    embeddings = []
    
    # Put model in eval mode (disables dropout, etc.)
    model.eval()

    # 'no_grad' saves memory since we aren't training
    with torch.no_grad():
        # tqdm wraps the dataloader to show a progress bar
        for batch in tqdm(dataloader, desc="Processing Batches"):
            batch = batch.to(device)

            # Transformers 5 returns a BaseModelOutputWithPooling here, whereas
            # earlier versions returned the tensor directly.
            image_features = model.get_image_features(pixel_values=batch)
            features = (
                image_features.pooler_output
                if hasattr(image_features, "pooler_output")
                else image_features
            )

            # 🔑 CRITICAL: Normalization
            # This ensures dot product = cosine similarity later
            features = features / features.norm(p=2, dim=-1, keepdim=True)

            # Move to CPU to save RAM and convert to numpy
            embeddings.append(features.cpu().numpy())

    # 6️⃣ Save Results
    # Concatenate all batches into one large matrix
    final_embeddings = np.vstack(embeddings).astype("float32")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.save(output_path, final_embeddings)
    print(f"✅ Success! Embeddings saved to: {output_path}")
    print(f"📊 Final Shape: {final_embeddings.shape}")

# -------------------------------------------------------------------
# 📌 ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        generate_image_embeddings(
            metadata_path=METADATA_PATH,
            image_folder=IMAGE_FOLDER,
            output_path=OUTPUT_PATH,
            batch_size=BATCH_SIZE
        )
        
        # 🔍 Verification Step
        print("\n🔍 Verifying output file...")
        data = np.load(OUTPUT_PATH)
        print(f"Verified: {data.shape} | Type: {data.dtype}")
        
    except Exception as main_error:
        print(f"❌ Script failed: {main_error}")




