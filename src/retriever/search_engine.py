import torch
import faiss
import pickle
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# Default Paths (Works for app.py)
DEFAULT_INDEX_PATH = "artifacts/fashion.index"
DEFAULT_METADATA_PATH = "artifacts/clean_metadata.pkl"
MODEL_ID = "openai/clip-vit-base-patch32"

class SearchEngine:
    def __init__(self, index_path=DEFAULT_INDEX_PATH, metadata_path=DEFAULT_METADATA_PATH):
        """
        Initializes the Search Engine.
            Args:
            index_path (str): Path to FAISS index
            metadata_path (str): Path to metadata
            Note: Uses Gemini 2.5 Flash VLM for image descriptions
            """
        
        print("⏳ Loading Search Engine...")
        
        # 1. Load FAISS Index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ FAISS Index not found at: {index_path}")
            
        try:
            self.index = faiss.read_index(index_path)
            print(f"✅ FAISS Index loaded: {self.index.ntotal} items")
        except Exception as e:
            print(f"❌ Error loading FAISS index: {e}")
            raise e

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"❌ Metadata not found at: {metadata_path}")
            
        try:
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
            
            # Convert DataFrame to List of Dictionaries
            if hasattr(data, 'to_dict'):
                print("🔄 Converting Pandas DataFrame to List...")
                self.metadata = data.to_dict('records')
            else:
                self.metadata = data
                
            print(f"✅ Metadata loaded: {len(self.metadata)} items")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            raise e

        # 3. Load CLIP Model
        print(f"🧠 Loading CLIP Model ({MODEL_ID})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        
        # 4. Load Vision Generator
        from src.models.gemini_client import GeminiClient
        self.vision_generator = GeminiClient()
        self.generator_name = "Gemini 2.5 Flash (API)"
        
        print(f"🎨 Vision Generator: {self.generator_name}")
        print("✅ System Ready.")

    def _get_query_embedding(self, text=None, image_path=None):
        """Helper: Converts Text OR Image into a vector"""
        inputs = None
        if text:
            # 🔧 FIXED: Added truncation=True to handle long captions from Gemini
            inputs = self.processor(
                text=[text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True,  # ← Truncate if text is too long
                max_length=77      # ← CLIP's max token limit
            ).to(self.device)
            with torch.no_grad():
                embedding = self.model.get_text_features(**inputs)
            if hasattr(embedding, "pooler_output"):
                embedding = embedding.pooler_output
        elif image_path:
            try:
                image = Image.open(image_path)
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embedding = self.model.get_image_features(**inputs)
                if hasattr(embedding, "pooler_output"):
                    embedding = embedding.pooler_output
            except Exception as e:
                print(f"❌ Error processing image {image_path}: {e}")
                return None

        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype('float32')

    def search_by_image(self, image_path, top_k=5, use_vlm_caption=False):
        """
        Input: Path to an image
        Output: List of similar products
        
        Args:
            use_vlm_caption: If True, generates caption first then searches by text
        """
        if use_vlm_caption:
            # NEW PATH: Image -> VLM Caption -> Text Search
            print(f"🔄 Using {self.generator_name} to describe image first...")
            caption = self.vision_generator.generate_caption(image_path)
            print(f"📝 Generated caption: {caption[:100]}...")
            
            # Now search by the generated text
            return self.search_by_text(caption, top_k=top_k)
        else:
            # ORIGINAL PATH: Direct image embedding
            query_vector = self._get_query_embedding(image_path=image_path)
            if query_vector is None: 
                return []

            distances, indices = self.index.search(query_vector, top_k)

            results = []
            for i in range(top_k):
                idx = indices[0][i]
                if idx < len(self.metadata):
                    product = self.metadata[idx]
                    product['score'] = float(distances[0][i])
                    results.append(product)
            return results

    def search_by_text(self, text_query, top_k=5):
        """Text search"""
        query_vector = self._get_query_embedding(text=text_query)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx < len(self.metadata):
                product = self.metadata[idx]
                product['score'] = float(distances[0][i])
                results.append(product)
        return results

if __name__ == "__main__":
    # Test block
    engine = SearchEngine()
    
    # Test Text Search
    print("\n🔍 Testing Text Search: 'Blue Denim Jeans'")
    results = engine.search_by_text("Blue Denim Jeans", top_k=3)
    for r in results:
        print(f"   → {r.get('productDisplayName', 'Unknown')} (Score: {r['score']:.3f})")