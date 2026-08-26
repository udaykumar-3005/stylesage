from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

class GeminiClient:
    def __init__(self):
        """
        Initializes Gemini 2.5 Flash (FREE tier).
        Requires GEMINI_API_KEY in .env file.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file!")
        
        self.client = genai.Client(api_key=api_key)
        print("✅ Gemini Vision initialized (FREE API, no billing required)")
    
    def generate_caption(self, image_path):
        """
        Input: Path to the cropped image
        Output: A detailed text description from Gemini
        """
        try:
            # Load image as PIL Image
            img = Image.open(image_path)
            
            # Prompt
            prompt = "Describe this clothing item in detail. Focus on color, texture, pattern, and style. Be specific and concise."
            
            # Call API with CORRECT model name (no 'models/' prefix!)
            print("💬 Gemini is thinking (FREE API call)...")
            response = self.client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[prompt, img]
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error with Gemini: {e}")
            return "A fashion item (API Error)"

# Quick Test Block
if __name__ == "__main__":
    try:
        gen = GeminiClient()
        
        # Test on a sample image
        test_image = "data/crop_temp.jpg"
        
        if os.path.exists(test_image):
            print(f"\n🖼️  Testing Gemini Vision on: {test_image}")
            description = gen.generate_caption(test_image)
            print(f"\n✨ RESULT:\n{description}")
        else:
            print("⚠️  No test image found at data/crop_temp.jpg")
    except Exception as e:
        print(f"🔥 Setup failed: {e}")