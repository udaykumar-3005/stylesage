# 👔 StyleSage — AI Fashion Search

Multi-modal fashion search powered by CLIP embeddings, FAISS vector search, and Gemini 2.5 Flash Vision.

Upload an image or describe what you're looking for to find similar fashion items from a 44,441-product catalog.

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Product images are served on demand from the companion dataset: [Udaykumar3005/stylesage-images](https://huggingface.co/datasets/Udaykumar3005/stylesage-images).

## Running locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
streamlit run app.py
```
