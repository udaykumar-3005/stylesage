import streamlit as st
from src.retriever.search_engine import SearchEngine
from PIL import Image
import time
import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

IMAGES_DIR = os.path.join("fashion-dataset", "images")
IMAGE_DATASET_REPO = "Udaykumar3005/stylesage-images"

def get_result_image_path(result):
    """Resolve a result's image: local dataset copy if present (fast, dev),
    otherwise fetch (and cache) from the Hugging Face dataset (deployed)."""
    filename = result.get("filename")
    if not filename:
        return None

    local_path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(local_path):
        return local_path

    try:
        stem = filename.rsplit(".", 1)[0]
        bucket = f"{int(stem) % 100:02d}"
        return hf_hub_download(
            repo_id=IMAGE_DATASET_REPO,
            repo_type="dataset",
            filename=f"images/{bucket}/{filename}",
        )
    except (HfHubHTTPError, OSError, ValueError):
        return None

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="StyleSage - AI Fashion Search",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .result-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE & DESCRIPTION
# ============================================================
st.title("👔 StyleSage - AI Fashion Search")
st.markdown("""
    Multi-modal fashion search powered by **Gemini 2.5 Flash Vision**.  
    Upload an image or describe what you're looking for!
""")

# ============================================================
# SIDEBAR - SETTINGS & INFO
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model info
    st.subheader("🤖 AI Model")
    st.info("""
    **Gemini 2.5 Flash**
    - Best-in-class caption quality
    - Fast inference (~1-2s)
    - Free API tier
    - Cloud-based
    """)
    
    st.divider()
    
    # Search settings
    st.subheader("🔍 Search Settings")
    top_k = st.slider("Number of results:", 3, 10, 5)
    use_vlm_caption = st.checkbox(
        "Use AI description", 
        value=True,
        help="Generate detailed description first, then search. Slower but more accurate."
    )
    
    st.divider()
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    Built with:
    - CLIP embeddings
    - FAISS vector search
    - Gemini 2.5 Flash VLM
    - 44k+ fashion products
    """)

# ============================================================
# LOAD SEARCH ENGINE (CACHED)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_search_engine():
    """Load and cache search engine (Gemini-only)"""
    return SearchEngine()  # 🔧 FIXED: Removed use_gemini_vision parameter

with st.spinner("Loading Gemini 2.5 Flash..."):
    try:
        engine = load_search_engine()
        st.sidebar.success("✅ Gemini ready!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load: {e}")
        st.stop()

# ============================================================
# MAIN INTERFACE - TABS
# ============================================================
tab1, tab2 = st.tabs(["📸 Image Search", "✍️ Text Search"])

# ============================================================
# TAB 1: IMAGE SEARCH
# ============================================================
with tab1:
    st.subheader("Upload a fashion image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo of clothing or fashion item"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Image", use_container_width=True)
            
            # Search button
            if st.button("🔍 Search Similar Items", type="primary"):
                # Save uploaded file temporarily
                temp_path = "temp_upload.jpg"
                image.convert("RGB").save(temp_path)
                
                # Perform search
                with st.spinner("Analyzing image with Gemini..."):
                    start_time = time.time()
                    
                    try:
                        results = engine.search_by_image(
                            temp_path, 
                            top_k=top_k, 
                            use_vlm_caption=use_vlm_caption
                        )
                        search_time = (time.time() - start_time) * 1000  # Convert to ms
                        
                        # Display in col2
                        with col2:
                            st.success(f"✅ Found {len(results)} results in {search_time:.0f}ms")
                            
                            # Display results
                            st.subheader("Similar Items")
                            for i, result in enumerate(results, 1):
                                with st.container():
                                    img_col, info_col = st.columns([1, 2])
                                    with img_col:
                                        result_img_path = get_result_image_path(result)
                                        if result_img_path:
                                            st.image(result_img_path, use_container_width=True)
                                        else:
                                            st.caption("📷 Image unavailable")
                                    with info_col:
                                        st.markdown(f"""
                                            <div class="result-card">
                                                <strong>#{i}. {result.get('productDisplayName', 'Unknown')}</strong><br>
                                                <small>Category: {result.get('articleType', 'N/A')}</small><br>
                                                <small>Color: {result.get('baseColour', 'N/A')}</small><br>
                                                <small>Similarity: {result['score']:.3f}</small>
                                            </div>
                                        """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"❌ Search failed: {e}")
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            with col2:
                st.info("👆 Upload an image to get started!")

# ============================================================
# TAB 2: TEXT SEARCH
# ============================================================
with tab2:
    st.subheader("Search by description")
    
    # Text input
    query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g., Blue denim jeans with ripped knees",
        help="Enter a detailed description of the fashion item"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Searching..."):
            start_time = time.time()
            
            try:
                results = engine.search_by_text(query, top_k=top_k)
                search_time = (time.time() - start_time) * 1000
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{len(results)}</h3>
                            <p>Results Found</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{search_time:.0f}ms</h3>
                            <p>Search Time</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>{results[0]['score']:.3f}</h3>
                            <p>Top Score</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # Results in columns
                cols = st.columns(min(5, len(results)))
                for i, result in enumerate(results):
                    with cols[i]:
                        st.markdown(f"""
                            **{i+1}. {result.get('productDisplayName', 'Unknown')[:30]}**
                        """)
                        st.caption(f"Category: {result.get('articleType', 'N/A')}")
                        st.caption(f"Color: {result.get('baseColour', 'N/A')}")
                        st.caption(f"Score: {result['score']:.3f}")
                        
                        # Show image if path exists
                        img_path = get_result_image_path(result)
                        if img_path:
                            st.image(img_path, use_container_width=True)
                        else:
                            st.caption("📷 Image unavailable")
            
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
    
    elif search_button:
        st.warning("⚠️ Please enter a search query")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with Streamlit • CLIP • FAISS • Gemini 2.5 Flash</p>
        <p>44,441 fashion products indexed</p>
    </div>
""", unsafe_allow_html=True)