import numpy as np
import faiss
from rag.loader import load_noaa_data, preprocess_events
from rag.embedder import embed_texts  # Your embedder function using OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# 1. Load and preprocess your NOAA data
df = load_noaa_data()
df = preprocess_events(df)
texts = df['text_chunk'].tolist()

# 2. Embed texts (assuming embed_texts returns a list of embeddings as numpy arrays)
embeddings = embed_texts(texts)

# 3. Convert embeddings to numpy array (shape: num_texts x embedding_dim)
vectors = np.array(embeddings).astype('float32')

# 4. Save embeddings and texts to npz
np.savez("embeddings.npz", vectors=vectors, texts=np.array(texts))

print("Saved embeddings.npz successfully.")
