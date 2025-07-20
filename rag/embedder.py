from openai import OpenAI
import numpy as np
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [d.embedding for d in response.data]

def save_embeddings(df, embeddings, output_path="embeddings.npz"):
    np.savez_compressed(output_path, texts=df['text_chunk'].tolist(), vectors=embeddings)
