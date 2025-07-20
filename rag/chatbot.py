import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Load API key once
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found")

client = OpenAI(api_key=api_key)


def embed_question(question: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def search_index(index: faiss.Index, query_vector: np.ndarray, k=5):
    D, I = index.search(np.array([query_vector]), k)
    return I[0]  # return top k indices


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = f"""You are a weather data assistant. Use the context below to answer the question.
Be accurate, cite specific storms or years if applicable.

Context:
{context}

Question: {question}
Answer:"""
    return prompt


def get_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
