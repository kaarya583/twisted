import streamlit as st
from rag.chatbot import embed_question, search_index, build_prompt, get_response
from rag.retriever import load_faiss_index  # Assuming you have this
# Load FAISS index and text chunks once outside of any function or inside cache if big
index, retrieved_chunks = load_faiss_index()

st.title("🧠 Historical Weather Chatbot")

user_question = st.text_input("Ask a question about historical severe weather:")

if user_question:
    query_vector = embed_question(user_question)
    top_ids = search_index(index, query_vector, k=5)
    relevant_chunks = [retrieved_chunks[i] for i in top_ids]

    prompt = build_prompt(user_question, relevant_chunks)
    answer = get_response(prompt)

    st.markdown("### Answer:")
    st.write(answer)
