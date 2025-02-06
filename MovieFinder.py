import streamlit as st
import pandas as pd
from RAG import RAGQA
import DenseRetriever as DR

def mainFrame():
    st.title("🎬 Movie Finder")
    st.write("This is a simple movie recommendation system based on the IMDb dataset.")
    st.write("Ask any question related to movies and I will do my best to provide you with relevant recommendations.")
    question = st.text_input("Ask me a question:")
    if st.button("Get Recommendations"):
        # Load the dataset
        pathCSV="./peliculasPopulares10k_CLEAN.csv"
        pathEmbeddings="./imdb_embeddings.npy"

        # Initialize the Dense Retriever
        retriever = DR.DenseRetriever(pathEmbeddings=pathEmbeddings, pathCSV=pathCSV)

        # Initialize the RAG-based QA system
        rag = RAGQA(retriever)

        # Generate the answer
        answer = rag.generate_answer(question, top_k=1)
        st.write(answer)
    
if __name__ == "__main__":
    mainFrame()
        
