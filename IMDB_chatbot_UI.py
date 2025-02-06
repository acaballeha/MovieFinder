"""
How to run:

pip install streamlit

(remember to add the Path)

streamlit run IMDB_chatbot_UI.py
"""

import streamlit as st
import pandas as pd
from RAG import RAGQA
import DenseRetriever as DR

# Función para simular la respuesta del chatbot (por ejemplo, basado en IMDB)
def chatbot_response(query, genre):
    return f"**Mostrando resultados para la solicitud:** '{query}' **en el género:** '{genre}'."

# Interfaz de usuario de Streamlit
def main():
    # Estilo de fondo, texto y otros ajustes de apariencia (Modo claro)
    st.set_page_config(page_title="MovieFinder", page_icon="🎬", layout="centered")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f4f4;  /* Fondo blanco */
            color: #333333;            /* Color de texto oscuro */
            font-family: 'Arial', sans-serif;
        }
        h1 {
            font-family: 'Helvetica', sans-serif;
            color: #1a73e8;           /* Título de color azul */
            text-align: center;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            background-color: #ffffff;  /* Fondo blanco de los cuadros de texto */
            color: #333333;            /* Color de texto oscuro */
            border: 1px solid #ddd;    /* Borde gris claro */
        }
        .stTextInput>div>div>input:focus {
            border: 1px solid #1a73e8;  /* Borde azul cuando está seleccionado */
        }
        .stSelectbox>div>div>div>div>div>div>input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #ddd;
        }
        .stSelectbox>div>div>div>div>div>div>input:focus {
            border: 1px solid #1a73e8;
        }
        .stWarning {
            background-color: #ffcc00;  /* Fondo amarillo para advertencia */
            color: black;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton>button#recommendations_button {
            background-color: green;  /* Botón verde */
            color: white;            /* Texto en blanco para contraste */
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
            display: block;
            margin: 0 auto;         /* Centrar el botón */
        }
        .stButton>button#recommendations_button:hover {
            background-color: lightgreen;  /* Botón al pasar el mouse */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🎬 Movie Finder")
    st.markdown(
        """
        <div style="background-color: green; color: white; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 18px; text-align: center;">
            Find the movie you do not remember the title.
        </div>
        """,
        unsafe_allow_html=True
    )

    question = st.text_input("Provide a description:", "A man bitten by a spider")
    
    if st.button("Guess", key="recommendations_button"):
        # Carga del dataset
        pathCSV = "./peliculasPopulares10k_CLEAN.csv"
        pathEmbeddings = "./imdb_embeddings.npy"

        # Inicializa el Dense Retriever
        retriever = DR.DenseRetriever(pathEmbeddings=pathEmbeddings, pathCSV=pathCSV)

        # Inicializa el sistema de QA basado en RAG
        rag = RAGQA(retriever)

        # Genera la respuesta
        answer, title, year, adult = rag.generate_answer(question, top_k=1)
        if adult:
            adult = "Yes"
        else:
            adult = "No"
        st.write(f"**Movie Title:** {title} **Year:** {year.split('-')[0]} **Adult Only:** {adult}")        
        st.write(answer)


if __name__ == "__main__":
    main()
