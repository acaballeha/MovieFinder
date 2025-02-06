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

queryHistory = []


GENRES_IMDB = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

GENRES_IMDB_INVERTED = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Sci-Fi": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37
}

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
    topK = st.number_input("Number of near movies:", min_value=1, max_value=10, value=3)
    selected_genre = None  
    genre = st.selectbox("Select a genre:", ["All"] + list(GENRES_IMDB.values()))
    selected_genre = None if genre == "All" else GENRES_IMDB_INVERTED[genre]
    
   
    
    if st.button("Guess", key="recommendations_button"):
        # Carga del dataset
        pathCSV = "./peliculasPopulares_CLEAN.csv"
        pathEmbeddings = "./imdb_embeddings.npy"

       
        queryHistory.append(question)


        # Inicializa el Dense Retriever
        retriever = DR.DenseRetriever(pathEmbeddings=pathEmbeddings, pathCSV=pathCSV) 
        rag = RAGQA(retriever)   
        
        # Busca las películas más similares    
        peliculasMatch = rag.denseSearch(question, top_k=topK, genre=selected_genre)
        # Genera la respuesta
        cols = st.columns(3)
        for idx, movie in enumerate(rag.generate_answer(peliculasMatch)):
            if len(movie) == 5:
                resumen, title, year, adult, pathPoster = movie
                col = idx % 3
                with cols[col]:
                    st.image(pathPoster, caption=title)
                    strWrite = f"**{title}** ({year})"
                    if adult:
                        strWrite += " - **Adult content**"
                    st.markdown(strWrite)
                    st.markdown(resumen)
            else:
                st.warning("Unexpected movie data format.")


if __name__ == "__main__":
    main()
