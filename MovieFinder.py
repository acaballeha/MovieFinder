import streamlit as st
import pandas as pd
import DenseRetriever as DR
import ast
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
        <div style="color: #4d4d4d; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 18px; text-align: center; font-family: 'Georgia', serif;">
            Find the movie you do not remember the title
        </div>
        """,
        unsafe_allow_html=True
    )
    pathCSV = "./peliculas_CLEAN.csv"
    resultRetriever = DR.DenseRetriever(pathCSV=pathCSV)
    
    question = st.text_input("Provide a description:", "A man bitten by a spider")
    
    topK = st.sidebar.number_input("Number of near movies:", min_value=1, max_value=10, value=3)
    genreSelector = st.sidebar.selectbox("Select a genre:", ["All"] + list(GENRES_IMDB.values()))
    genreSelector = None if genreSelector == "All" else GENRES_IMDB_INVERTED[genreSelector]
    
    startYear = st.sidebar.number_input("Start year:", min_value=1900, max_value=2025, value=1900)
    endYear = st.sidebar.number_input("End year:", min_value=1900, max_value=2025, value=2025)
    
    if st.button("Guess", key="recommendations_button"):
        # Carga del dataset
        
        pathEmbeddings = None

        
        # Búsqueda de películas similares
        resultados = resultRetriever.search(question, top_k=topK, specificGenre=genreSelector, start_year=startYear, end_year=endYear)
        
        with st.container():
            cols = st.columns(3)
            for i, (_, row) in enumerate(resultados.iterrows()):
                with cols[i % 3]:
                    year = row['release_date'].split('-')[0] if row['release_date'] else "Unknown"
                    st.markdown(f"**{row['title']}** {year}")
                    genre_ids = ast.literal_eval(row['genre_ids']) if isinstance(row['genre_ids'], str) else row['genre_ids']
                    generos = ", ".join([GENRES_IMDB.get(genre_id, "Unknown") for genre_id in genre_ids])
                    year = row['release_date'].split('-')[0] if row['release_date'] else "Unknown"
                    caption = f"**Similarity:** {row['similarity']:.2f}"
                    st.image(row['pathPoster'], use_container_width=True, caption=caption)
                 
                    
                   
                    
              
               


if __name__ == "__main__":
    main()
