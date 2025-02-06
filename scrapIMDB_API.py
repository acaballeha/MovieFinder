

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

import requests
import json
import time
import pandas as pd
import os
import argparse
from tqdm import tqdm

# Configuración de la API
API_KEY = "c436a0598ba40f517d94fa3c9cc217d6"  # Reemplaza con tu API Key de TMDB
BASE_URL = "https://api.themoviedb.org/3/movie/popular"
NUM_PELICULAS = 10000  # Número total de películas a descargar
PELICULAS_POR_PAGINA = 20  # TMDB devuelve 20 películas por página
paginas_a_descargar = (NUM_PELICULAS // PELICULAS_POR_PAGINA) + 1

# Argument parser configuration
parser = argparse.ArgumentParser(description="Descargar películas populares de TMDB y guardarlas en un archivo CSV.")
parser.add_argument("--path_save", type=str, default="./peliculasPopulares10k.csv", required=True, help="Ruta para guardar el archivo CSV de películas populares.")
args = parser.parse_args()

PATH_SAVE = args.path_save

# Ensure the directory exists
os.makedirs(os.path.dirname(PATH_SAVE), exist_ok=True)

def obtener_peliculas(n=NUM_PELICULAS):
    peliculas = []
    for pagina in tqdm(range(1, paginas_a_descargar + 1), desc="Descargando películas"):
        url = f"{BASE_URL}?api_key={API_KEY}&language=en-EN&page={pagina}"
        respuesta = requests.get(url)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            peliculas.extend(datos["results"])
            for pelicula in datos["results"]:
                poster_path = pelicula.get("poster_path")
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    poster_response = requests.get(poster_url)
                    if poster_response.status_code == 200:
                        poster_data = poster_response.content
                        titulo = pelicula["title"].replace(" ", "_")
                        año = pelicula["release_date"].split("-")[0]
                        poster_filename = f"./posters/{titulo}_{año}.jpg"
                        os.makedirs(os.path.dirname(poster_filename), exist_ok=True)
                        with open(poster_filename, "wb") as poster_file:
                            poster_file.write(poster_data)

        else:
            print(f"⚠ Error en la petición: {respuesta.status_code}")
            break
        # Pausa para no exceder los límites de la API (40 peticiones/10s)
        time.sleep(0.1)
        # Detener si alcanzamos el límite deseado
        if len(peliculas) >= n:
            break

    return peliculas[:n]

# Obtener las películas más populares
peliculas = obtener_peliculas(NUM_PELICULAS)

# Crear un DataFrame y guardar en un archivo CSV
df_peliculas = pd.DataFrame(peliculas)

df_peliculas.to_csv(PATH_SAVE, index=False, encoding="utf-8")

print(f"✅ Se han guardado {len(peliculas)} películas en {PATH_SAVE}")