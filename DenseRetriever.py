import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import argparse
from sklearn.preprocessing import MinMaxScaler

class DenseRetriever:
    def __init__(self,  model_name="sentence-transformers/all-MiniLM-L6-v2", pathEmbeddings=None, pathCSV=None):
        """
        Inicializa el modelo de embeddings y almacena los embeddings en memoria.
        :param df: DataFrame con las columnas ["title", "overview"].
        :param model_name: Nombre del modelo de Hugging Face.
        """
        self.df = pd.read_csv(pathCSV)
        scaler = MinMaxScaler()
        self.df["popularity"] = scaler.fit_transform(self.df[["popularity"]])
        self.model = SentenceTransformer(model_name)
        
        self.embeddings = None

        # Concatenar "title + overview" para generar embeddings
        self.df["text"] = self.df["title"] + " - " + self.df["overview"]

        # Generar embeddings para las películas
        self._generate_embeddings(pathEmbeddings) # Se puede añadir un path

    def _generate_embeddings(self, pathEmbeddings=None):
        """Genera embeddings y los almacena en memoria."""
        if pathEmbeddings is not None and os.path.exists(pathEmbeddings):
            self.embeddings = np.load(pathEmbeddings)
            print(f"Embeddings cargados desde: {pathEmbeddings}")
        else:
            pathEmbeddings="./imdb_embeddings.npy"
            if os.path.exists(pathEmbeddings):
                print("Borrados embeddings anteriores...")
                os.remove(pathEmbeddings)
            print("Generando nuevos embeddings...")
            self.embeddings = self.model.encode(self.df["text"].tolist(), convert_to_numpy=True)
            self.save_embeddings(pathEmbeddings)
            print(f"Embeddings generados y almacenados en {pathEmbeddings}.")
            

    def save_embeddings(self, path):
        """Guarda los embeddings en un archivo .npy."""
        np.save(path, self.embeddings)
        print(f"Embeddings guardados en: {path}")


    def search(self, query, top_k=5):
        """
        Realiza una búsqueda utilizando similitud del coseno.
        :param query: Texto de búsqueda.
        :param top_k: Número de resultados a devolver.
        :return: DataFrame con los resultados ordenados por similitud.
        """
        print(f"🔍 Buscando: {query}")

        # Convertir la query en embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Calcular similitud del coseno entre la query y los embeddings de las películas
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Obtener los índices de los mejores resultados
        best_indices = np.argsort(similarities)[::-1][:top_k]

        # Recuperar las películas coincidentes
        results = self.df.iloc[best_indices].copy()
        results["similarity"] = similarities[best_indices]

        return results.sort_values(by="similarity", ascending=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Dense Retriever for Movie Embeddings")
    parser.add_argument("--pathCSV", type=str, required=True, help="Path to the CSV file containing movie data")
    parser.add_argument("--pathEmbeddings", type=str, required=False, help="Path to the .npy file containing precomputed embeddings")

    args = parser.parse_args()
    pathCSV = args.pathCSV
    pathEmbeddings = args.pathEmbeddings

    retriever = DenseRetriever(pathCSV=pathCSV, pathEmbeddings=pathEmbeddings)
  

    while True:
        query = input("Ingresa la descripción de la pelicula aquí (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
        results = retriever.search(query, top_k=10)
        print(results)

