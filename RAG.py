
from transformers import pipeline
import pandas as pd
import DenseRetriever as DR

class RAGQA:
    def __init__(self, retriever, model_name="google/flan-t5-large"):
        """
        Inicializa el sistema de Question Answering con RAG.
        :param retriever: Instancia de DenseRetriever.
        :param model_name: Modelo de generación de Hugging Face.
        """
        self.retriever = retriever
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            max_length=100,  # 🔹 Permite respuestas más largas
            min_length=20,   # 🔹 Evita respuestas demasiado cortas
            do_sample=True,  # 🔹 Habilita el muestreo
            temperature=0.3, # 🔹 Hace la generación más creativa
            top_p=0.9,       # 🔹 Sampling más natural
            repetition_penalty=1.5 # 🔹 Evita repetir frases
        )
        
    def denseSearch(self, query, top_k=1, genre: int = None):
        """
        Recupera información y genera una respuesta en lenguaje natural.
        :param query: Pregunta del usuario.
        :param top_k: Número de documentos a recuperar.
        :return: Respuesta generada por el modelo.
        """
        #  Recuperar información con Dense Retriever
        retrieved_movies = self.retriever.search(query, top_k, genre)
            
        return retrieved_movies

    def generate_answer(self, retrieved_movies: pd.DataFrame):
        #  Recuperar información con Dense Retriever
        for _, row in retrieved_movies.iterrows():
            #  Construir el contexto
            context = f"- {row['title']} ({row['release_date']}): {row['overview']}"

            #  Construir el prompt para el modelo generativo
            prompt = (
                f"CONTEXT:\n{context}\n\n"
                "TALKS ABOUT THE MAIN PLOT OF THE MOVIE.\n"
                "DO NOT MENTION, FILM TITLE, FILM RELEASE YEAR, DIRECTOR, ACTORS.\n"
                "FOCUS YOUR DESCRIPTION ON THE PLOT, GENRE, OR THEME. ONLY PROVIDING A SHORT RESUME.\n"
            )

            # Generar la respuesta
            response = self.generator(prompt, max_length=100, truncation=True)
            resumen = response[0]["generated_text"]
            index = resumen.rfind(":")
            resumen = resumen[index+1:].strip()
            title = row["title"]
            year = row["release_date"]
            adult = row["adult"]
            pathPoster = row["pathPoster"]

            yield resumen, title, year, adult, pathPoster
        
    

if __name__ == "__main__":
    # Cargar el dataset
    pathCSV="./peliculasPopulares_CLEAN.csv"
    pathEmbeddings="./imdb_embeddings.npy"
 

    # Inicializar el Dense Retriever
    retriever = DR.DenseRetriever(pathEmbeddings=pathEmbeddings, pathCSV=pathCSV )

    # Inicializar el RAG QA
    rag = RAGQA(retriever)

    while True:
        query = input("Introduce tu pregunta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
        topK = rag.denseSearch(query, top_k=3)
        for resumen, title, year, adult, pathPoster in rag.generate_answer(topK):
            if adult:
                adult = "Yes"
            else:
                adult = "No"
            print(f"**Movie Title:** {title} **Year:** {year.split('-')[0]} **Adult Only:** {adult}")
            print(resumen)
            print(f"Poster: {pathPoster}")
            print("--------------------------------------------------")
        
