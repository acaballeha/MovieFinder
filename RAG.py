
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
            max_length=200,  # 🔹 Permite respuestas más largas
            min_length=50,   # 🔹 Evita respuestas demasiado cortas
            do_sample=True,  # 🔹 Habilita el muestreo
            temperature=0.2, # 🔹 Hace la generación más creativa
            top_p=0.9,       # 🔹 Sampling más natural
            repetition_penalty=1.7 # 🔹 Evita repetir frases
        )

    def generate_answer(self, query, top_k=1):
        """
        Recupera información y genera una respuesta en lenguaje natural.
        :param query: Pregunta del usuario.
        :param top_k: Número de documentos a recuperar.
        :return: Respuesta generada por el modelo.
        """
        #  Recuperar información con Dense Retriever
        retrieved_movies = self.retriever.search(query, top_k=top_k)

        #  Construir el contexto
        context = "\n".join([
            f"- {row['title']} ({row['release_date']}): {row['overview']}"
            for _, row in retrieved_movies.iterrows()
        ])

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
        index=resumen.find(":")
        resumen=resumen[index+1:]
        title = retrieved_movies.iloc[0]["title"]
        year = retrieved_movies.iloc[0]["release_date"]
        adult = retrieved_movies.iloc[0]["adult"]
       
        return resumen,  title, year, adult
    
    

if __name__ == "__main__":
    # Cargar el dataset
    pathCSV="./peliculasPopulares10k_CLEAN.csv"
    pathEmbeddings="./imdb_embeddings.npy"
 

    # Inicializar el Dense Retriever
    retriever = DR.DenseRetriever(pathEmbeddings=pathEmbeddings, pathCSV=pathCSV )

    # Inicializar el RAG QA
    rag = RAGQA(retriever)

    while True:
        query = input("Introduce tu pregunta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
        answer = rag.generate_answer(query)
        print(answer)
        print()
        print()
