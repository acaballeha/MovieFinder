import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import argparse
from sklearn.preprocessing import MinMaxScaler

class DenseRetriever:
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",  pathCSV=None):
        self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv(pathCSV)
        self.df["overview"] = self.df["title"] + " " + self.df["overview"]
        self.embeddings = None

    def _generate_embeddings(self):
        self.embeddings = self.model.encode(self.df["overview"].tolist(), convert_to_numpy=True)
        

    def save_embeddings(self, path):
        if self.embeddings is None:
            self._generate_embeddings()
        np.save(path, self.embeddings)
        

    def search(self, query, top_k=5, specificGenre=None, start_year=None, end_year=None):
        
        aux_df = self.df.copy()
       
        if specificGenre is not None:
            self.df = self.df[self.df["genre_ids"].astype(str).str.contains(str(specificGenre), na=False)]
        if start_year is not None:
            self.df = self.df[self.df["release_date"].str[:4].astype(int) > int(start_year)]
        
        if end_year is not None:
            self.df = self.df[self.df["release_date"].str[:4].astype(int) < int(end_year)]
        
        self._generate_embeddings()
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        cosine_similarities = cosine_similarity(query_embedding, self.embeddings)[0]
    
        best_indices = np.argsort(cosine_similarities)[::-1][:top_k]
        
        results = self.df.iloc[best_indices].copy()
        results["similarity"] = cosine_similarities[best_indices]
        
        self.df = aux_df
        
        return results.sort_values(by="similarity", ascending=False)
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate embeddings for a given CSV file.")
    parser.add_argument("pathCSV", type=str, help="Path to the input CSV file.")
    parser.add_argument("pathEmbeddings", type=str, help="Path to save the embeddings.")
    args = parser.parse_args()
    
    retriever = DenseRetriever(pathCSV=args.pathCSV)
    retriever.save_embeddings(args.pathEmbeddings)
    print(f"✅ Embeddings saved in {args.pathEmbeddings}")
    
    
    while True:
        query = input("Introduce tu pregunta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
        topK = retriever.search(query, top_k=3)
        
        print(topK[["title", "overview", "similarity"]])
