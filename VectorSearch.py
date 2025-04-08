from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json


class VectorSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the vector search class with a pre-trained embedding model.
        :param model_name: Name of the sentence transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.embeddings = None

    def ingest_file(self, file_path):
        """
        Ingest a file containing a list of strings (one string per line).
        :param file_path: Path to the text file to ingest.
        """
        with open(file_path, "r") as file:
            self.texts = [line.strip() for line in file if line.strip()]

        # Generate embeddings for all texts
        self.embeddings = self.model.encode(self.texts)
        print(f"Successfully ingested {len(self.texts)} items.")

    def save_embeddings(self, file_path):
        """
        Save texts and embeddings to a JSON file.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Please ingest data first.")

        data = {
            "texts": self.texts,
            "embeddings": self.embeddings.tolist()  # Convert NumPy array to list
        }
        with open(file_path, "w") as file:
            json.dump(data, file)

        print(f"Embeddings saved to {file_path}")

    def load_embeddings(self, file_path):
        """
        Load texts and embeddings from a JSON file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)

        self.texts = data["texts"]
        self.embeddings = np.array(data["embeddings"])  # Convert list back to NumPy array
        print(f"Loaded {len(self.texts)} embeddings from {file_path}")

    def search(self, query, top_k=1):
         """
         Search for the closest matches to the given query.
         :param query: The input text to search for.
         :param top_k: Number of top matches to return.
         :return: List of top_k (score, text) tuples.
         """
         #if not self.embeddings:
         #    raise ValueError("No data ingested. Please call ingest_file first.")

         # Generate embedding for the query
         query_embedding = self.model.encode([query])

         # Compute cosine similarities
         similarities = cosine_similarity(query_embedding, self.embeddings)[0]

         # Get top-k matches
         top_indices = np.argsort(similarities)[::-1][:top_k]
         results = [(similarities[i], self.texts[i]) for i in top_indices]

         return results
