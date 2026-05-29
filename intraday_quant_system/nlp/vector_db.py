"""
Milvus Lite Wrapper for FinBERT News Embeddings.
"""
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class NewsVectorDB:
    def __init__(self, db_path: str = "./milvus_lite.db", collection_name: str = "finbert_news", dim: int = 768):
        self.db_path = db_path
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self.connect()

    def connect(self):
        """Connect to Milvus Lite."""
        logger.info(f"Connecting to Milvus Lite at {self.db_path}")
        connections.connect("default", uri=self.db_path)
        self._init_collection()

    def _init_collection(self):
        """Initialize the collection for FinBERT embeddings."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} loaded.")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
                FieldSchema(name="ticker", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="sentiment", dtype=DataType.FLOAT),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="FinBERT News Embeddings Collection")
            self.collection = Collection(self.collection_name, schema)
            
            # Create an index on the embedding field
            index_params = {
                "metric_type": "L2",
                "index_type": "FLAT",
                "params": {}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Collection {self.collection_name} created and indexed.")
            
        self.collection.load()

    def insert_news(self, ticker: str, timestamp: float, sentiment: float, embedding: np.ndarray):
        """
        Insert a news embedding.
        
        Args:
            ticker: The stock ticker.
            timestamp: The timestamp of the news.
            sentiment: The sentiment score from FinBERT.
            embedding: The embedding vector from FinBERT.
        """
        if embedding.shape[0] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embedding.shape[0]}")
            
        data = [
            [timestamp],
            [ticker],
            [sentiment],
            [embedding.tolist()]
        ]
        
        res = self.collection.insert(data)
        self.collection.flush()
        return res

    def search_similar_news(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search for similar news based on the query embedding.
        
        Args:
            query_embedding: The embedding vector to search for.
            top_k: Number of results to return.
            
        Returns:
            List of matching records.
        """
        if query_embedding.shape[0] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {query_embedding.shape[0]}")
            
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["ticker", "timestamp", "sentiment"]
        )
        
        matches = []
        for hits in results:
            for hit in hits:
                matches.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "ticker": hit.entity.get("ticker"),
                    "timestamp": hit.entity.get("timestamp"),
                    "sentiment": hit.entity.get("sentiment")
                })
        return matches

    def close(self):
        """Disconnect from Milvus."""
        connections.disconnect("default")
